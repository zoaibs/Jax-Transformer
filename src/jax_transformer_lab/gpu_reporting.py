"""GPU-only findings, derived from saved records; never rewrites historical CPU results."""

import json
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .util import now, source_fingerprint, write_json

COLORS = {"components": "#b27a20", "whole": "#346b9e", "packed": "#148574", "cudnn": "#8556ad"}


def records(root):
    output = []
    for manifest_path in sorted(root.rglob("manifest.json")):
        if "test_tmp" in manifest_path.relative_to(root).parts:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for job in manifest.get("jobs", []):
            path = manifest_path.parent / job["result"]
            item = json.loads(path.read_text(encoding="utf-8"))
            item["_path"] = path
            if item.get("status") == "ok":
                platforms = {d["platform"] for d in item["environment"]["devices"]}
                if item["spec"]["backend"] != "gpu" or platforms != {"gpu"}:
                    raise ValueError(f"GPU report cannot include a CPU result: {path}")
            output.append(item)
    return output


def key(record):
    s = record["spec"]
    m = s["model"]
    return (
        m["hidden"],
        m["heads"],
        m.get("layers", 1),
        m.get("expansion", 4),
        m["dtype"],
        s["batch"],
        s["sequence"],
    )


def comparisons(rows, baseline, candidate):
    groups = {}
    for row in rows:
        groups.setdefault((key(row), row["spec"].get("trial", 1)), {})[row["spec"]["variant"]] = row
    return [
        (g[baseline], g[candidate], g[baseline]["warm"]["median_ms"] / g[candidate]["warm"]["median_ms"])
        for g in groups.values()
        if baseline in g and candidate in g
    ]


def summarize_pairs(pairs, root):
    """Ratios of process medians; per-trial ratios are retained separately."""
    first = pairs[0][0]["spec"]
    useful = first["batch"] * first["sequence"]
    baseline_ms = statistics.median(a["warm"]["median_ms"] for a, _, _ in pairs)
    candidate_ms = statistics.median(b["warm"]["median_ms"] for _, b, _ in pairs)
    baseline_queued = statistics.median(a["queued"]["per_batch"]["median_ms"] for a, _, _ in pairs)
    candidate_queued = statistics.median(b["queued"]["per_batch"]["median_ms"] for _, b, _ in pairs)
    depths = {r["queued"]["depth"] for pair in pairs for r in pair[:2]}
    if len(depths) != 1:
        raise ValueError("Cannot compare different queued submission depths")
    return {
        "shape": {
            "batch": first["batch"],
            "sequence": first["sequence"],
            "hidden": first["model"]["hidden"],
            "heads": first["model"]["heads"],
        },
        "dtype": first["model"]["dtype"],
        "baseline": pairs[0][0]["spec"]["variant"],
        "candidate": pairs[0][1]["spec"]["variant"],
        "trials": len(pairs),
        "baseline_median_ms": baseline_ms,
        "candidate_median_ms": candidate_ms,
        "speedup": baseline_ms / candidate_ms,
        "paired_trial_speedups": [ratio for _, _, ratio in pairs],
        "candidate_tokens_per_second": useful / (candidate_ms / 1000),
        "queue_depth": next(iter(depths)),
        "baseline_queued_ms_per_batch": baseline_queued,
        "candidate_queued_ms_per_batch": candidate_queued,
        "queued_throughput_ratio": baseline_queued / candidate_queued,
        "baseline_queued_tokens_per_second": useful / (baseline_queued / 1000),
        "candidate_queued_tokens_per_second": useful / (candidate_queued / 1000),
        "paired_trial_queued_ratios": [
            a["queued"]["per_batch"]["median_ms"] / b["queued"]["per_batch"]["median_ms"] for a, b, _ in pairs
        ],
        "source_files": [str(r["_path"].relative_to(root)) for pair in pairs for r in pair[:2]],
    }


def generate_gpu(root, output):
    root, output = Path(root), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = records(root)
    successful = [r for r in rows if r["status"] == "ok"]
    main = [r for r in successful if r["_path"].parent.name == "main" and r["kind"] == "case"]
    confirm = [r for r in successful if r["_path"].parent.name == "confirm" and r["kind"] == "case"]
    memory = [r for r in successful if r["kind"] == "memory"]
    workloads = [r for r in successful if r["kind"] == "workload"]
    policy_order = {"exact": 0, "bucketed": 1, "fine_buckets": 2, "refined_64": 3}
    workloads.sort(key=lambda r: policy_order.get(r["spec"].get("name"), 99))
    failures = [r for r in rows if r["status"] != "ok"]
    if not main:
        raise ValueError("Run the GPU main suite before generating findings")
    env = main[0]["environment"]
    hardware = env["devices"][0]["kind"]
    figures = output.parent / "gpu_figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "savefig.facecolor": "white",
        }
    )
    figure_names = []

    def save(fig, name, title, note=""):
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.text(
            0.5,
            0.012,
            hardware + " · WSL2 · synthetic inference" + (" · " + note if note else ""),
            ha="center",
            fontsize=8,
            color="#555555",
        )
        fig.tight_layout(rect=(0, 0.055, 1, 0.94))
        fig.savefig(figures / (name + ".png"), dpi=170)
        fig.savefig(figures / (name + ".svg"))
        plt.close(fig)
        figure_names.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for dtype, ax in zip(("fp32", "bf16"), axes, strict=True):
        for variant, color in COLORS.items():
            subset = sorted(
                [
                    r
                    for r in main
                    if r["spec"]["model"]["hidden"] == 256
                    and r["spec"]["batch"] == 1
                    and r["spec"]["model"]["dtype"] == dtype
                    and r["spec"]["variant"] == variant
                ],
                key=lambda r: r["spec"]["sequence"],
            )
            if subset:
                ax.plot(
                    [r["spec"]["sequence"] for r in subset],
                    [r["warm"]["median_ms"] for r in subset],
                    marker="o",
                    color=color,
                    label=variant,
                )
        ax.set(xlabel="Sequence length", ylabel="Warm synchronized median (ms)", title=dtype.upper())
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    save(fig, "warm_latency", "GPU inference latency across sequence lengths", "B=1, D=256, H=4")

    subset = [
        r
        for r in main
        if r["spec"]["batch"] == 1
        and r["spec"]["sequence"] == 1024
        and r["spec"]["model"]["hidden"] == 256
        and r["spec"]["model"]["dtype"] == "bf16"
    ]
    if subset:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
        names = [r["spec"]["variant"] for r in subset]
        axes[0].bar(names, [r["tokens_per_second"] for r in subset], color=[COLORS[n] for n in names])
        axes[0].set(ylabel="Tokens/s from synchronized median", title="One batch, then synchronization")
        queued = [r for r in subset if r.get("queued")]
        axes[1].bar(
            [r["spec"]["variant"] for r in queued],
            [1024 / (r["queued"]["per_batch"]["median_ms"] / 1000) for r in queued],
            color=[COLORS[r["spec"]["variant"]] for r in queued],
        )
        axes[1].set(
            ylabel="Tokens/s from queued group completion",
            title="Eight independent batches, then synchronization",
        )
        save(
            fig,
            "throughput",
            "Dispatch/synchronization boundaries affect measured throughput",
            "B=1, S=1024, BF16",
        )

    subset_mem = [
        r for r in memory if r["spec"]["model"]["hidden"] == 256 and r["spec"]["model"]["dtype"] == "bf16"
    ]
    if subset_mem:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
        names = [r["spec"]["variant"] for r in subset_mem]
        axes[0].bar(
            names,
            [r["executable_memory"]["estimated_buffer_bytes"] / 2**20 for r in subset_mem],
            color=[COLORS[n] for n in names],
        )
        axes[0].set(ylabel="Compiler-planned buffers (MiB)", title="Arguments + output + temporary - aliases")
        values = [r["snapshots"]["after_warm"].get("peak_bytes_in_use") for r in subset_mem]
        axes[1].bar(
            [n for n, v in zip(names, values, strict=True) if v is not None],
            [v / 2**20 for v in values if v is not None],
            color="#9a647f",
        )
        axes[1].set(
            ylabel="Allocator high-water bytes in use (MiB)",
            title="Includes initialization/compilation/autotuning",
        )
        save(
            fig,
            "memory",
            "Two different memory measurements; neither is a pure inference peak",
            "B=1, S=1024, BF16",
        )

    if workloads:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
        names = [r["spec"]["name"] for r in workloads]
        axes[0].bar(names, [r["cold_workload_ms"] / 1000 for r in workloads], color="#346b9e")
        axes[0].set(ylabel="Cold completion (seconds)", title="Includes uploads and new-shape compilation")
        axes[1].bar(names, [r["warm_compute"]["median_ms"] for r in workloads], color="#148574")
        axes[1].set(ylabel="Warm compute-only completion (ms)", title="Same ordered trace, resident inputs")
        for ax in axes:
            ax.tick_params(axis="x", rotation=18)
        save(fig, "variable_lengths", "GPU bucketing trades compilation reuse against padding")

    def link(row):
        return "[raw](" + os.path.relpath(row["_path"], output.parent).replace("\\", "/") + ")"

    fused_pairs = comparisons(confirm, "packed", "cudnn")
    primary = [(a, b, ratio) for a, b, ratio in fused_pairs if key(a) == (256, 4, 1, 4, "bf16", 1, 1024)]
    headline = None
    if primary:
        headline = summarize_pairs(primary, root)
    fusion_pairs = [
        (a, b, ratio)
        for a, b, ratio in comparisons(confirm, "components", "whole")
        if key(a) == (256, 4, 1, 4, "bf16", 1, 1024)
    ]
    fusion = summarize_pairs(fusion_pairs, root) if fusion_pairs else None
    lines = [
        "# GPU inference findings",
        "",
        f"Generated {now()} from `{root}`.",
        "",
        "## Outcome",
        "",
        f"Actual GPU execution verified on **{hardware}** under Ubuntu/WSL2. "
        "The historical 3.46x CPU cold-start result is preserved and is not a GPU speedup.",
        "",
    ]
    if fusion:
        lines += [
            f"**Whole-block JIT versus component JIT**, at B=1, S=1024, D=256, H=4 in BF16: "
            f"{fusion['baseline_median_ms']:.4f} → {fusion['candidate_median_ms']:.4f} ms synchronized "
            f"latency ({fusion['speedup']:.2f}x ratio). With {fusion['queue_depth']} independent batches "
            f"queued, throughput was **{fusion['baseline_queued_tokens_per_second']:,.0f} → "
            f"{fusion['candidate_queued_tokens_per_second']:,.0f} tokens/s "
            f"({fusion['queued_throughput_ratio']:.2f}x)** across {fusion['trials']} fresh-process comparisons. "
            "These variants share the same precision policy and mathematical architecture.",
            "",
        ]
    if headline:
        lines += [
            f"At B=1, S=1024, D=256, H=4 in BF16, {headline['trials']} fresh-process comparisons gave "
            f"**{headline['baseline_median_ms']:.4f} ms packed explicit attention versus "
            f"{headline['candidate_median_ms']:.4f} ms cuDNN attention**, a "
            f"**{headline['speedup']:.2f}x** ratio of process-median latencies. "
            f"The candidate's derived throughput is {headline['candidate_tokens_per_second']:,.0f} tokens/s.",
            f"Queued throughput at depth {headline['queue_depth']} was "
            f"{headline['baseline_queued_tokens_per_second']:,.0f} → "
            f"{headline['candidate_queued_tokens_per_second']:,.0f} tokens/s "
            f"({headline['queued_throughput_ratio']:.2f}x candidate/baseline throughput ratio).",
            "",
            "This is a fused implementation and precision-policy comparison: explicit attention keeps "
            "logits/probabilities/value accumulation in FP32; cuDNN operates on BF16 Q/K/V with its "
            "backend-specific accumulation and normalization. It is not a same-kernel speedup at "
            "identical intermediate precision.",
            "",
        ]
    lines += [
        "## Environment and methods",
        "",
        f"- OS: {env['os']}; Python {env['python'].splitlines()[0]}; JAX/jaxlib {env['jax']}.",
        "- Local RTX 5070 Laptop GPU, approximately 8 GiB VRAM. Driver and exact CUDA/cuDNN package versions are in every raw record and the Linux lock file.",
        "- Isolated `.venv-linux`; the original Windows environment and CPU output files are not replaced.",
        "- Fresh sequential workers; persistent JAX compilation cache disabled; preallocation disabled. Driver/library caches are not reset.",
        "- Main timings: individual synchronized batch calls on resident inputs; each record has sample count, warmup duration, raw samples, median, p95 and CV. Host dispatch and synchronization are included.",
        f"- At least 20 warmup calls plus a minimum {main[0]['spec'].get('warmup_seconds', 0):g} s warmup; confirmation uses 1 s and 200 samples. Twenty additional warm calls follow the external telemetry query. GPU snapshots are outside timing regions.",
        "- Queued timings separately enqueue eight independent executions of the same device-resident batch and wait for all outputs. They are group-average completion measurements, not request-level latency or a production serving benchmark.",
        "- Trace/lower/compile and first executable invocation are separated where supported; startup and transfers are excluded from those cold-path sums.",
        "- FP32 denotes storage/accumulation, with operand precision DEFAULT. Backend selection may use reduced-precision multiplication. Dense outputs and sensitive explicit-attention operations accumulate in FP32.",
        "- Every case is checked against the same seeded, unquantized FP32 master-weight path. BF16 tolerances remain atol=0.12, rtol=0.06; normalized error is reported. Padding and empty masks have GPU tests.",
        "- No clock/power settings, driver, security policy, or reboot changes were made during this extension.",
        "",
        "## Main GPU matrix",
        "",
        "| Variant | dtype | B | S | D | Median ms | p95 ms | CV % | Tokens/s | NRMSE vs FP32 | Source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in main:
        s, w = r["spec"], r["warm"]
        lines += [
            f"| {s['variant']} | {s['model']['dtype']} | {s['batch']} | {s['sequence']} | {s['model']['hidden']} | "
            f"{w['median_ms']:.4f} | {w['p95_ms']:.4f} | {w['cv'] * 100:.1f} | {r['tokens_per_second']:,.0f} | "
            f"{r['validation']['normalized_rmse']:.6f} | {link(r)} |"
        ]
    lines += [
        "",
        "## Fresh-process confirmation",
        "",
        "Trials alternate forward and reversed configuration order. Ratios below 1 mean the candidate was slower. "
        "Latency ratio is baseline time / candidate time; queued ratio is candidate throughput / baseline throughput. "
        "These are three process observations, not a confidence interval.",
        "",
        "| Comparison | dtype | B | S | Trial | Baseline ms | Candidate ms | Latency ratio | Queued throughput ratio | Sources |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for baseline, candidate in (("components", "whole"), ("whole", "packed"), ("packed", "cudnn")):
        for a, b, ratio in comparisons(confirm, baseline, candidate):
            s = a["spec"]
            lines += [
                f"| {baseline} / {candidate} | {s['model']['dtype']} | {s['batch']} | {s['sequence']} | "
                f"{s.get('trial', 1)} | {a['warm']['median_ms']:.4f} | {b['warm']['median_ms']:.4f} | "
                f"{ratio:.3f}x | {a['queued']['per_batch']['median_ms'] / b['queued']['per_batch']['median_ms']:.3f}x | {link(a)} {link(b)} |"
            ]
    lines += [
        "",
        "## Optimization investigation",
        "",
        "The main sweep showed that component boundaries limited queued throughput even when the "
        "GPU arithmetic and precision were unchanged. The follow-up confirmation therefore also compared "
        "component JIT with whole-block JIT at the same S=1024 point in three fresh processes. "
        "Both use separate Q/K/V projections and explicit FP32 attention intermediates. "
        "This tests compilation/dispatch boundaries without the cuDNN intermediate-precision confound.",
        "",
        "The initial GPU probe exposed a cuDNN integration error: its bias/mask path required both sequence "
        "axes, while the existing [B,1,1,S] broadcast mask had only one query position. The cuDNN path "
        "now explicitly represents [B,1,S,S] masking. GPU tests verify padding isolation, arbitrary mask "
        "holes, an empty batch row, and agreement with the explicit implementation.",
        "",
        "The explicit implementation materializes attention scores/probabilities and uses FP32 attention "
        "contractions even for BF16 model storage. The targeted candidate routes attention through "
        "cuDNN while retaining packed projections, weights and residual architecture. Saved GPU traces, "
        "optimized HLO and memory estimates provide evidence of the resulting execution structure. "
        "Repeated matched timings determine its measured value; no custom-kernel performance claim is made.",
        "",
    ]
    profiles = list(root.glob("profiles/*/summary.json"))
    verified_profiles = 0
    profile_counts = {}
    for path in profiles:
        data = json.loads(path.read_text())
        verified = data.get("gpu_activity_verified", False)
        verified_profiles += int(verified)
        trace_count = sum(t.get("gpu_duration_events", 0) for t in data.get("traces", []))
        profile_counts[data["variant"]] = trace_count
        lines += [
            f"- {data['variant']}: trace status `{data['trace_status']}`, GPU activity verified **{verified}**, "
            f"{trace_count} GPU duration events. [Summary]({os.path.relpath(path, output.parent).replace(chr(92), '/')})."
        ]
        tops = [t.get("top_gpu_events", []) for t in data.get("traces", [])]
        for name, metric in tops[0][:4] if tops else []:
            lines += [
                f"  - `{name}`: {metric['count']} events, {metric['summed_duration_us']:.1f} summed microseconds."
            ]
    lines += [
        "",
        "Event sums may overlap across streams and nested regions. They are not elapsed time, "
        "utilization, bandwidth, or occupancy. HLO instruction counts are not GPU kernel counts. "
        "Host-only captures remain explicitly unverified.",
        "",
        f"At the common profile point, component JIT produced {profile_counts.get('components', 'unavailable')} "
        f"GPU duration events and whole-block JIT produced {profile_counts.get('whole', 'unavailable')}. "
        "Equal event counts would not explain the throughput gain through fewer GPU events. The result is "
        "consistent with reduced Python/XLA dispatch boundaries and whole-program command-buffer execution. "
        "This is an interpretation of the code, timings and traces, not an occupancy or bandwidth measurement.",
        "",
        "## Memory: dedicated workers",
        "",
        "These workers allocate no FP32 reference model and keep no queued-output list. Compiler-planned "
        "buffers are a static estimate. Allocator high-water counters include backend initialization, "
        "compilation and autotuning; they were not reset and are not described as inference-only peaks. "
        "Reserved bytes are separately stored when available. NVIDIA snapshots are whole-device values "
        "that can include unrelated Windows activity. WSL/WDDM process memory may be unavailable.",
        "The backend also reports pool_bytes in the raw counters. Its bytes_reserved field can be zero "
        "while the allocator pool is nonzero; neither zero reserved bytes nor live-array bytes represents "
        "total device/process VRAM. No inference-only peak counter was reset or claimed.",
        "",
        "| Variant | dtype | B | S | D | Planned buffers MiB | Temporary MiB | Allocator bytes in use MiB | Lifetime high-water MiB | Source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]

    def mib(value):
        return "unavailable" if value is None else f"{value / 2**20:.2f}"

    for r in memory:
        s, m, a = r["spec"], r["executable_memory"], r["snapshots"]["after_warm"]
        lines += [
            f"| {s['variant']} | {s['model']['dtype']} | {s['batch']} | {s['sequence']} | {s['model']['hidden']} | "
            f"{mib(m.get('estimated_buffer_bytes'))} | {mib(m.get('temp_size_in_bytes'))} | "
            f"{mib(a.get('bytes_in_use'))} | {mib(a.get('peak_bytes_in_use'))} | {link(r)} |"
        ]
    lines += [
        "",
        "## Variable sequence lengths",
        "",
        "The original 25-request seed-42 trace and request-order batches of two are replayed on GPU. "
        "Inputs and valid token values match across policies, including the final partial batch. "
        "Cold completion includes input placement and explicit per-new-shape compilation; startup, host "
        "generation and parameter upload are excluded. Warm compute uses resident inputs; warm e2e "
        "includes input placement. Useful tokens exclude padding. These results are independent of the "
        "historical CPU 3.46x result.",
        "",
        "| Policy | Explicit compile calls | Extra tokens % | Cold s | Warm compute ms | Warm e2e ms | Useful tokens/s | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in workloads:
        lines += [
            f"| {r['spec']['name']} | {r['observed_explicit_compile_calls']} | {r['padding_overhead_fraction'] * 100:.1f} | "
            f"{r['cold_workload_ms'] / 1000:.3f} | {r['warm_compute']['median_ms']:.3f} | "
            f"{r['warm_e2e']['median_ms']:.3f} | {r['warm_useful_tokens_per_second']:,.0f} | {link(r)} |"
        ]
    lines += [
        "",
        "## Limitations and unsuccessful experiments",
        "",
        "- The baseline cuDNN probe failed before the mask fix; its error is preserved below. FP32 cuDNN is explicitly unsupported in this project's BF16 fused-attention variant.",
        "- Small synchronized calls can be dominated by host/WSL dispatch and synchronization. Queued throughput has different boundaries and must not be substituted for individual-call latency.",
        "- Laptop power management, thermals, Windows activity and unreset driver caches affect measurements. Telemetry snapshots do not capture every transient.",
        "- Synthetic random weights do not establish model quality. BF16 agreement does not guarantee task-level accuracy.",
        "- Cold workload timings have one observation per policy. No serving arrival model or request-latency SLO is evaluated.",
        "- Native-Windows CPU regression initially encountered an existing Application Control block on `_lapack.pyd`; no security settings were changed. CPU regression is also tested through WSL with CPU explicitly selected.",
        "",
    ]
    for r in failures:
        lines += [f"- {link(r)}: {r.get('error', 'worker failure')[:400]}"]
    diagnostics = root / "unsupported_fp32_cudnn.json"
    if diagnostics.exists():
        lines += [
            f"- Unsupported FP32 cuDNN diagnostic: [record]({os.path.relpath(diagnostics, output.parent).replace(chr(92), '/')})."
        ]
    provenance_note = root / "source_snapshot_note.json"
    if provenance_note.exists():
        lines += [
            f"- A functional smoke run with a source-snapshot mismatch was preserved separately and rerun. "
            f"It is not part of the main measurements. [Provenance note]({os.path.relpath(provenance_note, output.parent).replace(chr(92), '/')})."
        ]
    lines += ["", "## Figures", ""]
    for name in figure_names:
        lines += [f"![{name}](gpu_figures/{name}.png)", ""]
    acceptance = root / "acceptance.json"
    if acceptance.exists():
        data = json.loads(acceptance.read_text())
        lines += [
            "## Final verification and accounting",
            "",
            "```json",
            json.dumps(data, indent=2),
            "```",
            "",
        ]
    lines += [
        "## Run commands",
        "",
        "Ubuntu terminal; activation is optional:",
        "",
        "```bash",
        "python3 scripts/run_gpu.py doctor",
        "python3 scripts/run_gpu.py test",
        "python3 scripts/run_gpu.py smoke",
        "python3 scripts/run_gpu.py full",
        f"python3 scripts/run_gpu.py report --output {root}",
        "```",
        "",
        "From PowerShell, `.\\.venv\\Scripts\\python.exe scripts/run_gpu.py smoke` bridges to Ubuntu. "
        "If using bare `python`, first run `source .venv-linux/bin/activate`. System Ubuntu Python does "
        "not need JAX installed. Exact GPU packages are pinned in `requirements-linux.lock.txt`.",
        "",
        "## References",
        "",
        "- [JAX GPU installation](https://docs.jax.dev/en/latest/installation.html)",
        "- [JAX attention API](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html)",
        "- [JAX profiling and benchmarking](https://docs.jax.dev/en/latest/201/profiling.html)",
        "- [JAX multiplication precision](https://docs.jax.dev/en/latest/201/precision.html)",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "generated": now(),
        "gpu": hardware,
        "main_cases": len(main),
        "confirmation_cases": len(confirm),
        "memory_workers": len(memory),
        "workloads": len(workloads),
        "recorded_failures": len(failures),
        "verified_gpu_profiles": verified_profiles,
        "headline": headline,
        "confirmed_whole_block": fusion,
        "source_sha256": source_fingerprint(),
    }
    write_json(output.with_suffix(".json"), summary)
    return output
