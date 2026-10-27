import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent


def _project_root():
    # Editable and wheel installs both work when invoked from a project checkout.
    for candidate in (Path.cwd(), *Path.cwd().parents, PACKAGE_DIR.parents[1]):
        manifest = candidate / "pyproject.toml"
        if manifest.exists() and 'name = "jax-transformer-lab"' in manifest.read_text(encoding="utf-8"):
            return candidate
    return Path.cwd()


ROOT = _project_root()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def stats(samples):
    a = np.asarray(samples, dtype=float)
    if not len(a) or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Timing samples must be nonempty, finite, and nonnegative")
    mean = float(a.mean())
    return {
        "count": len(a),
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": mean,
        "std_ms": float(a.std()),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "cv": float(a.std() / mean) if mean else 0.0,
    }


def source_fingerprint():
    digest = hashlib.sha256()
    paths = [(f"src/jax_transformer_lab/{p.name}", p) for p in sorted(PACKAGE_DIR.glob("*.py"))]
    paths += [
        (str(p.relative_to(ROOT)).replace("\\", "/"), p) for p in sorted((ROOT / "configs").glob("*.json"))
    ]
    if (ROOT / "pyproject.toml").exists():
        paths.append(("pyproject.toml", ROOT / "pyproject.toml"))
    for name, path in paths:
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_text(args):
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def environment():
    import importlib.metadata

    import jax

    keys = (
        "JAX_PLATFORMS",
        "JAX_ENABLE_COMPILATION_CACHE",
        "JAX_COMPILATION_CACHE_DIR",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "TF_GPU_ALLOCATOR",
        "NVIDIA_TF32_OVERRIDE",
        "CUDA_MODULE_LOADING",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
    )
    packages = {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata.get("Name")
    }
    return {
        "timestamp": now(),
        "python": sys.version,
        "executable": sys.executable,
        "os": platform.platform(),
        "cpu": platform.processor(),
        "jax": jax.__version__,
        "devices": [{"platform": d.platform, "kind": d.device_kind, "id": d.id} for d in jax.devices()],
        "packages": packages,
        "env": {k: os.environ.get(k) for k in keys},
        "effective_jax": {
            "enable_compilation_cache": jax.config.jax_enable_compilation_cache,
            "default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        },
        "source_sha256": source_fingerprint(),
        "nvidia_smi": run_text(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,pstate,temperature.gpu",
                "--format=csv,noheader",
            ]
        ),
    }


def error_metrics(actual, expected, dtype):
    a, b = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    error = a - b
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {"passed": False, "finite": False}
    atol, rtol = (1e-3, 1e-3) if dtype == "fp32" else (0.12, 0.06)
    return {
        "passed": bool(np.allclose(a, b, atol=atol, rtol=rtol)),
        "finite": True,
        "max_abs_error": float(np.abs(error).max()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse": float(np.linalg.norm(error) / max(np.linalg.norm(b), 1e-12)),
        "atol": atol,
        "rtol": rtol,
    }
