from jax_transformer_lab.profiling import summarize_events
from jax_transformer_lab.telemetry import memory_description


def test_host_cuda_launch_is_not_gpu_trace_evidence():
    events = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "/host:CPU"}},
        {"ph": "X", "pid": 1, "name": "cudaLaunchKernel", "dur": 20},
    ]
    assert summarize_events(events)["gpu_duration_events"] == 0
    events += [
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "/device:GPU:0"}},
        {"ph": "X", "pid": 2, "name": "gemm", "dur": 10},
    ]
    assert summarize_events(events)["gpu_duration_events"] == 1


def test_missing_memory_counters_remain_unavailable():
    absent = memory_description(None)
    assert absent["peak_bytes_in_use"] is None
    assert absent["bytes_reserved"] is None
    present = memory_description({"bytes_in_use": 123, "peak_bytes_in_use": 456})
    assert present["bytes_in_use"] == 123
    assert present["peak_bytes_in_use"] == 456
    assert "including compilation" in present["peak_scope"]
