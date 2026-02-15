# Runner Module

ONNX Runtime execution with multiple backend support.

## Supported Backends

| Backend | Provider | Platform |
|---------|----------|----------|
| MIGraphX | MIGraphXExecutionProvider | Linux (ROCm) |
| ROCm | ROCMExecutionProvider | Linux (ROCm) |
| CUDA | CUDAExecutionProvider | Linux/Windows |
| CPU | CPUExecutionProvider | All platforms |

## Execution Modes

| Mode | Description |
|------|-------------|
| Standard | Normal inference execution |
| Warmup | Discard initial iterations |
| Timed | Measure iteration latencies |
| Profiled | Integrated rocprof profiling |

## Usage

```python
from aaco.runner import ORTRunner

runner = ORTRunner(
    model_path="models/resnet50.onnx",
    provider="MIGraphXExecutionProvider",
    device_id=0
)

runner.warmup(iterations=10)
latencies = runner.run(iterations=100, return_latencies=True)
```
