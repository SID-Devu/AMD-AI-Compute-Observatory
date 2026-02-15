# Runner Module

ONNX Runtime model execution with multi-backend support.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               RUNNER MODULE                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                             ORT RUNNER                                          │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         ort_runner.py                                    │   │
│  │                                                                          │   │
│  │    ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐    │   │
│  │    │           │    │           │    │           │    │           │    │   │
│  │    │   Load    │───▶│  Session  │───▶│  Warmup   │───▶│  Measure  │    │   │
│  │    │   Model   │    │   Init    │    │   Phase   │    │   Phase   │    │   │
│  │    │           │    │           │    │           │    │           │    │   │
│  │    └───────────┘    └───────────┘    └───────────┘    └───────────┘    │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION PROVIDERS                                    │
│                                                                                 │
│    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│    │                 │  │                 │  │                 │              │
│    │   MIGraphX EP   │  │    ROCm EP      │  │    CUDA EP      │              │
│    │                 │  │                 │  │                 │              │
│    │  AMD Instinct   │  │   AMD GPU       │  │  NVIDIA GPU     │              │
│    │  Optimized      │  │   Direct        │  │  Reference      │              │
│    │                 │  │                 │  │                 │              │
│    └─────────────────┘  └─────────────────┘  └─────────────────┘              │
│                                                                                 │
│    ┌─────────────────┐  ┌─────────────────┐                                   │
│    │                 │  │                 │                                   │
│    │    CPU EP       │  │  OpenVINO EP    │                                   │
│    │                 │  │                 │                                   │
│    │   Fallback      │  │   Intel Opt     │                                   │
│    │                 │  │                 │                                   │
│    └─────────────────┘  └─────────────────┘                                   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MODEL REGISTRY                                        │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        model_registry.py                                 │   │
│  │                                                                          │   │
│  │    configs/models.yaml                                                   │   │
│  │    ├── resnet50:                                                        │   │
│  │    │   ├── path: models/resnet50.onnx                                   │   │
│  │    │   ├── input_shapes: {input: [1, 3, 224, 224]}                     │   │
│  │    │   └── dtype: float16                                               │   │
│  │    │                                                                     │   │
│  │    └── bert-base:                                                       │   │
│  │        ├── path: models/bert-base.onnx                                  │   │
│  │        ├── input_shapes: {input_ids: [1, 128]}                         │   │
│  │        └── dtype: int64                                                 │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Components

### ort_runner.py

ONNX Runtime execution engine:

```python
from aaco.runner.ort_runner import ORTRunner

runner = ORTRunner(
    model_path="model.onnx",
    backend="migraphx",
    device_id=0
)

results = runner.run(
    warmup=50,
    iterations=100,
    batch_size=1
)

print(f"Mean latency: {results.mean_ms:.2f}ms")
print(f"Throughput: {results.throughput:.1f} samples/s")
```

### Execution Phases

| Phase | Purpose | Output |
|-------|---------|--------|
| Load | Parse ONNX model | Graph metadata |
| Init | Create ORT session | Execution provider |
| Warmup | Stabilize execution | Discarded iterations |
| Measure | Collect timing | Latency samples |

### Backend Selection

| Backend | Provider | Recommended For |
|---------|----------|-----------------|
| migraphx | MIGraphXExecutionProvider | AMD Instinct (production) |
| rocm | ROCMExecutionProvider | AMD GPU (direct) |
| cuda | CUDAExecutionProvider | NVIDIA GPU |
| cpu | CPUExecutionProvider | Fallback |

## Output

```python
@dataclass
class InferenceResult:
    latencies_ms: List[float]      # Per-iteration latencies
    mean_ms: float                 # Mean latency
    std_ms: float                  # Standard deviation
    p50_ms: float                  # Median latency
    p99_ms: float                  # 99th percentile
    throughput: float              # Samples per second
    warmup_iterations: int
    measure_iterations: int
```
