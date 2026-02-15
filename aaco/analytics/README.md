# Analytics Module

Performance metrics computation, bottleneck classification, and root cause analysis.

## Components

| Component | Description |
|-----------|-------------|
| `metrics.py` | Derived metrics computation (KAR, PFI, LTS, HEU) |
| `attribution.py` | Graph-to-kernel probabilistic attribution |
| `classify.py` | Bottleneck classification engine |
| `rootcause.py` | Bayesian root cause analysis |

## Key Metrics

| Metric | Description |
|--------|-------------|
| KAR | Kernel Amplification Ratio — kernel count per ONNX node |
| PFI | Partition Fragmentation Index — graph partitioning quality |
| LTS | Launch Tax Score — CPU-GPU synchronization overhead |
| HEU | Hardware Envelope Utilization — percentage of peak capability |

## Usage

```python
from aaco.analytics import MetricsEngine, BottleneckClassifier

engine = MetricsEngine()
metrics = engine.compute(session_data)
classification = BottleneckClassifier().classify(metrics)
```
