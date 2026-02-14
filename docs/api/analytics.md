# API Reference: Analytics Module

::: aaco.analytics
    options:
      show_root_heading: true
      show_source: true

## Overview

The analytics module provides statistical analysis, bottleneck classification, and root cause inference.

## Statistics

### StatisticalSummary

```python
from aaco.analytics import StatisticalSummary

summary = StatisticalSummary(data)

print(summary.mean)
print(summary.median)
print(summary.std)
print(summary.percentile(95))
print(summary.cv)  # Coefficient of variation
```

### Outlier Detection

```python
from aaco.analytics import OutlierDetector

detector = OutlierDetector(method="iqr", threshold=1.5)
clean_data = detector.filter(data)
outliers = detector.detect(data)
```

Methods:
- `iqr`: Interquartile range (default)
- `zscore`: Z-score based
- `mad`: Median absolute deviation

## Bottleneck Classification

### BottleneckClassifier

```python
from aaco.analytics import BottleneckClassifier

classifier = BottleneckClassifier()
result = classifier.classify(session)

print(result.category)    # 'compute', 'memory', 'io', etc.
print(result.confidence)  # 0.0 - 1.0
print(result.evidence)    # Supporting evidence
```

### Categories

| Category | Description | Evidence |
|----------|-------------|----------|
| `compute_bound` | GPU compute limited | High SQ_BUSY, low memory BW |
| `memory_bound` | Memory bandwidth limited | High L2 miss, saturated HBM |
| `latency_bound` | Memory latency limited | High L2 hit but slow |
| `kernel_launch` | Launch overhead | Many small kernels |
| `host_bound` | CPU/host limited | CPU busy, GPU idle gaps |
| `io_bound` | Data transfer limited | High PCIe utilization |

## Root Cause Analysis

### BayesianRootCause

```python
from aaco.analytics import BayesianRootCause

analyzer = BayesianRootCause(
    prior="uniform",  # or "empirical"
    min_confidence=0.7
)

result = analyzer.analyze(session)

for cause in result.ranked_causes:
    print(f"{cause.name}: {cause.posterior:.2%}")
    print(f"  Evidence: {cause.evidence}")
```

### Root Cause Categories

```python
class RootCause(Enum):
    KERNEL_INEFFICIENCY = "kernel_inefficiency"
    MEMORY_ACCESS_PATTERN = "memory_access_pattern"
    OCCUPANCY_LIMITED = "occupancy_limited"
    REGISTER_PRESSURE = "register_pressure"
    SHARED_MEMORY_BANK_CONFLICT = "shared_memory_bank_conflict"
    WARP_DIVERGENCE = "warp_divergence"
    LAUNCH_OVERHEAD = "launch_overhead"
    HOST_DEVICE_SYNC = "host_device_sync"
    DATA_TRANSFER = "data_transfer"
    THERMAL_THROTTLING = "thermal_throttling"
```

## Drift Detection

### DriftDetector

```python
from aaco.analytics import DriftDetector

detector = DriftDetector(
    method="ewma_cusum",
    alpha=0.3,
    threshold=5.0
)

# Check for drift
result = detector.detect(
    baseline=baseline_session,
    current=current_session
)

if result.has_drift:
    print(f"Drift detected at: {result.drift_point}")
    print(f"Magnitude: {result.magnitude:.2%}")
```

### Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `ewma` | Exponentially weighted moving average | Gradual drift |
| `cusum` | Cumulative sum | Step changes |
| `ewma_cusum` | Combined (default) | General purpose |
| `page_hinkley` | Page-Hinkley test | Online detection |

## Hardware Envelope

### HardwareEnvelopeAnalyzer

```python
from aaco.analytics import HardwareEnvelopeAnalyzer

analyzer = HardwareEnvelopeAnalyzer(device_id=0)

# Calibrate with microbenchmarks
analyzer.calibrate()

# Analyze session
heu = analyzer.analyze(session)

print(f"Compute utilization: {heu.compute:.1%}")
print(f"Memory BW utilization: {heu.memory_bandwidth:.1%}")
print(f"Overall HEU: {heu.overall:.1%}")
```

## Attribution

### KernelAttributor

Map graph operations to kernels.

```python
from aaco.analytics import KernelAttributor

attributor = KernelAttributor()
attribution = attributor.attribute(session)

for op in attribution.operations:
    print(f"{op.name}:")
    for kernel in op.kernels:
        print(f"  {kernel.name}: {kernel.kar:.2f} KAR")
```

Metrics:
- **KAR**: Kernel Attribution Ratio
- **PFI**: Performance Fraction Index
- **LTS**: Latency Time Share
