# API Reference: Collectors Module

::: aaco.collectors
    options:
      show_root_heading: true
      show_source: true

## Overview

Collectors are responsible for gathering performance data during profiling sessions.

## Available Collectors

| Collector | Description | Requirements |
|-----------|-------------|--------------|
| `TimingCollector` | Execution time measurement | None |
| `CounterCollector` | GPU hardware counters | ROCm |
| `TraceCollector` | Kernel execution traces | ROCm |
| `MemoryCollector` | Memory usage tracking | None |
| `PowerCollector` | Power consumption | ROCm |
| `SystemCollector` | System metrics | None |
| `EBPFCollector` | Kernel-level metrics | Linux, root |

## TimingCollector

```python
from aaco.collectors import TimingCollector

collector = TimingCollector(
    precision="ns",  # 'ns', 'us', 'ms'
    clock="monotonic"  # 'monotonic', 'perf_counter'
)

# Collect timing
with collector:
    model.run()
    
timing = collector.get_results()
```

## CounterCollector

Collects GPU hardware performance counters.

```python
from aaco.collectors import CounterCollector

collector = CounterCollector(
    device_id=0,
    counters=[
        "GRBM_COUNT",
        "GRBM_GUI_ACTIVE",
        "SQ_WAVES",
        "TA_BUSY_CU_SUM"
    ]
)

# Available counter groups
print(CounterCollector.available_counters())
```

### Counter Groups

| Group | Counters | Description |
|-------|----------|-------------|
| `compute` | SQ_*, CU_* | Shader/compute unit activity |
| `memory` | MC_*, L2_* | Memory controller, L2 cache |
| `graphics` | GRBM_*, PA_* | Graphics pipe activity |
| `thermal` | TEMP_*, POWER_* | Thermal and power |

## TraceCollector

Collects kernel execution traces.

```python
from aaco.collectors import TraceCollector

collector = TraceCollector(
    device_id=0,
    trace_level="kernel",  # 'api', 'kernel', 'full'
    output_format="perfetto"
)

# Run with tracing
collector.start()
model.run()
traces = collector.stop()

# Export to Perfetto
traces.export("trace.perfetto")
```

## MemoryCollector

```python
from aaco.collectors import MemoryCollector

collector = MemoryCollector(
    track_gpu=True,
    track_cpu=True,
    sample_interval=0.01
)

memory_data = collector.collect()
print(f"Peak GPU: {memory_data.gpu_peak_mb} MB")
print(f"Peak CPU: {memory_data.cpu_peak_mb} MB")
```

## PowerCollector

```python
from aaco.collectors import PowerCollector

collector = PowerCollector(
    device_id=0,
    sample_interval=0.001
)

power_data = collector.collect()
print(f"Average: {power_data.average_watts} W")
print(f"Peak: {power_data.peak_watts} W")
```

## EBPFCollector

Kernel-level metrics via eBPF (requires root).

```python
from aaco.collectors import EBPFCollector

collector = EBPFCollector(
    probes=["sched", "irq", "syscall"]
)

# Scheduler interference index
sii = collector.scheduler_interference_index()

# Context switches during measurement
context_switches = collector.context_switches()
```

## Custom Collectors

Create custom collectors by extending `BaseCollector`:

```python
from aaco.collectors import BaseCollector

class MyCollector(BaseCollector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def start(self):
        """Start collection."""
        pass
        
    def stop(self) -> dict:
        """Stop collection and return data."""
        return {}
        
    def reset(self):
        """Reset collector state."""
        pass
```

Register your collector:

```python
from aaco import Observatory

obs = Observatory()
obs.register_collector("my_collector", MyCollector)

session = obs.profile(
    model="model.onnx",
    collectors=["timing", "my_collector"]
)
```
