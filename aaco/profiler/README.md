# Profiler Module

rocprof integration for GPU kernel profiling.

## Components

| Component | Description |
|-----------|-------------|
| `rocprof.py` | rocprof wrapper and output parsing |
| `counters.py` | Hardware counter configuration |
| `trace.py` | HIP/HSA trace processing |

## Profiling Modes

| Mode | Data Collected |
|------|----------------|
| Trace | Kernel launches, memory operations, timestamps |
| Counters | Hardware performance counters |
| Combined | Both trace and counter data |

## Hardware Counters

Common counters for AMD Instinct:

| Counter | Description |
|---------|-------------|
| `SQ_WAVES` | Wavefronts executed |
| `SQ_INSTS_VALU` | VALU instructions |
| `SQ_INSTS_SALU` | SALU instructions |
| `TA_FLAT_READ_WAVEFRONTS` | Flat memory reads |
| `TA_FLAT_WRITE_WAVEFRONTS` | Flat memory writes |

## Usage

```python
from aaco.profiler import RocprofWrapper

profiler = RocprofWrapper(device_id=0)
profiler.configure(trace=True, counters=["SQ_WAVES", "SQ_INSTS_VALU"])

with profiler.profile():
    # Execute kernels
    pass

trace_data = profiler.get_trace()
counter_data = profiler.get_counters()
```
