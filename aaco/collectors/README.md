# Collectors Module

GPU and system telemetry data collection.

## Components

| Component | Description |
|-----------|-------------|
| `gpu.py` | GPU telemetry via ROCm SMI (clocks, power, temperature, utilization) |
| `system.py` | System metrics (CPU, memory, context switches) |
| `sampler.py` | Time-series sampling infrastructure |

## Collected Metrics

### GPU Telemetry

| Metric | Source | Unit |
|--------|--------|------|
| GPU Clock | rocm-smi | MHz |
| Memory Clock | rocm-smi | MHz |
| Power | rocm-smi | W |
| Temperature | rocm-smi | °C |
| Utilization | rocm-smi | % |
| VRAM Usage | rocm-smi | MB |

### System Telemetry

| Metric | Source | Unit |
|--------|--------|------|
| CPU Usage | psutil | % |
| Memory Usage | psutil | MB |
| Context Switches | psutil | count |

## Usage

```python
from aaco.collectors import GPUSampler, SystemSampler

gpu_sampler = GPUSampler(device_id=0, interval_ms=100)
system_sampler = SystemSampler(interval_ms=100)

with gpu_sampler, system_sampler:
    # Execute workload
    pass

gpu_data = gpu_sampler.get_samples()
system_data = system_sampler.get_samples()
```
