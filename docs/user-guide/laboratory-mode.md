# Laboratory Mode

Laboratory Mode provides **deterministic, reproducible profiling** by controlling system variability.

## Why Laboratory Mode?

Without isolation, measurements vary due to:

- CPU frequency scaling
- Background processes
- Memory pressure
- Thermal throttling
- Interrupt handling

Laboratory Mode eliminates these variables for scientific measurement.

## Quick Enable

```bash
# CLI
aaco profile --model model.onnx --lab-mode

# Python
session = obs.profile(model="model.onnx", lab_mode=True)
```

## Features

### CPU Isolation

```yaml
laboratory:
  cpu_isolation:
    enabled: true
    cores: [4, 5, 6, 7]
    pin_threads: true
```

- Pins workload to isolated CPU cores
- Prevents thread migration
- Uses `taskset` and `numactl`

### GPU Clock Lock

```yaml
laboratory:
  gpu:
    clock_lock: true
    target_frequency: max
```

- Locks GPU clock to fixed frequency
- Prevents dynamic frequency scaling
- Uses `rocm-smi --setperflevel`

### Process Isolation

```yaml
laboratory:
  process:
    nice: -20
    cgroup_isolation: true
```

- Maximum scheduling priority
- cgroups v2 resource isolation
- Memory locking (mlock)

### System Preparation

```yaml
laboratory:
  system:
    disable_turbo: true
    set_governor: performance
    drop_caches: true
```

- Disables CPU turbo boost
- Sets performance governor
- Clears filesystem caches

## Requirements

Laboratory Mode requires:

| Requirement | Why |
|-------------|-----|
| Root/sudo | CPU governor, nice values |
| cgroups v2 | Process isolation |
| ROCm 6.0+ | GPU clock control |
| Linux 5.15+ | Full eBPF support |

## Usage Examples

### Basic Lab Mode

```python
from aaco import Observatory, LabConfig

obs = Observatory()

# Use defaults
session = obs.profile(
    model="model.onnx",
    lab_mode=True
)
```

### Custom Lab Configuration

```python
lab_config = LabConfig(
    cpu_cores=[8, 9, 10, 11],
    gpu_clock_lock=True,
    gpu_frequency=1800,  # MHz
    disable_turbo=True,
    drop_caches=True
)

session = obs.profile(
    model="model.onnx",
    lab_config=lab_config
)
```

### Lab Mode with eBPF

```python
session = obs.profile(
    model="model.onnx",
    lab_mode=True,
    ebpf=True  # Enable eBPF forensics
)

# Check interference
sii = session.scheduler_interference_index
print(f"Scheduler interference: {sii:.4f}")
```

## Validation

Check lab mode effectiveness:

```python
# Coefficient of Variation (CV)
cv = session.metrics.latency.cv()
print(f"CV: {cv:.2%}")  # Should be < 1%

# Quality score
quality = session.quality_score
print(f"Quality: {quality}")  # 'excellent', 'good', 'fair', 'poor'
```

| CV | Quality |
|----|---------|
| < 1% | Excellent |
| 1-3% | Good |
| 3-5% | Fair |
| > 5% | Poor |

## Best Practices

1. **Warm up the system** before profiling
2. **Run multiple sessions** and compare
3. **Check quality scores** after each run
4. **Document system state** for reproducibility

## Troubleshooting

??? question "Permission denied"
    Run with sudo or add capabilities:
    ```bash
    sudo aaco profile --lab-mode ...
    ```

??? question "Clock lock not working"
    Check ROCm installation and GPU support:
    ```bash
    rocm-smi --showperflevel
    ```

??? question "High CV despite lab mode"
    - Check for thermal throttling
    - Verify no background processes
    - Try different CPU cores
