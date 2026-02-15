# Laboratory Module

Deterministic execution environment for reproducible measurements.

## Isolation Mechanisms

| Mechanism | Purpose |
|-----------|---------|
| cgroups v2 | Process resource isolation |
| CPU Affinity | Pin to specific cores |
| NUMA Binding | Memory locality control |
| GPU Clock Lock | Fixed clock frequencies |
| IRQ Affinity | Interrupt handling isolation |

## Requirements

Laboratory mode requires root privileges or appropriate capabilities:

- `CAP_SYS_ADMIN` for cgroups management
- ROCm SMI access for GPU clock control
- Kernel support for CPU isolation

## Configuration

```yaml
laboratory:
  enabled: true
  cpu_cores: [0, 1, 2, 3]
  numa_node: 0
  gpu_clock_mhz: 1700
  memory_clock_mhz: 1600
  isolation_level: strict
```

## Usage

```python
from aaco.laboratory import LaboratoryContext

with LaboratoryContext(config) as lab:
    # Measurements within deterministic environment
    results = run_benchmark()
```
