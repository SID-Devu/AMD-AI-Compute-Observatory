# Laboratory Module

Deterministic execution environment for reproducible performance measurements.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            LABORATORY MODULE                                    │
│                                                                                 │
│                    Deterministic Measurement Environment                        │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ISOLATION CONTROLLER                                   │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     isolation_controller.py                              │   │
│  │                                                                          │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │   │
│  │  │             │  │             │  │             │  │             │    │   │
│  │  │  cgroups v2 │  │  CPU Pin    │  │  NUMA Bind  │  │  IRQ Aff    │    │   │
│  │  │  Isolation  │  │             │  │             │  │             │    │   │
│  │  │             │  │  taskset    │  │  numactl    │  │  /proc/irq  │    │   │
│  │  │  Memory     │  │  isolcpus   │  │  membind    │  │  smp_aff    │    │   │
│  │  │  CPU Quota  │  │  cpuset     │  │  preferred  │  │             │    │   │
│  │  │             │  │             │  │             │  │             │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EXECUTION CAPSULE                                      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      execution_capsule.py                                │   │
│  │                                                                          │   │
│  │    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐            │   │
│  │    │         │    │         │    │         │    │         │            │   │
│  │    │ Pre-    │───▶│ Warmup  │───▶│ Measure │───▶│ Post-   │            │   │
│  │    │ flight  │    │ Phase   │    │ Phase   │    │ flight  │            │   │
│  │    │         │    │         │    │         │    │         │            │   │
│  │    └─────────┘    └─────────┘    └─────────┘    └─────────┘            │   │
│  │                                                                          │   │
│  │    • Env capture   • GPU warmup   • Iterations   • Cleanup              │   │
│  │    • Isolation     • Memory prime • Sampling     • Validation           │   │
│  │    • Clock lock    • JIT compile  • Determinism  • Artifact save        │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          STABILITY COMPONENTS                                   │
│                                                                                 │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐          │
│  │                   │  │                   │  │                   │          │
│  │  thermal_guard    │  │  noise_sentinel   │  │ stability_valid   │          │
│  │                   │  │                   │  │                   │          │
│  │  • Temperature    │  │  • Background     │  │  • CoV Check      │          │
│  │    monitoring     │  │    process scan   │  │  • Outlier Det    │          │
│  │  • Throttle det   │  │  • System load    │  │  • Trend Check    │          │
│  │  • Cool-down      │  │  • IRQ storms     │  │  • P-value Test   │          │
│  │                   │  │                   │  │                   │          │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Components

### isolation_controller.py

Process isolation mechanisms:

| Mechanism | Description | Linux Requirement |
|-----------|-------------|-------------------|
| cgroups v2 | Resource limits | Kernel 4.5+ |
| CPU Pinning | Core affinity | taskset |
| NUMA Binding | Memory locality | numactl |
| IRQ Affinity | Interrupt steering | /proc/irq |

### execution_capsule.py

Encapsulated execution environment:

```python
from aaco.laboratory.execution_capsule import ExecutionCapsule

capsule = ExecutionCapsule(
    cpu_cores=[4, 5, 6, 7],      # Isolated cores
    numa_node=0,                  # NUMA binding
    gpu_clock_level="high",       # Lock GPU clocks
    memory_limit_mb=32768         # Memory cap
)

with capsule:
    # Deterministic execution
    results = run_inference(model)
```

### thermal_guard.py

Thermal management:
- Monitor GPU temperature during execution
- Detect thermal throttling events
- Enforce cool-down periods between runs
- Report thermal impact on measurements

### noise_sentinel.py

System noise detection:
- Scan for background processes
- Monitor system load
- Detect IRQ storms
- Report noise score

### stability_validator.py

Measurement validation:
- Coefficient of variation (CoV) check
- Outlier detection (IQR method)
- Trend detection (Mann-Kendall)
- Statistical significance testing

## Usage

```python
from aaco.laboratory import LabMode

# Create deterministic environment
with LabMode(
    isolate_cpus=True,
    lock_gpu_clocks=True,
    numa_bind=True,
    warmup_iterations=50,
    measure_iterations=100
) as lab:
    results = lab.run(model_path="model.onnx")
    
    # Validation included
    assert results.cov < 0.05  # <5% variation
    assert results.noise_score < 0.1  # Low noise
```

## Isolation Levels

| Level | CPU Pin | NUMA | GPU Clock | cgroups |
|-------|---------|------|-----------|---------|
| None | No | No | No | No |
| Light | Yes | No | No | No |
| Standard | Yes | Yes | Yes | No |
| Full | Yes | Yes | Yes | Yes |
