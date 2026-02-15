# Calibration Module

Hardware performance calibration for theoretical peak measurement.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            CALIBRATION MODULE                                   │
│                                                                                 │
│                    Hardware Digital Twin Calibration                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MICROBENCHMARK SUITE                                    │
│                                                                                 │
│    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│    │                 │  │                 │  │                 │              │
│    │   Compute       │  │   Memory        │  │   Launch        │              │
│    │   Benchmark     │  │   Bandwidth     │  │   Overhead      │              │
│    │                 │  │                 │  │                 │              │
│    │  FP16/FP32/INT8 │  │  HBM Bandwidth  │  │  Empty Kernel   │              │
│    │  GEMM Peak      │  │  L2 Bandwidth   │  │  Launch Latency │              │
│    │  TFLOPS         │  │  GB/s           │  │  Microseconds   │              │
│    │                 │  │                 │  │                 │              │
│    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│             │                    │                    │                        │
│             └────────────────────┴────────────────────┘                        │
│                                  │                                              │
│                                  ▼                                              │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │                    Hardware Envelope                                 │     │
│    │                                                                      │     │
│    │   ┌─────────────────────────────────────────────────────────────┐  │     │
│    │   │                                                              │  │     │
│    │   │  compute_ceiling_tflops: 1000.0                            │  │     │
│    │   │  memory_ceiling_gbps: 5300.0                               │  │     │
│    │   │  launch_floor_us: 2.5                                      │  │     │
│    │   │                                                              │  │     │
│    │   └─────────────────────────────────────────────────────────────┘  │     │
│    │                                                                      │     │
│    └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         HEU CALCULATION                                         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                                                                          │   │
│  │   Hardware Envelope Utilization (HEU)                                   │   │
│  │                                                                          │   │
│  │                    actual_performance                                    │   │
│  │   HEU = ────────────────────────────────────                            │   │
│  │                calibrated_ceiling                                        │   │
│  │                                                                          │   │
│  │   Example:                                                               │   │
│  │   • Model achieves 800 TFLOPS                                          │   │
│  │   • Calibrated ceiling is 1000 TFLOPS                                  │   │
│  │   • HEU = 800/1000 = 80%                                               │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ROOFLINE MODEL                                          │
│                                                                                 │
│    Performance                                                                  │
│    (TFLOPS)                                                                     │
│         │                                                                       │
│    1000 ├─────────────────────────────────────────────────── Compute Ceiling   │
│         │                                              ╱                        │
│     800 ├────────────────────────────────────────────╱──── Measured (HEU=80%) │
│         │                                          ╱                            │
│     600 ├────────────────────────────────────────╱                             │
│         │                                      ╱                                │
│     400 ├──────────────────────────────────╱   (Memory Bound Region)           │
│         │                                ╱                                      │
│     200 ├────────────────────────────╱                                         │
│         │                          ╱                                            │
│       0 └────────────────────────╱────────────────────────────────────────────  │
│         0         50        100        150        200        250                │
│                      Arithmetic Intensity (FLOP/byte)                           │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Usage

```python
from aaco.calibration import HardwareCalibrator

calibrator = HardwareCalibrator(device_id=0)

# Run microbenchmarks
envelope = calibrator.calibrate()

print(f"Compute ceiling: {envelope.compute_tflops} TFLOPS")
print(f"Memory ceiling: {envelope.memory_gbps} GB/s")
print(f"Launch floor: {envelope.launch_us} μs")

# Calculate HEU for a workload
heu = calibrator.compute_heu(
    measured_tflops=800,
    arithmetic_intensity=150
)
print(f"HEU: {heu:.1%}")
```

## Calibration Output

```json
{
  "device": "AMD Instinct MI300X",
  "calibration_timestamp": "2026-02-15T12:00:00Z",
  "envelope": {
    "compute_ceiling_fp16_tflops": 1307.0,
    "compute_ceiling_fp32_tflops": 653.0,
    "memory_bandwidth_gbps": 5300.0,
    "launch_overhead_us": 2.5,
    "l2_bandwidth_gbps": 8000.0
  },
  "validation": {
    "iterations": 1000,
    "cov": 0.02,
    "stable": true
  }
}
```

## Hardware Profiles

| GPU | FP16 TFLOPS | FP32 TFLOPS | Memory BW |
|-----|-------------|-------------|-----------|
| MI100 | 185 | 92 | 1228 GB/s |
| MI210 | 362 | 181 | 1638 GB/s |
| MI250X | 762 | 381 | 3276 GB/s |
| MI300X | 1307 | 653 | 5300 GB/s |
