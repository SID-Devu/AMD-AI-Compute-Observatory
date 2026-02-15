# Calibration Module

Hardware envelope calibration via microbenchmarks.

## Calibration Targets

| Target | Benchmark | Unit |
|--------|-----------|------|
| Compute Peak | GEMM saturation | TFLOPS |
| Memory Bandwidth | Streaming copy | GB/s |
| Launch Overhead | Empty kernel | μs |
| L2 Bandwidth | Cache-resident access | GB/s |

## Hardware Profiles

| GPU | FP16 TFLOPS | FP32 TFLOPS | Memory BW |
|-----|-------------|-------------|-----------|
| MI300X | 1307 | 653 | 5300 GB/s |
| MI250X | 762 | 381 | 3276 GB/s |
| MI210 | 362 | 181 | 1638 GB/s |
| MI100 | 185 | 92 | 1228 GB/s |

## Usage

```python
from aaco.calibration import HardwareCalibrator

calibrator = HardwareCalibrator(device_id=0)
envelope = calibrator.calibrate()

print(f"Compute ceiling: {envelope.compute_tflops} TFLOPS")
print(f"Memory bandwidth: {envelope.memory_gbps} GB/s")
print(f"Launch overhead: {envelope.launch_us} μs")
```
