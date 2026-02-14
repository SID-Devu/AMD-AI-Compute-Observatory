# AACO Performance Baselines

This directory stores baseline performance measurements for regression detection.

## Structure

```
baselines/
├── README.md
├── mi300x/           # MI300X baseline data
│   └── *.json
├── mi250x/           # MI250X baseline data
│   └── *.json
└── rdna3/            # RDNA3 baseline data
    └── *.json
```

## Baseline Format

Each baseline file follows this schema:

```json
{
  "metadata": {
    "baseline_id": "string",
    "gpu_arch": "gfx942|gfx90a|gfx1100",
    "rocm_version": "6.x.x",
    "created_at": "ISO timestamp",
    "model_name": "string"
  },
  "kernels": {
    "kernel_name": {
      "latency_us": 123.45,
      "flops": 1e12,
      "memory_bw_gbps": 800.0,
      "occupancy": 0.85
    }
  },
  "workload": {
    "total_time_ms": 100.0,
    "throughput": 500.0
  }
}
```

## Adding New Baselines

Use the AACO CLI:

```bash
aaco baseline create --model resnet50 --gpu mi300x
aaco baseline compare --current results.json --baseline baselines/mi300x/resnet50.json
```
