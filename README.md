# AMD AI Compute Observatory

**Performance Analysis and Observability Platform for AMD Instinct Accelerators**

[![Build Status](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![ROCm 6.0+](https://img.shields.io/badge/ROCm-6.0%2B-red.svg)](https://rocm.docs.amd.com/)

---

## Introduction

AMD AI Compute Observatory (AACO) is an open-source platform for performance analysis and
observability of artificial intelligence and machine learning inference workloads on AMD Instinct
accelerators. The platform provides deterministic, reproducible performance measurements with
cross-layer visibility from ONNX graph operations through HIP kernel executions to hardware
performance counters.

AACO addresses critical challenges in production AI performance engineering:

- Establishing statistically rigorous performance baselines
- Detecting performance regressions with controlled false-positive rates
- Diagnosing root causes through evidence-based bottleneck classification
- Providing actionable optimization recommendations

The platform integrates natively with the ROCm software ecosystem, leveraging rocprof for
kernel-level profiling, ROCm SMI for device telemetry, and MIGraphX for graph-level optimization
insights.

## Key Features

| Feature | Description |
|---------|-------------|
| **Laboratory Mode** | Deterministic execution environment with cgroups v2 isolation, CPU affinity, and GPU clock locking |
| **Cross-Layer Attribution** | Probabilistic mapping from ONNX operations to HIP kernel executions |
| **Hardware Envelope Utilization** | Microbenchmark-based calibration for theoretical peak performance measurement |
| **Statistical Governance** | EWMA and CUSUM control charts for automated regression detection |
| **Root Cause Analysis** | Bayesian inference engine for evidence-based bottleneck classification |

## Supported Hardware

| Accelerator | Architecture | Status |
|-------------|--------------|--------|
| AMD Instinct MI300X | CDNA 3 | Fully Supported |
| AMD Instinct MI300A | CDNA 3 | Fully Supported |
| AMD Instinct MI250X | CDNA 2 | Fully Supported |
| AMD Instinct MI250 | CDNA 2 | Fully Supported |
| AMD Instinct MI210 | CDNA 2 | Fully Supported |
| AMD Instinct MI100 | CDNA | Fully Supported |

## Requirements

| Component | Version |
|-----------|---------|
| Python | 3.10 or higher |
| ROCm | 6.0 or higher |
| Operating System | Ubuntu 22.04+, RHEL 8+, Windows 10/11 |

## Installation

```bash
git clone https://github.com/SID-Devu/AMD-AI-Compute-Observatory.git
cd AMD-AI-Compute-Observatory
pip install -e ".[all]"
```

Verify installation:

```bash
aaco --version
```

## Quick Start

### Performance Profiling

```bash
aaco run --model models/resnet50.onnx --backend migraphx
```

### Generate Report

```bash
aaco report --session sessions/latest --format html
```

### Regression Detection

```bash
aaco diff --baseline baselines/production.json --current sessions/latest
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GOVERNANCE LAYER                               │
│              Statistical Regression | Root Cause Analysis                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                             INTELLIGENCE LAYER                              │
│         Attribution Engine | Hardware Digital Twin | Trace Lake             │
├─────────────────────────────────────────────────────────────────────────────┤
│                             MEASUREMENT LAYER                               │
│              Laboratory Mode | rocprof Integration | eBPF                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                              COLLECTION LAYER                               │
│              ONNX Runtime | MIGraphX | ROCm SMI | System Metrics            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Performance Metrics

| Metric | Description |
|--------|-------------|
| **KAR** | Kernel Amplification Ratio — kernel count per ONNX node |
| **PFI** | Partition Fragmentation Index — graph partitioning quality |
| **LTS** | Launch Tax Score — CPU-GPU synchronization overhead |
| **HEU** | Hardware Envelope Utilization — percentage of peak capability |

## Documentation

| Resource | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design and components |
| [Methodology](docs/methodology.md) | Statistical measurement methodology |
| [Bottleneck Taxonomy](docs/bottleneck_taxonomy.md) | Classification criteria |
| [Data Schema](docs/data_schema.md) | Session output format reference |
| [API Reference](docs/api/) | Python API documentation |

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/unit/ -v

# Code quality
ruff check aaco/
mypy aaco/
```

## Contributing

Contributions are welcome. Please review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Copyright (c) 2026 Sudheer Ibrahim Daniel Devu

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Sudheer Ibrahim Daniel Devu**

- GitHub: [github.com/SID-Devu](https://github.com/SID-Devu)
- Email: sudheerdevu4work@gmail.com

## Citation

If you use AMD AI Compute Observatory in your research or work, please cite:

```bibtex
@software{aaco2026,
  author = {Devu, Sudheer Ibrahim Daniel},
  title = {AMD AI Compute Observatory},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/SID-Devu/AMD-AI-Compute-Observatory}
}
```

## Acknowledgments

This project builds upon the AMD ROCm open-source ecosystem and integrates with industry-standard
frameworks including ONNX Runtime. We acknowledge the contributions of the open-source community.

## Related Projects

- [ROCm](https://github.com/ROCm/ROCm) — AMD open-source GPU compute platform
- [MIGraphX](https://github.com/ROCm/AMDMIGraphX) — AMD graph optimization engine
- [rocprofiler](https://github.com/ROCm/rocprofiler) — ROCm profiling infrastructure
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — Cross-platform inference engine
