# AMD AI Compute Observatory

AMD AI Compute Observatory (AACO) is a performance analysis platform for AI and machine learning
workloads running on AMD Instinct accelerators. The platform provides deterministic measurement
capabilities, cross-layer observability from ONNX graphs to HIP kernels, and automated performance
diagnostics with statistical rigor.

AACO enables performance engineers to establish reproducible baselines, detect regressions with
statistical governance, and diagnose bottlenecks through evidence-based root cause analysis. The
platform integrates with ROCm profiling tools including rocprof, ROCm SMI, and MIGraphX to provide
comprehensive visibility across the entire inference stack.

## Features

- **Deterministic Measurement** - Laboratory mode with cgroups v2 isolation, CPU pinning, and GPU
  clock locking for reproducible performance measurements
- **Cross-Layer Attribution** - Graph-to-kernel mapping with probabilistic attribution from ONNX
  operations to HIP kernel executions
- **Bottleneck Classification** - Automated classification of performance bottlenecks (launch-bound,
  memory-bound, compute-bound, CPU-bound, thermal-throttled) with evidence signals
- **Statistical Governance** - EWMA and CUSUM drift detection with robust baseline management for
  regression detection
- **Hardware Calibration** - Microbenchmark-based calibration to establish theoretical peak
  performance envelopes for Hardware Envelope Utilization (HEU) metrics

## Requirements

- Python 3.10 or higher
- ROCm 6.0 or higher (for GPU profiling features)
- Linux (Ubuntu 22.04 or later recommended) or Windows 10/11

## Installation

```bash
git clone https://github.com/SID-Devu/AMD-AI-Compute-Observatory.git
cd AMD-AI-Compute-Observatory
pip install -e ".[all]"
```

For minimal installation without optional dependencies:

```bash
pip install -e .
```

## Usage

### Basic Profiling

```bash
# Profile an ONNX model with MIGraphX backend
aaco run --model resnet50 --backend migraphx --batch 1

# Full-stack profiling with system telemetry
aaco run --model bert-base --backend migraphx --profile --telemetry
```

### Report Generation

```bash
# Generate performance report
aaco report --session sessions/latest --format html

# Export metrics for programmatic analysis
aaco report --session sessions/latest --format json
```

### Regression Analysis

```bash
# Compare against baseline with statistical tests
aaco diff --baseline baselines/production.json --session sessions/latest --threshold 5%
```

## Architecture

AACO implements a layered architecture for deterministic measurement and analysis:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Governance Layer                               │
│         Statistical Regression | Root Cause Analysis | Fleet Ops           │
├─────────────────────────────────────────────────────────────────────────────┤
│                             Intelligence Layer                              │
│      Kernel Fingerprint | Attribution Engine | Hardware Digital Twin       │
├─────────────────────────────────────────────────────────────────────────────┤
│                             Measurement Layer                               │
│          Laboratory Mode | rocprof Integration | eBPF Telemetry            │
├─────────────────────────────────────────────────────────────────────────────┤
│                              Collection Layer                               │
│            ONNX Runtime | MIGraphX | ROCm SMI | System Telemetry           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| KAR (Kernel Amplification Ratio) | Ratio of GPU kernels to ONNX nodes, measures kernel explosion |
| PFI (Partition Fragmentation Index) | Graph partitioning quality indicator |
| LTS (Launch Tax Score) | CPU-GPU synchronization overhead measurement |
| HEU (Hardware Envelope Utilization) | Percentage of calibrated peak performance achieved |

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Measurement Methodology](docs/methodology.md)
- [Bottleneck Taxonomy](docs/bottleneck_taxonomy.md)
- [Data Schema Reference](docs/data_schema.md)
- [API Documentation](docs/api/)

## Building from Source

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/unit -v

# Build documentation
mkdocs build
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines, code style requirements, and
the pull request process.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full license text.

## Related Projects

- [ROCm](https://github.com/ROCm/ROCm) - AMD open-source GPU compute platform
- [MIGraphX](https://github.com/ROCm/AMDMIGraphX) - AMD graph inference engine
- [rocprof](https://github.com/ROCm/rocprofiler) - ROCm profiling tools
