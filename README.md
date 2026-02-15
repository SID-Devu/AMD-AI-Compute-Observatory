# AMD AI Compute Observatory

AMD AI Compute Observatory (AACO) is an open-source performance analysis and observability platform
designed for artificial intelligence and machine learning inference workloads on AMD Instinct
accelerators. The platform delivers deterministic, reproducible performance measurements with
cross-layer visibility spanning from high-level ONNX graph operations through HIP kernel executions
to low-level hardware performance counters.

AACO addresses critical challenges in production AI performance engineering: establishing statistically
rigorous baselines, detecting performance regressions with controlled false-positive rates, and
diagnosing root causes through evidence-based bottleneck classification. The platform integrates
natively with the ROCm software ecosystem, leveraging rocprof for kernel-level profiling, ROCm SMI
for device telemetry, and MIGraphX for graph-level optimization insights.

## Overview

Modern AI inference deployments require continuous performance validation across software updates,
model changes, and hardware configurations. AACO provides the instrumentation and analysis
infrastructure to maintain performance guarantees at scale.

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Laboratory Mode** | Deterministic execution environment using cgroups v2 process isolation, explicit CPU core affinity, and GPU clock frequency locking to minimize measurement variance |
| **Cross-Layer Attribution** | Probabilistic mapping from ONNX graph operations to HIP kernel executions, enabling identification of which model operations contribute to observed latencies |
| **Hardware Envelope Utilization** | Microbenchmark-based calibration establishing theoretical peak throughput, allowing performance to be expressed as percentage of hardware capability |
| **Statistical Governance** | EWMA and CUSUM control charts for drift detection with configurable sensitivity, supporting automated regression detection in CI/CD pipelines |
| **Root Cause Analysis** | Bayesian inference engine combining multiple evidence signals to classify bottlenecks and recommend targeted optimizations |

### Performance Metrics

AACO computes derived metrics that provide actionable insights into inference performance:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **KAR** | `kernel_count / onnx_node_count` | Kernel Amplification Ratio indicates graph compilation efficiency; high values suggest opportunities for kernel fusion |
| **PFI** | `partition_count / node_count` | Partition Fragmentation Index measures graph partitioning quality; values approaching 1.0 indicate excessive fragmentation |
| **LTS** | `launch_overhead_us / kernel_duration_us` | Launch Tax Score quantifies CPU-GPU synchronization overhead; high values indicate launch-bound workloads |
| **HEU** | `measured_throughput / calibrated_peak` | Hardware Envelope Utilization expresses achieved performance relative to calibrated hardware capability |

## Requirements

### Supported Platforms

| Platform | Version | Notes |
|----------|---------|-------|
| Ubuntu | 22.04, 24.04 | Primary development and testing platform |
| RHEL | 8.x, 9.x | Enterprise Linux support |
| Windows | 10, 11 | Limited profiling capabilities without ROCm |

### Software Dependencies

| Component | Minimum Version | Purpose |
|-----------|-----------------|---------|
| Python | 3.10 | Runtime environment |
| ROCm | 6.0 | GPU profiling and device management |
| ONNX Runtime | 1.16 | Model execution framework |

### Hardware Compatibility

AACO supports AMD Instinct accelerators with full profiling capabilities:

- AMD Instinct MI300X, MI300A
- AMD Instinct MI250X, MI250, MI210
- AMD Instinct MI100

## Installation

### Standard Installation

```bash
git clone https://github.com/SID-Devu/AMD-AI-Compute-Observatory.git
cd AMD-AI-Compute-Observatory
pip install -e ".[all]"
```

### Minimal Installation

For environments without ROCm or optional dependencies:

```bash
pip install -e .
```

### Container Deployment

```bash
docker build -t aaco:latest .
docker run --device=/dev/kfd --device=/dev/dri -v $(pwd)/sessions:/app/sessions aaco:latest
```

### Verification

```bash
aaco --version
aaco doctor  # Verify ROCm and dependencies
```

## Usage

### Performance Profiling

```bash
# Basic inference profiling
aaco run --model models/resnet50.onnx --backend migraphx

# Full-stack profiling with hardware counters
aaco run --model models/bert-base.onnx --backend migraphx --profile --counters

# Laboratory mode for deterministic measurements
aaco run --model models/llama2-7b.onnx --backend migraphx --laboratory
```

### Report Generation

```bash
# Generate HTML performance report
aaco report --session sessions/latest --format html --output report.html

# Export structured data for analysis
aaco report --session sessions/latest --format json --output metrics.json
```

### Regression Detection

```bash
# Compare against established baseline
aaco diff --baseline baselines/v1.0.json --current sessions/latest

# CI/CD integration with exit code on regression
aaco diff --baseline baselines/production.json --current sessions/latest --fail-on-regression
```

### Baseline Management

```bash
# Establish new baseline from session
aaco baseline create --session sessions/latest --name production-v2.0

# List available baselines
aaco baseline list
```

## Architecture

AACO implements a layered architecture separating concerns across collection, measurement,
intelligence, and governance domains:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                GOVERNANCE LAYER                                 │
│                                                                                 │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐             │
│   │   Statistical   │   │   Root Cause    │   │     Fleet       │             │
│   │   Regression    │   │    Analysis     │   │   Operations    │             │
│   │   Detection     │   │     Engine      │   │                 │             │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                               INTELLIGENCE LAYER                                │
│                                                                                 │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐             │
│   │     Kernel      │   │  Probabilistic  │   │    Hardware     │             │
│   │   Fingerprint   │   │   Attribution   │   │   Digital Twin  │             │
│   │     Engine      │   │     Engine      │   │                 │             │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                               MEASUREMENT LAYER                                 │
│                                                                                 │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐             │
│   │   Laboratory    │   │     rocprof     │   │      eBPF       │             │
│   │      Mode       │   │   Integration   │   │    Telemetry    │             │
│   │                 │   │                 │   │                 │             │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                COLLECTION LAYER                                 │
│                                                                                 │
│   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐             │
│   │  ONNX Runtime   │   │    MIGraphX     │   │    ROCm SMI     │             │
│   │   Telemetry     │   │   Partitions    │   │     Metrics     │             │
│   │                 │   │                 │   │                 │             │
│   └─────────────────┘   └─────────────────┘   └─────────────────┘             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Model Input → Graph Analysis → Deterministic Execution → Multi-Layer Collection
                                                                    │
                                                                    ▼
                                                         Unified Trace Storage
                                                                    │
                                                                    ▼
              Optimization Recommendations ← Root Cause Analysis ← Attribution
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Detailed system architecture and design rationale |
| [Methodology](docs/methodology.md) | Statistical methodology and measurement theory |
| [Bottleneck Taxonomy](docs/bottleneck_taxonomy.md) | Classification criteria and evidence signals |
| [Data Schema](docs/data_schema.md) | Session output format and schema reference |
| [API Reference](docs/api/) | Python API documentation |

## Development

### Building from Source

```bash
git clone https://github.com/SID-Devu/AMD-AI-Compute-Observatory.git
cd AMD-AI-Compute-Observatory
pip install -e ".[dev]"
```

### Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires ROCm)
pytest tests/integration/ -v -m "not rocm" 

# Full test suite with coverage
pytest --cov=aaco --cov-report=html
```

### Code Quality

```bash
# Linting
ruff check aaco/

# Formatting
ruff format aaco/

# Type checking
mypy aaco/ --ignore-missing-imports
```

## Contributing

We welcome contributions from the community. Please review our [Contributing Guidelines](CONTRIBUTING.md)
before submitting pull requests. All contributions must adhere to the project's code style requirements
and include appropriate test coverage.

## License

Copyright (c) 2026 Sudheer Ibrahim Daniel Devu. All rights reserved.

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the complete license text.

## Author

**Sudheer Ibrahim Daniel Devu**

- GitHub: [@SID-Devu](https://github.com/SID-Devu)
- Email: sudheerdevu4work@gmail.com

AMD AI Compute Observatory was created and is maintained by Sudheer Ibrahim Daniel Devu.

## Support

- **Issue Tracker**: [GitHub Issues](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/discussions)

## Related Projects

- [ROCm](https://github.com/ROCm/ROCm) — AMD open-source GPU compute platform
- [MIGraphX](https://github.com/ROCm/AMDMIGraphX) — AMD graph optimization and inference engine
- [rocprofiler](https://github.com/ROCm/rocprofiler) — ROCm profiling infrastructure
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — Cross-platform inference engine

## Acknowledgments

AMD AI Compute Observatory builds upon the ROCm open-source ecosystem and integrates with
industry-standard frameworks including ONNX Runtime. We acknowledge the contributions of the
broader open-source community that make this work possible.
