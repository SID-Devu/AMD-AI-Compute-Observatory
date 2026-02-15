<p align="center">
  <img src="https://www.amd.com/content/dam/amd/en/images/logos/amd-logo.svg" alt="AMD" width="200"/>
</p>

<h1 align="center">AMD AI Compute Observatory</h1>

<p align="center">
  <strong>Model-to-Metal Performance Analysis Platform for AMD Instinct Accelerators</strong>
</p>

<p align="center">
  <a href="https://github.com/SID-Devu/AMD-AI-Compute-Observatory/actions"><img src="https://img.shields.io/github/actions/workflow/status/SID-Devu/AMD-AI-Compute-Observatory/ci-cd.yml?branch=master&label=CI%2FCD&style=flat-square" alt="Build Status"/></a>
  <img src="https://img.shields.io/badge/ROCm-6.0%2B-ED1C24?style=flat-square" alt="ROCm"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-ED1C24?style=flat-square" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-ED1C24?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/Version-4.0.0-ED1C24?style=flat-square" alt="Version"/>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#documentation">Documentation</a>
</p>

---

## Overview

AMD AI Compute Observatory (AACO) is a comprehensive performance analysis and optimization platform designed for AI/ML workloads running on AMD Instinct GPUs. The platform provides deterministic measurement capabilities, cross-layer observability, and automated performance diagnostics.

AACO enables engineers to:

- **Measure** — Deterministic, reproducible performance measurements with statistical rigor
- **Analyze** — Cross-layer attribution from ONNX graphs to HIP kernels to hardware counters
- **Diagnose** — Automated bottleneck classification with evidence-based root cause analysis
- **Optimize** — Data-driven optimization recommendations with regression governance

### Key Capabilities

| Capability | Description |
|------------|-------------|
| Laboratory Mode | Deterministic execution environment with cgroups v2 isolation, CPU pinning, and GPU clock locking |
| Hardware Calibration | Microbenchmark-based calibration for theoretical peak performance envelopes |
| Cross-Layer Attribution | Graph-to-kernel mapping with probabilistic attribution scores |
| Statistical Governance | EWMA/CUSUM drift detection with robust baseline management |
| Root Cause Analysis | Bayesian inference engine for performance regression diagnosis |

---

## Features

### Multi-Plane Observability

| Layer | Technology | Telemetry |
|-------|------------|-----------|
| Application | ONNX Runtime | Graph operations, shapes, execution order |
| Runtime | MIGraphX / HIP | Partition mapping, kernel launches, memory transfers |
| GPU | rocprof / ROCm SMI | Kernel execution, hardware counters, clocks, power |
| System | eBPF | Scheduler events, page faults, context switches |

### Performance Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| KAR | `GPU_kernels / ONNX_nodes` | Kernel Amplification Ratio — measures kernel explosion |
| PFI | `partitions / nodes` | Partition Fragmentation Index — graph partitioning quality |
| LTS | `launch_overhead / kernel_time` | Launch Tax Score — CPU-GPU synchronization cost |
| SII | `runqueue_wait / wall_time` | Scheduler Interference Index — OS scheduling impact |
| HEU | `actual_perf / calibrated_ceiling` | Hardware Envelope Utilization — peak utilization percentage |

### Bottleneck Classification

| Classification | Indicators | Resolution Strategy |
|---------------|------------|---------------------|
| Launch-bound | High kernel count, low average duration | Kernel fusion, operation batching |
| CPU-bound | High context switches, runqueue wait | Reduce host operations, async execution |
| Memory-bound | High memory operation ratio, bandwidth limited | Data layout optimization, prefetching |
| Compute-bound | High GPU utilization, stable timing | Workload scaling |
| Thermal-throttled | Clock variance, power drops | Thermal management, power configuration |

---

## Installation

### Requirements

- Python 3.10 or higher
- ROCm 6.0 or higher (for GPU profiling)
- Linux (Ubuntu 22.04+ recommended) or Windows 10/11

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/SID-Devu/AMD-AI-Compute-Observatory.git
cd AMD-AI-Compute-Observatory

# Install with all dependencies
pip install -e ".[all]"

# Verify installation
aaco --version
```

### Minimal Installation

```bash
# Core functionality only
pip install -e .
```

### Docker Deployment

```bash
# Build container
docker build -t aaco:latest -f Dockerfiles/rocm.dockerfile .

# Run with GPU access
docker run --device=/dev/kfd --device=/dev/dri \
           -v $(pwd)/sessions:/app/sessions \
           aaco:latest run --model bert-base
```

---

## Usage

### Basic Profiling

```bash
# Profile an ONNX model
aaco run --model resnet50 --backend migraphx --batch 1

# Full-stack profiling with telemetry
aaco run --model llama2-7b --backend migraphx \
         --profile --telemetry --ebpf
```

### Report Generation

```bash
# Generate HTML report
aaco report --session sessions/latest --format html

# Export to JSON for programmatic access
aaco report --session sessions/latest --format json
```

### Regression Analysis

```bash
# Compare against baseline
aaco diff --baseline baselines/production.json \
          --session sessions/latest \
          --threshold 5%
```

### Real-Time Monitoring

```bash
# Launch dashboard
aaco dashboard --port 8501
```

---

## Architecture

AACO implements a layered architecture designed for deterministic measurement and comprehensive analysis.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Governance Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Statistical │  │  Root Cause  │  │    Auto-     │  │    Fleet     │    │
│  │  Regression  │  │   Analysis   │  │ Optimization │  │  Operations  │    │
│  │  Governance  │  │    Engine    │  │    Engine    │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                             Intelligence Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Kernel     │  │ Probabilistic│  │   Hardware   │  │   Unified    │    │
│  │ Fingerprint  │  │ Attribution  │  │   Digital    │  │    Trace     │    │
│  │   Engine     │  │    Engine    │  │     Twin     │  │     Lake     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│                             Measurement Layer                               │
│  ┌────────────────────────────────┐  ┌────────────────────────────────┐    │
│  │        Laboratory Mode         │  │      eBPF Forensic Tracer      │    │
│  │  • cgroups v2 isolation        │  │  • Scheduler interference      │    │
│  │  • CPU pinning                 │  │  • Context switch tracking     │    │
│  │  • GPU clock control           │  │  • Page fault monitoring       │    │
│  └────────────────────────────────┘  └────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
ONNX Model ──► Graph Analysis ──► Deterministic Execution ──► Multi-Layer Collection
                                                                      │
    ┌─────────────────────────────────────────────────────────────────┘
    │
    ▼
Unified Trace Lake ──► Attribution Engine ──► Bottleneck Classification
                                                      │
    ┌─────────────────────────────────────────────────┘
    │
    ▼
Statistical Governance ──► Root Cause Analysis ──► Optimization Recommendations
```

---

## Session Output

Each profiling session generates a structured evidence bundle:

```
sessions/<date>/<session_id>/
├── session.json              # Session metadata and configuration
├── env.json                  # Environment reproducibility data
├── model/
│   ├── model_meta.json       # ONNX model metadata
│   ├── graph_nodes.parquet   # Graph node data
│   └── graph_edges.parquet   # Graph edge data
├── runtime/
│   ├── ort_config.json       # ONNX Runtime configuration
│   └── migraphx_partition.json
├── telemetry/
│   ├── system_events.parquet
│   └── gpu_events.parquet
├── profiler/
│   ├── rocprof_raw/
│   └── rocprof_kernels.parquet
├── attribution/
│   ├── kernel_groups.parquet
│   └── op_to_kernel_map.parquet
├── metrics/
│   ├── inference_iters.parquet
│   ├── inference_summary.json
│   ├── derived_metrics.json
│   └── bottleneck.json
├── regress/
│   ├── baseline_ref.json
│   ├── diff.json
│   └── verdict.json
└── report/
    ├── report.html
    └── plots/
```

---

## Configuration

### Model Registry

```yaml
# configs/models.yaml
models:
  resnet50:
    path: "models/resnet50.onnx"
    input_shapes:
      input: [1, 3, 224, 224]
    dtype: float16
    warmup: 10
    iterations: 100

  bert-base:
    path: "models/bert-base.onnx"
    input_shapes:
      input_ids: [1, 128]
      attention_mask: [1, 128]
    dtype: int64
    warmup: 5
    iterations: 50
```

### Backend Configuration

```yaml
# configs/backends.yaml
backends:
  migraphx:
    provider: "MIGraphXExecutionProvider"
    device_id: 0
    fp16_enable: true
    
  cpu:
    provider: "CPUExecutionProvider"
    intra_op_threads: 4
    inter_op_threads: 2
```

---

## Development

### Testing

```bash
# Unit tests
pytest tests/unit -v

# Integration tests (requires ROCm)
pytest tests/integration -v

# Coverage report
pytest --cov=aaco --cov-report=html
```

### Code Quality

```bash
# Lint and format
ruff check aaco/ --fix
ruff format aaco/

# Type checking
mypy aaco/ --strict
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design and component overview |
| [Methodology](docs/methodology.md) | Measurement methodology and statistical approach |
| [Bottleneck Taxonomy](docs/bottleneck_taxonomy.md) | Classification rules and evidence signals |
| [Data Schema](docs/data_schema.md) | Complete data schema and Parquet layouts |
| [API Reference](docs/api/) | Module and function documentation |

---

## Contributing

Contributions are welcome. Please review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Code style and formatting requirements
- Testing requirements
- Pull request process
- Issue reporting

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Support

- [Issue Tracker](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/issues) — Bug reports and feature requests
- [Discussions](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/discussions) — Questions and community support

---

<p align="center">
  <img src="https://img.shields.io/badge/AMD-ED1C24?style=for-the-badge&logo=amd&logoColor=white" alt="AMD"/>
</p>

<p align="center">
  <sub>AMD AI Compute Observatory is designed for AMD Instinct accelerators and the ROCm software stack.</sub>
</p>
