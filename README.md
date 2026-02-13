# AMD AI Compute Observatory (AACO)

<div align="center">

**From Kernel to Tokens: Full-Stack Observability + Performance Intelligence for AMD AI Workloads**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![ROCm](https://img.shields.io/badge/ROCm-6.0+-red.svg)](https://rocm.docs.amd.com/)

</div>

---

## 🎯 The North Star

AACO answers the questions that matter in AI compute performance:

| Question | AACO Delivers |
|----------|---------------|
| **Where did the time go?** | Kernel scheduling vs GPU kernels vs memory stalls |
| **Why did latency change?** | Driver/runtime changes, bandwidth saturation, kernel launch overhead |
| **What is the bottleneck class?** | Memory-bound / Compute-bound / Launch-bound / CPU-bound |
| **What should I optimize next?** | Top offenders ranked with confidence + evidence |

**This is not benchmarking. This is performance truth infrastructure.**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AACO System Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│   │    Model     │───▶│  ONNX Graph  │───▶│   ORT/EP     │                  │
│   │   (ONNX)     │    │  Extraction  │    │  (MIGraphX)  │                  │
│   └──────────────┘    └──────────────┘    └──────┬───────┘                  │
│                                                   │                          │
│                                                   ▼                          │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                      ROCm Runtime Layer                          │       │
│   │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │       │
│   │   │   rocprof   │   │  rocm-smi   │   │  HIP APIs   │           │       │
│   │   │   Traces    │   │  Telemetry  │   │   Events    │           │       │
│   │   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │       │
│   └──────────┼─────────────────┼─────────────────┼───────────────────┘       │
│              │                 │                 │                           │
│              ▼                 ▼                 ▼                           │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                    Correlation Engine                            │       │
│   │         session_id + timestamp alignment + attribution           │       │
│   └─────────────────────────────────┬───────────────────────────────┘       │
│                                     │                                        │
│              ┌──────────────────────┼──────────────────────┐                │
│              ▼                      ▼                      ▼                │
│   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐       │
│   │  Kernel Plane    │   │  Analytics Plane │   │  Regression Guard│       │
│   │  (eBPF/module)   │   │  (Metrics/Class) │   │  (Baseline Diff) │       │
│   └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘       │
│            │                      │                      │                  │
│            └──────────────────────┼──────────────────────┘                  │
│                                   ▼                                         │
│                        ┌──────────────────┐                                 │
│                        │   Report/UI      │                                 │
│                        │  (HTML/Streamlit)│                                 │
│                        └──────────────────┘                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

### 🔬 Multi-Plane Observability
- **Kernel Plane**: eBPF-based CPU scheduling, context switches, page faults
- **GPU Plane**: rocprof traces, rocm-smi telemetry, kernel execution profiling
- **Inference Plane**: Per-iteration latencies, token-level LLM profiling, throughput curves

### 📊 Performance Intelligence
- **Kernel Launch Tax Analyzer**: Detects "too many tiny kernels" anti-pattern
- **Kernel Amplification Ratio (KAR)**: `GPU kernels / ONNX nodes` - measures fusion efficiency
- **Bottleneck Classifier**: Rule-based + ML classification with evidence
- **Unified Timeline**: Correlate CPU jitter ↔ GPU bursts ↔ clock drops ↔ latency spikes

### 🚨 Regression Guard
- Baseline storage with reproducibility metadata
- Noise-aware confidence scoring
- Automatic root cause attribution
- CI/CD integration ready

### 📈 Reporting
- Auto-generated HTML executive reports
- Interactive Streamlit dashboard
- Diff mode for A/B comparison
- Publication-quality plots

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sudheerdevu/AMD-AI-Compute-Observatory.git
cd AMD-AI-Compute-Observatory

# Install with all dependencies
pip install -e ".[all]"

# Or minimal install
pip install -e .
```

### Run Your First Session

```bash
# Basic inference profiling
aaco run --model resnet50 --backend migraphx --batch 1

# With full profiling (rocprof + system telemetry)
aaco run --model resnet50 --backend migraphx --batch 1 --profile --telemetry

# Generate report
aaco report --session sessions/latest

# Compare against baseline
aaco diff --baseline baselines/resnet50_migraphx_b1.json --session sessions/latest
```

### One-Command Demo

```bash
./scripts/run_demo.sh
# Outputs: session bundle + HTML report + verdict.json
```

---

## 📁 Session Bundle Structure

Every AACO session produces a complete evidence artifact:

```
sessions/<date>/<session_id>/
├── session.json           # Metadata + config spine
├── env.json               # Reproducibility lockbox
├── model/
│   ├── model_meta.json    # ONNX model metadata
│   ├── graph_nodes.parquet
│   └── graph_edges.parquet
├── runtime/
│   ├── ort_config.json
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

## 🎯 Bottleneck Taxonomy

AACO classifies performance bottlenecks with evidence:

| Class | Indicators | Evidence Signals |
|-------|------------|------------------|
| **Launch-bound** | Too many tiny kernels | High kernel count, low avg duration, high CPU overhead |
| **CPU-bound** | Scheduling overhead | High context switches, low GPU active time, runqueue wait |
| **Memory-bound** | Bandwidth limited | High memory ops ratio, slow scaling with batch |
| **Compute-bound** | GPU saturated | High GPU utilization, stable kernel times, good scaling |
| **Throttling** | Power/thermal limits | Clock variance, power drops, correlated latency spikes |

---

## 📊 Key Metrics

### Kernel Launch Tax
```
launch_tax_score = microkernel_pct × kernel_launch_rate / 1000
```
High score indicates fusion opportunities.

### Kernel Amplification Ratio (KAR)
```
KAR = gpu_kernel_launches / onnx_graph_nodes
```
- KAR ≈ 1.0: Excellent fusion
- KAR > 2.0: Investigate partitioning
- KAR > 5.0: Severe launch overhead

### GPU Active Ratio
```
gpu_active_ratio = total_kernel_time / wall_clock_time
```
Low ratio indicates CPU/launch overhead.

---

## 🔧 Configuration

### Model Registry (`configs/models.yaml`)

```yaml
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

### Backend Configuration (`configs/backends.yaml`)

```yaml
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

## 📈 Example Output

### Regression Verdict

```json
{
  "regression": true,
  "severity": "high",
  "confidence": 0.92,
  "latency_delta_pct": 18.3,
  "suspected_cause": "launch-bound",
  "evidence": {
    "kernel_launch_count_delta": "+67%",
    "avg_kernel_duration_delta": "-35%",
    "cpu_overhead_delta": "+22%"
  },
  "recommendation": "Investigate graph partitioning changes. Consider operator fusion optimization."
}
```

### Bottleneck Classification

```json
{
  "bottleneck_class": "launch-bound",
  "confidence": 0.87,
  "top_evidence": [
    {"signal": "microkernel_pct", "value": 0.73, "weight": 0.35},
    {"signal": "kernel_launch_rate", "value": 12500, "weight": 0.28},
    {"signal": "cpu_overhead_ratio", "value": 0.31, "weight": 0.22}
  ]
}
```

---

## 🛠️ Development

### Running Tests

```bash
# Unit tests
pytest tests/unit -v

# Integration tests (requires ROCm)
pytest tests/integration -v

# Full test suite with coverage
pytest --cov=aaco --cov-report=html
```

### Code Quality

```bash
# Format
black aaco tests
isort aaco tests

# Lint
flake8 aaco tests
mypy aaco
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design and data flow |
| [Methodology](docs/methodology.md) | Measurement science and noise handling |
| [Bottleneck Taxonomy](docs/bottleneck_taxonomy.md) | Classification rules and evidence |
| [Data Schema](docs/data_schema.md) | Complete schema documentation |
| [Reproducibility Contract](docs/reproducibility.md) | What gets captured and why |
| [ROCm Profiling Playbook](docs/rocm_profiling.md) | rocprof recipes and interpretation |

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

Built with insights from:
- AMD ROCm team's profiling documentation
- Linux kernel tracing community
- ONNX Runtime performance optimization guides

---

<div align="center">

**AACO: Performance Truth Infrastructure for AMD AI Compute**

*"Most engineers can run a model. Some can profile. Very few can instrument kernel + GPU + analytics and produce a diagnosis."*

</div>
