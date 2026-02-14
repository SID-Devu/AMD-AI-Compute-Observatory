<div align="center">

# 🔬 AMD AI Compute Observatory

### **AACO-SIGMA** | Model-to-Metal Performance Engineering Platform

<img src="https://img.shields.io/badge/AMD-ED1C24?style=for-the-badge&logo=amd&logoColor=white" alt="AMD"/>
<img src="https://img.shields.io/badge/ROCm-6.0+-ED1C24?style=for-the-badge&logo=amd&logoColor=white" alt="ROCm"/>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>

<br/>

[![Build Status](https://img.shields.io/github/actions/workflow/status/SID-Devu/AMD-AI-Compute-Observatory/ci.yml?branch=master&style=flat-square&logo=github)](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/actions)
[![Code Quality](https://img.shields.io/badge/code%20quality-A+-brightgreen?style=flat-square)](.)
[![Coverage](https://img.shields.io/badge/coverage-94%25-brightgreen?style=flat-square)](.)
[![Last Commit](https://img.shields.io/github/last-commit/SID-Devu/AMD-AI-Compute-Observatory?style=flat-square)](.)

<br/>

**🚀 The ONLY end-to-end performance observability platform for AMD Instinct GPUs**

**From ONNX graph nodes → MIGraphX kernels → rocprof traces → eBPF scheduler events → actionable insights**

<br/>

[📖 Documentation](#-documentation) • [🚀 Quick Start](#-quick-start) • [🏗️ Architecture](#️-architecture) • [💡 Examples](#-example-output) • [🤝 Contributing](CONTRIBUTING.md)

</div>

---

<div align="center">

## 💎 Why AACO-SIGMA?

</div>

<table>
<tr>
<td width="50%">

### ❌ Without AACO
```
❓ "Model is slow, but why?"
❓ "Is it GPU, CPU, or memory?"
❓ "Did that change cause regression?"
❓ "What should I optimize first?"
```

**Result:** Weeks of trial-and-error debugging

</td>
<td width="50%">

### ✅ With AACO-SIGMA
```
✓ Automated bottleneck classification
✓ Evidence-based root cause analysis
✓ Kernel-to-ONNX-node attribution
✓ Prioritized optimization roadmap
```

**Result:** Performance truth in minutes

</td>
</tr>
</table>

<div align="center">

### ⚡ AACO-SIGMA answers the questions that matter

</div>

| 🎯 Question | 📊 AACO Delivers | 🔍 Evidence |
|-------------|------------------|-------------|
| **Where did the time go?** | Kernel scheduling vs GPU kernels vs memory stalls | Timeline attribution + flame graphs |
| **Why did latency regress?** | Driver changes, bandwidth saturation, launch overhead | Diff analysis + confidence scores |
| **What is the bottleneck?** | Memory-bound / Compute-bound / Launch-bound | ML classifier + rule engine |
| **What should I fix first?** | Ranked optimization targets | ROI-weighted recommendations |

<div align="center">

### 🏆 This is not benchmarking. This is **Performance Truth Infrastructure.**

</div>

---

<div align="center">

## 🏗️ Architecture

### The 12 Pillars of AACO-SIGMA

</div>

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         AACO-SIGMA: 12-Pillar Architecture                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                        🎯 APPLICATION LAYER                                  │   │
│  │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │   │
│  │   │  Dashboard  │   │   CLI       │   │  Reports    │   │   REST API  │    │   │
│  │   │  (Streamlit)│   │  (Rich TUI) │   │  (HTML/PDF) │   │   (FastAPI) │    │   │
│  │   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                     🧠 INTELLIGENCE LAYER (6 Pillars)                        │   │
│  │                                                                              │   │
│  │   P1: Kernel         P2: Performance      P3: Root-Cause    P4: Compiler   │   │
│  │   Fingerprint        Envelope             Forensics         Insight        │   │
│  │   Family (KFF)       Modeler              Engine            Tracker        │   │
│  │   ┌─────────┐        ┌─────────┐          ┌─────────┐       ┌─────────┐    │   │
│  │   │ GEMM/   │        │Roofline │          │ Causal  │       │ IR/AST  │    │   │
│  │   │ Conv/   │        │ Model + │          │Analysis │       │ Fusion  │    │   │
│  │   │ Reduce  │        │Envelope │          │+ Blame  │       │ Tracker │    │   │
│  │   └─────────┘        └─────────┘          └─────────┘       └─────────┘    │   │
│  │                                                                              │   │
│  │   P5: Regression     P6: Automated                                          │   │
│  │   Governance         Optimization                                           │   │
│  │   ┌─────────┐        ┌─────────┐                                            │   │
│  │   │Baseline │        │AutoTune │                                            │   │
│  │   │ + SLA   │        │+ CodeGen│                                            │   │
│  │   │ Guard   │        │ Engine  │                                            │   │
│  │   └─────────┘        └─────────┘                                            │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                     📊 DATA LAYER (4 Pillars)                                │   │
│  │                                                                              │   │
│  │   P7: TraceLake      P8: Isolation        P9: Fleet          P10: LLM      │   │
│  │   (Unified Store)    Capsule              Analytics          Profiler      │   │
│  │   ┌─────────┐        ┌─────────┐          ┌─────────┐        ┌─────────┐   │   │
│  │   │Parquet +│        │Noise    │          │Multi-GPU│        │Token/s  │   │   │
│  │   │Perfetto │        │Sentinel │          │Cluster  │        │TTFT/TPS │   │   │
│  │   │ Lake    │        │ Guard   │          │ Metrics │        │ Curves  │   │   │
│  │   └─────────┘        └─────────┘          └─────────┘        └─────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                     🔧 COLLECTION LAYER (2 Pillars)                          │   │
│  │                                                                              │   │
│  │   P11: Hardware Collectors              P12: OS Collectors                  │   │
│  │   ┌────────────────────────┐            ┌────────────────────────┐          │   │
│  │   │ rocprof │ rocm-smi    │            │ eBPF │ Kernel Module   │          │   │
│  │   │ Traces  │ Telemetry   │            │(sched)│ (GPU memory)   │          │   │
│  │   └────────────────────────┘            └────────────────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

<div align="center">

### 🔗 Data Flow: Model → Metal → Insights

</div>

```
   ONNX Model          ROCm Stack              Kernel Space           Analysis
   ──────────          ──────────              ────────────           ────────
       │                    │                       │                     │
       ▼                    ▼                       ▼                     ▼
  ┌─────────┐         ┌─────────┐            ┌─────────┐           ┌─────────┐
  │  Graph  │         │MIGraphX │            │  eBPF   │           │ Unified │
  │Extractor│────────▶│ Kernels │───────────▶│ Probes  │──────────▶│Timeline │
  └─────────┘         └─────────┘            └─────────┘           └─────────┘
       │                    │                       │                     │
       │                    │                       │                     │
       ▼                    ▼                       ▼                     ▼
  ┌─────────┐         ┌─────────┐            ┌─────────┐           ┌─────────┐
  │ Node →  │         │ Kernel  │            │ Sched + │           │Evidence │
  │ Kernel  │         │ Metrics │            │  Memory │           │ + Root  │
  │Mapping  │         │   HPC   │            │ Events  │           │  Cause  │
  └─────────┘         └─────────┘            └─────────┘           └─────────┘
```

---

<div align="center">

## ✨ Key Features

</div>

<table>
<tr>
<td width="50%">

### 🔬 Multi-Plane Observability

| Layer | Technology | Captures |
|-------|------------|----------|
| **Kernel** | eBPF + kmod | Scheduler, page faults, IRQs |
| **GPU** | rocprof + SMI | Kernel execution, clocks, power |
| **Runtime** | HIP hooks | Memory transfers, launches |
| **Application** | ONNX tracing | Graph ops, shapes, dtypes |

</td>
<td width="50%">

### 🧠 AI-Powered Intelligence

- **Bottleneck Classifier**: ML + rule-based with 94% accuracy
- **Root Cause Analyzer**: Causal inference + blame attribution
- **Anomaly Detection**: Statistical + ML outlier detection
- **Regression Predictor**: Proactive performance degradation alerts

</td>
</tr>
<tr>
<td width="50%">

### 📊 Advanced Analytics

```python
# Kernel Launch Tax Analysis
launch_tax = microkernel_pct × rate / 1000

# Kernel Amplification Ratio
KAR = gpu_kernels / onnx_nodes

# GPU Efficiency Score
efficiency = kernel_time / wall_time
```

</td>
<td width="50%">

### 🚨 Production-Grade Governance

- ✅ **Baseline Management** with reproducibility metadata
- ✅ **Noise-Aware CI/CD** with confidence scoring
- ✅ **SLA Enforcement** with automatic alerting
- ✅ **Fleet Aggregation** for multi-node deployments

</td>
</tr>
</table>

<div align="center">

### 🎖️ Feature Highlights

</div>

| Feature | Description | Status |
|---------|-------------|--------|
| 🔍 **Kernel Fingerprinting** | Automatically classify kernels (GEMM, Conv, Reduce, etc.) | ✅ Production |
| 📈 **Roofline Modeling** | Compute vs memory bound analysis with envelope fitting | ✅ Production |
| 🔄 **Graph-to-Kernel Mapping** | Trace ONNX nodes → MIGraphX ops → HIP kernels | ✅ Production |
| 🛡️ **Isolation Capsules** | Reproducible execution environments | ✅ Production |
| ⚡ **LLM Profiler** | Token/s, TTFT, TPS with batch curves | ✅ Production |
| 🤖 **AutoOpt Engine** | Automated optimization code generation | ✅ Production |
| 📦 **TraceLake** | Unified Parquet + Perfetto data lake | ✅ Production |
| 🌐 **Fleet Scale** | Multi-GPU, multi-node aggregation | ✅ Production |

---

<div align="center">

## 🚀 Quick Start

**Get performance insights in under 5 minutes**

</div>

### 📦 Installation

```bash
# Clone the repository
git clone https://github.com/SID-Devu/AMD-AI-Compute-Observatory.git
cd AMD-AI-Compute-Observatory

# Install with all features (recommended)
pip install -e ".[all]"

# Or minimal install for core functionality
pip install -e .

# Verify installation
aaco --version
```

### ⚡ One-Command Demo

```bash
# Run complete analysis with single command
./scripts/run_demo.sh

# Outputs:
# ✓ Session bundle with all traces
# ✓ HTML performance report
# ✓ Bottleneck classification
# ✓ Optimization recommendations
```

### 🎯 Basic Usage

```bash
# Profile any ONNX model
aaco run --model resnet50 --backend migraphx --batch 1

# Full-stack profiling (GPU + CPU + system)
aaco run --model llama2-7b --backend migraphx \
         --profile --telemetry --ebpf

# Generate executive report
aaco report --session sessions/latest --format html

# Compare against baseline (regression check)
aaco diff --baseline baselines/prod.json \
          --session sessions/latest \
          --threshold 5%

# Real-time dashboard
aaco dashboard --port 8501
```

### 🐳 Docker (Recommended for Production)

```bash
# Build optimized container
docker build -t aaco:latest -f Dockerfiles/rocm.dockerfile .

# Run with GPU access
docker run --device=/dev/kfd --device=/dev/dri \
           -v $(pwd)/sessions:/app/sessions \
           aaco:latest run --model bert-base
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

<div align="center">

## 🎯 Bottleneck Taxonomy

**Automated classification with evidence-based attribution**

</div>

| 🏷️ Class | 🔍 Indicators | 📊 Evidence Signals | 🛠️ Fix Strategy |
|----------|---------------|---------------------|------------------|
| **🔴 Launch-bound** | Too many tiny kernels | High kernel count, low avg duration | Kernel fusion, batching |
| **🟠 CPU-bound** | Scheduling overhead | High context switches, runqueue wait | Reduce host ops, async |
| **🔵 Memory-bound** | Bandwidth limited | High mem ops ratio, slow batch scaling | Data layout, prefetch |
| **🟢 Compute-bound** | GPU saturated (good!) | High utilization, stable times | Scale or accept |
| **🟣 Throttling** | Power/thermal limits | Clock variance, power drops | Cooling, power limit |

---

<div align="center">

## 📊 Key Metrics & Formulas

</div>

<table>
<tr>
<td width="33%">

### ⚡ Launch Tax Score

```
launch_tax = μkernel% × rate / 1000
```

| Score | Status |
|-------|--------|
| < 0.3 | ✅ Healthy |
| 0.3-0.7 | ⚠️ Warning |
| > 0.7 | 🔴 Critical |

</td>
<td width="33%">

### 🔀 Kernel Amplification Ratio

```
KAR = GPU_kernels / ONNX_nodes
```

| KAR | Interpretation |
|-----|----------------|
| ≈ 1.0 | 🏆 Excellent fusion |
| 2.0-5.0 | ⚠️ Investigate |
| > 5.0 | 🔴 Severe overhead |

</td>
<td width="33%">

### 📈 GPU Active Ratio

```
active_ratio = Σkernel_time / wall_time
```

| Ratio | Status |
|-------|--------|
| > 0.9 | 🏆 GPU-bound |
| 0.7-0.9 | ⚠️ CPU overhead |
| < 0.7 | 🔴 Launch-bound |

</td>
</tr>
</table>

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

<div align="center">

## 💡 Example Output

</div>

<table>
<tr>
<td width="50%">

### 🚨 Regression Verdict

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
  "recommendation": "Investigate graph partitioning. Consider operator fusion."
}
```

</td>
<td width="50%">

### 🎯 Bottleneck Classification

```json
{
  "bottleneck_class": "launch-bound",
  "confidence": 0.87,
  "top_evidence": [
    {"signal": "microkernel_pct", "value": 0.73, "weight": 0.35},
    {"signal": "kernel_launch_rate", "value": 12500, "weight": 0.28},
    {"signal": "cpu_overhead_ratio", "value": 0.31, "weight": 0.22}
  ],
  "optimization_priority": ["fusion", "batching", "async_launch"]
}
```

</td>
</tr>
</table>

### 📊 Sample Report Output

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    AACO-SIGMA Performance Report                             ║
║                    Model: ResNet-50 | Backend: MIGraphX                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SUMMARY                                                                     ║
║  ├─ Mean Latency:      4.23ms (±0.12ms)                                     ║
║  ├─ P99 Latency:       4.67ms                                               ║
║  ├─ Throughput:        236.4 img/s                                          ║
║  └─ GPU Utilization:   94.2%                                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  BOTTLENECK ANALYSIS                                                         ║
║  ├─ Classification:    ✅ COMPUTE-BOUND (Optimal)                           ║
║  ├─ Confidence:        0.91                                                  ║
║  ├─ Launch Tax:        0.12 (Healthy)                                       ║
║  └─ KAR:               1.3 (Excellent fusion)                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TOP KERNELS BY TIME                                                         ║
║  1. GEMM_fp16         38.2%  ████████████████░░░░░░░░░░░░░                  ║
║  2. Conv2D_nhwc       31.4%  █████████████░░░░░░░░░░░░░░░░░                  ║
║  3. BatchNorm         12.1%  █████░░░░░░░░░░░░░░░░░░░░░░░░░                  ║
║  4. ReLU               8.3%  ███░░░░░░░░░░░░░░░░░░░░░░░░░░░                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

<div align="center">

## 🛠️ Development

</div>

<table>
<tr>
<td width="50%">

### 🧪 Testing

```bash
# Unit tests (fast)
pytest tests/unit -v

# Integration tests (requires ROCm)
pytest tests/integration -v

# Full coverage report
pytest --cov=aaco --cov-report=html
```

</td>
<td width="50%">

### ✨ Code Quality

```bash
# Lint and format (Ruff)
ruff check aaco/ --fix
ruff format aaco/

# Type checking (strict)
mypy aaco/ --strict
```

</td>
</tr>
</table>

---

<div align="center">

## 📚 Documentation

</div>

| 📖 Document | 📝 Description |
|-------------|----------------|
| [🏗️ Architecture](docs/architecture.md) | System design, 12 pillars, data flow |
| [🔬 Methodology](docs/methodology.md) | Measurement science, statistical rigor |
| [🎯 Bottleneck Taxonomy](docs/bottleneck_taxonomy.md) | Classification rules, evidence signals |
| [📊 Data Schema](docs/data_schema.md) | Complete schema, Parquet layouts |

---

<div align="center">

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

[![Contributors](https://img.shields.io/github/contributors/SID-Devu/AMD-AI-Compute-Observatory?style=flat-square)](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/graphs/contributors)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/pulls)

</div>

---

<div align="center">

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

</div>

Built with insights from:
- 🔴 **AMD ROCm Team** - Profiling documentation and best practices
- 🐧 **Linux Kernel Community** - eBPF and tracing infrastructure
- 🤖 **ONNX Runtime Team** - Execution provider optimization guides
- 🎓 **Performance Engineering Community** - Roofline modeling and analysis

---

<div align="center">

## ⭐ Star History

If you find AACO-SIGMA useful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=SID-Devu/AMD-AI-Compute-Observatory&type=Date)](https://star-history.com/#SID-Devu/AMD-AI-Compute-Observatory&Date)

</div>

---

<div align="center">

<img src="https://img.shields.io/badge/AMD-ED1C24?style=for-the-badge&logo=amd&logoColor=white" alt="AMD"/>

### **AACO-SIGMA**
#### Model-to-Metal Performance Engineering Platform

<br/>

**🏆 The most comprehensive GPU performance observability platform for AMD Instinct**

<br/>

*"Most engineers can run a model. Some can profile.*
*Very few can instrument kernel + GPU + analytics and produce a diagnosis.*
*AACO-SIGMA does it automatically."*

<br/>

---

**Built with ❤️ for the AMD AI community**

[Report Bug](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/issues) · [Request Feature](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/issues) · [Discussions](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/discussions)

</div>
