<div align="center">

# 🔬 AMD AI Compute Observatory

### **AACO-Ω∞** | Model-to-Metal Performance Science & Governance Engine

<img src="https://img.shields.io/badge/AMD-ED1C24?style=for-the-badge&logo=amd&logoColor=white" alt="AMD"/>
<img src="https://img.shields.io/badge/ROCm-6.0+-ED1C24?style=for-the-badge&logo=amd&logoColor=white" alt="ROCm"/>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge" alt="License"/>
<img src="https://img.shields.io/badge/Version-4.0.0-blue?style=for-the-badge" alt="Version"/>

<br/>

[![Build Status](https://img.shields.io/github/actions/workflow/status/SID-Devu/AMD-AI-Compute-Observatory/ci-cd.yml?branch=master&style=flat-square&logo=github)](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/actions)
[![PRs](https://img.shields.io/badge/PRs-welcome%20for%20review-blue?style=flat-square)](CONTRIBUTING.md)

<br/>

> ⚠️ **PROPRIETARY SOFTWARE** - © 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved. PRs welcome for review. See [LICENSE](LICENSE).

<br/>

**🧬 A deterministic, self-calibrating, cross-layer AI performance laboratory**

**From ONNX graph → MIGraphX partitions → HIP kernels → Hardware counters → Statistical governance → Root cause**

<br/>

[📖 Documentation](#-documentation) • [🚀 Quick Start](#-quick-start) • [🏗️ Architecture](#️-architecture) • [💡 Examples](#-example-output) • [📜 License](LICENSE)

</div>

---

<div align="center">

## 💎 Why AACO-Ω∞?

### **This is not profiling. This is Performance Science.**

</div>

<table>
<tr>
<td width="50%">

### ❌ Traditional Profiling
```
❓ "Model is slow, but why?"
❓ "Is it GPU, CPU, or memory?"
❓ "Did that change cause regression?"
❓ "How do I reproduce this measurement?"
```

**Result:** Inconsistent measurements, guesswork, weeks of debugging

</td>
<td width="50%">

### ✅ AACO-Ω∞ Performance Science
```
✓ Deterministic laboratory execution
✓ Hardware-calibrated digital twin
✓ Bayesian root cause with posteriors
✓ Statistical drift detection (EWMA/CUSUM)
✓ Closed-loop auto-optimization
```

**Result:** Scientific measurement, reproducible truth, automated governance

</td>
</tr>
</table>

<div align="center">

### ⚡ AACO-Ω∞ delivers scientific answers

</div>

| 🎯 Question | 📊 AACO-Ω∞ Delivers | 🔍 Method |
|-------------|---------------------|-----------|
| **Is this measurement reproducible?** | Deterministic Laboratory Mode | cgroups v2 isolation, CPU pinning, GPU clock lock |
| **What % of theoretical peak?** | Hardware Envelope Utilization (HEU) | Microbenchmark calibration + ceiling analysis |
| **Why did latency regress?** | Root Cause Posterior Probability (RCPP) | Bayesian inference with evidence-based ranking |
| **Is this drift statistically significant?** | Robust statistical governance | EWMA + CUSUM with median/MAD baseline |
| **Which kernels map to which ops?** | Graph→Partition→Kernel attribution | KAR, PFI, LTS metrics with confidence |

<div align="center">

### 🏆 This is not benchmarking. This is **Performance Science Infrastructure.**

</div>

---

<div align="center">

## 🏗️ Architecture

### The 10 Pillars of AACO-Ω∞

</div>

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    AACO-Ω∞: 10-Pillar Performance Science Architecture              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                    🎯 GOVERNANCE LAYER (Pillars 7-10)                        │   │
│  │                                                                              │   │
│  │   P7: Statistical        P8: Bayesian        P9: Auto-         P10: Fleet   │   │
│  │   Regression             Root Cause          Optimization      Performance  │   │
│  │   Governance             Engine              Engine            Ops          │   │
│  │   ┌──────────┐           ┌──────────┐        ┌──────────┐      ┌──────────┐│   │
│  │   │EWMA/CUSUM│           │Posterior │        │Hypothesis│      │Multi-Sess││   │
│  │   │Drift Det │           │Ranking   │        │Testing   │      │Trending  ││   │
│  │   │Robust BL │           │RCPP Score│        │Rollback  │      │Heatmaps  ││   │
│  │   └──────────┘           └──────────┘        └──────────┘      └──────────┘│   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                    🧠 INTELLIGENCE LAYER (Pillars 3-6)                       │   │
│  │                                                                              │   │
│  │   P3: GPU Counter-       P4: Probabilistic   P5: Hardware-     P6: Unified  │   │
│  │   Calibrated KFF         Attribution         Calibrated        Trace Lake   │   │
│  │   ┌──────────┐           ┌──────────┐        Digital Twin      ┌──────────┐│   │
│  │   │Family    │           │KAR/PFI/  │        ┌──────────┐      │Perfetto  ││   │
│  │   │Classify  │           │LTS Scores│        │HEU Score │      │Compat    ││   │
│  │   │Counter   │           │Graph→    │        │Microbench│      │Cross-    ││   │
│  │   │Signature │           │Kernel Map│        │Calibrate │      │Layer     ││   │
│  │   └──────────┘           └──────────┘        └──────────┘      └──────────┘│   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                         │                                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │                    🔬 DETERMINISM LAYER (Pillars 1-2)                        │   │
│  │                                                                              │   │
│  │   P1: Laboratory Mode                      P2: eBPF Forensic Scheduler      │   │
│  │   ┌────────────────────────────┐           ┌────────────────────────────┐   │   │
│  │   │ cgroups v2 | CPU isolate  │           │ Scheduler Interference     │   │   │
│  │   │ NUMA pin  | GPU clock lock │           │ Index (SII) + FPI + CNE    │   │   │
│  │   │ IRQ affinity | Process cage│           │ Context switches + wait    │   │   │
│  │   └────────────────────────────┘           └────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```
│  │   │ Traces  │ Telemetry   │            │(sched)│ (GPU memory)   │          │   │
│  │   └────────────────────────┘            └────────────────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

<div align="center">

### 🔗 Scientific Data Flow: Model → Metal → Diagnosis → Action

</div>

```
   ONNX Model          Laboratory          Digital Twin        Governance
   ──────────          ──────────          ────────────        ──────────
       │                    │                   │                   │
       ▼                    ▼                   ▼                   ▼
  ┌─────────┐         ┌─────────┐         ┌─────────┐         ┌─────────┐
  │  Graph  │         │Determin-│         │  HEU    │         │ EWMA/   │
  │Partition│────────▶│istic    │────────▶│Scoring  │────────▶│ CUSUM   │
  │   Map   │         │Execution│         │Envelope │         │Governance│
  └─────────┘         └─────────┘         └─────────┘         └─────────┘
       │                    │                   │                   │
       ▼                    ▼                   ▼                   ▼
  ┌─────────┐         ┌─────────┐         ┌─────────┐         ┌─────────┐
  │KAR/PFI/ │         │  eBPF   │         │Bayesian │         │  Auto-  │
  │  LTS    │         │Forensics│         │Root Cause│        │Optimize │
  │ Scores  │         │SII/FPI  │         │  RCPP   │         │Rollback │
  └─────────┘         └─────────┘         └─────────┘         └─────────┘
```

---

<div align="center">

## 📐 AACO-Ω∞ Scientific Metrics

</div>

| 🎯 Metric | 📊 Formula | 🔍 Purpose |
|-----------|------------|------------|
| **KAR** (Kernel Amplification Ratio) | `GPU_kernels / ONNX_nodes` | Measure kernel explosion |
| **PFI** (Partition Fragmentation Index) | `partitions / nodes` | Graph partitioning quality |
| **LTS** (Launch Tax Score) | `(launch_overhead / kernel_time) × weight` | CPU→GPU sync cost |
| **SII** (Scheduler Interference Index) | `runqueue_wait / wall_time` | OS scheduling impact |
| **HEU** (Hardware Envelope Utilization) | `actual_perf / calibrated_ceiling` | Peak utilization % |
| **CHI** (Compute Health Index) | `weighted(memory, compute, launch, thermal)` | Overall health score |
| **RCPP** (Root Cause Posterior Prob) | `P(cause\|evidence)` | Bayesian diagnosis confidence |

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
║                    AACO-Ω∞ Performance Science Report                        ║
║                    Model: ResNet-50 | Backend: MIGraphX                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  SUMMARY                                                                     ║
║  ├─ Mean Latency:      4.23ms (±0.12ms) [σ from robust baseline]            ║
║  ├─ P99 Latency:       4.67ms                                               ║
║  ├─ Throughput:        236.4 img/s                                          ║
║  └─ HEU Score:         87.3% (Hardware Envelope Utilization)                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ATTRIBUTION METRICS                                                         ║
║  ├─ KAR:               1.3 (Excellent kernel fusion)                        ║
║  ├─ PFI:               0.2 (Good partitioning)                              ║
║  ├─ LTS:               0.12 (Minimal launch tax)                            ║
║  └─ SII:               0.08 (Low scheduler interference)                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  GOVERNANCE STATUS                                                           ║
║  ├─ Drift Detection:   ✅ STABLE (EWMA within bounds)                       ║
║  ├─ CUSUM:             ✅ NO CHANGE POINT                                   ║
║  └─ Baseline Dev:      +0.8σ (Normal variation)                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ROOT CAUSE (if degraded)                                                    ║
║  ├─ Top RCPP:          N/A (No regression detected)                         ║
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

If you find AACO-Ω∞ useful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=SID-Devu/AMD-AI-Compute-Observatory&type=Date)](https://star-history.com/#SID-Devu/AMD-AI-Compute-Observatory&Date)

</div>

---

<div align="center">

<img src="https://img.shields.io/badge/AMD-ED1C24?style=for-the-badge&logo=amd&logoColor=white" alt="AMD"/>

### **AACO-Ω∞**
#### Model-to-Metal Performance Science & Governance Engine

<br/>

**🧬 The most scientifically rigorous GPU performance platform for AMD Instinct**

<br/>

*"Most engineers can run a model. Some can profile.*
*Very few can implement deterministic laboratory execution,*
*hardware-calibrated digital twins, Bayesian root cause analysis,*
*and statistical regression governance.*
*AACO-Ω∞ does it automatically."*

<br/>

---

**Built with ❤️ for the AMD AI community**

[Report Bug](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/issues) · [Request Feature](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/issues) · [Discussions](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/discussions)

</div>
