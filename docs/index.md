# AMD AI Compute Observatory

<div align="center">

![AACO Logo](assets/logo.png){ width="200" }

**AACO-Ω∞ | Model-to-Metal Performance Science & Governance Engine**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=for-the-badge&logo=github)](https://github.com/SID-Devu/AMD-AI-Compute-Observatory)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge)](../LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)

</div>

---

## 🎯 What is AACO?

AMD AI Compute Observatory (AACO) is a **deterministic, self-calibrating, cross-layer AI performance laboratory** designed for AMD hardware. It provides scientific measurement, reproducible truth, and automated governance for AI workloads.

!!! tip "This is Performance Science"
    AACO is not just another profiler. It's a complete **Performance Science Infrastructure** that answers questions with statistical rigor.

## ✨ Key Features

<div class="grid cards" markdown>

-   :material-flask:{ .lg .middle } **Deterministic Laboratory**

    ---

    cgroups v2 isolation, CPU pinning, GPU clock lock for reproducible measurements

-   :material-chart-bell-curve:{ .lg .middle } **Statistical Governance**

    ---

    EWMA + CUSUM drift detection with robust baselines (median/MAD)

-   :material-brain:{ .lg .middle } **Bayesian Root Cause**

    ---

    Posterior probability ranking with evidence-based classification

-   :material-gpu:{ .lg .middle } **Hardware Digital Twin**

    ---

    Microbenchmark-calibrated ceiling analysis with HEU scoring

</div>

## 🚀 Quick Start

```bash
# Install AACO
pip install aaco

# Run your first profile
aaco profile --model resnet50.onnx --output results/

# View the dashboard
aaco dashboard --session results/latest
```

## 📊 The 10 Pillars

AACO is built on 10 architectural pillars:

| Pillar | Name | Description |
|--------|------|-------------|
| P1 | Laboratory Mode | Deterministic execution environment |
| P2 | eBPF Forensics | Kernel-level interference detection |
| P3 | GPU Counter KFF | Counter-calibrated kernel family fingerprinting |
| P4 | Attribution | Probabilistic graph→kernel mapping |
| P5 | Digital Twin | Hardware-calibrated performance ceiling |
| P6 | Trace Lake | Unified cross-layer trace storage |
| P7 | Governance | Statistical regression detection |
| P8 | Root Cause | Bayesian inference engine |
| P9 | Auto-Opt | Hypothesis-driven optimization |
| P10 | Fleet Ops | Multi-session trending |

## 📖 Documentation

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **[Installation](getting-started/installation.md)**

    Setup AACO on your system

-   :material-rocket-launch:{ .lg .middle } **[Quick Start](getting-started/quickstart.md)**

    Get started in 5 minutes

-   :material-book-open:{ .lg .middle } **[User Guide](user-guide/overview.md)**

    Complete usage documentation

-   :material-api:{ .lg .middle } **[API Reference](api/core.md)**

    Detailed API documentation

</div>

## 📜 License

!!! warning "Proprietary Software"
    This is proprietary source-available software. See [LICENSE](https://github.com/SID-Devu/AMD-AI-Compute-Observatory/blob/master/LICENSE) for terms.

---

<div align="center">

**© 2026 Sudheer Ibrahim Daniel Devu. All Rights Reserved.**

</div>
