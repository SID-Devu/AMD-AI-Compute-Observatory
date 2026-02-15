# Analytics Module

Performance analysis, bottleneck classification, and diagnostics.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             ANALYTICS MODULE                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │         Raw Trace Data          │
                    │                                 │
                    │  Kernel Traces, GPU Telemetry,  │
                    │  System Metrics, Inference Log  │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           METRICS ENGINE                                        │
│                                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
│  │   metrics.py    │  │  launch_tax.py  │  │   chi.py        │                │
│  │                 │  │                 │  │                 │                │
│  │  • Throughput   │  │  • LTS Score    │  │  • CHI Score    │                │
│  │  • Latency P99  │  │  • Microkernel% │  │  • Health Index │                │
│  │  • GPU Active   │  │  • Launch Rate  │  │  • Composite    │                │
│  │  • Memory BW    │  │                 │  │                 │                │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
│                                                                                 │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ATTRIBUTION ENGINE                                      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         attribution.py                                   │   │
│  │                                                                          │   │
│  │    ONNX Node ──────► MIGraphX Partition ──────► HIP Kernel              │   │
│  │                                                                          │   │
│  │    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │   │
│  │    │    KAR      │    │    PFI      │    │    LTS      │                │   │
│  │    │   Score     │    │   Score     │    │   Score     │                │   │
│  │    │             │    │             │    │             │                │   │
│  │    │  kernels/   │    │ partitions/ │    │  launch/    │                │   │
│  │    │  nodes      │    │   nodes     │    │  kernel     │                │   │
│  │    └─────────────┘    └─────────────┘    └─────────────┘                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      BOTTLENECK CLASSIFIER                                      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          classify.py                                     │   │
│  │                                                                          │   │
│  │  Input Features ──► Rule Engine ──► Confidence Score ──► Classification │   │
│  │                                                                          │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐            │   │
│  │  │  Launch   │  │   CPU     │  │  Memory   │  │  Compute  │            │   │
│  │  │  Bound    │  │   Bound   │  │  Bound    │  │  Bound    │            │   │
│  │  │           │  │           │  │           │  │           │            │   │
│  │  │ High ker- │  │ High ctx  │  │ High mem  │  │ High GPU  │            │   │
│  │  │ nel count │  │ switches  │  │ ops ratio │  │ utiliz.   │            │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       ROOT CAUSE ANALYSIS                                       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         root_cause.py                                    │   │
│  │                                                                          │   │
│  │    Evidence ──► Bayesian Inference ──► Posterior Ranking ──► RCPP      │   │
│  │                                                                          │   │
│  │    P(cause | evidence) = P(evidence | cause) × P(cause) / P(evidence)  │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     REGRESSION DETECTION                                        │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      regression_guard.py                                 │   │
│  │                                                                          │   │
│  │    Baseline ──► EWMA Detection ──► CUSUM Analysis ──► Verdict          │   │
│  │                                                                          │   │
│  │    ┌───────────────────────────────────────────────────────────────┐   │   │
│  │    │  EWMA: λ × current + (1-λ) × previous                          │   │   │
│  │    │  CUSUM: Σmax(0, x_i - μ - k)                                   │   │   │
│  │    │  Robust Baseline: median ± 3 × MAD                             │   │   │
│  │    └───────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Components

| File | Purpose |
|------|---------|
| `metrics.py` | Derived metrics computation |
| `classify.py` | Bottleneck classification |
| `attribution.py` | Graph-to-kernel attribution |
| `root_cause.py` | Bayesian root cause analysis |
| `regression_guard.py` | Statistical regression detection |
| `launch_tax.py` | Launch overhead analysis |
| `chi.py` | Compute Health Index |
| `diff.py` | A/B comparison |
| `kernel_fingerprint.py` | Kernel family classification |
| `recommendation_engine.py` | Optimization recommendations |

## Key Metrics

| Metric | Formula | Range |
|--------|---------|-------|
| KAR | `gpu_kernels / onnx_nodes` | 1.0 - 10.0+ |
| PFI | `partitions / nodes` | 0.0 - 1.0 |
| LTS | `launch_overhead / kernel_time` | 0.0 - 1.0 |
| HEU | `actual / calibrated_ceiling` | 0.0 - 1.0 |
| CHI | `weighted(mem, compute, launch, thermal)` | 0.0 - 1.0 |
