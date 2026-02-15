# Architecture

## System Overview

AMD AI Compute Observatory (AACO) implements a layered architecture for comprehensive performance analysis of AI workloads on AMD Instinct accelerators.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                        AMD AI COMPUTE OBSERVATORY                               │
│                                                                                 │
│                   Model-to-Metal Performance Analysis Platform                  │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ INTERFACE LAYER                                                                 │
│                                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │               │  │               │  │               │  │               │   │
│  │   CLI Tool    │  │  HTML Report  │  │   JSON API    │  │   Dashboard   │   │
│  │               │  │               │  │               │  │               │   │
│  │   aaco run    │  │  Interactive  │  │  Programmatic │  │   Streamlit   │   │
│  │   aaco diff   │  │  Visualization│  │    Access     │  │   Real-time   │   │
│  │               │  │               │  │               │  │               │   │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ GOVERNANCE LAYER                                                                │
│                                                                                 │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐     │
│  │                     │  │                     │  │                     │     │
│  │     Statistical     │  │     Root Cause      │  │       Fleet         │     │
│  │     Regression      │  │      Analysis       │  │     Operations      │     │
│  │     Governance      │  │       Engine        │  │                     │     │
│  │                     │  │                     │  │                     │     │
│  │  • EWMA Detection   │  │  • Bayesian RCPP    │  │  • Multi-session    │     │
│  │  • CUSUM Analysis   │  │  • Evidence Ranking │  │  • Trend Analysis   │     │
│  │  • Robust Baseline  │  │  • Causal Inference │  │  • Fleet Heatmaps   │     │
│  │                     │  │                     │  │                     │     │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ INTELLIGENCE LAYER                                                              │
│                                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │               │  │               │  │               │  │               │   │
│  │    Kernel     │  │ Probabilistic │  │   Hardware    │  │   Unified     │   │
│  │  Fingerprint  │  │  Attribution  │  │    Digital    │  │    Trace      │   │
│  │    Engine     │  │    Engine     │  │     Twin      │  │     Lake      │   │
│  │               │  │               │  │               │  │               │   │
│  │  • Family     │  │  • KAR Score  │  │  • HEU Score  │  │  • Parquet    │   │
│  │    Classify   │  │  • PFI Score  │  │  • Microbench │  │  • Perfetto   │   │
│  │  • Counter    │  │  • LTS Score  │  │  • Calibrate  │  │  • Cross-ref  │   │
│  │    Signature  │  │  • Graph Map  │  │               │  │               │   │
│  │               │  │               │  │               │  │               │   │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ MEASUREMENT LAYER                                                               │
│                                                                                 │
│  ┌─────────────────────────────────┐  ┌─────────────────────────────────────┐ │
│  │                                 │  │                                     │ │
│  │        Laboratory Mode          │  │       eBPF Forensic Tracer          │ │
│  │                                 │  │                                     │ │
│  │  • cgroups v2 Isolation         │  │  • Scheduler Interference (SII)     │ │
│  │  • CPU Core Pinning             │  │  • Fault Penalty Index (FPI)        │ │
│  │  • NUMA Memory Binding          │  │  • Context Switch Tracking          │ │
│  │  • GPU Clock Locking            │  │  • Cache Miss Events                │ │
│  │  • IRQ Affinity Control         │  │                                     │ │
│  │                                 │  │                                     │ │
│  └─────────────────────────────────┘  └─────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ COLLECTION LAYER                                                                │
│                                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │               │  │               │  │               │  │               │   │
│  │  ORT Runner   │  │   rocprof     │  │    System     │  │     GPU       │   │
│  │               │  │   Wrapper     │  │    Sampler    │  │    Sampler    │   │
│  │  • MIGraphX   │  │               │  │               │  │               │   │
│  │  • ROCm EP    │  │  • HIP Trace  │  │  • CPU Load   │  │  • Clocks     │   │
│  │  • CUDA EP    │  │  • HSA Trace  │  │  • Memory     │  │  • Power      │   │
│  │  • CPU EP     │  │  • Counters   │  │  • Context    │  │  • Temp       │   │
│  │               │  │               │  │               │  │               │   │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ INFRASTRUCTURE LAYER                                                            │
│                                                                                 │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐     │
│  │                     │  │                     │  │                     │     │
│  │   Session Manager   │  │     Data Schema     │  │      Utilities      │     │
│  │                     │  │                     │  │                     │     │
│  │  • Lifecycle        │  │  • Type-safe        │  │  • High-res timing  │     │
│  │  • Artifact Store   │  │  • Dataclasses      │  │  • Subprocess mgmt  │     │
│  │  • Environment      │  │  • Parquet Schema   │  │  • /proc readers    │     │
│  │                     │  │                     │  │                     │     │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ EXTERNAL DEPENDENCIES                                                           │
│                                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │               │  │               │  │               │  │               │   │
│  │ ONNX Runtime  │  │    rocprof    │  │   rocm-smi    │  │    Linux      │   │
│  │               │  │               │  │               │  │    Kernel     │   │
│  │  MIGraphX EP  │  │  GPU Profiler │  │   Telemetry   │  │  eBPF/proc    │   │
│  │               │  │               │  │               │  │               │   │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐
    │             │
    │ ONNX Model  │
    │             │
    └──────┬──────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      GRAPH ANALYSIS                              │
    │                                                                  │
    │   Model Loading ──► Node Extraction ──► Partition Mapping       │
    │                                                                  │
    │   Output: graph_nodes.parquet, graph_edges.parquet              │
    └─────────────────────────────────────────────────────────────────┘
           │
           ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   DETERMINISTIC EXECUTION                        │
    │                                                                  │
    │   Laboratory Setup ──► Warmup Phase ──► Measurement Phase       │
    │                                                                  │
    │   Output: inference_iters.parquet, inference_summary.json       │
    └─────────────────────────────────────────────────────────────────┘
           │
           ├───────────────────┬───────────────────┬──────────────────┐
           ▼                   ▼                   ▼                  ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    ┌─────────────┐
    │   rocprof   │     │   System    │     │    GPU      │    │    eBPF     │
    │   Tracing   │     │   Sampling  │     │   Sampling  │    │   Tracing   │
    │             │     │             │     │             │    │             │
    │ Kernel exec │     │  CPU, Mem   │     │ Clocks, Pwr │    │  Scheduler  │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘    └──────┬──────┘
           │                   │                   │                  │
           └───────────────────┴───────────────────┴──────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                      UNIFIED TRACE LAKE                          │
    │                                                                  │
    │   Parquet Storage ──► Cross-Reference ──► Perfetto Export       │
    │                                                                  │
    │   Schema: TimestampNs, EventType, Duration, Metadata            │
    └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                     ATTRIBUTION ENGINE                           │
    │                                                                  │
    │   Graph→Kernel Mapping ──► KAR/PFI/LTS Scoring ──► Grouping    │
    │                                                                  │
    │   Output: kernel_groups.parquet, op_to_kernel_map.parquet       │
    └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   BOTTLENECK CLASSIFICATION                      │
    │                                                                  │
    │   Feature Extraction ──► Rule Engine ──► Confidence Scoring    │
    │                                                                  │
    │   Categories: Launch-bound, CPU-bound, Memory-bound,            │
    │               Compute-bound, Thermal-throttled                   │
    └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                   STATISTICAL GOVERNANCE                         │
    │                                                                  │
    │   EWMA Detection ──► CUSUM Analysis ──► Baseline Comparison     │
    │                                                                  │
    │   Output: drift_status, change_points, regression_verdict       │
    └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    ROOT CAUSE ANALYSIS                           │
    │                                                                  │
    │   Evidence Collection ──► Bayesian Inference ──► RCPP Ranking  │
    │                                                                  │
    │   Output: suspected_cause, posterior_probability, evidence      │
    └─────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │                    REPORT GENERATION                             │
    │                                                                  │
    │   Template Rendering ──► Visualization ──► Export               │
    │                                                                  │
    │   Formats: HTML, JSON, Terminal, Perfetto                       │
    └─────────────────────────────────────────────────────────────────┘
```

---

## Module Structure

```
aaco/
├── core/                    # Infrastructure and session management
│   ├── session.py           # Session lifecycle management
│   ├── schema.py            # Data structures and type definitions
│   └── utils.py             # Common utilities
│
├── collectors/              # Data collection subsystem
│   ├── rocm_smi_sampler.py  # GPU telemetry collection
│   ├── sys_sampler.py       # System metrics collection
│   ├── clocks.py            # Clock management
│   └── driver_interface.py  # Low-level driver access
│
├── profiler/                # GPU profiling subsystem
│   └── rocprof_wrapper.py   # rocprof integration
│
├── analytics/               # Analysis and diagnostics
│   ├── classify.py          # Bottleneck classification
│   ├── metrics.py           # Derived metrics computation
│   ├── attribution.py       # Graph-to-kernel attribution
│   ├── root_cause.py        # Bayesian root cause analysis
│   ├── regression_guard.py  # Statistical regression detection
│   └── recommendation_engine.py
│
├── governance/              # Fleet and regression governance
├── laboratory/              # Deterministic execution environment
├── calibration/             # Hardware calibration subsystem
├── tracelake/               # Unified trace storage
├── report/                  # Report generation
├── dashboard/               # Real-time monitoring
└── cli.py                   # Command-line interface
```

---

## Key Design Principles

### Deterministic Measurement

All measurements are designed for reproducibility:
- Configurable warmup iterations to reach steady state
- Statistical aggregation over measurement iterations
- Environment capture for reproducibility verification
- Isolation mechanisms to minimize interference

### Cross-Layer Attribution

Performance data is correlated across execution layers:
- ONNX graph nodes map to MIGraphX partitions
- Partitions map to HIP kernel launches
- Kernel execution correlates with hardware counters
- System events provide context for anomalies

### Statistical Rigor

All comparisons use statistically sound methods:
- Robust baseline computation using median/MAD
- EWMA for drift detection with configurable sensitivity
- CUSUM for change point detection
- Confidence intervals for all reported metrics

### Extensibility

The architecture supports extension:
- Pluggable collectors for new data sources
- Modular analytics for new classification rules
- Template-based reporting for custom formats
- API access for integration with external systems

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | AMD Instinct MI100 | AMD Instinct MI300X |
| ROCm | 5.6+ | 6.0+ |
| Memory | 32GB | 64GB+ |
| Storage | SSD 100GB | NVMe 500GB+ |

---

## Performance Characteristics

| Operation | Typical Duration | Notes |
|-----------|-----------------|-------|
| Model load | 1-10s | Depends on model size |
| Warmup | 5-30s | Configurable iterations |
| Measurement | 10-60s | Configurable iterations |
| rocprof trace | +20-50% overhead | Profiling adds overhead |
| Report generation | 1-5s | Depends on data volume |

---

## Related Documentation

- [Methodology](methodology.md) - Measurement methodology details
- [Data Schema](data_schema.md) - Complete data schema reference
- [Bottleneck Taxonomy](bottleneck_taxonomy.md) - Classification rules
