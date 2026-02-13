# AACO Architecture

## System Overview

AMD AI Compute Observatory (AACO) is a full-stack performance observability platform for AMD AI workloads. It provides deep visibility into every layer of the AI inference pipeline, from kernel launches to end-to-end latency.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          User Interface                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  CLI Tool   │  │ HTML Report │  │   JSON API  │  │  Dashboard  │     │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                        Report & Analytics Layer                          │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌────────────────┐  │
│  │   Report Renderer   │  │ Regression Detector │  │   Bottleneck   │  │
│  │   (HTML/Terminal)   │  │   (A/B Comparison)  │  │   Classifier   │  │
│  └─────────────────────┘  └─────────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                         Metrics Engine                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Derived Metrics: Throughput, Efficiency, GPU Active Ratio, KAR │   │
│  │  Phase Analysis: Warmup vs Measurement separation               │   │
│  │  Statistical: Mean, Std, P50/P90/P99, CoV, IQR                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                       Data Collection Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │  ORT Runner  │  │   rocprof    │  │  System      │  │  GPU        │ │
│  │  (Inference) │  │   Wrapper    │  │  Sampler     │  │  Sampler    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                          Core Infrastructure                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │  Session Manager │  │   Data Schema    │  │      Utilities       │  │
│  │  (Lifecycle)     │  │   (Dataclasses)  │  │  (Timing, Proc, IO)  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                        External Dependencies                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ ONNX Runtime │  │    rocprof   │  │   rocm-smi   │  │  /proc FS   │ │
│  │ (MIGraphX EP)│  │  (profiler)  │  │  (telemetry) │  │  (Linux)    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Core Layer (`aaco/core/`)

#### Session Manager
- Creates unique session IDs (timestamp + UUID)
- Manages session folders and artifact lifecycle
- Captures environment "lockbox" (pip freeze, env vars, config)
- Provides artifact save/load with compression support

#### Data Schema
- 20+ dataclasses defining all AACO data structures
- Type-safe representation of:
  - Session metadata
  - Inference results
  - Kernel executions
  - GPU/System samples
  - Bottleneck classifications
  - Regression verdicts

#### Utilities
- High-resolution timing (monotonic nanoseconds)
- Subprocess execution with timeout
- /proc filesystem readers
- Data format conversion helpers

### 2. Data Collection Layer

#### Inference Runner (`aaco/runner/`)
- **ORTRunner**: ONNX Runtime execution with multi-backend support
  - Backends: MIGraphXExecutionProvider, ROCMExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider
  - Automatic input shape detection and dummy data generation
  - Separate warmup and measurement phases
  - Per-iteration latency capture

- **Model Registry**: Model configuration management
  - Input shape overrides
  - Backend-specific optimizations
  - Batch size handling

#### Profiler (`aaco/profiler/`)
- **RocprofWrapper**: GPU kernel profiling
  - Runs workloads under rocprof
  - Configurable tracing options (HIP, HSA, system)
  - Output file management

- **RocprofParser**: Trace analysis
  - Parses rocprof CSV output
  - Computes per-kernel statistics
  - Generates kernel summaries

#### Collectors (`aaco/collectors/`)
- **SystemSampler**: Background thread sampling from /proc
  - CPU utilization
  - Memory (RSS)
  - Context switches
  - Page faults
  - Load average

- **ROCmSMISampler**: GPU telemetry via rocm-smi
  - GPU/Memory utilization
  - Power consumption
  - Temperature
  - Clock frequencies
  - VRAM usage

- **ClockMonitor**: CPU/GPU governor state
  - Scaling governor readings
  - Performance mode validation
  - Clock stability tracking

### 3. Analytics Layer (`aaco/analytics/`)

#### Derived Metrics Engine
- Computes phase-specific statistics
- Combines multiple data sources into unified metrics
- Calculates efficiency ratios:
  - **GPU Active Ratio**: Kernel time / Wall time
  - **KAR**: Kernel count / ONNX node count
  - **Microkernel %**: Percentage of sub-10μs kernels

#### Bottleneck Classifier
- Rule-based classification with weighted indicators
- Categories:
  - Compute Bound
  - Memory Bound
  - Launch Overhead
  - CPU Bound
  - Thermal Throttle
  - Frequency Scaling
  - Warmup Instability
- Generates evidence and recommendations

#### Regression Detector
- A/B session comparison
- Statistical significance testing (Welch's t-test)
- Threshold-based verdict (regression/improvement/neutral)

### 4. Presentation Layer

#### CLI (`aaco/cli.py`)
Commands:
- `aaco run`: Execute benchmark session
- `aaco diff`: Compare two sessions
- `aaco report`: Generate session report
- `aaco ls`: List recent sessions
- `aaco info`: Display system information

#### Report Renderer (`aaco/report/`)
- Terminal: Colored text output
- HTML: Jinja2-based rich reports
- JSON: Structured data export

## Data Flow

```
1. Session Initialization
   └── Create session folder
   └── Capture environment lockbox
   
2. Warmup Phase
   └── Run N warmup iterations
   └── Collect latencies (but don't include in stats)
   
3. Measurement Phase
   └── Start telemetry collectors (background threads)
   └── Run M measurement iterations
   └── Capture per-iteration latency
   └── Stop telemetry collectors

4. Optional: Kernel Profiling
   └── Re-run with rocprof wrapper
   └── Parse kernel traces
   └── Compute kernel metrics

5. Metrics Computation
   └── Aggregate inference results
   └── Compute derived metrics
   └── Classify bottleneck

6. Artifact Storage
   └── Save all data as JSON
   └── Optional Parquet export
   └── Generate HTML report

7. Regression Analysis (optional)
   └── Load baseline session
   └── Compare metrics
   └── Statistical significance test
   └── Generate verdict
```

## Session Bundle Structure

```
sessions/
└── 20240115_143022_abc123/
    ├── session_meta.json      # Session metadata
    ├── environment.json       # pip freeze, env vars
    ├── inference_results.json # Per-iteration latencies
    ├── derived_metrics.json   # Computed metrics
    ├── bottleneck.json        # Classification result
    ├── kernel_summary.json    # Top kernel stats
    ├── sys_samples.json       # System telemetry
    ├── gpu_samples.json       # GPU telemetry
    ├── clock_state.json       # Governor/clock info
    └── report.html            # Generated report
```

## Extension Points

1. **New Backends**: Add new execution providers in `ort_runner.py`
2. **New Collectors**: Implement `start()/stop()/get_samples()` interface
3. **New Classifiers**: Extend `BottleneckClassifier` with additional rules
4. **New Report Formats**: Add renderer methods to `ReportRenderer`
5. **Custom Metrics**: Extend `DerivedMetricsEngine` with domain-specific calculations
