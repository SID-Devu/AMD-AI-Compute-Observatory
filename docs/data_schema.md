# Data Schema Reference

## Overview

All AACO data is strongly typed using Python dataclasses. This document provides the complete schema reference.

## Session Metadata

```python
@dataclass
class SessionMetadata:
    session_id: str          # Unique identifier (timestamp_uuid)
    timestamp: str           # ISO 8601 timestamp
    tag: Optional[str]       # User-provided tag for identification
    hostname: str            # Machine hostname
    platform: str            # OS platform string
    python_version: str      # Python version
    aaco_version: str        # AACO version
    gpu: Dict[str, Any]      # GPU information dict
    model_path: Optional[str] # Path to model being benchmarked
    backend: Optional[str]   # Execution provider used
    config: Dict[str, Any]   # Runtime configuration
```

## Inference Results

```python
@dataclass
class InferenceResult:
    iteration: int           # Iteration number (0-indexed)
    phase: str              # "warmup" or "measurement"
    latency_ms: float       # Per-iteration latency in milliseconds
    t_start_ns: int         # Start timestamp (monotonic ns)
    t_end_ns: int           # End timestamp (monotonic ns)
```

## Kernel Metrics

```python
@dataclass
class KernelExecution:
    t_start_ns: int         # Kernel start time
    t_end_ns: int           # Kernel end time
    dur_ns: int             # Duration in nanoseconds
    kernel_name: str        # Full kernel name
    queue_id: int           # GPU queue ID
    
@dataclass
class KernelSummary:
    kernel_name: str        # Kernel name
    calls: int              # Number of invocations
    total_time_ms: float    # Total time across all calls
    avg_time_us: float      # Average per-call time
    min_time_us: float      # Minimum call time
    max_time_us: float      # Maximum call time
    std_time_us: float      # Standard deviation
    pct_total: float        # Percentage of total kernel time

@dataclass
class KernelMetrics:
    total_kernel_count: int        # Total kernel launches
    unique_kernel_count: int       # Unique kernel names
    total_kernel_time_ms: float    # Sum of all kernel times
    avg_kernel_duration_us: float  # Average kernel duration
    microkernel_count: int         # Kernels < threshold
    microkernel_pct: float         # Percentage of microkernels
    microkernel_threshold_us: float # Threshold (default 10μs)
    launch_rate_per_sec: float     # Kernel launches per second
    launch_tax_score: float        # Combined launch overhead score
    kernel_amplification_ratio: float # Kernels per ONNX node
    gpu_active_ratio: float        # Kernel time / wall time
    top_kernels: List[KernelSummary] # Top N kernels by time
```

## Telemetry Samples

```python
@dataclass
class SystemSample:
    timestamp_ns: int       # Sample timestamp
    cpu_pct: Optional[float] # CPU utilization percentage
    rss_mb: Optional[float]  # Resident set size (MB)
    ctx_switches: Optional[int] # Context switches
    page_faults: Optional[int]  # Page faults
    load_avg_1m: Optional[float] # 1-minute load average

@dataclass
class GPUSample:
    timestamp_ns: int           # Sample timestamp
    device_id: int              # GPU device index
    gpu_util_pct: Optional[float]  # GPU utilization %
    mem_util_pct: Optional[float]  # Memory utilization %
    temp_c: Optional[float]        # Temperature Celsius
    power_w: Optional[float]       # Power consumption Watts
    sclk_mhz: Optional[int]        # GPU clock MHz
    mclk_mhz: Optional[int]        # Memory clock MHz
    vram_used_mb: Optional[float]  # VRAM used MB
    vram_total_mb: Optional[float] # VRAM total MB
```

## Phase Metrics

```python
@dataclass
class PhaseMetrics:
    name: str               # Phase name ("warmup" or "measurement")
    iterations: int         # Number of iterations
    total_time_ms: float    # Total phase time
    mean_ms: float          # Mean latency
    std_ms: float           # Standard deviation
    p50_ms: float           # Median (50th percentile)
    p90_ms: float           # 90th percentile
    p99_ms: float           # 99th percentile
    min_ms: float           # Minimum latency
    max_ms: float           # Maximum latency
    iqr_ms: float           # Interquartile range
    cov_pct: float          # Coefficient of variation (%)
```

## Derived Metrics

```python
@dataclass
class DerivedMetrics:
    warmup_phase: PhaseMetrics      # Warmup statistics
    measurement_phase: PhaseMetrics # Measurement statistics
    throughput: Dict[str, float]    # Throughput metrics
    efficiency: Dict[str, float]    # Efficiency metrics
    latency: Dict[str, float]       # Aggregate latency stats
    system: Dict[str, float]        # System utilization
    gpu: Dict[str, float]           # GPU utilization
```

## Bottleneck Classification

```python
@dataclass
class BottleneckClassification:
    primary: str                    # Primary bottleneck category
    secondary: List[str]            # Secondary factors
    confidence: float               # Classification confidence (0-1)
    indicators: Dict[str, float]    # Indicator values used
    evidence: List[str]             # Human-readable evidence strings
    recommendations: List[str]      # Optimization suggestions
```

## Regression Verdict

```python
@dataclass
class RegressionVerdict:
    verdict: str                    # "REGRESSION", "IMPROVEMENT", "NEUTRAL"
    confidence: float               # Verdict confidence (0-1)
    regressions: List[str]          # Metrics that regressed
    improvements: List[str]         # Metrics that improved
    comparisons: List[Dict]         # Per-metric comparisons
    p_value: Optional[float]        # Statistical significance
    summary: str                    # Human-readable summary
```

## JSON Serialization

All schema objects serialize to JSON via `__dict__`:

```python
# Writing
with open("inference_results.json", "w") as f:
    json.dump([r.__dict__ for r in results], f, indent=2)

# Reading
with open("inference_results.json") as f:
    data = json.load(f)
    results = [InferenceResult(**d) for d in data]
```

## Parquet Schema (Future)

For analytics at scale, results export to Parquet with these columns:

```
inference_results.parquet:
  - session_id: string
  - iteration: int64
  - phase: string (categorical)
  - latency_ms: float64
  - t_start_ns: int64
  - t_end_ns: int64

kernel_traces.parquet:
  - session_id: string
  - kernel_name: string
  - t_start_ns: int64
  - t_end_ns: int64
  - dur_ns: int64
  - queue_id: int32

telemetry.parquet:
  - session_id: string
  - timestamp_ns: int64
  - source: string (categorical: "system", "gpu")
  - metric: string
  - value: float64
```
