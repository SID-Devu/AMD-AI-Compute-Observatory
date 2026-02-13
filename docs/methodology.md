# AACO Methodology

## Performance Measurement Philosophy

### Principles

1. **Session-Based Evidence**: Every benchmark run produces a complete, self-contained "evidence bundle" that captures all context needed to understand and reproduce results.

2. **Wall Clock Reality**: We measure what matters - end-to-end latency as experienced by applications - while also capturing the decomposition needed for optimization.

3. **Statistical Rigor**: Performance measurements are inherently noisy. We use appropriate statistical methods (percentiles, confidence intervals, significance tests) rather than single-point comparisons.

4. **Full-Stack Visibility**: Performance issues can arise at any layer. AACO instruments from kernel launches through inference APIs to application-level timing.

## Measurement Phases

### Warmup Phase

**Purpose**: Allow the system to reach steady state before measurement.

**What happens during warmup**:
- JIT compilation completes
- Memory pools are allocated
- GPU caches warm up
- Frequency governors stabilize

**Configuration**: Default 10 iterations, adjustable via `--warmup`

**Key Insight**: Compare warmup vs measurement to detect:
- JIT compilation overhead
- Memory allocation delays
- Lazy initialization costs

### Measurement Phase

**Purpose**: Capture representative steady-state performance.

**Methodology**:
- Run N iterations (default 100)
- Record per-iteration latency with high-resolution timer
- Compute statistical aggregates (mean, std, percentiles)

**Why per-iteration matters**:
- Detects outliers and tail latency
- Reveals performance variability
- Enables statistical comparison

## Key Metrics

### Latency Metrics

| Metric | Definition | Use Case |
|--------|------------|----------|
| Mean | Arithmetic average | General comparison |
| Median (P50) | 50th percentile | Typical user experience |
| P90/P99 | 90th/99th percentile | Tail latency (SLA) |
| Std Dev | Standard deviation | Consistency |
| CoV | Std/Mean × 100% | Normalized variability |
| IQR | P75 - P25 | Robust spread measure |

### Efficiency Metrics

#### GPU Active Ratio
```
GPU Active Ratio = Total Kernel Time / Wall Clock Time
```

**Interpretation**:
- 1.0 = GPU is fully utilized (100% of time in kernels)
- 0.5 = GPU is idle half the time
- Low values indicate launch overhead, data transfer, or CPU bottlenecks

#### Kernel Amplification Ratio (KAR)
```
KAR = Total GPU Kernel Count / ONNX Node Count
```

**Interpretation**:
- 1.0 = Perfect 1:1 mapping (ideal)
- 10.0 = Each ONNX node spawns 10 kernels (fusion opportunity)
- High KAR + many microkernels = launch overhead bottleneck

#### Microkernel Percentage
```
Microkernel % = (Kernels < 10μs) / Total Kernels × 100%
```

**Interpretation**:
- <10% = Healthy (mostly substantial kernels)
- >30% = Launch overhead concern
- >50% = Severe fragmentation (optimization critical)

## Bottleneck Classification

### Taxonomy

| Category | Indicators | Root Cause |
|----------|------------|------------|
| **Compute Bound** | High GPU util (>85%), High GPU Active Ratio | Kernels are compute-intensive |
| **Memory Bound** | High mem util, Low GPU util | Bandwidth-limited kernels |
| **Launch Overhead** | High microkernel %, High KAR, Low GPU Active Ratio | Too many small kernels |
| **CPU Bound** | High CPU util, Low GPU util | Host code limiting GPU |
| **Thermal Throttle** | High temp (>85°C), Clock variation | Power/cooling limits |
| **Data Transfer** | Low GPU util, PCIe activity | Host-device copies |

### Evidence-Based Classification

AACO uses weighted indicator scoring:

```python
score = 0
if microkernel_pct > 30:
    score += 0.4
    evidence.append("High microkernel %")
if kar > 10:
    score += 0.3
    evidence.append("High kernel amplification")
if gpu_active_ratio < 0.3:
    score += 0.3
    evidence.append("Low GPU active ratio")

if score > 0.6:
    classification = "LAUNCH_OVERHEAD"
```

## Regression Detection

### Statistical Methodology

1. **Threshold-Based**: Simple percentage comparison
   - Regression: >5% slower
   - Improvement: >5% faster
   - Neutral: Within 5%

2. **Statistical Significance**: Welch's t-test
   - Compare latency distributions
   - Account for unequal variances
   - Report p-value

3. **Combined Verdict**:
   - Use both threshold and significance
   - Higher confidence when statistically significant
   - Conservative default (avoid false positives)

### Comparison Dimensions

| Metric | Higher is Better | Threshold |
|--------|------------------|-----------|
| Mean latency | No | 5% |
| P99 latency | No | 10% |
| Throughput | Yes | 5% |
| GPU Active Ratio | Yes | 5% |

## Best Practices

### Reproducible Benchmarks

1. **Lock clocks**: Use performance governor, fixed frequencies
2. **Isolate system**: Minimize background processes
3. **Warm up**: Sufficient warmup to reach steady state
4. **Repeat**: Multiple sessions for confidence

### Environment Capture

AACO automatically captures:
- pip freeze (Python environment)
- Environment variables
- GPU configuration (rocm-smi)
- Clock/governor state
- Host information

### Meaningful Comparisons

1. **Same hardware**: Compare on identical systems
2. **Same workload**: Identical models, inputs, config
3. **Same conditions**: Temperature, power, load
4. **Statistical validity**: Enough iterations for confidence

## Profiling Methodology

### When to Profile

- After identifying performance target
- When classifying bottleneck type
- Investigating specific optimization

### rocprof Usage

```bash
# Trace HIP API calls and kernels
aaco run model.onnx --profile

# Full kernel trace
rocprof --hip-trace --stats ./benchmark
```

### Kernel Analysis

1. **Identify top kernels**: Focus on highest total time
2. **Analyze distribution**: Look for outliers
3. **Check launch patterns**: Detect fragmentation
4. **Correlate with ONNX nodes**: Map to model architecture
