# Quick Start

Get started with AACO in 5 minutes!

## Basic Profiling

### Profile an ONNX Model

```bash
# Basic profile
aaco profile --model resnet50.onnx --output results/

# With more iterations for statistical significance
aaco profile --model resnet50.onnx --iterations 100 --warmup 10

# Laboratory mode for deterministic results
aaco profile --model resnet50.onnx --lab-mode --output results/lab/
```

### View Results

```bash
# Generate report
aaco report --session results/latest --format html

# Launch interactive dashboard
aaco dashboard --session results/latest

# Quick summary
aaco summary --session results/latest
```

## CLI Overview

```bash
# Get help
aaco --help

# Available commands
aaco profile    # Run profiling session
aaco analyze    # Analyze profiling data
aaco report     # Generate reports
aaco dashboard  # Launch interactive dashboard
aaco compare    # Compare multiple sessions
aaco doctor     # System diagnostics
```

## Python API

```python
from aaco import Observatory, Config

# Create observatory instance
obs = Observatory(config=Config.default())

# Run profiling
session = obs.profile(
    model="resnet50.onnx",
    iterations=100,
    warmup=10,
    lab_mode=True
)

# Analyze results
analysis = obs.analyze(session)

# Get root cause
root_cause = analysis.root_cause()
print(f"Bottleneck: {root_cause.category}")
print(f"Confidence: {root_cause.confidence:.2%}")

# Generate report
obs.report(session, format="html", output="report.html")
```

## Example Workflow

### 1. Profile Your Model

```python
from aaco import Observatory

# Initialize
obs = Observatory()

# Profile with lab mode for reproducible results
session = obs.profile(
    model="my_model.onnx",
    lab_mode=True,
    config={
        "iterations": 100,
        "warmup": 20,
        "collect_counters": True,
        "collect_traces": True
    }
)
```

### 2. Analyze Performance

```python
# Get comprehensive analysis
analysis = obs.analyze(session)

# Hardware utilization
heu = analysis.hardware_envelope_utilization()
print(f"GPU Compute: {heu.compute:.1%}")
print(f"Memory Bandwidth: {heu.memory:.1%}")

# Bottleneck classification
bottleneck = analysis.classify_bottleneck()
print(f"Type: {bottleneck.category}")
print(f"Evidence: {bottleneck.evidence}")
```

### 3. Root Cause Analysis

```python
# Bayesian root cause inference
root_cause = analysis.root_cause()

for cause in root_cause.ranked_causes:
    print(f"{cause.name}: {cause.posterior:.2%}")
    
# Get recommendations
for rec in root_cause.recommendations:
    print(f"- {rec}")
```

### 4. Track Regressions

```python
# Compare with baseline
comparison = obs.compare(
    baseline="sessions/baseline",
    current=session
)

# Check for regressions
if comparison.has_regression:
    print(f"Regression detected!")
    print(f"  Metric: {comparison.metric}")
    print(f"  Change: {comparison.change:+.2%}")
    print(f"  P-value: {comparison.p_value:.4f}")
```

## Next Steps

- [Configuration Guide](configuration.md) - Customize AACO settings
- [User Guide](../user-guide/overview.md) - Deep dive into features
- [API Reference](../api/core.md) - Complete API documentation
- [Examples](../examples/basic.md) - More usage examples
