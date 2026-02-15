# Governance Module

Statistical regression detection, baseline management, and CI/CD integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            GOVERNANCE MODULE                                    │
│                                                                                 │
│                Statistical Regression Governance Framework                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         BASELINE MANAGER                                        │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      baseline_manager.py                                 │   │
│  │                                                                          │   │
│  │    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐          │   │
│  │    │               │    │               │    │               │          │   │
│  │    │  Historical   │───▶│    Robust     │───▶│   Baseline    │          │   │
│  │    │   Sessions    │    │   Statistics  │    │    Store      │          │   │
│  │    │               │    │               │    │               │          │   │
│  │    │  Parquet DB   │    │  Median, MAD  │    │  JSON Export  │          │   │
│  │    │               │    │  IQR, P99     │    │  Versioned    │          │   │
│  │    │               │    │               │    │               │          │   │
│  │    └───────────────┘    └───────────────┘    └───────────────┘          │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                       REGRESSION DETECTOR                                       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     regression_detector.py                               │   │
│  │                                                                          │   │
│  │    Current Session ──────────────────────────────────────────────────   │   │
│  │                            │                                             │   │
│  │                            ▼                                             │   │
│  │    ┌───────────────────────────────────────────────────────────────┐   │   │
│  │    │                    EWMA Detection                              │   │   │
│  │    │                                                                │   │   │
│  │    │    EWMA_t = λ × X_t + (1 - λ) × EWMA_{t-1}                   │   │   │
│  │    │                                                                │   │   │
│  │    │    λ = 0.3 (default smoothing factor)                         │   │   │
│  │    │    Alert when |EWMA - baseline| > 3σ                          │   │   │
│  │    │                                                                │   │   │
│  │    └───────────────────────────────────────────────────────────────┘   │   │
│  │                            │                                             │   │
│  │                            ▼                                             │   │
│  │    ┌───────────────────────────────────────────────────────────────┐   │   │
│  │    │                    CUSUM Analysis                              │   │   │
│  │    │                                                                │   │   │
│  │    │    S_t^+ = max(0, S_{t-1}^+ + (x_t - μ - k))                 │   │   │
│  │    │    S_t^- = max(0, S_{t-1}^- + (μ - k - x_t))                 │   │   │
│  │    │                                                                │   │   │
│  │    │    k = 0.5σ (slack), h = 5σ (threshold)                       │   │   │
│  │    │    Change point when S > h                                     │   │   │
│  │    │                                                                │   │   │
│  │    └───────────────────────────────────────────────────────────────┘   │   │
│  │                            │                                             │   │
│  │                            ▼                                             │   │
│  │    ┌───────────────────────────────────────────────────────────────┐   │   │
│  │    │                  Regression Verdict                            │   │   │
│  │    │                                                                │   │   │
│  │    │    regression: true/false                                      │   │   │
│  │    │    severity: none/low/medium/high/critical                    │   │   │
│  │    │    confidence: 0.0 - 1.0                                      │   │   │
│  │    │    delta_pct: percentage change                               │   │   │
│  │    │                                                                │   │   │
│  │    └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         BAYESIAN ENGINE                                         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       bayesian_engine.py                                 │   │
│  │                                                                          │   │
│  │    Evidence ──► Prior ──► Likelihood ──► Posterior ──► RCPP Ranking    │   │
│  │                                                                          │   │
│  │    P(cause | evidence) ∝ P(evidence | cause) × P(cause)                │   │
│  │                                                                          │   │
│  │    Causes:                                                               │   │
│  │    • launch-bound (kernel explosion)                                    │   │
│  │    • memory-bound (bandwidth limited)                                   │   │
│  │    • compute-bound (GPU saturated)                                      │   │
│  │    • cpu-bound (host overhead)                                          │   │
│  │    • thermal-throttle (power/temp limited)                              │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CI INTEGRATION                                          │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       ci_integration.py                                  │   │
│  │                                                                          │   │
│  │    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │   │
│  │    │             │    │             │    │             │                │   │
│  │    │   GitHub    │    │   GitLab    │    │   Jenkins   │                │   │
│  │    │   Actions   │    │   CI/CD     │    │   Pipeline  │                │   │
│  │    │             │    │             │    │             │                │   │
│  │    │  Exit code  │    │  JUnit XML  │    │  JSON API   │                │   │
│  │    │  Artifacts  │    │  Artifacts  │    │  Artifacts  │                │   │
│  │    │             │    │             │    │             │                │   │
│  │    └─────────────┘    └─────────────┘    └─────────────┘                │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Components

### baseline_manager.py

Baseline creation and management:

```python
from aaco.governance.baseline_manager import BaselineManager

manager = BaselineManager()

# Create baseline from sessions
baseline = manager.create_baseline(
    sessions=["session_1", "session_2", "session_3"],
    method="robust"  # median + MAD
)

# Save baseline
manager.save("baselines/production_v1.json", baseline)
```

### regression_detector.py

Statistical regression detection:

| Method | Description | Use Case |
|--------|-------------|----------|
| EWMA | Exponentially weighted moving average | Trend detection |
| CUSUM | Cumulative sum control chart | Change point detection |
| Robust | Median + MAD comparison | Outlier-resistant |

### sla_engine.py

SLA enforcement:

```python
from aaco.governance.sla_engine import SLAEngine

sla = SLAEngine(
    p99_latency_ms=10.0,
    throughput_min=100.0,
    regression_threshold_pct=5.0
)

# Check compliance
result = sla.check(session)
if not result.compliant:
    print(result.violations)
```

## Output

```json
{
  "regression": true,
  "severity": "high",
  "confidence": 0.92,
  "latency_delta_pct": 18.3,
  "suspected_cause": "launch-bound",
  "evidence": {
    "kernel_count_delta": "+67%",
    "avg_kernel_duration_delta": "-35%"
  },
  "recommendation": "Investigate graph partitioning"
}
```

## CI/CD Integration

```yaml
# GitHub Actions example
- name: Performance Regression Check
  run: |
    aaco diff \
      --baseline baselines/main.json \
      --session sessions/latest \
      --threshold 5% \
      --fail-on-regression
```
