# Governance Module

Statistical regression detection and baseline management.

## Components

| Component | Description |
|-----------|-------------|
| `baseline.py` | Baseline creation and management |
| `detector.py` | Regression detection algorithms |
| `diff.py` | Session comparison and verdict generation |

## Detection Algorithms

| Algorithm | Use Case |
|-----------|----------|
| EWMA | Exponentially weighted moving average for trend detection |
| CUSUM | Cumulative sum for shift detection |
| Robust Baseline | Outlier-resistant baseline computation |

## Verdict Classification

| Verdict | Criteria |
|---------|----------|
| `PASS` | Within threshold, statistically insignificant change |
| `REGRESSED` | Exceeds threshold with statistical significance |
| `IMPROVED` | Performance improvement detected |
| `INCONCLUSIVE` | Insufficient data or high variance |

## Usage

```python
from aaco.governance import BaselineManager, RegressionDetector

manager = BaselineManager()
baseline = manager.load("production-v1.0")

detector = RegressionDetector(threshold_pct=5.0)
verdict = detector.compare(baseline, current_session)

if verdict.regressed:
    print(f"Regression detected: {verdict.delta_pct:.1f}%")
```
