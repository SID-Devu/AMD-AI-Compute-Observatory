# Report Module

Performance report generation in multiple formats.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              REPORT MODULE                                      │
│                                                                                 │
│                    Performance Report Generation Engine                         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA AGGREGATION                                      │
│                                                                                 │
│    Session Bundle ──┬──► Metrics           ──┐                                 │
│                     ├──► Bottleneck Result ──┼──► Report Data Model            │
│                     ├──► Regression Verdict ─┤                                 │
│                     └──► Attribution Scores ─┘                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TEMPLATE ENGINE                                        │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                          Jinja2 Templates                                │   │
│  │                                                                          │   │
│  │    templates/                                                           │   │
│  │    ├── report.html         ◄── Main HTML report template               │   │
│  │    ├── regression.html     ◄── Regression analysis template            │   │
│  │    ├── summary_card.html   ◄── Metric summary cards                    │   │
│  │    ├── timeline.html       ◄── Timeline visualization                  │   │
│  │    └── bottleneck.html     ◄── Bottleneck classification               │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT FORMATS                                         │
│                                                                                 │
│    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                    │
│    │               │  │               │  │               │                    │
│    │     HTML      │  │     JSON      │  │   Terminal    │                    │
│    │               │  │               │  │               │                    │
│    │  Interactive  │  │  Programmatic │  │  Rich Tables  │                    │
│    │  Charts       │  │  Access       │  │  ANSI Colors  │                    │
│    │  Styling      │  │               │  │               │                    │
│    │               │  │               │  │               │                    │
│    └───────────────┘  └───────────────┘  └───────────────┘                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Report Structure

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      AACO PERFORMANCE REPORT                                     │
│                                                                                  │
│  Model: ResNet-50        Backend: MIGraphX        Session: 20260215_abc123      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  SUMMARY                                                                         │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  Mean Latency:    4.23ms (±0.12ms)                                         │ │
│  │  P99 Latency:     4.67ms                                                   │ │
│  │  Throughput:      236.4 samples/s                                          │ │
│  │  HEU Score:       87.3%                                                    │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  ATTRIBUTION METRICS                                                             │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  KAR:  1.3 (Excellent kernel fusion)                                       │ │
│  │  PFI:  0.2 (Good partitioning)                                             │ │
│  │  LTS:  0.12 (Minimal launch tax)                                           │ │
│  │  SII:  0.08 (Low scheduler interference)                                   │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  BOTTLENECK CLASSIFICATION                                                       │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  Classification:  COMPUTE-BOUND                                            │ │
│  │  Confidence:      0.87                                                     │ │
│  │  Status:          Healthy - GPU utilization is optimal                    │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
│  REGRESSION STATUS                                                               │
│  ┌────────────────────────────────────────────────────────────────────────────┐ │
│  │  Drift Detection:    STABLE (EWMA within bounds)                          │ │
│  │  CUSUM:              NO CHANGE POINT                                       │ │
│  │  Baseline Deviation: +0.8σ (Normal variation)                             │ │
│  └────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Usage

```python
from aaco.report import ReportGenerator

generator = ReportGenerator()

# Generate HTML report
generator.generate(
    session_path="sessions/20260215_abc123",
    output_path="report.html",
    format="html"
)

# Generate JSON for programmatic access
generator.generate(
    session_path="sessions/20260215_abc123",
    output_path="report.json",
    format="json"
)
```

## CLI Usage

```bash
# Generate HTML report
aaco report --session sessions/latest --format html

# Generate JSON report
aaco report --session sessions/latest --format json --output report.json

# Terminal output
aaco report --session sessions/latest --format terminal
```
