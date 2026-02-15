# AACO Source Code

AMD AI Compute Observatory - Core Python Package

## Module Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               AACO MODULES                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │    cli.py   │
                              │   Entry     │
                              │   Point     │
                              └──────┬──────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          │                          │                          │
          ▼                          ▼                          ▼
┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│                 │        │                 │        │                 │
│     runner/     │        │    profiler/    │        │    report/      │
│                 │        │                 │        │                 │
│  Model Exec     │        │  GPU Profiling  │        │  Generation     │
│                 │        │                 │        │                 │
└────────┬────────┘        └────────┬────────┘        └────────┬────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│                 │       │                 │       │                 │
│   collectors/   │       │   analytics/    │       │   governance/   │
│                 │       │                 │       │                 │
│  Data Collect   │       │  Analysis       │       │  Regression     │
│                 │       │                 │       │                 │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         └─────────────────────────┼─────────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│                 │      │                 │      │                 │
│  laboratory/    │      │  calibration/   │      │   tracelake/    │
│                 │      │                 │      │                 │
│  Isolation      │      │  HW Calibrate   │      │  Trace Storage  │
│                 │      │                 │      │                 │
└────────┬────────┘      └────────┬────────┘      └────────┬────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │                 │
                        │     core/       │
                        │                 │
                        │  Session, Schema│
                        │  Utilities      │
                        │                 │
                        └─────────────────┘
```

## Module Descriptions

| Module | Purpose |
|--------|---------|
| `core/` | Session management, data schema, utilities |
| `collectors/` | System and GPU telemetry collection |
| `profiler/` | rocprof integration for kernel profiling |
| `analytics/` | Metrics computation, classification, attribution |
| `governance/` | Statistical regression detection, fleet operations |
| `laboratory/` | Deterministic execution environment |
| `calibration/` | Hardware performance calibration |
| `tracelake/` | Unified trace storage and export |
| `runner/` | ONNX Runtime model execution |
| `report/` | HTML/JSON report generation |
| `dashboard/` | Real-time Streamlit dashboard |
| `cli.py` | Command-line interface |

## Data Flow

```
Input Model ──► runner/ ──► collectors/ ──► tracelake/
                              │
                              ▼
                        analytics/ ──► governance/ ──► report/
```

## Usage

```python
from aaco.core.session import SessionManager
from aaco.runner.ort_runner import ORTRunner
from aaco.analytics.classify import BottleneckClassifier

# Initialize session
session = SessionManager.create_session()

# Run inference
runner = ORTRunner(model_path="model.onnx")
results = runner.run(iterations=100)

# Classify bottleneck
classifier = BottleneckClassifier()
bottleneck = classifier.classify(results.metrics)
```
