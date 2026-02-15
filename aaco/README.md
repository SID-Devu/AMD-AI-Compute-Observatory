# AACO Source Code

This directory contains the core Python package for AMD AI Compute Observatory.

## Package Structure

| Module | Description |
|--------|-------------|
| `core/` | Session management, configuration, and shared utilities |
| `analytics/` | Performance metrics computation and bottleneck classification |
| `collectors/` | GPU and system telemetry data collection |
| `profiler/` | rocprof integration and kernel profiling |
| `runner/` | ONNX Runtime execution with multiple backend support |
| `laboratory/` | Deterministic execution environment with process isolation |
| `calibration/` | Hardware envelope calibration via microbenchmarks |
| `governance/` | Statistical regression detection and baseline management |
| `tracelake/` | Unified trace storage and cross-layer correlation |
| `report/` | Report generation in HTML and JSON formats |

## Module Dependencies

```
cli.py
  ├── runner/
  │     └── core/, collectors/
  ├── profiler/
  │     └── core/
  ├── report/
  │     └── analytics/, tracelake/
  ├── analytics/
  │     └── collectors/, core/
  └── governance/
        └── analytics/, core/
```

## Entry Points

The package exposes the following CLI commands via `cli.py`:

- `aaco run` — Execute profiling session
- `aaco report` — Generate performance reports
- `aaco diff` — Compare sessions against baselines
- `aaco baseline` — Manage performance baselines
- `aaco dashboard` — Launch real-time monitoring interface

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests for specific module
pytest tests/unit/test_analytics.py -v

# Type check
mypy aaco/ --ignore-missing-imports
```
