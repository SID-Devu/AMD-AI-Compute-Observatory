# Core Module

Session management, configuration, and shared utilities.

## Components

| Component | Description |
|-----------|-------------|
| `session.py` | Session lifecycle management and state persistence |
| `config.py` | Configuration loading and validation |
| `schema.py` | Data schema definitions and validation |
| `utils.py` | Shared utility functions |

## Session Structure

Sessions are stored as structured directories:

```
sessions/<session_id>/
├── session.json          # Session metadata
├── env.json              # Environment snapshot
├── model/                # Model metadata
├── runtime/              # Runtime configuration
├── telemetry/            # Collected telemetry
├── profiler/             # Profiling data
├── metrics/              # Computed metrics
└── report/               # Generated reports
```

## Usage

```python
from aaco.core import SessionManager

manager = SessionManager()
session = manager.create(model="resnet50", backend="migraphx")
# ... execute profiling ...
manager.finalize(session)
```
