# AACO Profiling Sessions

This directory stores profiling session data for reproducibility and analysis.

## Structure

```
sessions/
├── README.md
├── session_schema.json     # Session file schema
└── examples/               # Example sessions
    └── *.json
```

## Session Format

Each session captures a complete profiling run with all data needed for reproduction.

## Creating Sessions

```bash
# Start a new session
aaco session start --name "llama2-perf-test"

# Run profiling within session
aaco profile --model llama2-7b --capture-all

# End session (saves automatically)
aaco session end
```

## Loading Sessions

```python
from aaco.core.session import Session

# Load existing session
session = Session.load("sessions/20260214_llama2_perf.json")

# Access captured data
for trace in session.traces:
    print(f"Trace: {trace.name}, Events: {len(trace.events)}")
```
