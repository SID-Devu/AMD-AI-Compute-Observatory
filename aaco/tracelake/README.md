# Trace Lake Module

Unified trace storage and cross-layer correlation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            TRACE LAKE MODULE                                    │
│                                                                                 │
│                      Unified Performance Data Storage                           │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION                                        │
│                                                                                 │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│    │             │  │             │  │             │  │             │         │
│    │   rocprof   │  │   rocm-smi  │  │   /proc     │  │    eBPF     │         │
│    │   traces    │  │   samples   │  │   samples   │  │   events    │         │
│    │             │  │             │  │             │  │             │         │
│    └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│           │                │                │                │                 │
│           └────────────────┴────────────────┴────────────────┘                 │
│                                    │                                            │
│                                    ▼                                            │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │                      Event Normalizer                                │     │
│    │                                                                      │     │
│    │  • Timestamp alignment (nanosecond precision)                       │     │
│    │  • Schema normalization                                             │     │
│    │  • Event type classification                                        │     │
│    │                                                                      │     │
│    └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          STORAGE LAYER                                          │
│                                                                                 │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │                       Parquet Store                                  │     │
│    │                                                                      │     │
│    │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │     │
│    │   │              │  │              │  │              │             │     │
│    │   │   Kernel     │  │     GPU      │  │    System    │             │     │
│    │   │   Events     │  │   Events     │  │   Events     │             │     │
│    │   │              │  │              │  │              │             │     │
│    │   │ .parquet     │  │ .parquet     │  │ .parquet     │             │     │
│    │   │              │  │              │  │              │             │     │
│    │   └──────────────┘  └──────────────┘  └──────────────┘             │     │
│    │                                                                      │     │
│    └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          QUERY INTERFACE                                        │
│                                                                                 │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │                        Trace Query API                               │     │
│    │                                                                      │     │
│    │   • Time range queries                                              │     │
│    │   • Event type filtering                                            │     │
│    │   • Cross-reference joins                                           │     │
│    │   • Aggregations (mean, sum, count)                                │     │
│    │                                                                      │     │
│    └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          EXPORT FORMATS                                         │
│                                                                                 │
│    ┌───────────────┐  ┌───────────────┐  ┌───────────────┐                    │
│    │               │  │               │  │               │                    │
│    │   Perfetto    │  │   Chrome      │  │     JSON      │                    │
│    │   Protobuf    │  │   Trace       │  │   Events      │                    │
│    │               │  │               │  │               │                    │
│    │   ui.perfetto │  │  chrome://    │  │  Programmatic │                    │
│    │   .dev        │  │  tracing      │  │   Access      │                    │
│    │               │  │               │  │               │                    │
│    └───────────────┘  └───────────────┘  └───────────────┘                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Usage

```python
from aaco.tracelake import TraceLake

# Initialize trace lake
lake = TraceLake(session_path="sessions/20260215_abc123")

# Query kernel events
kernels = lake.query(
    event_type="kernel",
    start_ns=0,
    end_ns=1_000_000_000
)

# Cross-reference with GPU telemetry
timeline = lake.correlate(
    kernel_events=kernels,
    gpu_samples=lake.query(event_type="gpu")
)

# Export to Perfetto
lake.export_perfetto("trace.perfetto")
```

## Schema

### Unified Event Schema

| Column | Type | Description |
|--------|------|-------------|
| timestamp_ns | int64 | Event timestamp (ns) |
| event_type | string | kernel/gpu/system/ebpf |
| duration_ns | int64 | Event duration |
| name | string | Event name |
| metadata | json | Additional properties |

### Perfetto Export

Compatible with [Perfetto UI](https://ui.perfetto.dev) for visualization:
- Kernel execution tracks
- GPU telemetry tracks
- System metrics tracks
- Cross-layer correlation
