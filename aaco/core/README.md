# Core Module

Infrastructure components for session management, data schemas, and utilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                CORE MODULE                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            SESSION MANAGER                                      │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │                 │    │                 │    │                 │            │
│  │   Session ID    │───▶│   Artifact      │───▶│   Environment   │            │
│  │   Generation    │    │   Storage       │    │   Capture       │            │
│  │                 │    │                 │    │                 │            │
│  │  timestamp_uuid │    │  .parquet       │    │  pip freeze     │            │
│  │                 │    │  .json          │    │  env vars       │            │
│  │                 │    │  compression    │    │  config         │            │
│  │                 │    │                 │    │                 │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA SCHEMA                                        │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                         DATACLASSES                                        │ │
│  │                                                                            │ │
│  │  SessionMetadata      InferenceResult      KernelExecution                │ │
│  │  GPUSample            SystemSample         BottleneckResult               │ │
│  │  RegressionVerdict    AttributionScore     DerivedMetrics                 │ │
│  │                                                                            │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                               UTILITIES                                         │
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐            │
│  │                 │    │                 │    │                 │            │
│  │   Timing        │    │   Subprocess    │    │   /proc         │            │
│  │                 │    │   Management    │    │   Readers       │            │
│  │  monotonic_ns   │    │                 │    │                 │            │
│  │  perf_counter   │    │  run_with_      │    │  cpu_info       │            │
│  │                 │    │  timeout        │    │  mem_info       │            │
│  │                 │    │                 │    │  numa_info      │            │
│  │                 │    │                 │    │                 │            │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Components

### session.py

Session lifecycle management:
- Unique session ID generation (timestamp + UUID)
- Artifact folder structure creation
- Environment "lockbox" capture
- Parquet/JSON artifact persistence

### schema.py

Type-safe dataclasses:
- 20+ structured data types
- Full type annotations
- Serialization support
- Validation methods

### utils.py

Common utilities:
- High-resolution timing functions
- Subprocess execution with timeout
- Linux /proc filesystem readers
- Data format conversion

## Session Structure

```
sessions/<date>/<session_id>/
├── session.json           # Session metadata
├── env.json               # Environment lockbox
├── model/                 # Model artifacts
├── runtime/               # Runtime configuration
├── telemetry/             # System telemetry
├── profiler/              # Profiler output
├── attribution/           # Attribution data
├── metrics/               # Computed metrics
├── regress/               # Regression analysis
└── report/                # Generated reports
```
