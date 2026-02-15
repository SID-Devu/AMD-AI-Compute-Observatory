# Architecture Diagrams

This directory contains architecture diagrams and visual documentation for AMD AI Compute Observatory.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              AACO ARCHITECTURE                              │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────┐
                    │         USER INTERFACE          │
                    │   CLI | Reports | Dashboard     │
                    └───────────────┬─────────────────┘
                                    │
┌───────────────────────────────────┴───────────────────────────────────────┐
│                           GOVERNANCE LAYER                                │
│              Regression Detection | Root Cause Analysis                   │
├───────────────────────────────────────────────────────────────────────────┤
│                          INTELLIGENCE LAYER                               │
│         Attribution Engine | Hardware Digital Twin | Trace Lake           │
├───────────────────────────────────────────────────────────────────────────┤
│                           MEASUREMENT LAYER                               │
│                Laboratory Mode | rocprof | eBPF Telemetry                 │
├───────────────────────────────────────────────────────────────────────────┤
│                           COLLECTION LAYER                                │
│               ONNX Runtime | MIGraphX | ROCm SMI | psutil                 │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │        AMD INSTINCT GPU       │
                    │      MI300X | MI250X | MI100  │
                    └───────────────────────────────┘
```

## Data Flow

```
Input Model ──► Graph Analysis ──► Execution ──► Collection ──► Attribution
                                                                     │
                                                                     ▼
     Recommendations ◄── Root Cause ◄── Classification ◄── Metrics Engine
```

## Session Bundle Structure

```
sessions/<session_id>/
├── session.json
├── env.json
├── model/
├── runtime/
├── telemetry/
├── profiler/
├── attribution/
├── metrics/
└── report/
```

## Related Documentation

- [Architecture Overview](../docs/architecture.md)
- [Data Schema Reference](../docs/data_schema.md)
- [Measurement Methodology](../docs/methodology.md)
