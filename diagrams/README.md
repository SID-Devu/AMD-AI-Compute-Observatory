# Architecture Diagrams

This directory contains architecture and workflow diagrams for AMD AI Compute Observatory.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│                         AMD AI COMPUTE OBSERVATORY                              │
│                                                                                 │
│              Model-to-Metal Performance Analysis Platform                       │
│                                                                                 │
│                        ┌─────────────────────┐                                 │
│                        │     AMD Instinct    │                                 │
│                        │     MI100/MI200/    │                                 │
│                        │       MI300X        │                                 │
│                        └─────────────────────┘                                 │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 5: INTERFACE                                                              │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│ │   CLI    │ │   HTML   │ │   JSON   │ │Dashboard │ │ Perfetto │              │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
├─────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 4: GOVERNANCE                                                             │
│ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐   │
│ │   Regression   │ │   Root Cause   │ │      SLA       │ │     Fleet      │   │
│ │   Detection    │ │    Analysis    │ │  Enforcement   │ │   Operations   │   │
│ └────────────────┘ └────────────────┘ └────────────────┘ └────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 3: INTELLIGENCE                                                           │
│ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐   │
│ │    Kernel      │ │  Attribution   │ │    Digital     │ │     Trace      │   │
│ │  Fingerprint   │ │    Engine      │ │     Twin       │ │     Lake       │   │
│ └────────────────┘ └────────────────┘ └────────────────┘ └────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 2: MEASUREMENT                                                            │
│ ┌─────────────────────────────────┐ ┌─────────────────────────────────────┐   │
│ │        Laboratory Mode          │ │         eBPF Forensics              │   │
│ │  Isolation, Determinism, Repr.  │ │  Scheduler, Faults, Context, IRQ   │   │
│ └─────────────────────────────────┘ └─────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 1: COLLECTION                                                             │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│ │   ORT    │ │ rocprof  │ │ rocm-smi │ │  /proc   │ │   eBPF   │              │
│ │  Runner  │ │  Trace   │ │ Telemetry│ │ Sampler  │ │  Tracer  │              │
│ └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘              │
├─────────────────────────────────────────────────────────────────────────────────┤
│ LAYER 0: INFRASTRUCTURE                                                         │
│ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐                       │
│ │    Session     │ │     Data       │ │   Utilities    │                       │
│ │    Manager     │ │    Schema      │ │                │                       │
│ └────────────────┘ └────────────────┘ └────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Pipeline

```
                              AMD AI COMPUTE OBSERVATORY
                                   DATA FLOW

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                              INPUT                                        │
    │                                                                           │
    │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
    │   │    ONNX     │    │   Config    │    │  Baseline   │                 │
    │   │    Model    │    │    YAML     │    │    JSON     │                 │
    │   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                 │
    │          │                  │                  │                         │
    └──────────┼──────────────────┼──────────────────┼─────────────────────────┘
               │                  │                  │
               ▼                  ▼                  ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                           PROCESSING                                      │
    │                                                                           │
    │   Graph Analysis ─► Laboratory Setup ─► Deterministic Execution          │
    │          │                                      │                         │
    │          ▼                                      ▼                         │
    │   Parallel Collection: rocprof + rocm-smi + /proc + eBPF                 │
    │                              │                                            │
    │                              ▼                                            │
    │   Trace Lake ─► Attribution ─► Classification ─► Governance              │
    │                                                                           │
    └──────────────────────────────────────────────────────────────────────────┘
               │
               ▼
    ┌──────────────────────────────────────────────────────────────────────────┐
    │                             OUTPUT                                        │
    │                                                                           │
    │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
    │   │   Session   │    │    HTML     │    │  Regression │                 │
    │   │   Bundle    │    │   Report    │    │   Verdict   │                 │
    │   └─────────────┘    └─────────────┘    └─────────────┘                 │
    │                                                                           │
    └──────────────────────────────────────────────────────────────────────────┘
```

## Attribution Flow

```
                           GRAPH-TO-KERNEL ATTRIBUTION

    ONNX Graph                MIGraphX                 HIP Runtime
    ──────────                ────────                 ───────────

    ┌─────────┐
    │  Conv2D │
    └────┬────┘
         │
    ┌────┴────┐         ┌─────────────┐
    │  BatchN │────────▶│  Partition  │
    └────┬────┘         │     #1      │         ┌─────────────────┐
         │              └──────┬──────┘         │ miopenConvFwd   │
    ┌────┴────┐                │          ┌────▶│ (kernel)        │
    │  ReLU   │                │          │     └─────────────────┘
    └────┬────┘                │          │
         │              ┌──────┴──────┐   │     ┌─────────────────┐
    ┌────┴────┐         │   Fused     │───┼────▶│ miopenBatchNorm │
    │ MaxPool │────────▶│   Op #1     │   │     │ (kernel)        │
    └─────────┘         └──────┬──────┘   │     └─────────────────┘
                               │          │
                        ┌──────┴──────┐   │     ┌─────────────────┐
                        │  Partition  │───┴────▶│ miopenPoolFwd   │
                        │     #2      │         │ (kernel)        │
                        └─────────────┘         └─────────────────┘


    KAR = 3 kernels / 4 nodes = 0.75 (good fusion)
    PFI = 2 partitions / 4 nodes = 0.50
```

## Bottleneck Classification

```
                         BOTTLENECK DECISION TREE

                              ┌─────────────┐
                              │   Metrics   │
                              │   Input     │
                              └──────┬──────┘
                                     │
                         ┌───────────┴───────────┐
                         │                       │
                         ▼                       ▼
              ┌──────────────────┐    ┌──────────────────┐
              │ microkernel_pct  │    │  gpu_active_pct  │
              │     > 0.5?       │    │     > 0.9?       │
              └────────┬─────────┘    └────────┬─────────┘
                   Yes │                   Yes │
                       ▼                       ▼
              ┌──────────────────┐    ┌──────────────────┐
              │   LAUNCH-BOUND   │    │  COMPUTE-BOUND   │
              └──────────────────┘    └──────────────────┘

                         │ No                   │ No
                         ▼                       ▼
              ┌──────────────────┐    ┌──────────────────┐
              │  ctx_switches    │    │  mem_ops_ratio   │
              │     > 10000?     │    │     > 0.7?       │
              └────────┬─────────┘    └────────┬─────────┘
                   Yes │                   Yes │
                       ▼                       ▼
              ┌──────────────────┐    ┌──────────────────┐
              │    CPU-BOUND     │    │   MEMORY-BOUND   │
              └──────────────────┘    └──────────────────┘

                         │ No
                         ▼
              ┌──────────────────┐
              │  clock_variance  │
              │     > 0.1?       │
              └────────┬─────────┘
                   Yes │
                       ▼
              ┌──────────────────┐
              │ THERMAL-THROTTLE │
              └──────────────────┘
```

## Statistical Governance

```
                         REGRESSION DETECTION FLOW

    Historical Baseline                    New Session
    ───────────────────                    ───────────

    ┌───────────────────┐                 ┌───────────────────┐
    │ median(latency)   │                 │ current_latency   │
    │ MAD(latency)      │                 │                   │
    │ P99(latency)      │                 │                   │
    └─────────┬─────────┘                 └─────────┬─────────┘
              │                                     │
              └──────────────┬──────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────────────┐
              │           EWMA Detection             │
              │                                      │
              │   EWMA_t = λ×X_t + (1-λ)×EWMA_{t-1} │
              │   Alert if |EWMA - baseline| > 3σ   │
              │                                      │
              └────────────────┬─────────────────────┘
                               │
                               ▼
              ┌──────────────────────────────────────┐
              │           CUSUM Analysis             │
              │                                      │
              │   S⁺ = max(0, S⁺ + (x - μ - k))    │
              │   S⁻ = max(0, S⁻ + (μ - k - x))    │
              │   Change point if S > h             │
              │                                      │
              └────────────────┬─────────────────────┘
                               │
                               ▼
              ┌──────────────────────────────────────┐
              │         Regression Verdict           │
              │                                      │
              │   • regression: true/false           │
              │   • severity: low/medium/high        │
              │   • confidence: 0.0-1.0              │
              │   • delta_pct: percentage change     │
              │                                      │
              └──────────────────────────────────────┘
```

## Hardware Stack

```
                         AMD INSTINCT HARDWARE STACK

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                                                                         │
    │                        APPLICATION LAYER                                │
    │                                                                         │
    │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │
    │   │   PyTorch   │  │    ONNX     │  │  TensorFlow │                    │
    │   │             │  │   Runtime   │  │             │                    │
    │   └─────────────┘  └─────────────┘  └─────────────┘                    │
    │                                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │                         RUNTIME LAYER                                   │
    │                                                                         │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │                        MIGraphX                                  │  │
    │   │                   Graph Optimization                             │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │                          HIP                                     │  │
    │   │              Heterogeneous-computing Interface                   │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │                                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │                        LIBRARY LAYER                                    │
    │                                                                         │
    │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
    │   │ rocBLAS  │  │ rocFFT   │  │ MIOpen   │  │ rocRAND  │              │
    │   │          │  │          │  │          │  │          │              │
    │   │  GEMM    │  │   FFT    │  │   DNN    │  │  Random  │              │
    │   └──────────┘  └──────────┘  └──────────┘  └──────────┘              │
    │                                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │                        DRIVER LAYER                                     │
    │                                                                         │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │                        amdgpu                                    │  │
    │   │                    Linux Kernel Driver                           │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │                                                                         │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │                        HARDWARE LAYER                                   │
    │                                                                         │
    │   ┌─────────────────────────────────────────────────────────────────┐  │
    │   │                   AMD Instinct MI300X                            │  │
    │   │                                                                  │  │
    │   │   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐              │  │
    │   │   │   CU    │ │   CU    │ │   CU    │ │   CU    │   x 228     │  │
    │   │   └─────────┘ └─────────┘ └─────────┘ └─────────┘              │  │
    │   │                                                                  │  │
    │   │   ┌─────────────────────────────────────────────────────────┐  │  │
    │   │   │                    HBM3 Memory                           │  │  │
    │   │   │                  192GB @ 5.3 TB/s                        │  │  │
    │   │   └─────────────────────────────────────────────────────────┘  │  │
    │   │                                                                  │  │
    │   └─────────────────────────────────────────────────────────────────┘  │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘
```

## Session Bundle Structure

```
                              SESSION ARTIFACT STRUCTURE

    sessions/<date>/<session_id>/
    │
    ├── session.json                 ◄── Session metadata
    │
    ├── env.json                     ◄── Environment lockbox
    │
    ├── model/
    │   ├── model_meta.json          ◄── ONNX model metadata
    │   ├── graph_nodes.parquet      ◄── Graph node data
    │   └── graph_edges.parquet      ◄── Graph edge data
    │
    ├── runtime/
    │   ├── ort_config.json          ◄── ORT configuration
    │   └── migraphx_partition.json  ◄── MIGraphX partitions
    │
    ├── telemetry/
    │   ├── gpu_events.parquet       ◄── GPU samples (100Hz)
    │   └── system_events.parquet    ◄── System samples (10Hz)
    │
    ├── profiler/
    │   ├── rocprof_raw/             ◄── Raw rocprof output
    │   └── rocprof_kernels.parquet  ◄── Parsed kernel data
    │
    ├── attribution/
    │   ├── kernel_groups.parquet    ◄── Kernel groupings
    │   └── op_to_kernel_map.parquet ◄── Attribution mapping
    │
    ├── metrics/
    │   ├── inference_iters.parquet  ◄── Per-iteration data
    │   ├── inference_summary.json   ◄── Summary statistics
    │   ├── derived_metrics.json     ◄── KAR, PFI, LTS, etc.
    │   └── bottleneck.json          ◄── Classification result
    │
    ├── regress/
    │   ├── baseline_ref.json        ◄── Baseline reference
    │   ├── diff.json                ◄── Delta analysis
    │   └── verdict.json             ◄── Regression verdict
    │
    └── report/
        ├── report.html              ◄── HTML report
        └── plots/                   ◄── Visualization assets
```
