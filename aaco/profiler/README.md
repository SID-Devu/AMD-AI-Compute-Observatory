# Profiler Module

GPU kernel profiling via AMD rocprof integration.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PROFILER MODULE                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            ROCPROF WRAPPER                                      │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         rocprof_wrap.py                                  │   │
│  │                                                                          │   │
│  │    ┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐    │   │
│  │    │           │    │           │    │           │    │           │    │   │
│  │    │  Setup    │───▶│  Execute  │───▶│  Collect  │───▶│  Parse    │    │   │
│  │    │  Config   │    │  Workload │    │  Traces   │    │  Output   │    │   │
│  │    │           │    │           │    │           │    │           │    │   │
│  │    └───────────┘    └───────────┘    └───────────┘    └───────────┘    │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          TRACE COLLECTION                                       │
│                                                                                 │
│    ┌─────────────────────────────────────────────────────────────────────┐     │
│    │                       rocprof Options                                │     │
│    │                                                                      │     │
│    │  --hip-trace          HIP API calls                                 │     │
│    │  --hsa-trace          HSA runtime calls                             │     │
│    │  --sys-trace          System calls                                  │     │
│    │  --stats              Kernel statistics                             │     │
│    │  -i input.txt         Hardware counter input                        │     │
│    │  -o output.csv        Output file                                   │     │
│    │                                                                      │     │
│    └─────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            TRACE PARSER                                         │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        rocprof_parse.py                                  │   │
│  │                                                                          │   │
│  │    Raw CSV ──► Timestamp Alignment ──► Kernel Extraction ──► Parquet   │   │
│  │                                                                          │   │
│  │    ┌───────────────────────────────────────────────────────────────┐   │   │
│  │    │  Extracted Fields:                                             │   │   │
│  │    │                                                                │   │   │
│  │    │  • Kernel Name                                                 │   │   │
│  │    │  • Start Timestamp (ns)                                        │   │   │
│  │    │  • Duration (ns)                                               │   │   │
│  │    │  • Grid Size (x, y, z)                                         │   │   │
│  │    │  • Block Size (x, y, z)                                        │   │   │
│  │    │  • Registers per Thread                                        │   │   │
│  │    │  • Shared Memory (bytes)                                       │   │   │
│  │    │  • Stream ID                                                   │   │   │
│  │    │                                                                │   │   │
│  │    └───────────────────────────────────────────────────────────────┘   │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Usage

```python
from aaco.profiler.rocprof_wrap import RocprofWrapper

# Initialize profiler
profiler = RocprofWrapper(
    output_dir="profiler_output",
    hip_trace=True,
    hsa_trace=False,
    stats=True
)

# Profile a command
result = profiler.profile(
    command=["python", "inference.py"],
    timeout=300
)

# Parse results
kernels = profiler.parse_results()
```

## Output Structure

```
profiler/
├── rocprof_raw/
│   ├── results.csv          # Raw rocprof output
│   ├── results.hip_stats.csv
│   └── results.copy_stats.csv
└── rocprof_kernels.parquet  # Parsed kernel data
```

## Parquet Schema

| Column | Type | Description |
|--------|------|-------------|
| kernel_name | string | Kernel function name |
| start_ns | int64 | Start timestamp |
| duration_ns | int64 | Execution duration |
| grid_x | int32 | Grid dimension X |
| grid_y | int32 | Grid dimension Y |
| grid_z | int32 | Grid dimension Z |
| block_x | int32 | Block dimension X |
| block_y | int32 | Block dimension Y |
| block_z | int32 | Block dimension Z |
| regs | int32 | Registers per thread |
| shared_mem | int32 | Shared memory bytes |
| stream_id | int32 | CUDA/HIP stream |

## Hardware Counters

Configurable counters via input.txt:

```
pmc: GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES
pmc: TCC_HIT TCC_MISS TCC_READ TCC_WRITE
pmc: TCP_READ TCP_WRITE
```
