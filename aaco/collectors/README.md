# Collectors Module

System and GPU telemetry collection for AMD Instinct accelerators.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            COLLECTORS MODULE                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA COLLECTION LAYER                                 │
│                                                                                 │
│         ┌───────────────────────────────────────────────────────┐              │
│         │                  Sampling Controller                   │              │
│         │                                                        │              │
│         │  start() ──► sample_loop() ──► stop() ──► get_data()  │              │
│         └───────────────────────────────────────────────────────┘              │
│                                    │                                            │
│                    ┌───────────────┼───────────────┐                           │
│                    │               │               │                           │
│                    ▼               ▼               ▼                           │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐      │
│  │                     │ │                     │ │                     │      │
│  │  rocm_smi_sampler   │ │    sys_sampler      │ │      clocks         │      │
│  │                     │ │                     │ │                     │      │
│  │  GPU Telemetry      │ │  System Metrics     │ │  Clock Management   │      │
│  │                     │ │                     │ │                     │      │
│  └──────────┬──────────┘ └──────────┬──────────┘ └──────────┬──────────┘      │
│             │                       │                       │                  │
│             ▼                       ▼                       ▼                  │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐      │
│  │                     │ │                     │ │                     │      │
│  │  • GPU Clock        │ │  • CPU Utilization  │ │  • Lock GPU Clock   │      │
│  │  • Memory Clock     │ │  • Memory Usage     │ │  • Query Current    │      │
│  │  • Temperature      │ │  • Context Switches │ │  • Reset to Default │      │
│  │  • Power Draw       │ │  • Page Faults      │ │                     │      │
│  │  • Memory Used      │ │  • IRQ Count        │ │                     │      │
│  │  • Fan Speed        │ │  • Load Average     │ │                     │      │
│  │  • Utilization      │ │                     │ │                     │      │
│  │                     │ │                     │ │                     │      │
│  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DRIVER INTERFACE                                       │
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                       driver_interface.py                                │   │
│  │                                                                          │   │
│  │    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                │   │
│  │    │             │    │             │    │             │                │   │
│  │    │  rocm-smi   │    │   /sys/     │    │   ioctl     │                │   │
│  │    │  CLI        │    │   class/    │    │   Calls     │                │   │
│  │    │             │    │   drm/      │    │             │                │   │
│  │    └─────────────┘    └─────────────┘    └─────────────┘                │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Components

### rocm_smi_sampler.py

GPU telemetry collection via ROCm SMI:

| Metric | Source | Unit |
|--------|--------|------|
| GPU Clock | `rocm-smi --showclocks` | MHz |
| Memory Clock | `rocm-smi --showclocks` | MHz |
| Temperature | `rocm-smi --showtemp` | °C |
| Power | `rocm-smi --showpower` | W |
| Memory Used | `rocm-smi --showmeminfo` | MB |
| Utilization | `rocm-smi --showuse` | % |

### sys_sampler.py

System metrics collection via /proc:

| Metric | Source | Unit |
|--------|--------|------|
| CPU Usage | `/proc/stat` | % |
| Memory | `/proc/meminfo` | MB |
| Context Switches | `/proc/vmstat` | count |
| Page Faults | `/proc/vmstat` | count |
| Load Average | `/proc/loadavg` | ratio |

### clocks.py

GPU clock management:

```python
# Lock clocks for deterministic measurement
lock_gpu_clocks(device_id=0, level="high")

# Query current clock state
clocks = get_current_clocks(device_id=0)

# Reset to automatic
reset_gpu_clocks(device_id=0)
```

## Data Output

```
telemetry/
├── gpu_events.parquet     # GPU samples at 100Hz
└── system_events.parquet  # System samples at 10Hz
```

## Parquet Schema

### GPU Events

| Column | Type | Description |
|--------|------|-------------|
| timestamp_ns | int64 | Nanosecond timestamp |
| gpu_clock_mhz | int32 | GPU clock frequency |
| mem_clock_mhz | int32 | Memory clock frequency |
| temperature_c | float32 | GPU temperature |
| power_w | float32 | Power consumption |
| memory_used_mb | int64 | VRAM usage |
| utilization_pct | float32 | GPU utilization |

### System Events

| Column | Type | Description |
|--------|------|-------------|
| timestamp_ns | int64 | Nanosecond timestamp |
| cpu_pct | float32 | CPU utilization |
| memory_mb | int64 | Memory usage |
| ctx_switches | int64 | Context switches |
| page_faults | int64 | Page faults |
