# Configuration

AACO uses YAML configuration files for customization.

## Configuration File Locations

AACO searches for configuration in this order:

1. Command-line specified: `--config path/to/config.yaml`
2. Current directory: `./aaco.yaml`
3. User config: `~/.config/aaco/config.yaml`
4. System config: `/etc/aaco/config.yaml`

## Default Configuration

```yaml
# aaco.yaml - AACO Configuration
# © 2026 Sudheer Ibrahim Daniel Devu

# =============================================================================
# General Settings
# =============================================================================
general:
  log_level: INFO
  session_dir: ./sessions
  output_dir: ./results
  temp_dir: /tmp/aaco

# =============================================================================
# Profiling Settings
# =============================================================================
profiling:
  # Iteration settings
  default_iterations: 100
  default_warmup: 10
  min_iterations: 10
  max_iterations: 10000
  
  # Timing
  timeout: 3600  # seconds
  sample_interval: 0.001  # seconds
  
  # Data collection
  collect_counters: true
  collect_traces: true
  collect_memory: true
  collect_power: true

# =============================================================================
# Laboratory Mode
# =============================================================================
laboratory:
  enabled: false
  
  # CPU isolation
  cpu_isolation:
    enabled: true
    cores: [4, 5, 6, 7]  # Isolated cores to use
    pin_threads: true
  
  # GPU settings
  gpu:
    clock_lock: true
    target_frequency: max  # 'max', 'min', or specific MHz
    power_limit: null  # Watts, null for default
  
  # Process isolation
  process:
    nice: -20
    ionice_class: realtime
    cgroup_isolation: true
  
  # System preparation
  system:
    disable_turbo: true
    set_governor: performance
    drop_caches: true

# =============================================================================
# Analysis Settings
# =============================================================================
analysis:
  # Statistical settings
  statistics:
    confidence_level: 0.95
    outlier_method: iqr  # 'iqr', 'zscore', 'mad'
    outlier_threshold: 1.5
  
  # Baseline settings
  baseline:
    method: median  # 'mean', 'median', 'trimmed_mean'
    robust: true
  
  # Drift detection
  drift:
    ewma_alpha: 0.3
    cusum_threshold: 5.0
    window_size: 20
  
  # Root cause
  root_cause:
    prior_method: uniform  # 'uniform', 'empirical'
    min_confidence: 0.7
    max_causes: 5

# =============================================================================
# Hardware Settings
# =============================================================================
hardware:
  # GPU detection
  gpu:
    auto_detect: true
    device_ids: null  # null for all, or [0, 1, ...]
  
  # ROCm settings
  rocm:
    path: /opt/rocm
    version: auto
  
  # MIGraphX settings
  migraphx:
    exhaustive_tune: false
    fast_math: true

# =============================================================================
# Report Settings
# =============================================================================
reporting:
  # Output formats
  default_format: html
  
  # HTML report
  html:
    theme: auto  # 'light', 'dark', 'auto'
    interactive: true
    include_raw_data: false
  
  # JSON report
  json:
    pretty: true
    include_session: true
  
  # Charts
  charts:
    style: seaborn
    dpi: 150
    figsize: [12, 8]

# =============================================================================
# Dashboard Settings
# =============================================================================
dashboard:
  host: localhost
  port: 8501
  theme: dark
  auto_refresh: true
  refresh_interval: 5  # seconds

# =============================================================================
# Storage Settings
# =============================================================================
storage:
  # Session storage
  session:
    format: parquet  # 'parquet', 'json', 'pickle'
    compression: zstd
  
  # Data retention
  retention:
    max_sessions: 100
    max_age_days: 30
    auto_cleanup: true
```

## Environment Variables

Override configuration with environment variables:

| Variable | Description |
|----------|-------------|
| `AACO_CONFIG` | Path to config file |
| `AACO_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) |
| `AACO_SESSION_DIR` | Session storage directory |
| `AACO_OUTPUT_DIR` | Output directory |
| `AACO_LAB_MODE` | Enable laboratory mode (1/0) |
| `ROCM_PATH` | ROCm installation path |

## Configuration Profiles

### Performance Profile

```yaml
# High-accuracy profiling
profiling:
  default_iterations: 500
  default_warmup: 50
  
laboratory:
  enabled: true
  cpu_isolation:
    enabled: true
  gpu:
    clock_lock: true
```

### Quick Profile

```yaml
# Fast profiling for development
profiling:
  default_iterations: 20
  default_warmup: 5
  
laboratory:
  enabled: false

analysis:
  statistics:
    outlier_method: none
```

### CI Profile

```yaml
# CI/CD optimized
profiling:
  default_iterations: 50
  timeout: 600
  
reporting:
  default_format: json
  json:
    include_session: false
```

## Next Steps

- [Laboratory Mode](../user-guide/laboratory-mode.md) - Deterministic profiling
- [Analysis Guide](../user-guide/analysis.md) - Understanding results
