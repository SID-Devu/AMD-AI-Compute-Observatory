# Installation

This guide covers installing AMD AI Compute Observatory on your system.

## Requirements

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.11 |
| OS | Linux (Ubuntu 20.04+) | Ubuntu 22.04 LTS |
| RAM | 8 GB | 16 GB+ |
| GPU | AMD ROCm-compatible | MI100/MI200/MI300 |
| ROCm | 5.7+ | 6.0+ |

### Software Dependencies

- Python 3.10 or higher
- ROCm 6.0+ (for GPU profiling features)
- Linux kernel 5.15+ (for eBPF features)

## Installation Methods

### Method 1: pip (Recommended)

```bash
# Basic installation
pip install aaco

# With all optional dependencies
pip install aaco[all]

# With specific extras
pip install aaco[onnx,dashboard,ml]
```

### Method 2: From Source

```bash
# Clone the repository
git clone https://github.com/SID-Devu/AMD-AI-Compute-Observatory.git
cd AMD-AI-Compute-Observatory

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### Method 3: Docker

```bash
# Pull the image
docker pull ghcr.io/sid-devu/amd-ai-compute-observatory:latest

# Or build locally
docker build -t aaco:latest .

# Run
docker run -it aaco:latest --help
```

### Method 4: Docker Compose

```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d dashboard jupyter
```

## Optional Dependencies

| Extra | Packages | Use Case |
|-------|----------|----------|
| `onnx` | onnx, onnxruntime | ONNX model support |
| `dashboard` | streamlit, plotly | Interactive dashboard |
| `ml` | scikit-learn, shap | ML-powered analysis |
| `ebpf` | bcc | Kernel-level profiling |
| `dev` | pytest, mypy, ruff | Development tools |
| `all` | All of the above | Full installation |

## Verification

Verify your installation:

```bash
# Check CLI
aaco --version

# Run self-test
aaco doctor

# Test import
python -c "import aaco; print(aaco.__version__)"
```

## ROCm Setup

For GPU profiling features, ensure ROCm is properly installed:

```bash
# Check ROCm installation
rocm-smi

# Verify HIP
hipcc --version

# Check MIGraphX
migraphx-driver --version
```

## Troubleshooting

### Common Issues

??? question "Import Error: No module named 'aaco'"
    Ensure you've activated your virtual environment:
    ```bash
    source venv/bin/activate
    ```

??? question "ROCm not detected"
    Check ROCm installation:
    ```bash
    ls /opt/rocm
    rocm-smi --showproductname
    ```

??? question "Permission denied for eBPF"
    eBPF features require root or CAP_BPF:
    ```bash
    sudo aaco profile --ebpf ...
    # Or add capabilities
    sudo setcap cap_bpf+ep $(which python)
    ```

## Next Steps

- [Quick Start Guide](quickstart.md) - Get started in 5 minutes
- [Configuration](configuration.md) - Configure AACO for your environment
- [User Guide](../user-guide/overview.md) - Complete usage documentation
