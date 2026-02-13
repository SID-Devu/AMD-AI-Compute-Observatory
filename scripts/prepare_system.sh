#!/bin/bash
# System Preparation Script
# Prepares system for reproducible benchmarking

set -e

echo "AACO System Preparation"
echo "======================"
echo ""

# Check root/sudo
if [ "$EUID" -ne 0 ]; then
    echo "Warning: Not running as root. Some operations may fail."
    SUDO="sudo"
else
    SUDO=""
fi

# 1. Set CPU governor to performance
echo "1. Setting CPU governor to performance..."
if command -v cpupower &> /dev/null; then
    $SUDO cpupower frequency-set -g performance || echo "  Failed (may need cpupower installed)"
elif [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo "performance" | $SUDO tee "$gov" > /dev/null 2>&1 || true
    done
    echo "  Set via sysfs"
else
    echo "  Could not set CPU governor"
fi
echo ""

# 2. Set GPU to high performance
echo "2. Setting GPU performance level..."
if command -v rocm-smi &> /dev/null; then
    $SUDO rocm-smi --setperflevel high || echo "  Failed"
    echo "  GPU set to high performance"
else
    echo "  rocm-smi not available"
fi
echo ""

# 3. Check and report GPU clocks
echo "3. Current GPU state:"
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showclocks 2>/dev/null || true
fi
echo ""

# 4. Disable CPU frequency scaling (optional)
echo "4. Current CPU frequency:"
if [ -f /proc/cpuinfo ]; then
    grep "MHz" /proc/cpuinfo | head -4
fi
echo ""

# 5. Check for interfering processes
echo "5. Checking for GPU-using processes..."
if command -v rocm-smi &> /dev/null; then
    rocm-smi --showpidgpus 2>/dev/null || echo "  No GPU processes found"
fi
echo ""

# 6. Report system load
echo "6. Current system load:"
uptime
echo ""

# 7. Memory status
echo "7. Memory status:"
free -h
echo ""

# 8. Verify Python environment
echo "8. Python environment:"
python3 --version
pip show onnxruntime 2>/dev/null | grep -E "^(Name|Version)" || echo "  onnxruntime not installed"
echo ""

echo "System preparation complete!"
echo ""
echo "Recommendations for reproducible benchmarks:"
echo "  - Close unnecessary applications"
echo "  - Disable automatic updates"
echo "  - Ensure adequate cooling"
echo "  - Run multiple iterations for statistical confidence"
