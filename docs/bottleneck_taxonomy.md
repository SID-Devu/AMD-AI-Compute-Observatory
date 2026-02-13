# Bottleneck Taxonomy

## Overview

This document defines the complete taxonomy of performance bottlenecks that AACO can detect and classify. Understanding these categories is essential for systematic performance optimization.

## Bottleneck Categories

### 1. Compute Bound

**Definition**: The GPU compute units are the limiting factor; work is genuinely compute-intensive.

**Indicators**:
- GPU utilization > 85%
- GPU Active Ratio > 0.7
- Low memory bandwidth utilization
- Kernels have high arithmetic intensity

**Evidence**:
```
• High GPU utilization: 92%
• High GPU active ratio: 0.85
• Memory bandwidth not saturated
```

**Optimizations**:
- FP16/INT8 quantization (reduce compute requirements)
- Algorithm optimization (fewer FLOPS)
- Kernel fusion (reduce overhead)
- Consider tensor cores if available

**This is actually good**: Being compute-bound means you're using the hardware efficiently. Optimization focuses on reducing work, not fixing inefficiencies.

---

### 2. Memory Bound

**Definition**: Memory bandwidth is the limiting factor; compute units wait for data.

**Indicators**:
- High memory bandwidth utilization (>80%)
- Low GPU utilization (<50%)
- Large tensor operations
- High VRAM traffic

**Evidence**:
```
• High memory utilization: 88%
• Low GPU compute utilization: 45%
• Large intermediate tensors detected
```

**Optimizations**:
- Operator fusion (reduce memory traffic)
- In-place operations where possible
- Smaller data types (FP16 reduces bandwidth 2x)
- Optimize memory access patterns
- Layer reordering to improve locality

---

### 3. Launch Overhead

**Definition**: Too many small GPU kernel launches; launch latency dominates.

**Indicators**:
- Microkernel % > 30%
- Kernel Amplification Ratio > 10x
- GPU Active Ratio < 0.3
- High launch rate (>5000 kernels/sec)

**Evidence**:
```
• High microkernel %: 45%
• High kernel amplification: 15x
• Low GPU active ratio: 0.25
• Launch rate: 8000 kernels/sec
```

**Why this happens**:
- Elementwise operations spawn separate kernels
- Lack of operator fusion
- Suboptimal graph optimization
- Many small operations in model architecture

**Optimizations**:
- Enable aggressive operator fusion in the EP
- Use MIGraphX EP's fusion capabilities
- Review ONNX graph for optimization opportunities
- Consider graph-level optimizations (e.g., opset version)
- Batch small operations where possible

---

### 4. Data Transfer

**Definition**: Host-device memory transfers are the bottleneck.

**Indicators**:
- Low GPU utilization
- High PCIe activity
- Large input/output tensors on host
- Frequent synchronization points

**Evidence**:
```
• Low GPU utilization during inference
• Large tensors copied per iteration
• Multiple D2H/H2D transfers detected
```

**Optimizations**:
- Pin host memory for faster transfers
- Use async transfers with pipelining
- Keep tensors on device between operations
- Reduce data movement in application code

---

### 5. CPU Bound

**Definition**: Host CPU processing limits GPU utilization.

**Indicators**:
- High CPU utilization (>80%)
- Low GPU utilization (<50%)
- Single-threaded pre/post processing
- Inefficient data preparation

**Evidence**:
```
• High CPU utilization: 95%
• Low GPU utilization: 35%
• CPU appears to be limiting GPU work submission
```

**Optimizations**:
- Profile CPU code for hotspots
- Parallelize data preprocessing
- Use async inference APIs
- Move preprocessing to GPU if possible
- Optimize Python code (NumPy vectorization, Cython)

---

### 6. Thermal Throttle

**Definition**: GPU reducing performance due to thermal limits.

**Indicators**:
- Temperature > 85°C (junction)
- Clock frequency reduction observed
- Performance degradation over time
- Power limit reached

**Evidence**:
```
• High temperature: 92°C
• Clock frequency dropped from 2100 to 1800 MHz
• Performance degraded 15% over 5 minutes
```

**Optimizations**:
- Improve cooling (better thermal paste, airflow)
- Reduce ambient temperature
- Set conservative power limits
- Add pauses between benchmark iterations
- Consider undervolting (if supported)

---

### 7. Frequency Scaling

**Definition**: Clock frequency instability affecting performance consistency.

**Indicators**:
- Large clock frequency variation (>200 MHz range)
- Power management enabled
- Non-performance governor
- Inconsistent per-iteration latencies

**Evidence**:
```
• Clock variation: 1600-2100 MHz during benchmark
• Scaling governor: ondemand (not performance)
• High latency coefficient of variation: 25%
```

**Optimizations**:
- Lock GPU clocks: `rocm-smi --setperflevel high`
- Set performance governor: `cpupower frequency-set -g performance`
- Disable power management during benchmarks
- Set fixed clock frequencies

---

### 8. Warmup Instability

**Definition**: Performance differs between warmup and measurement phases.

**Indicators**:
- Warmup latency > 20% higher than measurement
- First few iterations significantly slower
- JIT compilation detected
- Memory allocation spikes during warmup

**Evidence**:
```
• Warmup mean: 15.2ms, Measurement mean: 12.1ms
• Warmup effect: +25%
• High variance in early iterations
```

**Causes**:
- JIT compilation (first-time kernel compilation)
- Lazy memory allocation
- Cache warming
- GPU power state transition

**Optimizations**:
- Increase warmup iterations
- Pre-compile kernels if supported
- Use persistent execution modes
- Pre-allocate memory pools

---

### 9. Kernel Fragmentation

**Definition**: Many unique kernels with poor reuse.

**Indicators**:
- High unique kernel count
- Low kernel reuse ratio
- Many kernels with single call
- Dynamic shapes causing kernel regeneration

**Evidence**:
```
• 500 unique kernels in single inference
• Average calls per kernel: 1.2
• Many dynamically shaped operations
```

**Optimizations**:
- Use static shapes where possible
- Enable kernel caching
- Group similar operations
- Review model architecture for regularization

---

### 10. Balanced (No Bottleneck)

**Definition**: No single dominant bottleneck; system is reasonably optimized.

**Indicators**:
- Moderate GPU utilization (50-80%)
- Reasonable GPU Active Ratio (0.5-0.8)
- Low microkernel percentage
- Stable latencies

**Evidence**:
```
• GPU utilization: 72%
• GPU Active Ratio: 0.65
• Microkernel %: 8%
• Performance is consistent
```

**Next Steps**:
- System is well-balanced
- Further optimization requires architecture changes
- Consider model distillation for further gains
- Hardware upgrade for substantial improvement

## Classification Logic

```
Priority Order:
1. Thermal Throttle (if temp > 85°C)
2. Launch Overhead (if microkernel% > 30% AND GPU active < 0.4)
3. Memory Bound (if mem_util > 80% AND gpu_util < 60%)
4. Compute Bound (if gpu_util > 85%)
5. CPU Bound (if cpu_util > 80% AND gpu_util < 50%)
6. Frequency Scaling (if clock_range > 200MHz)
7. Warmup Instability (if warmup_effect > 20%)
8. Balanced (default)
```

## Bottleneck Combinations

Often multiple bottlenecks coexist. AACO reports:
- **Primary**: Most impactful bottleneck
- **Secondary**: Contributing factors

Common combinations:
- Launch Overhead + CPU Bound: Python preprocessing + many small kernels
- Memory Bound + Thermal: Large tensors causing high power draw
- Compute Bound + Thermal: Sustained heavy computation
