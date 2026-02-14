"""
AACO-SIGMA Counter-Based Performance Model

Models GPU performance using hardware counter readings.
Provides component-level breakdown of execution time.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
from enum import Enum, auto


class BottleneckType(Enum):
    """Types of performance bottlenecks."""

    COMPUTE = auto()  # ALU/FPU limited
    MEMORY_BANDWIDTH = auto()  # HBM bandwidth limited
    CACHE = auto()  # L1/L2 cache misses
    LATENCY = auto()  # Memory latency bound
    OCCUPANCY = auto()  # Low wave occupancy
    LAUNCH_OVERHEAD = auto()  # Kernel launch overhead
    SYNC = auto()  # Synchronization overhead
    UNKNOWN = auto()


class PerformanceComponent(Enum):
    """Components of GPU execution time."""

    COMPUTE_VALU = auto()  # Vector ALU
    COMPUTE_SALU = auto()  # Scalar ALU
    COMPUTE_MFMA = auto()  # Matrix operations (CDNA)
    MEMORY_LOAD = auto()  # Memory loads
    MEMORY_STORE = auto()  # Memory stores
    CACHE_ACCESS = auto()  # Cache operations
    SYNC_BARRIER = auto()  # Barrier synchronization
    LAUNCH = auto()  # Kernel launch
    OTHER = auto()  # Unattributed


@dataclass
class ResourceUtilization:
    """GPU resource utilization metrics."""

    # Compute utilization (0-100%)
    valu_utilization: float = 0.0
    salu_utilization: float = 0.0
    mfma_utilization: float = 0.0

    # Memory utilization
    memory_bandwidth_utilization: float = 0.0
    l1_cache_hit_rate: float = 0.0
    l2_cache_hit_rate: float = 0.0

    # Occupancy
    achieved_occupancy: float = 0.0
    theoretical_occupancy: float = 0.0

    # Waves
    active_waves_per_cu: float = 0.0
    max_waves_per_cu: int = 32

    def get_occupancy_ratio(self) -> float:
        """Get achieved vs theoretical occupancy ratio."""
        if self.theoretical_occupancy == 0:
            return 0.0
        return self.achieved_occupancy / self.theoretical_occupancy

    def is_compute_bound(self) -> bool:
        """Check if workload is compute bound."""
        compute_util = max(self.valu_utilization, self.mfma_utilization)
        return compute_util > self.memory_bandwidth_utilization

    def is_memory_bound(self) -> bool:
        """Check if workload is memory bound."""
        compute_util = max(self.valu_utilization, self.mfma_utilization)
        return self.memory_bandwidth_utilization > compute_util


@dataclass
class CounterReading:
    """A set of hardware counter readings."""

    # Wave counters
    waves_launched: int = 0
    waves_completed: int = 0

    # Instruction counters
    valu_instructions: int = 0
    salu_instructions: int = 0
    mfma_instructions: int = 0
    lds_instructions: int = 0

    # Memory counters
    memory_read_bytes: int = 0
    memory_write_bytes: int = 0

    # Cache counters
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0

    # GPU cycles
    gpu_cycles: int = 0
    active_cycles: int = 0
    stall_cycles: int = 0

    # Timing
    duration_ns: int = 0


class CounterBasedModel:
    """
    Models GPU performance using hardware counters.

    Breaks down execution time into components and identifies bottlenecks.
    """

    # AMD GPU architecture parameters (RDNA3/CDNA2 defaults)
    DEFAULT_PARAMS = {
        "gpu_clock_mhz": 2100,
        "memory_clock_mhz": 1000,
        "memory_bus_width_bits": 256,  # HBM2e
        "num_cus": 60,
        "valu_ops_per_cycle": 64,  # FP32 per CU
        "mfma_ops_per_cycle": 256,  # Matrix ops per CU
        "l1_cache_size_kb": 16,
        "l2_cache_size_mb": 8,
        "max_waves_per_cu": 32,
    }

    def __init__(self, **params):
        """Initialize with GPU parameters."""
        self.params = {**self.DEFAULT_PARAMS, **params}
        self._calculate_peak()

    def _calculate_peak(self):
        """Calculate peak performance metrics."""
        # Peak compute (TFLOPS)
        self.peak_valu_tflops = (
            self.params["num_cus"]
            * self.params["valu_ops_per_cycle"]
            * self.params["gpu_clock_mhz"]
            * 1e-6
        )

        self.peak_mfma_tflops = (
            self.params["num_cus"]
            * self.params["mfma_ops_per_cycle"]
            * self.params["gpu_clock_mhz"]
            * 1e-6
        )

        # Peak memory bandwidth (GB/s)
        # For HBM: effective bandwidth considers ECC overhead
        self.peak_memory_gbps = (
            self.params["memory_clock_mhz"]
            * self.params["memory_bus_width_bits"]
            / 8
            * 2
            / 1000  # DDR factor
        )

    def analyze(self, counters: CounterReading) -> Dict[str, Any]:
        """
        Analyze counter readings and produce performance model.
        """
        duration_s = counters.duration_ns / 1e9

        if duration_s == 0:
            return {"error": "Zero duration"}

        # Calculate achieved metrics
        total_bytes = counters.memory_read_bytes + counters.memory_write_bytes
        achieved_bandwidth_gbps = total_bytes / duration_s / 1e9

        achieved_valu_ops = counters.valu_instructions * 64  # Ops per instruction
        achieved_valu_tflops = achieved_valu_ops / duration_s / 1e12

        achieved_mfma_ops = counters.mfma_instructions * 256
        achieved_mfma_tflops = achieved_mfma_ops / duration_s / 1e12

        # Calculate utilization
        utilization = ResourceUtilization(
            valu_utilization=min(100, achieved_valu_tflops / self.peak_valu_tflops * 100),
            mfma_utilization=min(100, achieved_mfma_tflops / self.peak_mfma_tflops * 100)
            if self.peak_mfma_tflops > 0
            else 0,
            memory_bandwidth_utilization=min(
                100, achieved_bandwidth_gbps / self.peak_memory_gbps * 100
            ),
            l1_cache_hit_rate=self._calc_hit_rate(counters.l1_hits, counters.l1_misses),
            l2_cache_hit_rate=self._calc_hit_rate(counters.l2_hits, counters.l2_misses),
            active_waves_per_cu=counters.waves_launched / self.params["num_cus"]
            if self.params["num_cus"] > 0
            else 0,
        )

        # Calculate occupancy
        if counters.gpu_cycles > 0:
            utilization.achieved_occupancy = counters.active_cycles / counters.gpu_cycles * 100

        # Determine bottleneck
        bottleneck = self._identify_bottleneck(counters, utilization)

        # Time breakdown
        breakdown = self._compute_time_breakdown(counters, duration_s)

        return {
            "duration_ns": counters.duration_ns,
            "achieved_bandwidth_gbps": achieved_bandwidth_gbps,
            "achieved_valu_tflops": achieved_valu_tflops,
            "achieved_mfma_tflops": achieved_mfma_tflops,
            "utilization": utilization,
            "bottleneck": bottleneck.name,
            "time_breakdown": breakdown,
            "efficiency_pct": self._compute_efficiency(utilization, bottleneck),
        }

    def _calc_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate."""
        total = hits + misses
        if total == 0:
            return 0.0
        return hits / total * 100

    def _identify_bottleneck(
        self, counters: CounterReading, utilization: ResourceUtilization
    ) -> BottleneckType:
        """Identify the primary performance bottleneck."""

        # High memory util + low compute = memory bound
        if utilization.memory_bandwidth_utilization > 80:
            if utilization.valu_utilization < 50:
                return BottleneckType.MEMORY_BANDWIDTH

        # High compute util = compute bound
        if utilization.valu_utilization > 80 or utilization.mfma_utilization > 80:
            return BottleneckType.COMPUTE

        # Low cache hit rates = cache issues
        if utilization.l2_cache_hit_rate < 50 and utilization.l1_cache_hit_rate < 50:
            return BottleneckType.CACHE

        # High stall cycles = latency issues
        if counters.gpu_cycles > 0:
            stall_ratio = counters.stall_cycles / counters.gpu_cycles
            if stall_ratio > 0.3:
                return BottleneckType.LATENCY

        # Low occupancy
        if utilization.achieved_occupancy < 30:
            return BottleneckType.OCCUPANCY

        return BottleneckType.UNKNOWN

    def _compute_time_breakdown(
        self, counters: CounterReading, duration_s: float
    ) -> Dict[str, float]:
        """Break down time into components."""
        breakdown: Dict[str, float] = {}

        if duration_s == 0:
            return breakdown

        # Estimate time per component based on instruction counts
        total_insts = (
            counters.valu_instructions
            + counters.salu_instructions
            + counters.mfma_instructions
            + counters.lds_instructions
        )

        if total_insts > 0:
            breakdown[PerformanceComponent.COMPUTE_VALU.name] = (
                counters.valu_instructions / total_insts * 100
            )
            breakdown[PerformanceComponent.COMPUTE_SALU.name] = (
                counters.salu_instructions / total_insts * 100
            )
            breakdown[PerformanceComponent.COMPUTE_MFMA.name] = (
                counters.mfma_instructions / total_insts * 100
            )

        # Memory time estimate
        total_bytes = counters.memory_read_bytes + counters.memory_write_bytes
        if total_bytes > 0:
            estimated_mem_time = total_bytes / (self.peak_memory_gbps * 1e9)
            breakdown[PerformanceComponent.MEMORY_LOAD.name] = (
                counters.memory_read_bytes / total_bytes * estimated_mem_time / duration_s * 100
            )
            breakdown[PerformanceComponent.MEMORY_STORE.name] = (
                counters.memory_write_bytes / total_bytes * estimated_mem_time / duration_s * 100
            )

        return breakdown

    def _compute_efficiency(
        self, utilization: ResourceUtilization, bottleneck: BottleneckType
    ) -> float:
        """Compute overall efficiency percentage."""

        # Efficiency is based on resource utilization at the bottleneck
        if bottleneck == BottleneckType.COMPUTE:
            return max(utilization.valu_utilization, utilization.mfma_utilization)
        elif bottleneck == BottleneckType.MEMORY_BANDWIDTH:
            return utilization.memory_bandwidth_utilization
        elif bottleneck == BottleneckType.CACHE:
            return (utilization.l1_cache_hit_rate + utilization.l2_cache_hit_rate) / 2
        elif bottleneck == BottleneckType.OCCUPANCY:
            return utilization.achieved_occupancy
        else:
            # Average of key metrics
            return (utilization.valu_utilization + utilization.memory_bandwidth_utilization) / 2

    def compute_arithmetic_intensity(self, counters: CounterReading) -> float:
        """
        Compute arithmetic intensity (FLOP/Byte).

        Key metric for roofline analysis.
        """
        total_flops = (
            counters.valu_instructions * 64  # FP32 ops
            + counters.mfma_instructions * 256  # Matrix ops
        )

        total_bytes = counters.memory_read_bytes + counters.memory_write_bytes

        if total_bytes == 0:
            return float("inf")

        return total_flops / total_bytes

    def suggest_optimizations(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest optimizations based on analysis."""
        suggestions = []
        bottleneck = analysis.get("bottleneck", "UNKNOWN")

        if bottleneck == "MEMORY_BANDWIDTH":
            suggestions.extend(
                [
                    "Consider data layout optimizations (coalescing)",
                    "Use shared memory for data reuse",
                    "Reduce precision (FP16/BF16) to halve bandwidth",
                    "Fuse kernels to reduce memory traffic",
                ]
            )

        elif bottleneck == "COMPUTE":
            suggestions.extend(
                [
                    "Use MFMA instructions for matrix operations",
                    "Vectorize operations for better ALU utilization",
                    "Consider algorithmic optimizations",
                ]
            )

        elif bottleneck == "CACHE":
            suggestions.extend(
                [
                    "Improve data locality",
                    "Use blocking/tiling to fit working set in cache",
                    "Prefetch data when possible",
                ]
            )

        elif bottleneck == "OCCUPANCY":
            suggestions.extend(
                [
                    "Reduce register usage per thread",
                    "Reduce shared memory per workgroup",
                    "Use smaller workgroup sizes if register-bound",
                ]
            )

        elif bottleneck == "LATENCY":
            suggestions.extend(
                [
                    "Increase occupancy to hide latency",
                    "Use asynchronous memory operations",
                    "Overlap compute and memory operations",
                ]
            )

        return suggestions
