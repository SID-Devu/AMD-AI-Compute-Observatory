"""
AACO-SIGMA Performance Envelope

Models the achievable performance boundaries for a workload.
Defines the hardware envelope that bounds possible performance.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum, auto

from .gpu_model import GPUModel


class BoundType(Enum):
    """Type of performance bound."""

    COMPUTE_FP32 = auto()
    COMPUTE_FP16 = auto()
    COMPUTE_INT8 = auto()
    MEMORY_HBM = auto()
    MEMORY_L2 = auto()
    MEMORY_L1 = auto()
    LAUNCH_OVERHEAD = auto()
    OCCUPANCY = auto()


@dataclass
class EnvelopePoint:
    """A point in the performance envelope."""

    # Coordinates
    arithmetic_intensity: float  # FLOPS / byte
    achieved_gflops: float  # Actual performance

    # Context
    kernel_name: Optional[str] = None
    bound_type: BoundType = BoundType.COMPUTE_FP32

    # Efficiency
    efficiency: float = 0.0  # Achieved / theoretical peak

    # Distance from ceiling
    gap_to_ceiling_pct: float = 0.0


@dataclass
class EnvelopeBoundary:
    """A boundary/ceiling in the envelope."""

    name: str
    bound_type: BoundType

    # Roofline parameters
    ceiling_gflops: float = 0.0  # Peak compute
    bandwidth_gbps: float = 0.0  # Memory bandwidth
    ridge_point: float = 0.0  # Transition point

    # Valid range
    min_ai: float = 0.0
    max_ai: float = float("inf")

    def get_bound_at(self, arithmetic_intensity: float) -> float:
        """Get performance bound at given AI."""
        if arithmetic_intensity < self.ridge_point:
            # Memory bound: performance = bandwidth * AI
            return self.bandwidth_gbps * arithmetic_intensity
        else:
            # Compute bound: performance = ceiling
            return self.ceiling_gflops


@dataclass
class EnvelopeRegion(Enum):
    """Regions in the performance envelope."""

    MEMORY_BOUND = "memory_bound"
    COMPUTE_BOUND = "compute_bound"
    LATENCY_BOUND = "latency_bound"
    OPTIMAL = "optimal"


@dataclass
class EnvelopeAnalysis:
    """Analysis of a point relative to envelope."""

    point: EnvelopePoint

    # Region
    region: EnvelopeRegion = EnvelopeRegion.MEMORY_BOUND

    # Limiting factor
    limiting_bound: Optional[EnvelopeBoundary] = None
    limiting_factor: str = ""

    # Gaps
    gaps: Dict[str, float] = field(default_factory=dict)  # bound_name -> gap_pct

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class PerformanceEnvelope:
    """
    Performance envelope model for GPU workloads.

    The envelope defines the achievable performance space
    bounded by compute capacity and memory bandwidth.
    """

    def __init__(self, gpu_model: GPUModel):
        self.gpu = gpu_model
        self.boundaries: List[EnvelopeBoundary] = []
        self.points: List[EnvelopePoint] = []

        # Initialize standard boundaries
        self._init_boundaries()

    def _init_boundaries(self) -> None:
        """Initialize performance boundaries from GPU specs."""
        specs = self.gpu.specs

        # FP32 roofline
        fp32_peak = specs.peak_fp32_tflops * 1000  # GFLOPS
        hbm_bw = specs.memory.memory_bandwidth_gbps

        self.boundaries.append(
            EnvelopeBoundary(
                name="FP32 Roofline",
                bound_type=BoundType.COMPUTE_FP32,
                ceiling_gflops=fp32_peak,
                bandwidth_gbps=hbm_bw,
                ridge_point=fp32_peak / hbm_bw if hbm_bw > 0 else 0,
            )
        )

        # FP16 roofline
        fp16_peak = specs.peak_fp16_tflops * 1000
        self.boundaries.append(
            EnvelopeBoundary(
                name="FP16 Roofline",
                bound_type=BoundType.COMPUTE_FP16,
                ceiling_gflops=fp16_peak,
                bandwidth_gbps=hbm_bw,
                ridge_point=fp16_peak / hbm_bw if hbm_bw > 0 else 0,
            )
        )

        # L2 Cache roofline
        l2_bw = hbm_bw * 2.5  # Typical L2 is ~2.5x HBM
        self.boundaries.append(
            EnvelopeBoundary(
                name="L2 Cache Roofline",
                bound_type=BoundType.MEMORY_L2,
                ceiling_gflops=fp32_peak,
                bandwidth_gbps=l2_bw,
                ridge_point=fp32_peak / l2_bw if l2_bw > 0 else 0,
            )
        )

        # L1/LDS roofline
        l1_bw = specs.compute_units * specs.memory.l1_cache_kb_per_cu * specs.boost_clock_mhz / 1000
        self.boundaries.append(
            EnvelopeBoundary(
                name="L1/LDS Roofline",
                bound_type=BoundType.MEMORY_L1,
                ceiling_gflops=fp32_peak,
                bandwidth_gbps=l1_bw,
                ridge_point=fp32_peak / l1_bw if l1_bw > 0 else 0,
            )
        )

    def add_point(
        self,
        arithmetic_intensity: float,
        achieved_gflops: float,
        kernel_name: Optional[str] = None,
    ) -> EnvelopePoint:
        """
        Add a measured point to the envelope.

        Args:
            arithmetic_intensity: FLOPS per byte
            achieved_gflops: Achieved performance in GFLOPS
            kernel_name: Optional kernel identifier
        """
        # Determine which bound applies
        bound = self._get_active_bound(arithmetic_intensity)
        theoretical = bound.get_bound_at(arithmetic_intensity) if bound else achieved_gflops

        efficiency = achieved_gflops / theoretical if theoretical > 0 else 0
        gap = (1 - efficiency) * 100

        # Determine bound type
        if bound:
            if arithmetic_intensity < bound.ridge_point:
                bound_type = BoundType.MEMORY_HBM
            else:
                bound_type = bound.bound_type
        else:
            bound_type = BoundType.COMPUTE_FP32

        point = EnvelopePoint(
            arithmetic_intensity=arithmetic_intensity,
            achieved_gflops=achieved_gflops,
            kernel_name=kernel_name,
            bound_type=bound_type,
            efficiency=efficiency,
            gap_to_ceiling_pct=gap,
        )

        self.points.append(point)
        return point

    def _get_active_bound(self, ai: float) -> Optional[EnvelopeBoundary]:
        """Get the active (lowest) bound at given AI."""
        if not self.boundaries:
            return None

        # Get FP32 roofline as primary
        for bound in self.boundaries:
            if bound.bound_type == BoundType.COMPUTE_FP32:
                return bound

        return self.boundaries[0]

    def analyze_point(self, point: EnvelopePoint) -> EnvelopeAnalysis:
        """
        Analyze a point relative to the envelope.
        """
        analysis = EnvelopeAnalysis(point=point)

        # Determine region
        primary_bound = self._get_active_bound(point.arithmetic_intensity)
        if primary_bound:
            if point.arithmetic_intensity < primary_bound.ridge_point * 0.1:
                analysis.region = EnvelopeRegion.LATENCY_BOUND
            elif point.arithmetic_intensity < primary_bound.ridge_point:
                analysis.region = EnvelopeRegion.MEMORY_BOUND
            else:
                analysis.region = EnvelopeRegion.COMPUTE_BOUND

        # Calculate gaps to all ceilings
        for bound in self.boundaries:
            ceiling = bound.get_bound_at(point.arithmetic_intensity)
            gap = (ceiling - point.achieved_gflops) / ceiling * 100 if ceiling > 0 else 0
            analysis.gaps[bound.name] = max(0, gap)

        # Find limiting bound
        min_ceiling = float("inf")
        for bound in self.boundaries:
            ceiling = bound.get_bound_at(point.arithmetic_intensity)
            if ceiling < min_ceiling:
                min_ceiling = ceiling
                analysis.limiting_bound = bound
                analysis.limiting_factor = bound.name

        # Generate recommendations
        analysis.recommendations = self._generate_recommendations(point, analysis)

        return analysis

    def _generate_recommendations(
        self, point: EnvelopePoint, analysis: EnvelopeAnalysis
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recs = []

        if analysis.region == EnvelopeRegion.MEMORY_BOUND:
            recs.append("Memory-bound: Consider increasing arithmetic intensity")
            recs.append("Options: kernel fusion, tiling, data reuse")

            # Check if close to L2 ceiling
            if "L2 Cache Roofline" in analysis.gaps:
                if analysis.gaps["L2 Cache Roofline"] < 20:
                    recs.append("Near L2 ceiling: Good cache utilization")
                else:
                    recs.append("Far from L2 ceiling: Check cache hit rates")

        elif analysis.region == EnvelopeRegion.COMPUTE_BOUND:
            recs.append("Compute-bound: Consider precision reduction")

            # Check FP16 potential
            if "FP16 Roofline" in analysis.gaps:
                fp16_ceiling = None
                for bound in self.boundaries:
                    if bound.bound_type == BoundType.COMPUTE_FP16:
                        fp16_ceiling = bound.ceiling_gflops
                        break

                if fp16_ceiling:
                    potential = (fp16_ceiling - point.achieved_gflops) / point.achieved_gflops * 100
                    recs.append(f"FP16 potential: {potential:.0f}% speedup possible")

        elif analysis.region == EnvelopeRegion.LATENCY_BOUND:
            recs.append("Latency-bound: Kernel launch overhead may dominate")
            recs.append("Consider batching or fusing small kernels")

        # Efficiency recommendations
        if point.efficiency < 0.5:
            recs.append(
                f"Low efficiency ({point.efficiency * 100:.0f}%): Significant optimization opportunity"
            )
        elif point.efficiency > 0.8:
            recs.append(
                f"High efficiency ({point.efficiency * 100:.0f}%): Near optimal for this AI"
            )

        return recs

    def get_optimal_ai_range(self) -> Tuple[float, float]:
        """
        Get the arithmetic intensity range for optimal performance.

        Returns range where performance is within 90% of peak.
        """
        primary = self._get_active_bound(0)
        if not primary:
            return (0, float("inf"))

        # Find where we achieve 90% of peak
        target = primary.ceiling_gflops * 0.9

        # Below ridge point: AI where memory_bound = target
        # bandwidth * AI = target -> AI = target / bandwidth
        lower_ai = target / primary.bandwidth_gbps if primary.bandwidth_gbps > 0 else 0

        # Upper bound: effectively no limit for compute bound
        upper_ai = primary.ridge_point * 10

        return (lower_ai, upper_ai)

    def export_for_plotting(self) -> Dict[str, Any]:
        """
        Export envelope data for visualization.
        """
        data = {
            "boundaries": [],
            "points": [],
            "gpu_name": self.gpu.specs.name,
        }

        # Generate roofline curves
        for bound in self.boundaries:
            ai_values = []
            perf_values = []

            # Generate points along the roofline
            ai_range = [0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
            for ai in ai_range:
                ai_values.append(ai)
                perf_values.append(bound.get_bound_at(ai))

            data["boundaries"].append(
                {
                    "name": bound.name,
                    "type": bound.bound_type.name,
                    "ai_values": ai_values,
                    "perf_values": perf_values,
                    "ridge_point": bound.ridge_point,
                    "peak_gflops": bound.ceiling_gflops,
                }
            )

        # Export measured points
        for point in self.points:
            data["points"].append(
                {
                    "kernel": point.kernel_name,
                    "ai": point.arithmetic_intensity,
                    "gflops": point.achieved_gflops,
                    "efficiency": point.efficiency,
                    "gap_pct": point.gap_to_ceiling_pct,
                }
            )

        return data
