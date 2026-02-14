"""
AACO-SIGMA Roofline Analysis

Implements roofline model for performance analysis.
Visualizes achieved vs peak performance for GPU kernels.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum, auto
import math

from .counter_model import CounterReading, CounterBasedModel


class PerformanceBound(Enum):
    """Type of performance bound."""
    COMPUTE = auto()        # Compute-bound region
    MEMORY = auto()         # Memory-bound region
    CACHE_L1 = auto()       # L1 cache bound
    CACHE_L2 = auto()       # L2 cache bound
    BALANCED = auto()       # Near the ridge point


@dataclass
class RooflinePoint:
    """A point on the roofline plot."""
    
    # Identifier
    name: str
    kernel_name: str = ""
    
    # Coordinates
    arithmetic_intensity: float = 0.0  # FLOP/Byte
    achieved_performance: float = 0.0  # GFLOPS
    
    # Ceilings
    compute_ceiling: float = 0.0  # Peak GFLOPS (slanted roof)
    memory_ceiling: float = 0.0   # Memory-bound ceiling
    
    # Analysis
    bound: PerformanceBound = PerformanceBound.MEMORY
    efficiency_pct: float = 0.0
    headroom_pct: float = 0.0  # Room for improvement
    
    # Optional metadata
    duration_ns: int = 0
    total_flops: int = 0
    total_bytes: int = 0


@dataclass
class RooflineCeiling:
    """A ceiling line on the roofline model."""
    name: str
    slope: float  # 0 for horizontal (compute), bandwidth for sloped (memory)
    intercept: float  # Peak value
    is_compute_bound: bool = True


class RooflineModel:
    """
    Roofline performance model.
    
    Models the relationship between:
    - Arithmetic intensity (FLOP/Byte)
    - Achieved performance (GFLOP/s)
    - Hardware ceilings (compute and memory bandwidth)
    """
    
    # AMD GPU defaults (RDNA3/CDNA2-like)
    DEFAULT_PARAMS = {
        "peak_fp32_gflops": 25000,      # Peak FP32 GFLOPS
        "peak_fp16_gflops": 50000,      # Peak FP16 GFLOPS
        "peak_mfma_gflops": 100000,     # Peak matrix GFLOPS
        "peak_hbm_gbps": 1600,          # HBM bandwidth
        "peak_l2_gbps": 3200,           # L2 cache bandwidth
        "peak_l1_gbps": 6400,           # L1 cache bandwidth
    }
    
    def __init__(self, **params):
        """Initialize roofline model with hardware parameters."""
        self.params = {**self.DEFAULT_PARAMS, **params}
        self.ceilings: List[RooflineCeiling] = []
        self._build_ceilings()
    
    def _build_ceilings(self):
        """Build roofline ceiling lines."""
        self.ceilings = [
            # Compute ceilings (horizontal)
            RooflineCeiling(
                name="Peak FP32",
                slope=0,
                intercept=self.params["peak_fp32_gflops"],
                is_compute_bound=True,
            ),
            RooflineCeiling(
                name="Peak FP16",
                slope=0,
                intercept=self.params["peak_fp16_gflops"],
                is_compute_bound=True,
            ),
            RooflineCeiling(
                name="Peak MFMA",
                slope=0,
                intercept=self.params["peak_mfma_gflops"],
                is_compute_bound=True,
            ),
            
            # Memory ceilings (sloped)
            RooflineCeiling(
                name="HBM Bandwidth",
                slope=self.params["peak_hbm_gbps"],
                intercept=0,
                is_compute_bound=False,
            ),
            RooflineCeiling(
                name="L2 Cache",
                slope=self.params["peak_l2_gbps"],
                intercept=0,
                is_compute_bound=False,
            ),
            RooflineCeiling(
                name="L1 Cache",
                slope=self.params["peak_l1_gbps"],
                intercept=0,
                is_compute_bound=False,
            ),
        ]
    
    def get_ridge_point(self, compute_ceiling: str = "Peak FP32",
                        memory_ceiling: str = "HBM Bandwidth") -> Tuple[float, float]:
        """
        Get the ridge point where compute and memory ceilings meet.
        
        Returns (arithmetic_intensity, performance)
        """
        compute = next(c for c in self.ceilings if c.name == compute_ceiling)
        memory = next(c for c in self.ceilings if c.name == memory_ceiling)
        
        # Ridge point: AI where compute ceiling meets memory ceiling
        # compute_ceiling = memory_slope * AI
        # AI = compute_ceiling / memory_slope
        
        ai = compute.intercept / memory.slope if memory.slope > 0 else float('inf')
        return (ai, compute.intercept)
    
    def get_ceiling_at(self, arithmetic_intensity: float,
                       compute_type: str = "Peak FP32") -> float:
        """Get the performance ceiling at a given arithmetic intensity."""
        compute = next(c for c in self.ceilings if c.name == compute_type)
        hbm = next(c for c in self.ceilings if c.name == "HBM Bandwidth")
        
        memory_bound_perf = hbm.slope * arithmetic_intensity
        compute_bound_perf = compute.intercept
        
        return min(memory_bound_perf, compute_bound_perf)
    
    def analyze_point(self, ai: float, achieved_gflops: float,
                      name: str = "", kernel_name: str = "") -> RooflinePoint:
        """Analyze a performance point on the roofline."""
        point = RooflinePoint(
            name=name,
            kernel_name=kernel_name,
            arithmetic_intensity=ai,
            achieved_performance=achieved_gflops,
        )
        
        # Get ceilings
        fp32_ceiling = self.params["peak_fp32_gflops"]
        hbm_ceiling = self.params["peak_hbm_gbps"] * ai
        
        point.compute_ceiling = fp32_ceiling
        point.memory_ceiling = hbm_ceiling
        
        # Determine which ceiling applies
        effective_ceiling = min(fp32_ceiling, hbm_ceiling)
        
        if hbm_ceiling < fp32_ceiling:
            point.bound = PerformanceBound.MEMORY
        else:
            point.bound = PerformanceBound.COMPUTE
        
        # Check if near ridge point
        ridge_ai, _ = self.get_ridge_point()
        if 0.8 * ridge_ai <= ai <= 1.2 * ridge_ai:
            point.bound = PerformanceBound.BALANCED
        
        # Calculate efficiency
        point.efficiency_pct = (achieved_gflops / effective_ceiling * 100) if effective_ceiling > 0 else 0
        
        # Calculate headroom
        point.headroom_pct = max(0, 100 - point.efficiency_pct)
        
        return point


class RooflineAnalyzer:
    """
    Analyzes GPU kernels using the roofline model.
    """
    
    def __init__(self, model: Optional[RooflineModel] = None):
        self.model = model or RooflineModel()
        self.counter_model = CounterBasedModel()
        self.points: List[RooflinePoint] = []
    
    def analyze_kernel(self, kernel_name: str,
                       counters: CounterReading) -> RooflinePoint:
        """
        Analyze a kernel using counter data.
        """
        # Calculate arithmetic intensity
        total_flops = counters.valu_instructions * 64 + counters.mfma_instructions * 256
        total_bytes = counters.memory_read_bytes + counters.memory_write_bytes
        
        ai = total_flops / total_bytes if total_bytes > 0 else 0.0
        
        # Calculate achieved performance
        duration_s = counters.duration_ns / 1e9
        achieved_gflops = total_flops / duration_s / 1e9 if duration_s > 0 else 0.0
        
        # Analyze point
        point = self.model.analyze_point(ai, achieved_gflops, 
                                         name=kernel_name, 
                                         kernel_name=kernel_name)
        point.duration_ns = counters.duration_ns
        point.total_flops = total_flops
        point.total_bytes = total_bytes
        
        self.points.append(point)
        return point
    
    def analyze_batch(self, 
                      kernel_data: List[Tuple[str, CounterReading]]) -> List[RooflinePoint]:
        """Analyze multiple kernels."""
        results = []
        for kernel_name, counters in kernel_data:
            point = self.analyze_kernel(kernel_name, counters)
            results.append(point)
        return results
    
    def get_summary(self) -> Dict[str, Any]:
        """Get analysis summary across all points."""
        if not self.points:
            return {}
        
        # Group by bound type
        bound_counts = {}
        for point in self.points:
            bound_counts[point.bound.name] = bound_counts.get(point.bound.name, 0) + 1
        
        # Calculate averages
        avg_efficiency = sum(p.efficiency_pct for p in self.points) / len(self.points)
        avg_headroom = sum(p.headroom_pct for p in self.points) / len(self.points)
        
        # Find extremes
        highest_ai = max(self.points, key=lambda p: p.arithmetic_intensity)
        lowest_ai = min(self.points, key=lambda p: p.arithmetic_intensity)
        most_efficient = max(self.points, key=lambda p: p.efficiency_pct)
        least_efficient = min(self.points, key=lambda p: p.efficiency_pct)
        
        return {
            "total_points": len(self.points),
            "bound_distribution": bound_counts,
            "average_efficiency_pct": avg_efficiency,
            "average_headroom_pct": avg_headroom,
            "highest_ai": {
                "name": highest_ai.name,
                "ai": highest_ai.arithmetic_intensity,
            },
            "lowest_ai": {
                "name": lowest_ai.name,
                "ai": lowest_ai.arithmetic_intensity,
            },
            "most_efficient": {
                "name": most_efficient.name,
                "efficiency_pct": most_efficient.efficiency_pct,
            },
            "least_efficient": {
                "name": least_efficient.name,
                "efficiency_pct": least_efficient.efficiency_pct,
            },
        }
    
    def suggest_optimizations(self) -> Dict[str, List[str]]:
        """Suggest optimizations per kernel."""
        suggestions: Dict[str, List[str]] = {}
        
        for point in self.points:
            kernel_suggestions = []
            
            if point.bound == PerformanceBound.MEMORY:
                kernel_suggestions.extend([
                    "Increase arithmetic intensity through algorithmic changes",
                    "Use data reuse via shared memory",
                    "Consider reduced precision (FP16/BF16)",
                    "Improve memory access patterns for coalescing",
                ])
            
            elif point.bound == PerformanceBound.COMPUTE:
                kernel_suggestions.extend([
                    "Use vectorized operations",
                    "Leverage matrix instructions (MFMA)",
                    "Check for instruction-level parallelism",
                ])
            
            if point.efficiency_pct < 50:
                kernel_suggestions.append(
                    f"Kernel is only {point.efficiency_pct:.1f}% efficient - "
                    "significant optimization opportunity"
                )
            
            if point.arithmetic_intensity < 1.0:
                kernel_suggestions.append(
                    "Very low arithmetic intensity - consider kernel fusion"
                )
            
            if kernel_suggestions:
                suggestions[point.name] = kernel_suggestions
        
        return suggestions
    
    def export_for_plotting(self) -> Dict[str, Any]:
        """Export data for external visualization."""
        ridge = self.model.get_ridge_point()
        
        return {
            "model": {
                "peak_fp32_gflops": self.model.params["peak_fp32_gflops"],
                "peak_hbm_gbps": self.model.params["peak_hbm_gbps"],
                "ridge_point": {"ai": ridge[0], "gflops": ridge[1]},
            },
            "ceilings": [
                {
                    "name": c.name,
                    "slope": c.slope,
                    "intercept": c.intercept,
                    "is_compute_bound": c.is_compute_bound,
                }
                for c in self.model.ceilings
            ],
            "points": [
                {
                    "name": p.name,
                    "kernel_name": p.kernel_name,
                    "ai": p.arithmetic_intensity,
                    "gflops": p.achieved_performance,
                    "bound": p.bound.name,
                    "efficiency_pct": p.efficiency_pct,
                }
                for p in self.points
            ],
        }
