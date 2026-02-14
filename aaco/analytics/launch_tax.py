"""
Launch Tax Analyzer
Deep analysis of GPU kernel launch overhead and microkernel anti-patterns.
The "signature ROCm feature" for AACO.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from aaco.core.schema import KernelSummary

logger = logging.getLogger(__name__)


@dataclass
class LaunchTaxMetrics:
    """Core metrics for launch tax analysis."""
    total_kernel_count: int
    total_kernel_time_ms: float
    wall_clock_time_ms: float
    
    # Duration distribution
    avg_kernel_duration_us: float
    p50_kernel_duration_us: float
    p90_kernel_duration_us: float
    min_kernel_duration_us: float
    max_kernel_duration_us: float
    
    # Microkernel analysis (kernels < threshold)
    microkernel_count: int
    microkernel_pct: float
    microkernel_threshold_us: float
    microkernel_time_ms: float
    microkernel_time_pct: float
    
    # Launch rate
    launch_rate_per_sec: float
    launch_rate_per_inference: float
    
    # Tax metrics
    launch_tax_score: float  # 0-100, higher = worse
    estimated_launch_overhead_ms: float
    estimated_launch_overhead_pct: float
    
    # Efficiency
    gpu_active_ratio: float
    useful_compute_ratio: float  # Excluding microkernel overhead
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KernelDurationBucket:
    """Duration bucket for histogram analysis."""
    range_us: Tuple[float, float]
    count: int
    total_time_ms: float
    pct_of_calls: float
    pct_of_time: float
    is_microkernel: bool


@dataclass
class LaunchTaxReport:
    """Complete launch tax analysis report."""
    metrics: LaunchTaxMetrics
    duration_histogram: List[KernelDurationBucket]
    top_microkernels: List[Dict[str, Any]]  # Most called tiny kernels
    top_heavy_kernels: List[Dict[str, Any]]  # Longest running kernels
    kernel_name_patterns: Dict[str, int]  # Pattern -> count
    assessment: str  # "healthy", "concerning", "critical"
    recommendations: List[str]
    detailed_diagnosis: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "duration_histogram": [asdict(b) for b in self.duration_histogram],
            "top_microkernels": self.top_microkernels,
            "top_heavy_kernels": self.top_heavy_kernels,
            "kernel_name_patterns": self.kernel_name_patterns,
            "assessment": self.assessment,
            "recommendations": self.recommendations,
            "detailed_diagnosis": self.detailed_diagnosis,
        }


class LaunchTaxAnalyzer:
    """
    Analyzes GPU kernel launch overhead and microkernel patterns.
    
    Key metrics:
    - Microkernel percentage (kernels under threshold)
    - Launch rate (kernels per second)
    - Launch tax score (composite overhead indicator)
    - Duration distribution analysis
    
    This is the signature feature for GPU performance analysis:
    - Identifies "too many tiny kernels" anti-pattern
    - Quantifies launch overhead impact
    - Provides fusion recommendations
    """
    
    # Configurable thresholds
    DEFAULT_MICROKERNEL_THRESHOLD_US = 10.0  # Kernels < 10μs are "micro"
    
    # Duration bucket boundaries (microseconds)
    DURATION_BUCKETS = [
        (0, 1),
        (1, 5),
        (5, 10),
        (10, 50),
        (50, 100),
        (100, 500),
        (500, 1000),
        (1000, 5000),
        (5000, float("inf")),
    ]
    
    # Assessment thresholds
    ASSESSMENT_THRESHOLDS = {
        "healthy": 20,      # Launch tax score < 20
        "concerning": 50,   # Launch tax score < 50
        "critical": 100,    # Launch tax score >= 50
    }
    
    # Estimated overhead per kernel launch (microseconds)
    # This varies by GPU/driver but 5-20μs is typical
    ESTIMATED_LAUNCH_OVERHEAD_US = 10.0
    
    def __init__(
        self,
        kernel_summaries: List[KernelSummary],
        wall_clock_ms: float,
        num_inferences: int = 1,
        microkernel_threshold_us: float = DEFAULT_MICROKERNEL_THRESHOLD_US,
    ):
        """
        Args:
            kernel_summaries: List of kernel summary stats
            wall_clock_ms: Total wall clock time
            num_inferences: Number of inference iterations
            microkernel_threshold_us: Threshold for "tiny" kernels
        """
        self.kernels = kernel_summaries
        self.wall_clock_ms = wall_clock_ms
        self.num_inferences = num_inferences
        self.threshold_us = microkernel_threshold_us
    
    def analyze(self) -> LaunchTaxReport:
        """Perform complete launch tax analysis."""
        if not self.kernels:
            return self._empty_report()
        
        # Compute core metrics
        metrics = self._compute_metrics()
        
        # Build duration histogram
        histogram = self._build_duration_histogram()
        
        # Find top microkernels
        top_micro = self._find_top_microkernels()
        
        # Find top heavy kernels
        top_heavy = self._find_top_heavy_kernels()
        
        # Extract kernel name patterns
        patterns = self._extract_kernel_patterns()
        
        # Determine assessment
        assessment = self._assess_severity(metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, top_micro)
        
        # Generate detailed diagnosis
        diagnosis = self._generate_diagnosis(metrics, histogram, top_micro)
        
        return LaunchTaxReport(
            metrics=metrics,
            duration_histogram=histogram,
            top_microkernels=top_micro,
            top_heavy_kernels=top_heavy,
            kernel_name_patterns=patterns,
            assessment=assessment,
            recommendations=recommendations,
            detailed_diagnosis=diagnosis,
        )
    
    def _compute_metrics(self) -> LaunchTaxMetrics:
        """Compute all launch tax metrics."""
        # Basic aggregates
        total_calls = sum(k.calls for k in self.kernels)
        total_time_ms = sum(k.total_time_ms for k in self.kernels)
        
        # All durations (expanded by calls)
        all_durations_us = []
        for k in self.kernels:
            all_durations_us.extend([k.avg_time_us] * k.calls)
        
        if not all_durations_us:
            all_durations_us = [0]
        
        arr = np.array(all_durations_us)
        
        # Microkernel analysis
        micro_calls = sum(
            k.calls for k in self.kernels 
            if k.avg_time_us < self.threshold_us
        )
        micro_time = sum(
            k.total_time_ms for k in self.kernels 
            if k.avg_time_us < self.threshold_us
        )
        
        micro_pct = (micro_calls / total_calls * 100) if total_calls > 0 else 0
        micro_time_pct = (micro_time / total_time_ms * 100) if total_time_ms > 0 else 0
        
        # Launch rate
        wall_s = self.wall_clock_ms / 1000
        launch_rate = total_calls / wall_s if wall_s > 0 else 0
        launch_per_inf = total_calls / self.num_inferences if self.num_inferences > 0 else total_calls
        
        # Estimated launch overhead
        overhead_us = total_calls * self.ESTIMATED_LAUNCH_OVERHEAD_US
        overhead_ms = overhead_us / 1000
        overhead_pct = (overhead_ms / self.wall_clock_ms * 100) if self.wall_clock_ms > 0 else 0
        
        # GPU active ratio
        gpu_active = total_time_ms / self.wall_clock_ms if self.wall_clock_ms > 0 else 0
        
        # Useful compute ratio (excluding microkernel time as less useful)
        useful_time = total_time_ms - micro_time
        useful_ratio = useful_time / self.wall_clock_ms if self.wall_clock_ms > 0 else 0
        
        # Launch tax score (0-100, composite)
        # Factors: microkernel %, launch rate, avg duration
        score = self._compute_launch_tax_score(
            micro_pct, launch_rate, float(np.mean(arr)), overhead_pct
        )
        
        return LaunchTaxMetrics(
            total_kernel_count=total_calls,
            total_kernel_time_ms=total_time_ms,
            wall_clock_time_ms=self.wall_clock_ms,
            avg_kernel_duration_us=float(np.mean(arr)),
            p50_kernel_duration_us=float(np.percentile(arr, 50)),
            p90_kernel_duration_us=float(np.percentile(arr, 90)),
            min_kernel_duration_us=float(np.min(arr)),
            max_kernel_duration_us=float(np.max(arr)),
            microkernel_count=micro_calls,
            microkernel_pct=micro_pct,
            microkernel_threshold_us=self.threshold_us,
            microkernel_time_ms=micro_time,
            microkernel_time_pct=micro_time_pct,
            launch_rate_per_sec=launch_rate,
            launch_rate_per_inference=launch_per_inf,
            launch_tax_score=score,
            estimated_launch_overhead_ms=overhead_ms,
            estimated_launch_overhead_pct=overhead_pct,
            gpu_active_ratio=gpu_active,
            useful_compute_ratio=useful_ratio,
        )
    
    def _compute_launch_tax_score(
        self,
        micro_pct: float,
        launch_rate: float,
        avg_duration_us: float,
        overhead_pct: float,
    ) -> float:
        """
        Compute composite launch tax score (0-100).
        
        Factors:
        - High microkernel % is bad
        - High launch rate is bad
        - Low average duration is bad
        - High overhead % is bad
        """
        # Normalize each factor to 0-25 range
        
        # Microkernel score (30% > threshold → concerning)
        micro_score = min(25, micro_pct * 0.8)
        
        # Launch rate score (> 10000/sec is concerning)
        rate_score = min(25, launch_rate / 400)
        
        # Average duration score (< 50μs is concerning)
        if avg_duration_us < 50:
            duration_score = 25 * (1 - avg_duration_us / 50)
        else:
            duration_score = 0
        
        # Overhead score
        overhead_score = min(25, overhead_pct * 2.5)
        
        total = micro_score + rate_score + duration_score + overhead_score
        return min(100, total)
    
    def _build_duration_histogram(self) -> List[KernelDurationBucket]:
        """Build histogram of kernel durations."""
        total_calls = sum(k.calls for k in self.kernels)
        total_time = sum(k.total_time_ms for k in self.kernels)
        
        buckets = []
        for low, high in self.DURATION_BUCKETS:
            # Find kernels in this bucket
            bucket_kernels = [
                k for k in self.kernels
                if low <= k.avg_time_us < high
            ]
            
            count = sum(k.calls for k in bucket_kernels)
            time = sum(k.total_time_ms for k in bucket_kernels)
            
            bucket = KernelDurationBucket(
                range_us=(low, high if high != float("inf") else 999999),
                count=count,
                total_time_ms=time,
                pct_of_calls=(count / total_calls * 100) if total_calls > 0 else 0,
                pct_of_time=(time / total_time * 100) if total_time > 0 else 0,
                is_microkernel=(high <= self.threshold_us),
            )
            buckets.append(bucket)
        
        return buckets
    
    def _find_top_microkernels(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find most called tiny kernels."""
        micro_kernels = [
            k for k in self.kernels
            if k.avg_time_us < self.threshold_us
        ]
        
        # Sort by call count
        sorted_kernels = sorted(micro_kernels, key=lambda k: -k.calls)
        
        return [
            {
                "kernel_name": k.kernel_name,
                "calls": k.calls,
                "avg_time_us": k.avg_time_us,
                "total_time_ms": k.total_time_ms,
                "waste_potential": k.calls * self.ESTIMATED_LAUNCH_OVERHEAD_US / 1000,  # ms
            }
            for k in sorted_kernels[:top_n]
        ]
    
    def _find_top_heavy_kernels(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Find kernels consuming most time."""
        sorted_kernels = sorted(self.kernels, key=lambda k: -k.total_time_ms)
        
        total_time = sum(k.total_time_ms for k in self.kernels)
        
        return [
            {
                "kernel_name": k.kernel_name,
                "calls": k.calls,
                "avg_time_us": k.avg_time_us,
                "total_time_ms": k.total_time_ms,
                "pct_of_total": (k.total_time_ms / total_time * 100) if total_time > 0 else 0,
            }
            for k in sorted_kernels[:top_n]
        ]
    
    def _extract_kernel_patterns(self) -> Dict[str, int]:
        """Extract common kernel name patterns."""
        patterns: Dict[str, int] = {}
        
        for k in self.kernels:
            # Extract pattern (first word or known prefix)
            name = k.kernel_name
            
            # Common pattern extraction
            if "_" in name:
                prefix = name.split("_")[0]
            else:
                prefix = name[:min(20, len(name))]
            
            patterns[prefix] = patterns.get(prefix, 0) + k.calls
        
        # Sort by count and return top 20
        sorted_patterns = sorted(patterns.items(), key=lambda x: -x[1])
        return dict(sorted_patterns[:20])
    
    def _assess_severity(self, metrics: LaunchTaxMetrics) -> str:
        """Determine assessment severity."""
        score = metrics.launch_tax_score
        
        if score < self.ASSESSMENT_THRESHOLDS["healthy"]:
            return "healthy"
        elif score < self.ASSESSMENT_THRESHOLDS["concerning"]:
            return "concerning"
        else:
            return "critical"
    
    def _generate_recommendations(
        self, metrics: LaunchTaxMetrics, top_micro: List[Dict]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        if metrics.microkernel_pct > 30:
            recs.append(
                f"HIGH MICROKERNEL WARNING: {metrics.microkernel_pct:.1f}% of kernels are under "
                f"{self.threshold_us}μs. Consider operator fusion to reduce kernel count."
            )
        
        if metrics.launch_rate_per_sec > 10000:
            recs.append(
                f"LAUNCH RATE WARNING: {metrics.launch_rate_per_sec:.0f} kernel launches/sec. "
                "This may cause CPU-side bottleneck. Review graph partitioning."
            )
        
        if metrics.estimated_launch_overhead_pct > 10:
            recs.append(
                f"LAUNCH OVERHEAD: Estimated {metrics.estimated_launch_overhead_pct:.1f}% time "
                "spent in launch overhead. Target fusing these kernels:"
            )
            for micro in top_micro[:3]:
                recs.append(f"  - {micro['kernel_name']} ({micro['calls']} calls)")
        
        if metrics.gpu_active_ratio < 0.5:
            recs.append(
                f"LOW GPU ACTIVITY: GPU active only {metrics.gpu_active_ratio*100:.1f}% of time. "
                "Check for CPU bottleneck or data transfer issues."
            )
        
        if not recs:
            recs.append(
                "Kernel launch overhead is well-controlled. No immediate concerns."
            )
        
        return recs
    
    def _generate_diagnosis(
        self,
        metrics: LaunchTaxMetrics,
        histogram: List[KernelDurationBucket],
        top_micro: List[Dict],
    ) -> str:
        """Generate detailed diagnostic text."""
        lines = [
            "=== LAUNCH TAX ANALYSIS ===",
            "",
            f"Total Kernels: {metrics.total_kernel_count:,}",
            f"Total Kernel Time: {metrics.total_kernel_time_ms:.2f} ms",
            f"Wall Clock Time: {metrics.wall_clock_time_ms:.2f} ms",
            f"GPU Active Ratio: {metrics.gpu_active_ratio*100:.1f}%",
            "",
            "--- Duration Distribution ---",
        ]
        
        for bucket in histogram:
            if bucket.count > 0:
                low, high = bucket.range_us
                high_str = f"{high:.0f}" if high < 999999 else "∞"
                micro_flag = " [MICRO]" if bucket.is_microkernel else ""
                lines.append(
                    f"  {low:>6.0f}-{high_str:>6}μs: "
                    f"{bucket.count:>6} calls ({bucket.pct_of_calls:>5.1f}%), "
                    f"{bucket.total_time_ms:>8.2f}ms ({bucket.pct_of_time:>5.1f}%){micro_flag}"
                )
        
        lines.extend([
            "",
            "--- Launch Tax Metrics ---",
            f"Microkernel Count: {metrics.microkernel_count:,} ({metrics.microkernel_pct:.1f}%)",
            f"Launch Rate: {metrics.launch_rate_per_sec:,.0f} kernels/sec",
            f"Launch Tax Score: {metrics.launch_tax_score:.1f}/100",
            f"Estimated Launch Overhead: {metrics.estimated_launch_overhead_ms:.2f} ms "
            f"({metrics.estimated_launch_overhead_pct:.1f}%)",
            "",
        ])
        
        if top_micro:
            lines.append("--- Top Microkernel Offenders ---")
            for micro in top_micro[:5]:
                lines.append(
                    f"  {micro['kernel_name'][:50]}: "
                    f"{micro['calls']} calls @ {micro['avg_time_us']:.1f}μs avg"
                )
        
        return "\n".join(lines)
    
    def _empty_report(self) -> LaunchTaxReport:
        """Return empty report when no kernels."""
        metrics = LaunchTaxMetrics(
            total_kernel_count=0,
            total_kernel_time_ms=0,
            wall_clock_time_ms=self.wall_clock_ms,
            avg_kernel_duration_us=0,
            p50_kernel_duration_us=0,
            p90_kernel_duration_us=0,
            min_kernel_duration_us=0,
            max_kernel_duration_us=0,
            microkernel_count=0,
            microkernel_pct=0,
            microkernel_threshold_us=self.threshold_us,
            microkernel_time_ms=0,
            microkernel_time_pct=0,
            launch_rate_per_sec=0,
            launch_rate_per_inference=0,
            launch_tax_score=0,
            estimated_launch_overhead_ms=0,
            estimated_launch_overhead_pct=0,
            gpu_active_ratio=0,
            useful_compute_ratio=0,
        )
        
        return LaunchTaxReport(
            metrics=metrics,
            duration_histogram=[],
            top_microkernels=[],
            top_heavy_kernels=[],
            kernel_name_patterns={},
            assessment="no_data",
            recommendations=["No kernel data available for analysis."],
            detailed_diagnosis="No kernel data.",
        )
    
    def plot_duration_histogram(self, output_path: Optional[str] = None):
        """Generate kernel duration histogram plot."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return None
        
        histogram = self._build_duration_histogram()
        
        # Filter non-empty buckets
        data = [(b.range_us, b.count, b.pct_of_time, b.is_microkernel) for b in histogram if b.count > 0]
        
        if not data:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Kernel Duration Analysis", fontsize=14, fontweight="bold")
        
        # Left: Call count by duration
        labels = [f"{d[0][0]}-{d[0][1] if d[0][1] < 999999 else '∞'}μs" for d in data]
        counts = [d[1] for d in data]
        colors = ["red" if d[3] else "steelblue" for d in data]
        
        ax1.barh(labels, counts, color=colors, alpha=0.7)
        ax1.set_xlabel("Kernel Call Count")
        ax1.set_title("Calls by Duration Bucket")
        ax1.axvline(x=0, color="gray", linewidth=0.5)
        
        # Add microkernel annotation
        ax1.text(
            0.95, 0.05,
            "Red = Microkernel (<10μs)",
            transform=ax1.transAxes,
            ha="right",
            fontsize=9,
            color="red",
        )
        
        # Right: Time contribution
        time_pcts = [d[2] for d in data]
        
        ax2.barh(labels, time_pcts, color=colors, alpha=0.7)
        ax2.set_xlabel("% of Total GPU Time")
        ax2.set_title("Time Contribution by Duration")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved launch tax plot to {output_path}")
        
        return fig


def analyze_launch_tax(
    kernel_summaries: List[KernelSummary],
    wall_clock_ms: float,
    num_inferences: int = 1,
) -> LaunchTaxReport:
    """
    Convenience function for launch tax analysis.
    
    Args:
        kernel_summaries: Kernel profiling data
        wall_clock_ms: Total wall clock time
        num_inferences: Number of inference runs
        
    Returns:
        LaunchTaxReport with full analysis.
    """
    analyzer = LaunchTaxAnalyzer(
        kernel_summaries=kernel_summaries,
        wall_clock_ms=wall_clock_ms,
        num_inferences=num_inferences,
    )
    return analyzer.analyze()
