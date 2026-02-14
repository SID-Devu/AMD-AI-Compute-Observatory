"""
Batch Scaling Analyzer
Analyzes throughput and latency scaling across batch sizes.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BatchPoint:
    """Single measurement point at a specific batch size."""

    batch_size: int
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_std_ms: float
    throughput_samples_per_sec: float
    throughput_batches_per_sec: float
    vram_usage_mb: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    power_w: Optional[float] = None


@dataclass
class ScalingAnalysis:
    """Complete batch scaling analysis results."""

    batch_sizes: List[int]
    points: List[BatchPoint]
    scaling_efficiency: float  # 0-1, how well throughput scales
    saturation_batch: Optional[int]  # Batch where throughput plateaus
    memory_bound_batch: Optional[int]  # Batch where VRAM becomes limiting
    optimal_batch: int  # Best batch for throughput/latency tradeoff
    throughput_curve_fit: Dict[str, float]  # Polynomial fit coefficients
    latency_curve_fit: Dict[str, float]
    bottleneck_transition: Optional[Dict[str, Any]]  # Where bottleneck changes
    analysis_summary: str
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_sizes": self.batch_sizes,
            "points": [asdict(p) for p in self.points],
            "scaling_efficiency": self.scaling_efficiency,
            "saturation_batch": self.saturation_batch,
            "memory_bound_batch": self.memory_bound_batch,
            "optimal_batch": self.optimal_batch,
            "throughput_curve_fit": self.throughput_curve_fit,
            "latency_curve_fit": self.latency_curve_fit,
            "bottleneck_transition": self.bottleneck_transition,
            "analysis_summary": self.analysis_summary,
            "recommendations": self.recommendations,
        }


class BatchScalingAnalyzer:
    """
    Analyzes how model performance scales with batch size.

    Key insights:
    - Throughput scaling efficiency (how close to linear)
    - Saturation point (where throughput plateaus)
    - Memory pressure thresholds
    - Optimal batch size for throughput/latency tradeoff
    """

    # Default batch sizes to sweep
    DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32]

    # Thresholds
    SATURATION_THRESHOLD = 0.1  # <10% throughput gain = saturated
    EFFICIENCY_IDEAL = 0.9  # 90% of linear scaling = good

    def __init__(self):
        self.points: List[BatchPoint] = []
        self.vram_limit_mb: Optional[float] = None

    def add_measurement(
        self,
        batch_size: int,
        latencies_ms: List[float],
        vram_mb: Optional[float] = None,
        gpu_util: Optional[float] = None,
        power_w: Optional[float] = None,
    ) -> BatchPoint:
        """
        Add a measurement at a specific batch size.

        Args:
            batch_size: Batch size used
            latencies_ms: List of per-inference latencies
            vram_mb: VRAM usage at this batch
            gpu_util: GPU utilization %
            power_w: Power consumption

        Returns:
            BatchPoint with computed metrics.
        """
        arr = np.array(latencies_ms)

        mean_ms = float(np.mean(arr))
        throughput_batches = 1000.0 / mean_ms  # batches/sec
        throughput_samples = throughput_batches * batch_size  # samples/sec

        point = BatchPoint(
            batch_size=batch_size,
            latency_p50_ms=float(np.percentile(arr, 50)),
            latency_p90_ms=float(np.percentile(arr, 90)),
            latency_p99_ms=float(np.percentile(arr, 99)),
            latency_mean_ms=mean_ms,
            latency_std_ms=float(np.std(arr)),
            throughput_samples_per_sec=throughput_samples,
            throughput_batches_per_sec=throughput_batches,
            vram_usage_mb=vram_mb,
            gpu_util_pct=gpu_util,
            power_w=power_w,
        )

        # Insert in sorted order
        self.points.append(point)
        self.points.sort(key=lambda p: p.batch_size)

        return point

    def analyze(self) -> ScalingAnalysis:
        """
        Perform complete scaling analysis.

        Returns:
            ScalingAnalysis with all insights.
        """
        if len(self.points) < 2:
            return self._minimal_analysis()

        batch_sizes = [p.batch_size for p in self.points]
        throughputs = [p.throughput_samples_per_sec for p in self.points]
        latencies = [p.latency_mean_ms for p in self.points]

        # Compute scaling efficiency
        efficiency = self._compute_scaling_efficiency(batch_sizes, throughputs)

        # Find saturation point
        saturation = self._find_saturation_point(batch_sizes, throughputs)

        # Find memory bound point
        memory_bound = self._find_memory_bound_point()

        # Find optimal batch
        optimal = self._find_optimal_batch()

        # Fit curves
        throughput_fit = self._fit_curve(batch_sizes, throughputs)
        latency_fit = self._fit_curve(batch_sizes, latencies)

        # Detect bottleneck transitions
        transition = self._detect_bottleneck_transition()

        # Generate analysis
        summary = self._generate_summary(efficiency, saturation, memory_bound, optimal)
        recommendations = self._generate_recommendations(efficiency, saturation, memory_bound)

        return ScalingAnalysis(
            batch_sizes=batch_sizes,
            points=self.points,
            scaling_efficiency=efficiency,
            saturation_batch=saturation,
            memory_bound_batch=memory_bound,
            optimal_batch=optimal,
            throughput_curve_fit=throughput_fit,
            latency_curve_fit=latency_fit,
            bottleneck_transition=transition,
            analysis_summary=summary,
            recommendations=recommendations,
        )

    def _compute_scaling_efficiency(
        self, batch_sizes: List[int], throughputs: List[float]
    ) -> float:
        """
        Compute how well throughput scales with batch size.
        Perfect linear scaling = 1.0
        """
        if len(batch_sizes) < 2:
            return 1.0

        # Ideal scaling: throughput proportional to batch
        base_throughput = throughputs[0]
        base_batch = batch_sizes[0]

        ideal_throughputs = [base_throughput * (b / base_batch) for b in batch_sizes]

        # Compute ratio of actual to ideal
        ratios = [
            actual / ideal if ideal > 0 else 0
            for actual, ideal in zip(throughputs, ideal_throughputs)
        ]

        # Average efficiency (excluding first point which is always 1.0)
        return float(np.mean(ratios[1:])) if len(ratios) > 1 else 1.0

    def _find_saturation_point(
        self, batch_sizes: List[int], throughputs: List[float]
    ) -> Optional[int]:
        """Find batch size where throughput gains drop below threshold."""
        if len(batch_sizes) < 2:
            return None

        for i in range(1, len(throughputs)):
            gain = (throughputs[i] - throughputs[i - 1]) / throughputs[i - 1]
            if gain < self.SATURATION_THRESHOLD:
                return batch_sizes[i]

        return None  # Never saturated in tested range

    def _find_memory_bound_point(self) -> Optional[int]:
        """Find batch size where VRAM usage becomes concerning."""
        if not self.vram_limit_mb:
            return None

        for point in self.points:
            if point.vram_usage_mb and point.vram_usage_mb > 0.9 * self.vram_limit_mb:
                return point.batch_size

        return None

    def _find_optimal_batch(self) -> int:
        """
        Find optimal batch size balancing throughput and latency.
        Uses efficiency-weighted throughput.
        """
        if not self.points:
            return 1

        # Score each point: throughput / (latency * batch)
        # This favors high throughput with reasonable latency
        best_score = 0
        best_batch = self.points[0].batch_size

        for point in self.points:
            score = point.throughput_samples_per_sec / (
                point.latency_mean_ms * np.log2(point.batch_size + 1)
            )
            if score > best_score:
                best_score = score
                best_batch = point.batch_size

        return best_batch

    def _fit_curve(self, x: List[int], y: List[float]) -> Dict[str, float]:
        """Fit polynomial curve to data."""
        try:
            # Log scale for better fit
            log_x = np.log2([max(1, v) for v in x])
            coeffs = np.polyfit(log_x, y, deg=min(2, len(x) - 1))

            return {f"c{i}": float(c) for i, c in enumerate(coeffs)}
        except Exception:
            return {}

    def _detect_bottleneck_transition(self) -> Optional[Dict[str, Any]]:
        """Detect where the bottleneck type changes."""
        if len(self.points) < 3:
            return None

        # Look for inflection points in throughput curve
        throughputs = [p.throughput_samples_per_sec for p in self.points]

        # Compute gains
        gains = []
        for i in range(1, len(throughputs)):
            gain = (throughputs[i] - throughputs[i - 1]) / throughputs[i - 1]
            gains.append(gain)

        # Find largest gain drop
        if len(gains) < 2:
            return None

        gain_drops = []
        for i in range(1, len(gains)):
            drop = gains[i - 1] - gains[i]
            gain_drops.append(drop)

        max_drop_idx = np.argmax(gain_drops) if gain_drops else None
        if max_drop_idx is None:
            return None

        transition_batch = self.points[max_drop_idx + 1].batch_size

        return {
            "transition_batch": transition_batch,
            "gain_before": gains[max_drop_idx],
            "gain_after": gains[max_drop_idx + 1] if max_drop_idx + 1 < len(gains) else 0,
            "likely_cause": self._infer_transition_cause(max_drop_idx),
        }

    def _infer_transition_cause(self, transition_idx: int) -> str:
        """Infer likely cause of bottleneck transition."""
        point_before = self.points[transition_idx]
        point_after = self.points[transition_idx + 1]

        # Check VRAM increase
        if point_before.vram_usage_mb and point_after.vram_usage_mb:
            vram_increase = point_after.vram_usage_mb / point_before.vram_usage_mb
            if vram_increase > 1.8:
                return "memory_capacity"

        # Check GPU utilization plateau
        if point_before.gpu_util_pct and point_after.gpu_util_pct:
            if point_after.gpu_util_pct > 90 and point_before.gpu_util_pct < 85:
                return "compute_saturation"

        return "bandwidth_saturation"

    def _generate_summary(
        self,
        efficiency: float,
        saturation: Optional[int],
        memory_bound: Optional[int],
        optimal: int,
    ) -> str:
        """Generate human-readable summary."""
        parts = []

        # Efficiency assessment
        if efficiency >= 0.9:
            parts.append("Excellent batch scaling efficiency (≥90% of linear).")
        elif efficiency >= 0.7:
            parts.append(f"Good batch scaling efficiency ({efficiency:.0%} of linear).")
        elif efficiency >= 0.5:
            parts.append(
                f"Moderate batch scaling ({efficiency:.0%} of linear). Consider optimization."
            )
        else:
            parts.append(f"Poor batch scaling ({efficiency:.0%} of linear). Bottleneck present.")

        # Saturation point
        if saturation:
            parts.append(f"Throughput saturates at batch {saturation}.")
        else:
            parts.append("No saturation observed in tested range.")

        # Memory bound
        if memory_bound:
            parts.append(f"VRAM-limited above batch {memory_bound}.")

        # Optimal
        parts.append(f"Optimal batch size: {optimal}.")

        return " ".join(parts)

    def _generate_recommendations(
        self,
        efficiency: float,
        saturation: Optional[int],
        memory_bound: Optional[int],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        if efficiency < 0.7:
            recs.append(
                "Low scaling efficiency suggests bandwidth or launch overhead bottleneck. "
                "Consider operator fusion or attention to memory access patterns."
            )

        if saturation and saturation <= 4:
            recs.append(
                f"Early saturation at batch {saturation} indicates limited GPU utilization. "
                "Check for CPU bottlenecks or kernel launch overhead."
            )

        if memory_bound:
            recs.append(
                f"Memory pressure above batch {memory_bound}. "
                "Consider mixed precision (FP16) or gradient checkpointing."
            )

        if not recs:
            recs.append("Scaling behavior is healthy across tested batch sizes.")

        return recs

    def _minimal_analysis(self) -> ScalingAnalysis:
        """Return minimal analysis when insufficient data."""
        batch_sizes = [p.batch_size for p in self.points]
        return ScalingAnalysis(
            batch_sizes=batch_sizes,
            points=self.points,
            scaling_efficiency=1.0,
            saturation_batch=None,
            memory_bound_batch=None,
            optimal_batch=batch_sizes[0] if batch_sizes else 1,
            throughput_curve_fit={},
            latency_curve_fit={},
            bottleneck_transition=None,
            analysis_summary="Insufficient data for full analysis.",
            recommendations=["Run more batch sizes for complete analysis."],
        )

    def get_scaling_table(self) -> List[Dict[str, Any]]:
        """Get tabular view of scaling data."""
        return [
            {
                "batch": p.batch_size,
                "latency_ms": f"{p.latency_mean_ms:.2f}",
                "throughput": f"{p.throughput_samples_per_sec:.1f}",
                "vram_mb": f"{p.vram_usage_mb:.0f}" if p.vram_usage_mb else "-",
                "gpu_util": f"{p.gpu_util_pct:.1f}%" if p.gpu_util_pct else "-",
            }
            for p in self.points
        ]

    def plot_scaling_curves(self, output_path: Optional[str] = None):
        """
        Generate scaling plots.

        Plots:
        1. Throughput vs Batch Size
        2. Latency vs Batch Size
        3. Efficiency curve
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plots")
            return None

        if len(self.points) < 2:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Batch Scaling Analysis", fontsize=14, fontweight="bold")

        batches = [p.batch_size for p in self.points]
        throughputs = [p.throughput_samples_per_sec for p in self.points]
        latencies = [p.latency_mean_ms for p in self.points]

        # Plot 1: Throughput
        ax1 = axes[0, 0]
        ax1.plot(batches, throughputs, "b-o", linewidth=2, markersize=8)
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Throughput (samples/sec)")
        ax1.set_title("Throughput Scaling")
        ax1.grid(True, alpha=0.3)

        # Add ideal linear line
        ideal = [throughputs[0] * b / batches[0] for b in batches]
        ax1.plot(batches, ideal, "g--", alpha=0.5, label="Ideal Linear")
        ax1.legend()

        # Plot 2: Latency
        ax2 = axes[0, 1]
        ax2.plot(batches, latencies, "r-o", linewidth=2, markersize=8)
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Latency vs Batch Size")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Efficiency
        ax3 = axes[1, 0]
        efficiencies = [
            throughputs[i] / (throughputs[0] * batches[i] / batches[0]) for i in range(len(batches))
        ]
        ax3.bar(range(len(batches)), efficiencies, color="purple", alpha=0.7)
        ax3.set_xticks(range(len(batches)))
        ax3.set_xticklabels([str(b) for b in batches])
        ax3.set_xlabel("Batch Size")
        ax3.set_ylabel("Scaling Efficiency")
        ax3.set_title("Scaling Efficiency (1.0 = Perfect)")
        ax3.axhline(y=1.0, color="g", linestyle="--", alpha=0.5)
        ax3.set_ylim(0, 1.2)

        # Plot 4: VRAM if available
        ax4 = axes[1, 1]
        vrams = [p.vram_usage_mb for p in self.points if p.vram_usage_mb]
        if vrams and len(vrams) == len(batches):
            ax4.plot(batches, vrams, "orange", marker="s", linewidth=2, markersize=8)
            ax4.set_xlabel("Batch Size")
            ax4.set_ylabel("VRAM Usage (MB)")
            ax4.set_title("Memory Scaling")
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(
                0.5,
                0.5,
                "VRAM data not available",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title("Memory Scaling")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved scaling plots to {output_path}")

        return fig


def analyze_batch_scaling(
    measurements: List[Dict[str, Any]],
    vram_limit_mb: Optional[float] = None,
) -> ScalingAnalysis:
    """
    Convenience function to analyze batch scaling from measurements.

    Args:
        measurements: List of dicts with batch_size, latencies_ms, etc.
        vram_limit_mb: GPU VRAM limit for memory bound detection

    Returns:
        ScalingAnalysis with complete analysis.
    """
    analyzer = BatchScalingAnalyzer()
    analyzer.vram_limit_mb = vram_limit_mb

    for m in measurements:
        analyzer.add_measurement(
            batch_size=m["batch_size"],
            latencies_ms=m["latencies_ms"],
            vram_mb=m.get("vram_mb"),
            gpu_util=m.get("gpu_util"),
            power_w=m.get("power_w"),
        )

    return analyzer.analyze()
