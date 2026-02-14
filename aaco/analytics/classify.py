"""
Bottleneck Classifier
Rule-based and ML-assisted classification of performance bottlenecks.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from aaco.core.schema import (
    BottleneckClassification,
    DerivedMetrics,
    KernelMetrics,
)

logger = logging.getLogger(__name__)


class BottleneckCategory(Enum):
    """Categories of performance bottlenecks."""

    COMPUTE_BOUND = "compute_bound"
    MEMORY_BOUND = "memory_bound"
    LAUNCH_OVERHEAD = "launch_overhead"
    DATA_TRANSFER = "data_transfer"
    CPU_BOUND = "cpu_bound"
    THERMAL_THROTTLE = "thermal_throttle"
    FREQUENCY_SCALING = "frequency_scaling"
    WARMUP_INSTABILITY = "warmup_instability"
    KERNEL_FRAGMENTATION = "kernel_fragmentation"
    IO_BOUND = "io_bound"
    BALANCED = "balanced"
    UNKNOWN = "unknown"


@dataclass
class IndicatorWeight:
    """Weight configuration for an indicator."""

    name: str
    weight: float
    threshold_low: float
    threshold_high: float


class BottleneckClassifier:
    """
    Classifies workload bottlenecks based on collected metrics.

    Uses a combination of:
    1. Rule-based heuristics with thresholds
    2. Weighted indicator scoring
    3. Multi-dimensional bottleneck fingerprinting
    """

    # Threshold configurations
    THRESHOLDS = {
        # GPU utilization thresholds
        "gpu_util_high": 85.0,  # Above this = compute bound
        "gpu_util_low": 40.0,  # Below this = likely bottleneck elsewhere
        # Launch overhead indicators
        "microkernel_pct_high": 30.0,  # High % of tiny kernels
        "kar_high": 10.0,  # Many kernels per ONNX node
        "launch_rate_high": 5000.0,  # Kernels/sec
        # Memory indicators
        "mem_util_high": 80.0,  # Memory bandwidth bound
        "vram_high_pct": 90.0,  # Near VRAM capacity
        # Thermal/power
        "temp_high_c": 85.0,  # Potential throttling
        "power_high_pct": 90.0,  # Near power limit
        # Latency stability
        "cov_high_pct": 15.0,  # High coefficient of variation
        "warmup_effect_high": 20.0,  # Large warmup effect
        # CPU indicators
        "cpu_high_pct": 80.0,  # CPU possibly limiting
    }

    def __init__(self):
        self.indicators: Dict[str, float] = {}
        self.scores: Dict[BottleneckCategory, float] = {}
        self.evidence: List[str] = []

    def classify(
        self,
        metrics: Optional[DerivedMetrics] = None,
        kernel_metrics: Optional[KernelMetrics] = None,
        custom_indicators: Optional[Dict[str, float]] = None,
    ) -> BottleneckClassification:
        """
        Classify the bottleneck based on metrics.

        Args:
            metrics: Derived performance metrics
            kernel_metrics: Kernel-level metrics
            custom_indicators: Additional indicators to consider

        Returns:
            BottleneckClassification with category, confidence, and evidence.
        """
        self.indicators = {}
        self.scores = {cat: 0.0 for cat in BottleneckCategory}
        self.evidence = []

        # Extract indicators from metrics
        if metrics:
            self._extract_from_derived_metrics(metrics)

        if kernel_metrics:
            self._extract_from_kernel_metrics(kernel_metrics)

        if custom_indicators:
            self.indicators.update(custom_indicators)

        # Apply classification rules
        self._apply_compute_rules()
        self._apply_memory_rules()
        self._apply_launch_overhead_rules()
        self._apply_thermal_rules()
        self._apply_cpu_rules()
        self._apply_stability_rules()

        # Determine primary bottleneck
        primary, confidence = self._determine_primary()

        # Build secondary indicators list
        secondary = [
            cat.value for cat, score in self.scores.items() if score > 0.3 and cat != primary
        ]

        return BottleneckClassification(
            primary=primary.value,
            secondary=secondary,
            confidence=confidence,
            indicators=self.indicators.copy(),
            evidence=self.evidence.copy(),
            recommendations=self._generate_recommendations(primary),
        )

    def _extract_from_derived_metrics(self, metrics: DerivedMetrics) -> None:
        """Extract indicator values from derived metrics."""
        # Efficiency indicators
        self.indicators["gpu_active_ratio"] = metrics.efficiency.get("gpu_active_ratio", 0)
        self.indicators["kar"] = metrics.efficiency.get("kernel_amplification_ratio", 0)
        self.indicators["microkernel_pct"] = metrics.efficiency.get("microkernel_pct", 0)

        # GPU utilization
        self.indicators["gpu_util_pct"] = metrics.gpu.get("gpu_util_mean_pct", 0)
        self.indicators["mem_util_pct"] = metrics.gpu.get("mem_util_mean_pct", 0)
        self.indicators["temp_c"] = metrics.gpu.get("temp_max_c", 0)
        self.indicators["power_w"] = metrics.gpu.get("power_max_w", 0)
        self.indicators["sclk_range_mhz"] = metrics.gpu.get("sclk_range_mhz", 0)

        # CPU/System
        self.indicators["cpu_pct"] = metrics.system.get("cpu_max_pct", 0)
        self.indicators["rss_delta_mb"] = metrics.system.get("rss_delta_mb", 0)

        # Latency stability
        self.indicators["cov_pct"] = metrics.measurement_phase.cov_pct
        self.indicators["warmup_effect_pct"] = metrics.latency.get("warmup_effect_pct", 0)

    def _extract_from_kernel_metrics(self, km: KernelMetrics) -> None:
        """Extract indicators from kernel metrics."""
        self.indicators["total_kernels"] = km.total_kernel_count
        self.indicators["unique_kernels"] = km.unique_kernel_count
        self.indicators["kernel_time_ms"] = km.total_kernel_time_ms
        self.indicators["avg_kernel_us"] = km.avg_kernel_duration_us
        self.indicators["microkernel_count"] = km.microkernel_count
        self.indicators["microkernel_pct"] = km.microkernel_pct
        self.indicators["launch_rate"] = km.launch_rate_per_sec
        self.indicators["launch_tax"] = km.launch_tax_score
        self.indicators["kar"] = km.kernel_amplification_ratio
        self.indicators["gpu_active_ratio"] = km.gpu_active_ratio

    def _apply_compute_rules(self) -> None:
        """Apply rules for compute-bound detection."""
        gpu_util = self.indicators.get("gpu_util_pct", 0)
        gpu_active = self.indicators.get("gpu_active_ratio", 0)

        if gpu_util > self.THRESHOLDS["gpu_util_high"]:
            self.scores[BottleneckCategory.COMPUTE_BOUND] += 0.5
            self.evidence.append(f"High GPU utilization: {gpu_util:.1f}%")

        if gpu_active > 0.7:
            self.scores[BottleneckCategory.COMPUTE_BOUND] += 0.3
            self.evidence.append(f"High GPU active ratio: {gpu_active:.2f}")

    def _apply_memory_rules(self) -> None:
        """Apply rules for memory-bound detection."""
        mem_util = self.indicators.get("mem_util_pct", 0)
        gpu_util = self.indicators.get("gpu_util_pct", 0)

        if mem_util > self.THRESHOLDS["mem_util_high"]:
            self.scores[BottleneckCategory.MEMORY_BOUND] += 0.5
            self.evidence.append(f"High memory utilization: {mem_util:.1f}%")

        # Low GPU util + moderate memory util = memory bound
        if gpu_util < 50 and mem_util > 50:
            self.scores[BottleneckCategory.MEMORY_BOUND] += 0.3
            self.evidence.append("Low GPU util with moderate memory util suggests memory bound")

    def _apply_launch_overhead_rules(self) -> None:
        """Apply rules for kernel launch overhead detection."""
        microkernel_pct = self.indicators.get("microkernel_pct", 0)
        launch_rate = self.indicators.get("launch_rate", 0)
        kar = self.indicators.get("kar", 0)
        gpu_active = self.indicators.get("gpu_active_ratio", 0)

        score = 0

        if microkernel_pct > self.THRESHOLDS["microkernel_pct_high"]:
            score += 0.4
            self.evidence.append(f"High microkernel %: {microkernel_pct:.1f}%")

        if kar > self.THRESHOLDS["kar_high"]:
            score += 0.3
            self.evidence.append(f"High kernel amplification: {kar:.1f}x")

        if launch_rate > self.THRESHOLDS["launch_rate_high"]:
            score += 0.2
            self.evidence.append(f"High launch rate: {launch_rate:.0f}/sec")

        # Low GPU active ratio + high launch rate = launch overhead
        if gpu_active < 0.3 and launch_rate > 1000:
            score += 0.3
            self.evidence.append("Low GPU active ratio with high launch rate")

        self.scores[BottleneckCategory.LAUNCH_OVERHEAD] += min(1.0, score)

    def _apply_thermal_rules(self) -> None:
        """Apply rules for thermal throttling detection."""
        temp = self.indicators.get("temp_c", 0)
        sclk_range = self.indicators.get("sclk_range_mhz", 0)

        if temp > self.THRESHOLDS["temp_high_c"]:
            self.scores[BottleneckCategory.THERMAL_THROTTLE] += 0.6
            self.evidence.append(f"High temperature: {temp:.0f}°C")

        if sclk_range > 200:  # Large clock variation suggests throttling
            self.scores[BottleneckCategory.FREQUENCY_SCALING] += 0.4
            self.evidence.append(f"Large clock variation: {sclk_range:.0f}MHz")

    def _apply_cpu_rules(self) -> None:
        """Apply rules for CPU-bound detection."""
        cpu_pct = self.indicators.get("cpu_pct", 0)
        gpu_util = self.indicators.get("gpu_util_pct", 0)

        if cpu_pct > self.THRESHOLDS["cpu_high_pct"]:
            self.scores[BottleneckCategory.CPU_BOUND] += 0.4
            self.evidence.append(f"High CPU utilization: {cpu_pct:.1f}%")

        # High CPU + low GPU = CPU limiting GPU work
        if cpu_pct > 80 and gpu_util < 50:
            self.scores[BottleneckCategory.CPU_BOUND] += 0.4
            self.evidence.append("CPU appears to be limiting GPU utilization")

    def _apply_stability_rules(self) -> None:
        """Apply rules for warmup/stability issues."""
        cov = self.indicators.get("cov_pct", 0)
        warmup_effect = self.indicators.get("warmup_effect_pct", 0)

        if cov > self.THRESHOLDS["cov_high_pct"]:
            self.evidence.append(f"High latency variance: CoV={cov:.1f}%")

        if abs(warmup_effect) > self.THRESHOLDS["warmup_effect_high"]:
            self.scores[BottleneckCategory.WARMUP_INSTABILITY] += 0.5
            self.evidence.append(f"Significant warmup effect: {warmup_effect:.1f}%")

    def _determine_primary(self) -> Tuple[BottleneckCategory, float]:
        """Determine the primary bottleneck category."""
        if not any(self.scores.values()):
            return BottleneckCategory.BALANCED, 0.5

        # Find highest scoring category
        sorted_scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)

        top_cat, top_score = sorted_scores[0]

        # If top score is low, classify as balanced
        if top_score < 0.3:
            return BottleneckCategory.BALANCED, 0.6

        # Confidence based on margin over second place
        if len(sorted_scores) > 1:
            second_score = sorted_scores[1][1]
            margin = top_score - second_score
            confidence = min(0.95, 0.5 + margin)
        else:
            confidence = min(0.95, 0.5 + top_score * 0.5)

        return top_cat, confidence

    def _generate_recommendations(self, category: BottleneckCategory) -> List[str]:
        """Generate optimization recommendations based on bottleneck type."""
        recommendations = {
            BottleneckCategory.COMPUTE_BOUND: [
                "Consider using FP16/INT8 quantization if accuracy permits",
                "Enable kernel fusion optimizations",
                "Profile individual kernels for optimization opportunities",
            ],
            BottleneckCategory.MEMORY_BOUND: [
                "Reduce model memory footprint with quantization",
                "Optimize memory access patterns",
                "Consider operator fusion to reduce memory traffic",
            ],
            BottleneckCategory.LAUNCH_OVERHEAD: [
                "Enable kernel batching/fusion to reduce launch count",
                "Review ONNX graph for suboptimal patterns",
                "Consider CUDA graphs or persistent execution modes",
                "Investigate MIGraphX EP operator fusion options",
            ],
            BottleneckCategory.DATA_TRANSFER: [
                "Pin host memory for faster transfers",
                "Use async data transfers with pipelining",
                "Minimize host-device synchronization points",
            ],
            BottleneckCategory.CPU_BOUND: [
                "Profile CPU code for optimization opportunities",
                "Consider using async inference APIs",
                "Reduce pre/post processing overhead",
            ],
            BottleneckCategory.THERMAL_THROTTLE: [
                "Improve cooling solution",
                "Reduce sustained workload intensity",
                "Set conservative power limits",
            ],
            BottleneckCategory.FREQUENCY_SCALING: [
                "Lock GPU clocks for consistent performance",
                "Disable power management during benchmarks",
                "Ensure performance governor is set",
            ],
            BottleneckCategory.WARMUP_INSTABILITY: [
                "Increase warmup iterations",
                "Ensure JIT compilation completes before measurement",
                "Clear GPU caches between runs for reproducibility",
            ],
            BottleneckCategory.BALANCED: [
                "System is reasonably balanced",
                "Consider model architecture optimizations",
                "Explore hardware upgrade options for further gains",
            ],
            BottleneckCategory.UNKNOWN: [
                "Collect additional profiling data",
                "Review system configuration for anomalies",
            ],
        }

        return recommendations.get(category, [])

    def explain(self) -> str:
        """Generate human-readable explanation of classification."""
        result = self.classify()

        lines = [
            f"Primary Bottleneck: {result.primary.upper()}",
            f"Confidence: {result.confidence:.0%}",
        ]

        if result.secondary:
            lines.append(f"Secondary Factors: {', '.join(result.secondary)}")

        lines.append("\nEvidence:")
        for ev in result.evidence:
            lines.append(f"  • {ev}")

        lines.append("\nRecommendations:")
        for rec in result.recommendations:
            lines.append(f"  → {rec}")

        return "\n".join(lines)
