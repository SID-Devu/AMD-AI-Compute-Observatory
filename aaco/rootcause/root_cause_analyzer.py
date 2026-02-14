"""
AACO-SIGMA Root Cause Analyzer

Analyzes performance data to identify root causes of performance issues.
Uses multi-layered analysis across hardware, software, and configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from enum import Enum, auto


class CauseCategory(Enum):
    """Categories of root causes."""

    # Hardware-level
    HARDWARE_MEMORY_BANDWIDTH = auto()
    HARDWARE_COMPUTE_BOUND = auto()
    HARDWARE_CACHE_MISS = auto()
    HARDWARE_OCCUPANCY = auto()
    HARDWARE_THERMAL = auto()

    # Software-level
    SOFTWARE_KERNEL_INEFFICIENCY = auto()
    SOFTWARE_MEMORY_PATTERN = auto()
    SOFTWARE_SYNCHRONIZATION = auto()
    SOFTWARE_FUSION_MISSED = auto()

    # Configuration-level
    CONFIG_BATCH_SIZE = auto()
    CONFIG_WORKGROUP_SIZE = auto()
    CONFIG_PRECISION = auto()
    CONFIG_ENVIRONMENT = auto()

    # Model-level
    MODEL_ARCHITECTURE = auto()
    MODEL_LAYER_BOTTLENECK = auto()
    MODEL_ATTENTION_OVERHEAD = auto()

    # System-level
    SYSTEM_DRIVER = auto()
    SYSTEM_OS_SCHEDULER = auto()
    SYSTEM_POWER_STATE = auto()
    SYSTEM_INTERFERENCE = auto()

    # Unknown
    UNKNOWN = auto()


@dataclass
class RootCause:
    """A identified root cause."""

    # Identity
    cause_id: str
    category: CauseCategory

    # Description
    title: str
    description: str

    # Impact
    impact_pct: float = 0.0  # Estimated % of total latency
    confidence: float = 0.0  # 0-1 confidence level

    # Location
    layer_name: str = ""
    kernel_name: str = ""
    code_location: str = ""

    # Evidence
    evidence_ids: List[str] = field(default_factory=list)

    # Related causes
    related_causes: List[str] = field(default_factory=list)

    # Actionable
    suggested_fixes: List[str] = field(default_factory=list)
    fix_complexity: str = ""  # "low", "medium", "high"

    # Metadata
    detected_at: float = 0.0


@dataclass
class AnalysisContext:
    """Context for root cause analysis."""

    # Performance data
    kernel_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    layer_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Counter data
    hardware_counters: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Timing
    total_latency_ms: float = 0.0
    kernel_latencies: Dict[str, float] = field(default_factory=dict)

    # Environment
    gpu_model: str = ""
    driver_version: str = ""

    # Baseline comparison
    baseline_latencies: Dict[str, float] = field(default_factory=dict)


class RootCauseAnalyzer:
    """
    Analyzes performance data to identify root causes.

    Analysis layers:
    1. Hardware analysis (counters, utilization)
    2. Software analysis (kernel patterns, memory access)
    3. Configuration analysis (settings, environment)
    4. Comparative analysis (vs baseline, vs expected)
    """

    # Thresholds for detection
    MEMORY_BOUND_THRESHOLD = 0.8  # 80% memory BW utilization
    COMPUTE_BOUND_THRESHOLD = 0.8  # 80% compute utilization
    CACHE_MISS_THRESHOLD = 0.3  # 30% cache miss rate
    OCCUPANCY_LOW_THRESHOLD = 0.5  # 50% occupancy
    REGRESSION_THRESHOLD = 0.1  # 10% regression

    def __init__(self):
        self._analysis_rules: List[callable] = []
        self._setup_default_rules()

    def _setup_default_rules(self):
        """Setup default analysis rules."""
        self._analysis_rules = [
            self._analyze_memory_bandwidth,
            self._analyze_compute_utilization,
            self._analyze_cache_efficiency,
            self._analyze_occupancy,
            self._analyze_kernel_patterns,
            self._analyze_synchronization,
            self._analyze_configuration,
            self._analyze_regressions,
        ]

    def analyze(self, context: AnalysisContext) -> List[RootCause]:
        """
        Run full root cause analysis.

        Returns list of identified root causes sorted by impact.
        """
        causes: List[RootCause] = []

        # Run all analysis rules
        for rule in self._analysis_rules:
            rule_causes = rule(context)
            causes.extend(rule_causes)

        # Deduplicate and merge related causes
        causes = self._merge_related_causes(causes)

        # Sort by impact
        causes.sort(key=lambda c: c.impact_pct, reverse=True)

        return causes

    def _analyze_memory_bandwidth(self, context: AnalysisContext) -> List[RootCause]:
        """Analyze memory bandwidth issues."""
        causes = []

        for kernel, counters in context.hardware_counters.items():
            # Check memory bandwidth utilization
            mem_util = counters.get("memory_bandwidth_utilization", 0)
            compute_util = counters.get("compute_utilization", 0)

            if mem_util > self.MEMORY_BOUND_THRESHOLD and compute_util < 0.5:
                kernel_time = context.kernel_latencies.get(kernel, 0)
                impact = (
                    (kernel_time / context.total_latency_ms * 100)
                    if context.total_latency_ms > 0
                    else 0
                )

                cause = RootCause(
                    cause_id=f"mem_bound_{kernel}",
                    category=CauseCategory.HARDWARE_MEMORY_BANDWIDTH,
                    title=f"Memory Bandwidth Bottleneck: {kernel}",
                    description=f"Kernel '{kernel}' is memory-bound with {mem_util * 100:.1f}% "
                    f"bandwidth utilization but only {compute_util * 100:.1f}% compute.",
                    impact_pct=impact,
                    confidence=0.9,
                    kernel_name=kernel,
                    suggested_fixes=[
                        "Increase arithmetic intensity through loop fusion",
                        "Use shared memory for data reuse",
                        "Consider reduced precision (FP16/BF16)",
                        "Optimize memory access patterns for coalescing",
                    ],
                    fix_complexity="medium",
                )
                causes.append(cause)

        return causes

    def _analyze_compute_utilization(self, context: AnalysisContext) -> List[RootCause]:
        """Analyze compute utilization issues."""
        causes = []

        for kernel, counters in context.hardware_counters.items():
            compute_util = counters.get("compute_utilization", 0)
            mfma_util = counters.get("mfma_utilization", 0)

            if compute_util > self.COMPUTE_BOUND_THRESHOLD:
                kernel_time = context.kernel_latencies.get(kernel, 0)
                impact = (
                    (kernel_time / context.total_latency_ms * 100)
                    if context.total_latency_ms > 0
                    else 0
                )

                # Check if MFMA could be used
                if mfma_util < 0.1 and "gemm" in kernel.lower():
                    cause = RootCause(
                        cause_id=f"no_mfma_{kernel}",
                        category=CauseCategory.SOFTWARE_KERNEL_INEFFICIENCY,
                        title=f"MFMA Not Utilized: {kernel}",
                        description=f"Matrix kernel '{kernel}' is not using MFMA instructions. "
                        f"MFMA utilization: {mfma_util * 100:.1f}%",
                        impact_pct=impact * 0.5,  # Potential improvement
                        confidence=0.85,
                        kernel_name=kernel,
                        suggested_fixes=[
                            "Use rocBLAS with MFMA-optimized kernels",
                            "Enable matrix instruction generation in compiler",
                            "Ensure matrix dimensions are MFMA-friendly (multiples of 16/32)",
                        ],
                        fix_complexity="low",
                    )
                    causes.append(cause)

        return causes

    def _analyze_cache_efficiency(self, context: AnalysisContext) -> List[RootCause]:
        """Analyze cache efficiency issues."""
        causes = []

        for kernel, counters in context.hardware_counters.items():
            l1_hit_rate = counters.get("l1_hit_rate", 1.0)
            l2_hit_rate = counters.get("l2_hit_rate", 1.0)

            if l1_hit_rate < (1 - self.CACHE_MISS_THRESHOLD):
                kernel_time = context.kernel_latencies.get(kernel, 0)
                impact = (
                    (kernel_time / context.total_latency_ms * 100)
                    if context.total_latency_ms > 0
                    else 0
                )

                cause = RootCause(
                    cause_id=f"cache_miss_{kernel}",
                    category=CauseCategory.HARDWARE_CACHE_MISS,
                    title=f"High Cache Miss Rate: {kernel}",
                    description=f"Kernel '{kernel}' has {(1 - l1_hit_rate) * 100:.1f}% L1 cache miss rate "
                    f"and {(1 - l2_hit_rate) * 100:.1f}% L2 miss rate.",
                    impact_pct=impact * 0.3,
                    confidence=0.8,
                    kernel_name=kernel,
                    suggested_fixes=[
                        "Apply loop tiling to improve cache locality",
                        "Reorder data access patterns",
                        "Use shared memory as explicit cache",
                        "Prefetch data when possible",
                    ],
                    fix_complexity="medium",
                )
                causes.append(cause)

        return causes

    def _analyze_occupancy(self, context: AnalysisContext) -> List[RootCause]:
        """Analyze GPU occupancy issues."""
        causes = []

        for kernel, counters in context.hardware_counters.items():
            occupancy = counters.get("achieved_occupancy", 1.0)

            if occupancy < self.OCCUPANCY_LOW_THRESHOLD:
                kernel_time = context.kernel_latencies.get(kernel, 0)
                impact = (
                    (kernel_time / context.total_latency_ms * 100)
                    if context.total_latency_ms > 0
                    else 0
                )

                cause = RootCause(
                    cause_id=f"low_occupancy_{kernel}",
                    category=CauseCategory.HARDWARE_OCCUPANCY,
                    title=f"Low GPU Occupancy: {kernel}",
                    description=f"Kernel '{kernel}' achieves only {occupancy * 100:.1f}% occupancy, "
                    f"leaving GPU underutilized.",
                    impact_pct=impact * 0.2,
                    confidence=0.75,
                    kernel_name=kernel,
                    suggested_fixes=[
                        "Reduce register usage per thread",
                        "Reduce shared memory per workgroup",
                        "Adjust workgroup size",
                        "Split kernel if resource bound",
                    ],
                    fix_complexity="medium",
                )
                causes.append(cause)

        return causes

    def _analyze_kernel_patterns(self, context: AnalysisContext) -> List[RootCause]:
        """Analyze kernel execution patterns."""
        causes = []

        # Look for small kernels that could be fused
        small_kernels = []
        for kernel, latency in context.kernel_latencies.items():
            if latency < 0.01:  # < 10us
                small_kernels.append(kernel)

        if len(small_kernels) > 5:
            total_small_time = sum(context.kernel_latencies.get(k, 0) for k in small_kernels)
            impact = (
                (total_small_time / context.total_latency_ms * 100)
                if context.total_latency_ms > 0
                else 0
            )

            cause = RootCause(
                cause_id="many_small_kernels",
                category=CauseCategory.SOFTWARE_FUSION_MISSED,
                title="Excessive Small Kernel Launches",
                description=f"Found {len(small_kernels)} kernels with <10µs duration. "
                f"Launch overhead may dominate execution time.",
                impact_pct=impact,
                confidence=0.7,
                suggested_fixes=[
                    "Enable kernel fusion in the compiler",
                    "Use graph execution mode for batching",
                    "Manually fuse elementwise operations",
                ],
                fix_complexity="medium",
            )
            causes.append(cause)

        return causes

    def _analyze_synchronization(self, context: AnalysisContext) -> List[RootCause]:
        """Analyze synchronization overhead."""
        causes = []

        for kernel, counters in context.hardware_counters.items():
            sync_time = counters.get("sync_time_pct", 0)

            if sync_time > 0.1:  # 10% of time in sync
                kernel_time = context.kernel_latencies.get(kernel, 0)
                impact = (
                    (kernel_time * sync_time / context.total_latency_ms * 100)
                    if context.total_latency_ms > 0
                    else 0
                )

                cause = RootCause(
                    cause_id=f"sync_overhead_{kernel}",
                    category=CauseCategory.SOFTWARE_SYNCHRONIZATION,
                    title=f"High Synchronization Overhead: {kernel}",
                    description=f"Kernel '{kernel}' spends {sync_time * 100:.1f}% of time "
                    f"in synchronization barriers.",
                    impact_pct=impact,
                    confidence=0.75,
                    kernel_name=kernel,
                    suggested_fixes=[
                        "Reduce barrier frequency",
                        "Use warp-level primitives where possible",
                        "Restructure algorithm to reduce dependencies",
                    ],
                    fix_complexity="high",
                )
                causes.append(cause)

        return causes

    def _analyze_configuration(self, context: AnalysisContext) -> List[RootCause]:
        """Analyze configuration issues."""
        causes = []

        # Check for suboptimal batch size
        for layer, metrics in context.layer_metrics.items():
            batch_size = metrics.get("batch_size", 0)

            if batch_size == 1:
                impact = (
                    (metrics.get("time_ms", 0) / context.total_latency_ms * 100)
                    if context.total_latency_ms > 0
                    else 0
                )

                cause = RootCause(
                    cause_id=f"batch_1_{layer}",
                    category=CauseCategory.CONFIG_BATCH_SIZE,
                    title=f"Single-Item Batching: {layer}",
                    description=f"Layer '{layer}' is executing with batch_size=1, "
                    f"which typically underutilizes GPU parallelism.",
                    impact_pct=impact * 0.4,
                    confidence=0.6,
                    layer_name=layer,
                    suggested_fixes=[
                        "Use larger batch sizes (8, 16, 32)",
                        "Enable dynamic batching",
                        "Use continuous batching for serving",
                    ],
                    fix_complexity="low",
                )
                causes.append(cause)

        return causes

    def _analyze_regressions(self, context: AnalysisContext) -> List[RootCause]:
        """Analyze regressions vs baseline."""
        causes = []

        for kernel, current_time in context.kernel_latencies.items():
            baseline_time = context.baseline_latencies.get(kernel, 0)

            if baseline_time > 0:
                regression = (current_time - baseline_time) / baseline_time

                if regression > self.REGRESSION_THRESHOLD:
                    impact = (
                        (current_time - baseline_time) / context.total_latency_ms * 100
                        if context.total_latency_ms > 0
                        else 0
                    )

                    cause = RootCause(
                        cause_id=f"regression_{kernel}",
                        category=CauseCategory.SOFTWARE_KERNEL_INEFFICIENCY,
                        title=f"Performance Regression: {kernel}",
                        description=f"Kernel '{kernel}' is {regression * 100:.1f}% slower than baseline "
                        f"({baseline_time * 1000:.2f}µs -> {current_time * 1000:.2f}µs).",
                        impact_pct=impact,
                        confidence=0.9,
                        kernel_name=kernel,
                        suggested_fixes=[
                            "Check for driver/compiler version changes",
                            "Review recent code changes",
                            "Compare hardware counters with baseline",
                        ],
                        fix_complexity="varies",
                    )
                    causes.append(cause)

        return causes

    def _merge_related_causes(self, causes: List[RootCause]) -> List[RootCause]:
        """Merge related causes to avoid duplicates."""
        # Group by kernel
        kernel_causes: Dict[str, List[RootCause]] = {}
        other_causes: List[RootCause] = []

        for cause in causes:
            if cause.kernel_name:
                if cause.kernel_name not in kernel_causes:
                    kernel_causes[cause.kernel_name] = []
                kernel_causes[cause.kernel_name].append(cause)
            else:
                other_causes.append(cause)

        # Link related causes
        for kernel, kernel_cause_list in kernel_causes.items():
            for i, cause in enumerate(kernel_cause_list):
                for j, other_cause in enumerate(kernel_cause_list):
                    if i != j:
                        cause.related_causes.append(other_cause.cause_id)

        # Flatten and return
        result = other_causes
        for cause_list in kernel_causes.values():
            result.extend(cause_list)

        return result
