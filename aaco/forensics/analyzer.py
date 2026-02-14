"""
AACO-SIGMA Forensic Bundle Analyzer

Analyzes forensic bundles for post-mortem debugging and comparison.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto
import time

from .bundle import (
    ForensicBundle,
)


class AnalysisType(Enum):
    """Type of analysis to perform."""

    SUMMARY = auto()  # Basic summary
    COMPARISON = auto()  # Compare two bundles
    REGRESSION = auto()  # Check for regressions
    ANOMALY = auto()  # Detect anomalies
    POST_MORTEM = auto()  # Full post-mortem


@dataclass
class Finding:
    """A finding from analysis."""

    category: str = ""  # "performance", "anomaly", "regression"
    severity: str = "info"  # "info", "warning", "error", "critical"
    title: str = ""
    description: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class AnalysisReport:
    """Report from bundle analysis."""

    bundle_id: str = ""
    analysis_type: AnalysisType = AnalysisType.SUMMARY
    timestamp: float = field(default_factory=time.time)

    # Summary stats
    total_traces: int = 0
    total_kernels: int = 0
    total_events: int = 0

    # Key metrics
    latency_ms: float = 0.0
    throughput: float = 0.0

    # Findings
    findings: List[Finding] = field(default_factory=list)

    # Scores
    health_score: float = 100.0  # 0-100
    performance_score: float = 100.0

    # Raw analysis data
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of comparing two bundles."""

    bundle_a_id: str = ""
    bundle_b_id: str = ""

    # Metric deltas
    latency_delta_pct: float = 0.0
    throughput_delta_pct: float = 0.0
    memory_delta_pct: float = 0.0

    # Kernel-level changes
    kernel_changes: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Significant differences
    regressions: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)

    # Overall assessment
    is_regression: bool = False
    regression_severity: str = "none"  # "none", "minor", "moderate", "severe"


@dataclass
class PostMortemResult:
    """Result of post-mortem analysis."""

    bundle_id: str = ""

    # Timeline reconstruction
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    # Root cause candidates
    root_causes: List[Dict[str, Any]] = field(default_factory=list)

    # Environment factors
    env_factors: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Confidence
    confidence: float = 0.0


class BundleAnalyzer:
    """
    Analyzes forensic bundles for insights and debugging.

    Capabilities:
    - Summary analysis
    - Bundle comparison
    - Regression detection
    - Anomaly detection
    - Post-mortem analysis
    """

    def __init__(self):
        # Thresholds
        self.regression_threshold_pct = 5.0  # 5% slowdown = regression
        self.anomaly_threshold_std = 2.0  # 2 std devs = anomaly

    def analyze(self, bundle: ForensicBundle) -> AnalysisReport:
        """
        Perform summary analysis of a bundle.

        Args:
            bundle: Bundle to analyze

        Returns:
            Analysis report
        """
        report = AnalysisReport(
            bundle_id=bundle.metadata.bundle_id,
            analysis_type=AnalysisType.SUMMARY,
        )

        # Gather stats
        report.total_traces = len(bundle.traces)
        report.total_events = sum(t.event_count for t in bundle.traces)

        # Count unique kernels
        kernels = set()
        if bundle.counters.kernel_counters:
            kernels.update(bundle.counters.kernel_counters.keys())
        if bundle.metrics.kernel_metrics:
            kernels.update(bundle.metrics.kernel_metrics.keys())
        report.total_kernels = len(kernels)

        # Metrics
        report.latency_ms = bundle.metrics.latency_ms
        report.throughput = bundle.metrics.throughput

        # Generate findings
        self._analyze_metrics(bundle, report)
        self._analyze_environment(bundle, report)
        self._analyze_traces(bundle, report)

        # Compute scores
        self._compute_scores(report)

        return report

    def _analyze_metrics(
        self,
        bundle: ForensicBundle,
        report: AnalysisReport,
    ) -> None:
        """Analyze metrics for issues."""
        metrics = bundle.metrics

        # Check for very high latency
        if metrics.latency_ms > 1000:
            report.findings.append(
                Finding(
                    category="performance",
                    severity="warning",
                    title="High latency detected",
                    description=f"Latency of {metrics.latency_ms:.1f}ms is unusually high",
                    evidence={"latency_ms": metrics.latency_ms},
                    recommendation="Profile individual kernels to identify bottlenecks",
                )
            )

        # Check memory usage
        if metrics.memory_peak_mb > 0:
            # High memory relative to typical workloads
            if metrics.memory_peak_mb > 16000:  # 16 GB
                report.findings.append(
                    Finding(
                        category="resource",
                        severity="info",
                        title="High memory usage",
                        description=f"Peak memory: {metrics.memory_peak_mb:.0f} MB",
                        evidence={"memory_peak_mb": metrics.memory_peak_mb},
                    )
                )

        # Check time breakdown
        if metrics.compute_time_ms > 0 and metrics.memory_time_ms > 0:
            total = metrics.compute_time_ms + metrics.memory_time_ms
            mem_pct = metrics.memory_time_ms / total * 100

            if mem_pct > 80:
                report.findings.append(
                    Finding(
                        category="performance",
                        severity="info",
                        title="Memory-bound workload",
                        description=f"{mem_pct:.0f}% of time spent on memory operations",
                        evidence={"memory_pct": mem_pct},
                        recommendation="Consider memory optimizations like fusion or tiling",
                    )
                )

    def _analyze_environment(
        self,
        bundle: ForensicBundle,
        report: AnalysisReport,
    ) -> None:
        """Analyze environment for potential issues."""
        env = bundle.environment

        # Check ROCm version
        if env.rocm_version:
            major = int(env.rocm_version.split(".")[0])
            if major < 6:
                report.findings.append(
                    Finding(
                        category="environment",
                        severity="info",
                        title="Older ROCm version",
                        description=f"ROCm {env.rocm_version} detected, consider upgrading",
                        evidence={"rocm_version": env.rocm_version},
                    )
                )

        # Check GPU count
        if env.gpu_count > 1:
            report.findings.append(
                Finding(
                    category="environment",
                    severity="info",
                    title="Multi-GPU system",
                    description=f"{env.gpu_count} GPUs detected",
                    evidence={"gpu_count": env.gpu_count},
                )
            )

    def _analyze_traces(
        self,
        bundle: ForensicBundle,
        report: AnalysisReport,
    ) -> None:
        """Analyze trace data for patterns."""
        if not bundle.traces:
            return

        total_events = sum(t.event_count for t in bundle.traces)

        if total_events == 0:
            report.findings.append(
                Finding(
                    category="data",
                    severity="warning",
                    title="Empty trace data",
                    description="No trace events captured",
                    recommendation="Verify profiler configuration",
                )
            )

    def _compute_scores(self, report: AnalysisReport) -> None:
        """Compute health and performance scores."""
        # Health score based on findings
        deductions = {
            "critical": 30,
            "error": 20,
            "warning": 10,
            "info": 2,
        }

        for finding in report.findings:
            severity = finding.severity
            report.health_score -= deductions.get(severity, 0)

        report.health_score = max(0, report.health_score)

        # Performance score - placeholder heuristic
        report.performance_score = report.health_score

    def compare(
        self,
        bundle_a: ForensicBundle,
        bundle_b: ForensicBundle,
    ) -> ComparisonResult:
        """
        Compare two bundles.

        Args:
            bundle_a: First bundle (baseline)
            bundle_b: Second bundle (current)

        Returns:
            Comparison result
        """
        result = ComparisonResult(
            bundle_a_id=bundle_a.metadata.bundle_id,
            bundle_b_id=bundle_b.metadata.bundle_id,
        )

        # Compute deltas
        if bundle_a.metrics.latency_ms > 0:
            result.latency_delta_pct = (
                (bundle_b.metrics.latency_ms - bundle_a.metrics.latency_ms)
                / bundle_a.metrics.latency_ms
                * 100
            )

        if bundle_a.metrics.throughput > 0:
            result.throughput_delta_pct = (
                (bundle_b.metrics.throughput - bundle_a.metrics.throughput)
                / bundle_a.metrics.throughput
                * 100
            )

        if bundle_a.metrics.memory_peak_mb > 0:
            result.memory_delta_pct = (
                (bundle_b.metrics.memory_peak_mb - bundle_a.metrics.memory_peak_mb)
                / bundle_a.metrics.memory_peak_mb
                * 100
            )

        # Compare kernel-level metrics
        kernels_a = set(bundle_a.metrics.kernel_metrics.keys())
        kernels_b = set(bundle_b.metrics.kernel_metrics.keys())
        common_kernels = kernels_a & kernels_b

        for kernel in common_kernels:
            metrics_a = bundle_a.metrics.kernel_metrics[kernel]
            metrics_b = bundle_b.metrics.kernel_metrics[kernel]

            changes = {}
            for metric_name in set(metrics_a.keys()) & set(metrics_b.keys()):
                if metrics_a[metric_name] > 0:
                    delta_pct = (
                        (metrics_b[metric_name] - metrics_a[metric_name])
                        / metrics_a[metric_name]
                        * 100
                    )
                    changes[metric_name] = delta_pct

            if changes:
                result.kernel_changes[kernel] = changes

        # Identify regressions and improvements
        if result.latency_delta_pct > self.regression_threshold_pct:
            result.regressions.append(f"Latency increased by {result.latency_delta_pct:.1f}%")
        elif result.latency_delta_pct < -self.regression_threshold_pct:
            result.improvements.append(f"Latency decreased by {abs(result.latency_delta_pct):.1f}%")

        if result.throughput_delta_pct < -self.regression_threshold_pct:
            result.regressions.append(
                f"Throughput decreased by {abs(result.throughput_delta_pct):.1f}%"
            )
        elif result.throughput_delta_pct > self.regression_threshold_pct:
            result.improvements.append(
                f"Throughput increased by {result.throughput_delta_pct:.1f}%"
            )

        # Overall assessment
        result.is_regression = len(result.regressions) > 0

        if result.is_regression:
            max_regression = max(
                result.latency_delta_pct,
                -result.throughput_delta_pct,
            )
            if max_regression > 20:
                result.regression_severity = "severe"
            elif max_regression > 10:
                result.regression_severity = "moderate"
            else:
                result.regression_severity = "minor"

        return result

    def post_mortem(
        self,
        bundle: ForensicBundle,
        baseline: Optional[ForensicBundle] = None,
    ) -> PostMortemResult:
        """
        Perform full post-mortem analysis.

        Args:
            bundle: Bundle to analyze
            baseline: Optional baseline for comparison

        Returns:
            Post-mortem result
        """
        result = PostMortemResult(bundle_id=bundle.metadata.bundle_id)

        # Reconstruct timeline from traces
        result.timeline = self._reconstruct_timeline(bundle)

        # Identify root cause candidates
        result.root_causes = self._identify_root_causes(bundle, baseline)

        # Check environment factors
        result.env_factors = self._check_env_factors(bundle)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(
            bundle, result.root_causes, result.env_factors
        )

        # Compute confidence
        result.confidence = self._compute_confidence(result)

        return result

    def _reconstruct_timeline(
        self,
        bundle: ForensicBundle,
    ) -> List[Dict[str, Any]]:
        """Reconstruct execution timeline from traces."""
        timeline = []

        # Add bundle creation
        timeline.append(
            {
                "timestamp": bundle.metadata.created_at,
                "event": "bundle_created",
                "description": "Forensic capture started",
            }
        )

        # Process trace events chronologically
        all_events = []
        for trace in bundle.traces:
            for event in trace.events:
                if isinstance(event, dict) and "ts" in event:
                    all_events.append(event)

        # Sort by timestamp
        all_events.sort(key=lambda e: e.get("ts", 0))

        # Add significant events to timeline
        for event in all_events[:50]:  # Limit to first 50
            timeline.append(
                {
                    "timestamp": event.get("ts", 0),
                    "event": event.get("name", "unknown"),
                    "duration": event.get("dur", 0),
                }
            )

        return timeline

    def _identify_root_causes(
        self,
        bundle: ForensicBundle,
        baseline: Optional[ForensicBundle],
    ) -> List[Dict[str, Any]]:
        """Identify potential root causes."""
        causes = []

        # Memory-bound analysis
        if bundle.metrics.memory_time_ms > bundle.metrics.compute_time_ms:
            causes.append(
                {
                    "category": "memory",
                    "description": "Memory-bound execution",
                    "confidence": 0.8,
                    "evidence": {
                        "memory_time_ms": bundle.metrics.memory_time_ms,
                        "compute_time_ms": bundle.metrics.compute_time_ms,
                    },
                }
            )

        # Compare with baseline if provided
        if baseline:
            comparison = self.compare(baseline, bundle)

            if comparison.latency_delta_pct > 10:
                causes.append(
                    {
                        "category": "regression",
                        "description": f"Latency regression of {comparison.latency_delta_pct:.1f}%",
                        "confidence": 0.9,
                        "evidence": {"latency_delta_pct": comparison.latency_delta_pct},
                    }
                )

            # Check kernel-specific regressions
            for kernel, changes in comparison.kernel_changes.items():
                for metric, delta in changes.items():
                    if delta > 20:  # 20% regression
                        causes.append(
                            {
                                "category": "kernel_regression",
                                "description": f"Kernel {kernel} {metric} regressed {delta:.1f}%",
                                "confidence": 0.85,
                                "evidence": {
                                    "kernel": kernel,
                                    "metric": metric,
                                    "delta": delta,
                                },
                            }
                        )

        # Sort by confidence
        causes.sort(key=lambda c: c.get("confidence", 0), reverse=True)

        return causes

    def _check_env_factors(self, bundle: ForensicBundle) -> List[str]:
        """Check for environmental factors."""
        factors = []

        env = bundle.environment

        # Check for known problematic configurations
        if env.rocm_version:
            version_parts = env.rocm_version.split(".")
            if len(version_parts) >= 2:
                major, minor = int(version_parts[0]), int(version_parts[1])
                if (major, minor) == (5, 4):
                    factors.append("ROCm 5.4 has known performance issues with some kernels")

        # Multi-GPU considerations
        if env.gpu_count > 1:
            factors.append("Multi-GPU setup may have communication overhead")

        return factors

    def _generate_recommendations(
        self,
        bundle: ForensicBundle,
        root_causes: List[Dict[str, Any]],
        env_factors: List[str],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        for cause in root_causes:
            category = cause.get("category", "")

            if category == "memory":
                recommendations.append(
                    "Consider memory optimizations: kernel fusion, "
                    "tiling, or async memory transfers"
                )
            elif category == "regression":
                recommendations.append(
                    "Investigate recent changes in code, dependencies, or system configuration"
                )
            elif category == "kernel_regression":
                kernel = cause.get("evidence", {}).get("kernel", "unknown")
                recommendations.append(
                    f"Profile kernel {kernel} in detail to identify the source of regression"
                )

        for factor in env_factors:
            if "ROCm" in factor:
                recommendations.append("Consider upgrading to latest ROCm version")
            if "Multi-GPU" in factor:
                recommendations.append("Profile inter-GPU communication to check for bottlenecks")

        # Deduplicate
        return list(dict.fromkeys(recommendations))

    def _compute_confidence(self, result: PostMortemResult) -> float:
        """Compute overall confidence in analysis."""
        if not result.root_causes:
            return 0.0

        # Average confidence of root causes
        confidences = [c.get("confidence", 0.5) for c in result.root_causes]
        avg_confidence = sum(confidences) / len(confidences)

        # Boost if we have timeline
        if len(result.timeline) > 5:
            avg_confidence = min(1.0, avg_confidence + 0.1)

        return avg_confidence
