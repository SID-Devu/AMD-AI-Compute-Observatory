"""
Recommendation Engine
Evidence-based performance optimization suggestions.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .feature_store import SessionFeatures, IterationFeatures

logger = logging.getLogger(__name__)


# ============================================================================
# Bottleneck Classification
# ============================================================================

class BottleneckClass(str, Enum):
    """Classification of performance bottlenecks."""
    LAUNCH_OVERHEAD = "launch_overhead"  # High kernel launch tax
    MEMORY_BANDWIDTH = "memory_bandwidth"  # Memory-bound workload
    COMPUTE_BOUND = "compute_bound"  # Compute-limited
    THERMAL_THROTTLE = "thermal_throttle"  # Thermal throttling
    CLOCK_INSTABILITY = "clock_instability"  # Unstable GPU clocks
    OS_INTERFERENCE = "os_interference"  # Context switches, scheduling
    DATA_TRANSFER = "data_transfer"  # Host-device transfer overhead
    SYNCHRONIZATION = "synchronization"  # Excessive sync points
    FRAMEWORK_OVERHEAD = "framework_overhead"  # Inference framework overhead
    UNKNOWN = "unknown"


class RecommendationPriority(str, Enum):
    """Priority level for recommendations."""
    CRITICAL = "critical"  # Must address immediately
    HIGH = "high"  # Should address soon
    MEDIUM = "medium"  # Worth investigating
    LOW = "low"  # Nice to have


# ============================================================================
# Evidence and Recommendations
# ============================================================================

@dataclass
class Evidence:
    """Evidence supporting a bottleneck diagnosis."""
    metric: str
    observed_value: float
    threshold: float
    description: str
    weight: float = 1.0  # Contribution to bottleneck confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "observed_value": self.observed_value,
            "threshold": self.threshold,
            "description": self.description,
            "weight": self.weight,
        }


@dataclass
class Recommendation:
    """Actionable performance recommendation."""
    title: str
    description: str
    bottleneck: BottleneckClass
    priority: RecommendationPriority
    confidence: float  # [0-1] confidence in recommendation
    
    # Supporting evidence
    evidence: List[Evidence] = field(default_factory=list)
    
    # Actionable steps
    actions: List[str] = field(default_factory=list)
    
    # Expected impact
    expected_improvement: str = ""
    effort_estimate: str = ""  # "low", "medium", "high"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "bottleneck": self.bottleneck.value,
            "priority": self.priority.value,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
            "actions": self.actions,
            "expected_improvement": self.expected_improvement,
            "effort_estimate": self.effort_estimate,
        }


@dataclass
class BottleneckAnalysis:
    """Complete bottleneck analysis for a session."""
    session_id: str
    primary_bottleneck: Optional[BottleneckClass] = None
    secondary_bottlenecks: List[BottleneckClass] = field(default_factory=list)
    
    # Bottleneck confidence scores
    bottleneck_scores: Dict[str, float] = field(default_factory=dict)
    
    # All recommendations, sorted by priority
    recommendations: List[Recommendation] = field(default_factory=list)
    
    # Top 5 drivers of performance issues
    top_drivers: List[Tuple[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "primary_bottleneck": self.primary_bottleneck.value if self.primary_bottleneck else None,
            "secondary_bottlenecks": [b.value for b in self.secondary_bottlenecks],
            "bottleneck_scores": self.bottleneck_scores,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "top_drivers": self.top_drivers,
        }


# ============================================================================
# Bottleneck Detector
# ============================================================================

class BottleneckDetector:
    """
    Detects performance bottlenecks from session features.
    
    Uses rule-based analysis with configurable thresholds to identify
    the primary limiting factor in model performance.
    """
    
    # Thresholds for bottleneck detection
    THRESHOLDS = {
        # Launch overhead
        "kar_low": 0.7,  # KAR below this indicates launch overhead
        "launch_tax_high": 0.3,  # Launch tax above this is significant
        
        # Thermal
        "temp_warning": 80,  # Celsius
        "temp_critical": 90,
        
        # Clock stability
        "clock_cv_warning": 5,  # CV% above this indicates instability
        "clock_cv_critical": 10,
        
        # OS interference
        "ctx_switches_high": 100,  # Per iteration
        "page_faults_high": 10,
        
        # Variability
        "cv_warning": 10,  # CV% for latency
        "cv_critical": 20,
        "spike_ratio_warning": 0.05,  # 5% spikes
        "spike_ratio_critical": 0.10,
        
        # GPU utilization
        "gpu_util_low": 50,  # Below this is underutilized
        "gpu_util_high": 90,  # Above this is well utilized
    }
    
    def detect(self, features: SessionFeatures) -> Dict[BottleneckClass, float]:
        """
        Detect bottlenecks and return confidence scores.
        
        Returns:
            Dict mapping bottleneck class to confidence score [0-1]
        """
        scores = {b: 0.0 for b in BottleneckClass}
        
        # Launch overhead detection
        if features.kar > 0:
            if features.kar < self.THRESHOLDS["kar_low"]:
                scores[BottleneckClass.LAUNCH_OVERHEAD] = 1.0 - features.kar
            if features.launch_tax_score > self.THRESHOLDS["launch_tax_high"]:
                scores[BottleneckClass.LAUNCH_OVERHEAD] = max(
                    scores[BottleneckClass.LAUNCH_OVERHEAD],
                    features.launch_tax_score
                )
        
        # Thermal throttling detection
        if features.temp_max_c >= self.THRESHOLDS["temp_critical"]:
            scores[BottleneckClass.THERMAL_THROTTLE] = 0.9
        elif features.temp_max_c >= self.THRESHOLDS["temp_warning"]:
            temp_severity = (features.temp_max_c - self.THRESHOLDS["temp_warning"]) / 10
            scores[BottleneckClass.THERMAL_THROTTLE] = min(0.7, temp_severity)
        
        # Clock instability detection
        if features.clock_std_mhz > 0 and features.clock_mean_mhz > 0:
            clock_cv = (features.clock_std_mhz / features.clock_mean_mhz) * 100
            if clock_cv >= self.THRESHOLDS["clock_cv_critical"]:
                scores[BottleneckClass.CLOCK_INSTABILITY] = 0.8
            elif clock_cv >= self.THRESHOLDS["clock_cv_warning"]:
                scores[BottleneckClass.CLOCK_INSTABILITY] = 0.5
        
        # OS interference detection
        if features.ctx_switches_per_iter >= self.THRESHOLDS["ctx_switches_high"]:
            scores[BottleneckClass.OS_INTERFERENCE] = min(1.0, 
                features.ctx_switches_per_iter / (self.THRESHOLDS["ctx_switches_high"] * 2))
        if features.page_faults_per_iter >= self.THRESHOLDS["page_faults_high"]:
            scores[BottleneckClass.OS_INTERFERENCE] = max(
                scores[BottleneckClass.OS_INTERFERENCE],
                min(1.0, features.page_faults_per_iter / (self.THRESHOLDS["page_faults_high"] * 2))
            )
        
        # Framework overhead detection (high CV + low KAR)
        if features.cv_pct >= self.THRESHOLDS["cv_warning"] and features.kar < self.THRESHOLDS["kar_low"]:
            scores[BottleneckClass.FRAMEWORK_OVERHEAD] = 0.6
        
        # Compute vs memory bound (simplified heuristic)
        if features.gpu_util_mean_pct >= self.THRESHOLDS["gpu_util_high"]:
            scores[BottleneckClass.COMPUTE_BOUND] = 0.7
        elif features.gpu_util_mean_pct < self.THRESHOLDS["gpu_util_low"] and features.kar > 0.8:
            scores[BottleneckClass.MEMORY_BANDWIDTH] = 0.6
        
        return scores


# ============================================================================
# Recommendation Engine
# ============================================================================

class RecommendationEngine:
    """
    Generates evidence-based optimization recommendations.
    
    Analyzes session features to identify bottlenecks and provide
    actionable suggestions with supporting evidence.
    """
    
    def __init__(self):
        self.detector = BottleneckDetector()
    
    def analyze(self, features: SessionFeatures,
                iterations: Optional[List[IterationFeatures]] = None) -> BottleneckAnalysis:
        """
        Perform complete bottleneck analysis and generate recommendations.
        
        Args:
            features: Session-level features
            iterations: Optional iteration-level features for detailed analysis
            
        Returns:
            BottleneckAnalysis with recommendations
        """
        analysis = BottleneckAnalysis(session_id=features.session_id)
        
        # Detect bottlenecks
        bottleneck_scores = self.detector.detect(features)
        analysis.bottleneck_scores = {b.value: s for b, s in bottleneck_scores.items()}
        
        # Sort bottlenecks by score
        sorted_bottlenecks = sorted(
            [(b, s) for b, s in bottleneck_scores.items() if s > 0.1],
            key=lambda x: x[1],
            reverse=True
        )
        
        if sorted_bottlenecks:
            analysis.primary_bottleneck = sorted_bottlenecks[0][0]
            analysis.secondary_bottlenecks = [b for b, _ in sorted_bottlenecks[1:4]]
        
        # Generate recommendations
        analysis.recommendations = self._generate_recommendations(features, sorted_bottlenecks)
        
        # Compute top drivers
        analysis.top_drivers = self._compute_top_drivers(features)
        
        return analysis
    
    def _generate_recommendations(
        self,
        features: SessionFeatures,
        bottlenecks: List[Tuple[BottleneckClass, float]]
    ) -> List[Recommendation]:
        """Generate recommendations based on detected bottlenecks."""
        recommendations = []
        
        for bottleneck, score in bottlenecks:
            if score < 0.1:
                continue
            
            rec = self._create_recommendation(bottleneck, features, score)
            if rec:
                recommendations.append(rec)
        
        # Add general recommendations if no specific bottlenecks found
        if not recommendations:
            recommendations.append(self._create_baseline_recommendation(features))
        
        # Sort by priority and confidence
        priority_order = [RecommendationPriority.CRITICAL, RecommendationPriority.HIGH,
                        RecommendationPriority.MEDIUM, RecommendationPriority.LOW]
        recommendations.sort(
            key=lambda r: (priority_order.index(r.priority), -r.confidence)
        )
        
        return recommendations
    
    def _create_recommendation(
        self,
        bottleneck: BottleneckClass,
        features: SessionFeatures,
        score: float
    ) -> Optional[Recommendation]:
        """Create recommendation for a specific bottleneck."""
        
        if bottleneck == BottleneckClass.LAUNCH_OVERHEAD:
            return Recommendation(
                title="Reduce Kernel Launch Overhead",
                description="High kernel launch tax is limiting GPU utilization. "
                           "The GPU spends significant time between kernels.",
                bottleneck=bottleneck,
                priority=RecommendationPriority.HIGH if score > 0.5 else RecommendationPriority.MEDIUM,
                confidence=score,
                evidence=[
                    Evidence(
                        metric="kar",
                        observed_value=features.kar,
                        threshold=0.7,
                        description=f"Kernel Active Ratio is {features.kar:.2f} (target: >0.7)"
                    ),
                    Evidence(
                        metric="launch_tax_score",
                        observed_value=features.launch_tax_score,
                        threshold=0.3,
                        description=f"Launch tax score is {features.launch_tax_score:.2f}"
                    ),
                ],
                actions=[
                    "Consider using CUDA graphs / HIP graphs for static workloads",
                    "Batch multiple operations into larger kernels",
                    "Enable kernel fusion in the framework if available",
                    "Use async operations to overlap compute and launch",
                ],
                expected_improvement="10-30% latency reduction",
                effort_estimate="medium",
            )
        
        elif bottleneck == BottleneckClass.THERMAL_THROTTLE:
            priority = RecommendationPriority.CRITICAL if features.temp_max_c >= 90 else RecommendationPriority.HIGH
            return Recommendation(
                title="Address Thermal Throttling",
                description="GPU temperature is causing clock throttling, reducing performance.",
                bottleneck=bottleneck,
                priority=priority,
                confidence=score,
                evidence=[
                    Evidence(
                        metric="temp_max_c",
                        observed_value=features.temp_max_c,
                        threshold=80,
                        description=f"Max GPU temperature reached {features.temp_max_c}°C"
                    ),
                ],
                actions=[
                    "Improve case airflow and ensure fans are working",
                    "Clean dust from heatsinks and fans",
                    "Consider undervolting the GPU",
                    "Add rest periods between inference batches",
                    "Check thermal paste condition if temperatures are extreme",
                ],
                expected_improvement="5-20% sustained performance improvement",
                effort_estimate="low to medium",
            )
        
        elif bottleneck == BottleneckClass.CLOCK_INSTABILITY:
            return Recommendation(
                title="Stabilize GPU Clocks",
                description="GPU clock frequency is unstable, causing performance variability.",
                bottleneck=bottleneck,
                priority=RecommendationPriority.MEDIUM,
                confidence=score,
                evidence=[
                    Evidence(
                        metric="clock_stability_score",
                        observed_value=features.clock_stability_score,
                        threshold=0.9,
                        description=f"Clock stability score is {features.clock_stability_score:.2f} (target: >0.9)"
                    ),
                    Evidence(
                        metric="clock_std_mhz",
                        observed_value=features.clock_std_mhz,
                        threshold=50,
                        description=f"Clock standard deviation is {features.clock_std_mhz:.0f} MHz"
                    ),
                ],
                actions=[
                    "Lock GPU clocks to a stable frequency: rocm-smi --setperflevel high",
                    "Ensure adequate power delivery (check PSU capacity)",
                    "Address thermal issues if present",
                    "Consider using rocm-smi to set fixed clock speeds",
                ],
                expected_improvement="5-15% reduction in latency variability",
                effort_estimate="low",
            )
        
        elif bottleneck == BottleneckClass.OS_INTERFERENCE:
            return Recommendation(
                title="Reduce OS Scheduling Interference",
                description="High context switches and system calls are interrupting inference.",
                bottleneck=bottleneck,
                priority=RecommendationPriority.MEDIUM,
                confidence=score,
                evidence=[
                    Evidence(
                        metric="ctx_switches_per_iter",
                        observed_value=features.ctx_switches_per_iter,
                        threshold=100,
                        description=f"Context switches per iteration: {features.ctx_switches_per_iter:.1f}"
                    ),
                ],
                actions=[
                    "Pin inference process to specific CPU cores (taskset)",
                    "Reduce background process activity",
                    "Consider using real-time scheduling (SCHED_FIFO)",
                    "Disable CPU power management (set governor to 'performance')",
                    "Close unnecessary applications during benchmarking",
                ],
                expected_improvement="5-10% reduction in tail latency",
                effort_estimate="low",
            )
        
        elif bottleneck == BottleneckClass.FRAMEWORK_OVERHEAD:
            return Recommendation(
                title="Optimize Framework Configuration",
                description="Inference framework overhead is contributing to latency.",
                bottleneck=bottleneck,
                priority=RecommendationPriority.MEDIUM,
                confidence=score,
                evidence=[
                    Evidence(
                        metric="cv_pct",
                        observed_value=features.cv_pct,
                        threshold=10,
                        description=f"Latency variability (CV) is {features.cv_pct:.1f}%"
                    ),
                    Evidence(
                        metric="kar",
                        observed_value=features.kar,
                        threshold=0.7,
                        description=f"Kernel Active Ratio is {features.kar:.2f}"
                    ),
                ],
                actions=[
                    "Enable framework-level optimizations (graph optimization)",
                    "Pre-allocate memory and disable dynamic allocation",
                    "Use optimized execution providers (MIGraphX, TensorRT)",
                    "Profile framework overhead with detailed tracing",
                ],
                expected_improvement="10-25% latency reduction",
                effort_estimate="medium",
            )
        
        elif bottleneck == BottleneckClass.COMPUTE_BOUND:
            return Recommendation(
                title="Optimize Compute-Bound Workload",
                description="GPU compute is the primary bottleneck. Consider model optimization.",
                bottleneck=bottleneck,
                priority=RecommendationPriority.LOW,
                confidence=score,
                evidence=[
                    Evidence(
                        metric="gpu_util_mean_pct",
                        observed_value=features.gpu_util_mean_pct,
                        threshold=90,
                        description=f"GPU utilization is {features.gpu_util_mean_pct:.1f}%"
                    ),
                ],
                actions=[
                    "Use quantization (INT8, FP16) to reduce compute requirements",
                    "Consider model distillation for smaller, faster models",
                    "Enable operator fusion in the framework",
                    "Upgrade to a GPU with more compute units",
                ],
                expected_improvement="20-50% with quantization",
                effort_estimate="high",
            )
        
        elif bottleneck == BottleneckClass.MEMORY_BANDWIDTH:
            return Recommendation(
                title="Address Memory Bandwidth Limitation",
                description="Workload appears memory-bound despite low GPU utilization.",
                bottleneck=bottleneck,
                priority=RecommendationPriority.MEDIUM,
                confidence=score,
                evidence=[
                    Evidence(
                        metric="gpu_util_mean_pct",
                        observed_value=features.gpu_util_mean_pct,
                        threshold=50,
                        description=f"GPU utilization is only {features.gpu_util_mean_pct:.1f}%"
                    ),
                ],
                actions=[
                    "Use larger batch sizes to improve memory efficiency",
                    "Enable memory layout optimizations (NCHW vs NHWC)",
                    "Consider memory-efficient attention implementations",
                    "Profile memory access patterns with rocprof",
                ],
                expected_improvement="10-30% with optimization",
                effort_estimate="medium",
            )
        
        return None
    
    def _create_baseline_recommendation(self, features: SessionFeatures) -> Recommendation:
        """Create general recommendation when no specific bottleneck is detected."""
        return Recommendation(
            title="Performance Baseline Established",
            description="No significant bottlenecks detected. Performance is within expected parameters.",
            bottleneck=BottleneckClass.UNKNOWN,
            priority=RecommendationPriority.LOW,
            confidence=0.5,
            evidence=[
                Evidence(
                    metric="chi_score",
                    observed_value=features.chi_score,
                    threshold=70,
                    description=f"Compute Health Index is {features.chi_score:.1f}/100"
                ),
            ],
            actions=[
                "Continue monitoring for regressions",
                "Consider profiling at higher load levels",
                "Document current configuration as reference baseline",
            ],
            expected_improvement="N/A",
            effort_estimate="low",
        )
    
    def _compute_top_drivers(self, features: SessionFeatures) -> List[Tuple[str, float]]:
        """
        Compute top 5 factors contributing to performance issues.
        
        Returns list of (factor_name, impact_score) tuples.
        """
        drivers = []
        
        # Launch overhead impact
        if features.launch_tax_score > 0.1:
            drivers.append(("kernel_launch_overhead", features.launch_tax_score))
        
        # Latency variability impact
        if features.cv_pct > 5:
            drivers.append(("latency_variability", min(1.0, features.cv_pct / 30)))
        
        # Spike impact
        if features.spike_ratio > 0.01:
            drivers.append(("latency_spikes", min(1.0, features.spike_ratio * 10)))
        
        # Thermal impact
        if features.temp_max_c > 70:
            thermal_impact = (features.temp_max_c - 70) / 30
            drivers.append(("thermal_pressure", min(1.0, thermal_impact)))
        
        # Clock instability impact
        if features.clock_stability_score < 0.95:
            drivers.append(("clock_instability", 1.0 - features.clock_stability_score))
        
        # OS interference impact
        if features.ctx_switches_per_iter > 10:
            sched_impact = min(1.0, features.ctx_switches_per_iter / 200)
            drivers.append(("os_scheduling", sched_impact))
        
        # GPU underutilization
        if features.gpu_util_mean_pct < 80 and features.gpu_util_mean_pct > 0:
            underutil = (80 - features.gpu_util_mean_pct) / 80
            drivers.append(("gpu_underutilization", underutil))
        
        # Sort by impact and return top 5
        drivers.sort(key=lambda x: x[1], reverse=True)
        return drivers[:5]
    
    def get_summary(self, analysis: BottleneckAnalysis) -> str:
        """Generate human-readable summary of analysis."""
        if not analysis.primary_bottleneck:
            return "No significant bottlenecks detected. System is operating efficiently."
        
        primary = analysis.primary_bottleneck.value.replace("_", " ")
        
        summary_parts = [f"Primary bottleneck: {primary}"]
        
        if analysis.secondary_bottlenecks:
            secondary = ", ".join(b.value.replace("_", " ") for b in analysis.secondary_bottlenecks[:2])
            summary_parts.append(f"Secondary factors: {secondary}")
        
        if analysis.recommendations:
            top_rec = analysis.recommendations[0]
            summary_parts.append(f"Top recommendation: {top_rec.title}")
        
        return ". ".join(summary_parts) + "."
