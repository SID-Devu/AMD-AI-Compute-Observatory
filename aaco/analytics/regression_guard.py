"""
Statistical Regression Guard
Baseline modeling, regression detection, and confidence scoring.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .feature_store import SessionFeatures, FeatureStore

logger = logging.getLogger(__name__)


# ============================================================================
# Regression Severity
# ============================================================================

class RegressionSeverity(str, Enum):
    """Severity levels for detected regressions."""
    NONE = "none"  # No regression
    MILD = "mild"  # Minor degradation, likely within noise
    MODERATE = "moderate"  # Notable degradation, investigate
    SEVERE = "severe"  # Major regression, action required
    CRITICAL = "critical"  # Critical regression, blocking


class MetricDirection(str, Enum):
    """Direction indicating improvement vs regression."""
    LOWER_IS_BETTER = "lower"  # Latency, temperature
    HIGHER_IS_BETTER = "higher"  # Throughput, KAR, CHI


# ============================================================================
# Regression Result
# ============================================================================

@dataclass
class RegressionResult:
    """Result of regression analysis for a single metric."""
    metric_name: str
    baseline_value: float
    current_value: float
    
    # Statistical measures
    z_score: float = 0.0
    percent_change: float = 0.0
    mad_score: float = 0.0  # Median Absolute Deviation score
    
    # Detection results
    is_regression: bool = False
    severity: RegressionSeverity = RegressionSeverity.NONE
    confidence: float = 0.0  # [0-1] confidence in detection
    
    # Context
    direction: MetricDirection = MetricDirection.LOWER_IS_BETTER
    threshold_used: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline_value": self.baseline_value,
            "current_value": self.current_value,
            "z_score": self.z_score,
            "percent_change": self.percent_change,
            "is_regression": self.is_regression,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "direction": self.direction.value,
        }


@dataclass
class RegressionReport:
    """Complete regression analysis report."""
    session_id: str
    model_name: str
    baseline_session_count: int
    
    # Overall verdict
    has_regression: bool = False
    overall_severity: RegressionSeverity = RegressionSeverity.NONE
    overall_confidence: float = 0.0
    
    # Per-metric results
    metrics: List[RegressionResult] = field(default_factory=list)
    
    # Top regressions
    top_regressions: List[RegressionResult] = field(default_factory=list)
    
    # Explanations
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model_name": self.model_name,
            "baseline_session_count": self.baseline_session_count,
            "has_regression": self.has_regression,
            "overall_severity": self.overall_severity.value,
            "overall_confidence": self.overall_confidence,
            "metrics": [m.to_dict() for m in self.metrics],
            "top_regressions": [m.to_dict() for m in self.top_regressions],
            "summary": self.summary,
            "recommendations": self.recommendations,
        }


# ============================================================================
# Baseline Model
# ============================================================================

@dataclass
class BaselineModel:
    """
    Statistical model of baseline performance.
    
    Built from historical sessions to enable regression detection.
    """
    model_name: str
    session_count: int
    
    # Per-metric statistics
    metric_means: Dict[str, float] = field(default_factory=dict)
    metric_stds: Dict[str, float] = field(default_factory=dict)
    metric_medians: Dict[str, float] = field(default_factory=dict)
    metric_mads: Dict[str, float] = field(default_factory=dict)  # Median Absolute Deviation
    
    # Direction indicators
    metric_directions: Dict[str, MetricDirection] = field(default_factory=dict)
    
    @classmethod
    def build_from_sessions(cls, sessions: List[SessionFeatures]) -> "BaselineModel":
        """Build baseline model from historical sessions."""
        if not sessions:
            return cls(model_name="", session_count=0)
        
        model = cls(
            model_name=sessions[0].model_name,
            session_count=len(sessions),
        )
        
        # Define which metrics to track and their directions
        metric_configs = {
            # Latency metrics (lower is better)
            "latency_mean_ms": MetricDirection.LOWER_IS_BETTER,
            "latency_std_ms": MetricDirection.LOWER_IS_BETTER,
            "latency_p50_ms": MetricDirection.LOWER_IS_BETTER,
            "latency_p90_ms": MetricDirection.LOWER_IS_BETTER,
            "latency_p95_ms": MetricDirection.LOWER_IS_BETTER,
            "latency_p99_ms": MetricDirection.LOWER_IS_BETTER,
            
            # Throughput (higher is better)
            "throughput_its": MetricDirection.HIGHER_IS_BETTER,
            
            # Stability (lower is better)
            "cv_pct": MetricDirection.LOWER_IS_BETTER,
            "spike_count": MetricDirection.LOWER_IS_BETTER,
            "spike_ratio": MetricDirection.LOWER_IS_BETTER,
            "noise_score": MetricDirection.LOWER_IS_BETTER,
            
            # Efficiency (higher is better)
            "kar": MetricDirection.HIGHER_IS_BETTER,
            "launch_tax_score": MetricDirection.LOWER_IS_BETTER,
            "gpu_util_mean_pct": MetricDirection.HIGHER_IS_BETTER,
            
            # Power/Thermal (lower is better)
            "power_mean_w": MetricDirection.LOWER_IS_BETTER,
            "temp_mean_c": MetricDirection.LOWER_IS_BETTER,
            "temp_max_c": MetricDirection.LOWER_IS_BETTER,
            
            # Clock stability (higher is better)
            "clock_stability_score": MetricDirection.HIGHER_IS_BETTER,
            
            # CHI (higher is better)
            "chi_score": MetricDirection.HIGHER_IS_BETTER,
        }
        
        for metric_name, direction in metric_configs.items():
            values = []
            for session in sessions:
                value = getattr(session, metric_name, None)
                if value is not None and value != 0:
                    values.append(value)
            
            if not values:
                continue
            
            arr = np.array(values)
            
            model.metric_means[metric_name] = float(np.mean(arr))
            model.metric_stds[metric_name] = float(np.std(arr))
            model.metric_medians[metric_name] = float(np.median(arr))
            model.metric_mads[metric_name] = float(np.median(np.abs(arr - np.median(arr))))
            model.metric_directions[metric_name] = direction
        
        return model


# ============================================================================
# Regression Guard
# ============================================================================

class RegressionGuard:
    """
    Statistical regression detection system.
    
    Features:
    - Z-score based detection with configurable thresholds
    - MAD-based detection (robust to outliers)
    - Multi-metric aggregation
    - Confidence scoring
    - Severity classification
    """
    
    # Default thresholds for severity classification
    THRESHOLDS = {
        RegressionSeverity.MILD: 1.5,  # 1.5 std deviations
        RegressionSeverity.MODERATE: 2.0,
        RegressionSeverity.SEVERE: 3.0,
        RegressionSeverity.CRITICAL: 4.0,
    }
    
    # Percent change thresholds (fallback when std is very small)
    PCT_THRESHOLDS = {
        RegressionSeverity.MILD: 5.0,  # 5% change
        RegressionSeverity.MODERATE: 10.0,
        RegressionSeverity.SEVERE: 20.0,
        RegressionSeverity.CRITICAL: 50.0,
    }
    
    # Metric weights for overall severity
    METRIC_WEIGHTS = {
        "latency_mean_ms": 2.0,
        "latency_p95_ms": 1.5,
        "throughput_its": 2.0,
        "chi_score": 1.5,
        "cv_pct": 1.0,
        "kar": 1.0,
    }
    
    def __init__(
        self,
        feature_store: Optional[FeatureStore] = None,
        z_threshold: float = 2.0,
        min_baseline_sessions: int = 3,
    ):
        self.feature_store = feature_store
        self.z_threshold = z_threshold
        self.min_baseline_sessions = min_baseline_sessions
        self._baseline_cache: Dict[str, BaselineModel] = {}
    
    def get_baseline(self, model_name: str) -> Optional[BaselineModel]:
        """Get or build baseline model for a model."""
        if model_name in self._baseline_cache:
            return self._baseline_cache[model_name]
        
        if not self.feature_store:
            return None
        
        sessions = self.feature_store.get_sessions_for_model(model_name)
        
        if len(sessions) < self.min_baseline_sessions:
            return None
        
        # Use most recent sessions for baseline (excluding current)
        baseline_sessions = sorted(
            sessions, 
            key=lambda s: s.timestamp_utc, 
            reverse=True
        )[1:11]  # Up to 10 sessions, excluding most recent
        
        baseline = BaselineModel.build_from_sessions(baseline_sessions)
        self._baseline_cache[model_name] = baseline
        
        return baseline
    
    def check_regression(
        self,
        current: SessionFeatures,
        baseline: Optional[BaselineModel] = None,
    ) -> RegressionReport:
        """
        Check current session for regressions against baseline.
        
        Args:
            current: Current session features
            baseline: Baseline model (if None, uses feature store)
            
        Returns:
            RegressionReport with detection results
        """
        if baseline is None:
            baseline = self.get_baseline(current.model_name)
        
        report = RegressionReport(
            session_id=current.session_id,
            model_name=current.model_name,
            baseline_session_count=baseline.session_count if baseline else 0,
        )
        
        if baseline is None or baseline.session_count < self.min_baseline_sessions:
            report.summary = f"Insufficient baseline data (need {self.min_baseline_sessions} sessions)"
            return report
        
        # Analyze each metric
        for metric_name in baseline.metric_means.keys():
            result = self._analyze_metric(
                metric_name=metric_name,
                current_value=getattr(current, metric_name, 0),
                baseline=baseline,
            )
            report.metrics.append(result)
        
        # Determine overall regression status
        regressions = [m for m in report.metrics if m.is_regression]
        
        if regressions:
            report.has_regression = True
            
            # Sort by severity and confidence
            sorted_regressions = sorted(
                regressions,
                key=lambda m: (
                    list(RegressionSeverity).index(m.severity),
                    -m.confidence,
                    -abs(m.z_score)
                ),
                reverse=True,
            )
            
            report.top_regressions = sorted_regressions[:5]
            report.overall_severity = sorted_regressions[0].severity
            report.overall_confidence = self._compute_overall_confidence(sorted_regressions)
            report.summary = self._generate_summary(report)
            report.recommendations = self._generate_recommendations(report)
        else:
            report.summary = "No significant regressions detected"
        
        return report
    
    def _analyze_metric(
        self,
        metric_name: str,
        current_value: float,
        baseline: BaselineModel,
    ) -> RegressionResult:
        """Analyze a single metric for regression."""
        baseline_mean = baseline.metric_means.get(metric_name, 0)
        baseline_std = baseline.metric_stds.get(metric_name, 0)
        baseline_median = baseline.metric_medians.get(metric_name, 0)
        baseline_mad = baseline.metric_mads.get(metric_name, 0)
        direction = baseline.metric_directions.get(metric_name, MetricDirection.LOWER_IS_BETTER)
        
        result = RegressionResult(
            metric_name=metric_name,
            baseline_value=baseline_mean,
            current_value=current_value,
            direction=direction,
        )
        
        # Compute percent change
        if baseline_mean != 0:
            result.percent_change = ((current_value - baseline_mean) / baseline_mean) * 100
        
        # Compute Z-score
        if baseline_std > 0:
            result.z_score = (current_value - baseline_mean) / baseline_std
        
        # Compute MAD score (robust alternative to Z-score)
        if baseline_mad > 0:
            result.mad_score = (current_value - baseline_median) / (baseline_mad * 1.4826)
        
        # Determine if this is a regression based on direction
        if direction == MetricDirection.LOWER_IS_BETTER:
            is_worse = current_value > baseline_mean
        else:
            is_worse = current_value < baseline_mean
        
        # Check thresholds
        abs_z = abs(result.z_score)
        abs_pct = abs(result.percent_change)
        
        severity = RegressionSeverity.NONE
        
        if is_worse:
            # Z-score based detection
            for sev in [RegressionSeverity.CRITICAL, RegressionSeverity.SEVERE,
                       RegressionSeverity.MODERATE, RegressionSeverity.MILD]:
                if abs_z >= self.THRESHOLDS[sev]:
                    severity = sev
                    break
            
            # Fallback to percent change if std is very small
            if severity == RegressionSeverity.NONE and baseline_std < 0.001:
                for sev in [RegressionSeverity.CRITICAL, RegressionSeverity.SEVERE,
                           RegressionSeverity.MODERATE, RegressionSeverity.MILD]:
                    if abs_pct >= self.PCT_THRESHOLDS[sev]:
                        severity = sev
                        break
        
        result.is_regression = severity != RegressionSeverity.NONE
        result.severity = severity
        result.threshold_used = self.THRESHOLDS.get(severity, 0) if severity != RegressionSeverity.NONE else 0
        
        # Compute confidence based on Z-score and consistency with MAD
        if result.is_regression:
            z_confidence = min(1.0, abs_z / 4.0)
            mad_confidence = min(1.0, abs(result.mad_score) / 4.0) if result.mad_score != 0 else z_confidence
            
            # Higher confidence if both methods agree
            if (result.z_score > 0) == (result.mad_score > 0):
                result.confidence = (z_confidence + mad_confidence) / 2
            else:
                result.confidence = z_confidence * 0.7  # Lower confidence if methods disagree
        
        return result
    
    def _compute_overall_confidence(self, regressions: List[RegressionResult]) -> float:
        """Compute overall confidence from individual regressions."""
        if not regressions:
            return 0.0
        
        # Weighted average of confidences
        total_weight = 0.0
        weighted_conf = 0.0
        
        for r in regressions:
            weight = self.METRIC_WEIGHTS.get(r.metric_name, 1.0)
            weighted_conf += r.confidence * weight
            total_weight += weight
        
        return weighted_conf / total_weight if total_weight > 0 else 0.0
    
    def _generate_summary(self, report: RegressionReport) -> str:
        """Generate human-readable summary."""
        if not report.has_regression:
            return "No significant regressions detected"
        
        top = report.top_regressions[0] if report.top_regressions else None
        if not top:
            return "Regression detected"
        
        severity_text = {
            RegressionSeverity.MILD: "Minor",
            RegressionSeverity.MODERATE: "Notable",
            RegressionSeverity.SEVERE: "Significant",
            RegressionSeverity.CRITICAL: "Critical",
        }
        
        sev = severity_text.get(report.overall_severity, "")
        
        return (
            f"{sev} regression detected. "
            f"Primary driver: {top.metric_name} ({top.percent_change:+.1f}%, "
            f"Z={top.z_score:.2f}). "
            f"Confidence: {report.overall_confidence:.0%}"
        )
    
    def _generate_recommendations(self, report: RegressionReport) -> List[str]:
        """Generate actionable recommendations based on regressions."""
        recommendations = []
        
        for r in report.top_regressions[:3]:
            metric = r.metric_name
            
            if "latency" in metric:
                if r.percent_change > 20:
                    recommendations.append(
                        f"Latency regression ({r.percent_change:+.1f}%): "
                        "Check for clock throttling, thermal issues, or driver updates"
                    )
                else:
                    recommendations.append(
                        f"Minor latency increase ({r.percent_change:+.1f}%): "
                        "May be due to system load or background processes"
                    )
            
            elif metric == "throughput_its":
                recommendations.append(
                    f"Throughput drop ({r.percent_change:+.1f}%): "
                    "Verify GPU utilization and check for memory bandwidth bottlenecks"
                )
            
            elif metric == "kar":
                recommendations.append(
                    f"Kernel Active Ratio decreased ({r.percent_change:+.1f}%): "
                    "High kernel launch overhead. Consider kernel fusion or batching"
                )
            
            elif metric == "cv_pct" or metric == "noise_score":
                recommendations.append(
                    f"Increased variability: "
                    "Check for thermal throttling, clock instability, or OS interference"
                )
            
            elif metric == "chi_score":
                recommendations.append(
                    f"Compute Health Index dropped ({r.percent_change:+.1f}%): "
                    "Overall system health degraded. Review all metrics for root cause"
                )
            
            elif "temp" in metric:
                recommendations.append(
                    f"Temperature increase: "
                    "Check cooling system, ensure adequate ventilation"
                )
            
            elif "clock" in metric:
                recommendations.append(
                    f"Clock instability: "
                    "Check power delivery and thermal conditions"
                )
        
        if not recommendations:
            recommendations.append("Run additional diagnostics to identify root cause")
        
        return recommendations


# ============================================================================
# CI/CD Integration Helpers
# ============================================================================

def check_regression_threshold(
    report: RegressionReport,
    max_severity: RegressionSeverity = RegressionSeverity.MODERATE,
) -> Tuple[bool, str]:
    """
    Check if regression exceeds threshold for CI/CD.
    
    Args:
        report: Regression report
        max_severity: Maximum allowed severity
        
    Returns:
        Tuple of (passed, message)
    """
    severity_order = list(RegressionSeverity)
    
    if not report.has_regression:
        return True, "PASS: No regressions detected"
    
    max_idx = severity_order.index(max_severity)
    actual_idx = severity_order.index(report.overall_severity)
    
    if actual_idx <= max_idx:
        return True, f"PASS: Regression severity ({report.overall_severity.value}) within threshold"
    else:
        return False, f"FAIL: Regression severity ({report.overall_severity.value}) exceeds threshold ({max_severity.value})"


def get_regression_exit_code(report: RegressionReport) -> int:
    """
    Get exit code for CI/CD based on regression severity.
    
    Returns:
        0: No regression
        1: Mild regression
        2: Moderate regression
        3: Severe regression  
        4: Critical regression
    """
    exit_codes = {
        RegressionSeverity.NONE: 0,
        RegressionSeverity.MILD: 1,
        RegressionSeverity.MODERATE: 2,
        RegressionSeverity.SEVERE: 3,
        RegressionSeverity.CRITICAL: 4,
    }
    
    return exit_codes.get(report.overall_severity, 0)
