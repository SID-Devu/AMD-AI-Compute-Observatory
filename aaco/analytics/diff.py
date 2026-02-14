"""
Regression Detector
Compares sessions to detect performance regressions.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from aaco.core.schema import (
    RegressionVerdict,
    PhaseMetrics,
    DerivedMetrics,
)

logger = logging.getLogger(__name__)


@dataclass
class RegressionThresholds:
    """Thresholds for regression detection."""
    latency_regression_pct: float = 5.0    # >5% slower = regression
    latency_improvement_pct: float = 5.0   # >5% faster = improvement
    p99_regression_pct: float = 10.0       # Higher threshold for p99
    throughput_regression_pct: float = 5.0
    statistical_significance: float = 0.05  # p-value threshold
    min_samples: int = 10                   # Minimum samples for comparison
    
    # Noise tolerance for "expected" variance
    expected_variance_pct: float = 3.0


@dataclass
class MetricComparison:
    """Comparison result for a single metric."""
    metric_name: str
    baseline_value: float
    current_value: float
    delta: float
    delta_pct: float
    verdict: str  # "regression", "improvement", "neutral", "noise"
    p_value: Optional[float] = None


class RegressionDetector:
    """
    Detects performance regressions between benchmark sessions.
    
    Supports:
    1. A/B comparison (baseline vs current)
    2. Trend analysis over multiple sessions
    3. Statistical significance testing
    """
    
    def __init__(self, thresholds: Optional[RegressionThresholds] = None):
        self.thresholds = thresholds or RegressionThresholds()
        self.comparisons: List[MetricComparison] = []
    
    def compare_metrics(
        self,
        baseline: DerivedMetrics,
        current: DerivedMetrics,
        baseline_raw: Optional[List[float]] = None,
        current_raw: Optional[List[float]] = None,
    ) -> RegressionVerdict:
        """
        Compare two sets of metrics and produce a regression verdict.
        
        Args:
            baseline: Metrics from baseline (reference) run
            current: Metrics from current (new) run
            baseline_raw: Raw latency values from baseline (for stats)
            current_raw: Raw latency values from current (for stats)
            
        Returns:
            RegressionVerdict with overall assessment.
        """
        self.comparisons = []
        
        # Compare phase metrics
        self._compare_phase(
            baseline.measurement_phase,
            current.measurement_phase,
            "measurement",
        )
        
        # Compare key metrics
        key_comparisons = [
            ("throughput_ips", 
             baseline.throughput.get("inferences_per_sec", 0),
             current.throughput.get("inferences_per_sec", 0),
             True),  # higher is better
            ("gpu_active_ratio",
             baseline.efficiency.get("gpu_active_ratio", 0),
             current.efficiency.get("gpu_active_ratio", 0),
             True),
            ("microkernel_pct",
             baseline.efficiency.get("microkernel_pct", 0),
             current.efficiency.get("microkernel_pct", 0),
             False),  # lower is better
            ("gpu_util_pct",
             baseline.gpu.get("gpu_util_mean_pct", 0),
             current.gpu.get("gpu_util_mean_pct", 0),
             True),
            ("power_w",
             baseline.gpu.get("power_mean_w", 0),
             current.gpu.get("power_mean_w", 0),
             False),  # lower is better (efficiency)
        ]
        
        for name, base_val, curr_val, higher_better in key_comparisons:
            self._compare_metric(name, base_val, curr_val, higher_better)
        
        # Statistical test if raw data available
        p_value = None
        if baseline_raw and current_raw:
            p_value = self._statistical_test(baseline_raw, current_raw)
        
        # Determine overall verdict
        verdict, confidence = self._determine_verdict(p_value)
        
        # Build detailed summary
        regressions = [c for c in self.comparisons if c.verdict == "regression"]
        improvements = [c for c in self.comparisons if c.verdict == "improvement"]
        
        return RegressionVerdict(
            verdict=verdict,
            confidence=confidence,
            regressions=[c.metric_name for c in regressions],
            improvements=[c.metric_name for c in improvements],
            comparisons=[
                {
                    "metric": c.metric_name,
                    "baseline": c.baseline_value,
                    "current": c.current_value,
                    "delta_pct": c.delta_pct,
                    "verdict": c.verdict,
                }
                for c in self.comparisons
            ],
            p_value=p_value,
            summary=self._build_summary(verdict, regressions, improvements),
        )
    
    def _compare_phase(
        self,
        baseline: PhaseMetrics,
        current: PhaseMetrics,
        phase_name: str,
    ) -> None:
        """Compare phase-level metrics."""
        metrics = [
            (f"{phase_name}_mean_ms", baseline.mean_ms, current.mean_ms, False),
            (f"{phase_name}_p99_ms", baseline.p99_ms, current.p99_ms, False),
            (f"{phase_name}_std_ms", baseline.std_ms, current.std_ms, False),
        ]
        
        for name, base_val, curr_val, higher_better in metrics:
            self._compare_metric(name, base_val, curr_val, higher_better)
    
    def _compare_metric(
        self,
        name: str,
        baseline: float,
        current: float,
        higher_is_better: bool,
    ) -> None:
        """Compare a single metric and classify the change."""
        if baseline == 0:
            if current == 0:
                verdict = "neutral"
                delta_pct = 0
            else:
                verdict = "improvement" if (higher_is_better and current > 0) else "regression"
                delta_pct = float('inf')
        else:
            delta = current - baseline
            delta_pct = 100 * delta / abs(baseline)
            
            # Determine verdict based on direction
            if higher_is_better:
                # Higher is better: positive delta = improvement
                if delta_pct > self.thresholds.latency_improvement_pct:
                    verdict = "improvement"
                elif delta_pct < -self.thresholds.latency_regression_pct:
                    verdict = "regression"
                elif abs(delta_pct) < self.thresholds.expected_variance_pct:
                    verdict = "noise"
                else:
                    verdict = "neutral"
            else:
                # Lower is better (latency): positive delta = regression
                if delta_pct > self.thresholds.latency_regression_pct:
                    verdict = "regression"
                elif delta_pct < -self.thresholds.latency_improvement_pct:
                    verdict = "improvement"
                elif abs(delta_pct) < self.thresholds.expected_variance_pct:
                    verdict = "noise"
                else:
                    verdict = "neutral"
        
        self.comparisons.append(MetricComparison(
            metric_name=name,
            baseline_value=baseline,
            current_value=current,
            delta=current - baseline,
            delta_pct=delta_pct if delta_pct != float('inf') else 999,
            verdict=verdict,
        ))
    
    def _statistical_test(
        self,
        baseline: List[float],
        current: List[float],
    ) -> float:
        """
        Perform statistical significance test (Welch's t-test).
        Returns p-value.
        """
        if len(baseline) < self.thresholds.min_samples or len(current) < self.thresholds.min_samples:
            logger.warning("Not enough samples for statistical test")
            return 1.0
        
        try:
            from scipy import stats
            _, p_value = stats.ttest_ind(baseline, current, equal_var=False)
            return float(p_value)
        except ImportError:
            # Manual calculation if scipy not available
            return self._manual_ttest(baseline, current)
    
    def _manual_ttest(
        self,
        baseline: List[float],
        current: List[float],
    ) -> float:
        """Manual Welch's t-test implementation."""
        n1, n2 = len(baseline), len(current)
        mean1, mean2 = np.mean(baseline), np.mean(current)
        var1, var2 = np.var(baseline, ddof=1), np.var(current, ddof=1)
        
        se = np.sqrt(var1/n1 + var2/n2)
        if se == 0:
            return 1.0
        
        t = (mean1 - mean2) / se
        
        # Degrees of freedom (Welch-Satterthwaite)
        df_num = (var1/n1 + var2/n2) ** 2
        df_den = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
        df = df_num / df_den if df_den > 0 else 1
        
        # Approximate p-value using normal distribution for large df
        if df > 30:
            p_value = 2 * (1 - self._normal_cdf(abs(t)))
        else:
            # Conservative estimate
            p_value = 2 * (1 - self._normal_cdf(abs(t) * 0.8))
        
        return p_value
    
    def _normal_cdf(self, x: float) -> float:
        """Approximate normal CDF."""
        return 0.5 * (1 + np.tanh(0.8 * x))
    
    def _determine_verdict(
        self,
        p_value: Optional[float],
    ) -> Tuple[str, float]:
        """Determine overall verdict from metric comparisons."""
        regressions = [c for c in self.comparisons if c.verdict == "regression"]
        improvements = [c for c in self.comparisons if c.verdict == "improvement"]
        
        # Priority metrics for overall verdict
        latency_comp = next(
            (c for c in self.comparisons if c.metric_name == "measurement_mean_ms"),
            None
        )
        
        # Check statistical significance
        is_significant = (
            p_value is not None and 
            p_value < self.thresholds.statistical_significance
        )
        
        # Determine verdict
        if latency_comp:
            if latency_comp.verdict == "regression":
                verdict = "REGRESSION"
                confidence = 0.9 if is_significant else 0.7
            elif latency_comp.verdict == "improvement":
                verdict = "IMPROVEMENT"
                confidence = 0.9 if is_significant else 0.7
            else:
                verdict = "NEUTRAL"
                confidence = 0.8
        else:
            # Fall back to counting
            if len(regressions) > len(improvements) + 2:
                verdict = "REGRESSION"
                confidence = 0.6
            elif len(improvements) > len(regressions) + 2:
                verdict = "IMPROVEMENT"
                confidence = 0.6
            else:
                verdict = "NEUTRAL"
                confidence = 0.7
        
        return verdict, confidence
    
    def _build_summary(
        self,
        verdict: str,
        regressions: List[MetricComparison],
        improvements: List[MetricComparison],
    ) -> str:
        """Build human-readable summary."""
        lines = [f"Overall Verdict: {verdict}"]
        
        if regressions:
            lines.append("\nRegressions detected:")
            for r in regressions[:5]:
                lines.append(f"  • {r.metric_name}: {r.delta_pct:+.1f}%")
        
        if improvements:
            lines.append("\nImprovements detected:")
            for i in improvements[:5]:
                lines.append(f"  • {i.metric_name}: {i.delta_pct:+.1f}%")
        
        return "\n".join(lines)
    
    def compare_sessions(
        self,
        baseline_path: Path,
        current_path: Path,
    ) -> RegressionVerdict:
        """
        Compare two session bundles.
        
        Args:
            baseline_path: Path to baseline session folder
            current_path: Path to current session folder
            
        Returns:
            RegressionVerdict
        """
        # Load session metadata
        def load_metrics(session_path: Path) -> Tuple[Optional[DerivedMetrics], List[float]]:
            metrics_file = session_path / "metrics.json"
            inference_file = session_path / "inference_results.json"
            
            derived = None
            raw_latencies = []
            
            if metrics_file.exists():
                with open(metrics_file) as f:
                    # Reconstruct DerivedMetrics from JSON
                    pass  # Would need proper deserialization
            
            if inference_file.exists():
                with open(inference_file) as f:
                    data = json.load(f)
                    raw_latencies = [
                        r["latency_ms"] for r in data
                        if r.get("phase") == "measurement"
                    ]
            
            return derived, raw_latencies
        
        base_metrics, base_raw = load_metrics(baseline_path)
        curr_metrics, curr_raw = load_metrics(current_path)
        
        if base_metrics and curr_metrics:
            return self.compare_metrics(base_metrics, curr_metrics, base_raw, curr_raw)
        
        # Fall back to raw comparison
        if base_raw and curr_raw:
            return self._compare_raw_latencies(base_raw, curr_raw)
        
        return RegressionVerdict(
            verdict="UNKNOWN",
            confidence=0,
            regressions=[],
            improvements=[],
            comparisons=[],
            summary="Insufficient data for comparison",
        )
    
    def _compare_raw_latencies(
        self,
        baseline: List[float],
        current: List[float],
    ) -> RegressionVerdict:
        """Compare just raw latency arrays."""
        base_mean = np.mean(baseline)
        curr_mean = np.mean(current)
        delta_pct = 100 * (curr_mean - base_mean) / base_mean if base_mean > 0 else 0
        
        p_value = self._statistical_test(baseline, current)
        
        if delta_pct > self.thresholds.latency_regression_pct:
            verdict = "REGRESSION"
        elif delta_pct < -self.thresholds.latency_improvement_pct:
            verdict = "IMPROVEMENT"
        else:
            verdict = "NEUTRAL"
        
        confidence = 0.9 if p_value < 0.05 else 0.6
        
        return RegressionVerdict(
            verdict=verdict,
            confidence=confidence,
            regressions=["latency_ms"] if verdict == "REGRESSION" else [],
            improvements=["latency_ms"] if verdict == "IMPROVEMENT" else [],
            comparisons=[{
                "metric": "mean_latency_ms",
                "baseline": base_mean,
                "current": curr_mean,
                "delta_pct": delta_pct,
                "verdict": verdict.lower(),
            }],
            p_value=p_value,
            summary=f"Mean latency: {base_mean:.2f}ms → {curr_mean:.2f}ms ({delta_pct:+.1f}%)",
        )


def diff_sessions(
    baseline: str,
    current: str,
    thresholds: Optional[Dict[str, float]] = None,
) -> RegressionVerdict:
    """
    Convenience function to diff two session paths.
    
    Args:
        baseline: Path to baseline session
        current: Path to current session
        thresholds: Optional threshold overrides
        
    Returns:
        RegressionVerdict
    """
    config = RegressionThresholds()
    if thresholds:
        for k, v in thresholds.items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    detector = RegressionDetector(config)
    return detector.compare_sessions(Path(baseline), Path(current))
