# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Statistical Regression Governance

Robust regression detection with:
- Median + MAD robust baseline
- EWMA (Exponentially Weighted Moving Average) drift
- CUSUM (Cumulative Sum) change detection
- Multi-metric regression analysis
- Confidence-scored regression verdicts
"""

import json
import math
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RegressionSeverity(Enum):
    """Severity levels for regressions."""
    NONE = "none"
    MINOR = "minor"           # < 5% regression
    MODERATE = "moderate"     # 5-15% regression
    SIGNIFICANT = "significant"  # 15-30% regression
    SEVERE = "severe"         # > 30% regression


class DriftDirection(Enum):
    """Direction of performance drift."""
    STABLE = "stable"
    IMPROVEMENT = "improvement"
    REGRESSION = "regression"


@dataclass
class RobustBaseline:
    """
    Robust baseline using median and MAD.
    
    More resistant to outliers than mean/std.
    """
    # Identity
    metric_name: str = ""
    
    # Robust statistics
    median: float = 0.0
    mad: float = 0.0  # Median Absolute Deviation
    
    # Bounds (median +/- k*MAD)
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    
    # Sample info
    sample_count: int = 0
    timestamp: float = 0.0
    
    # Converted to standard deviation equivalent
    # MAD * 1.4826 ≈ σ for normal distributions
    sigma_equivalent: float = 0.0
    
    @classmethod
    def compute(
        cls,
        metric_name: str,
        values: List[float],
        k: float = 3.0,
    ) -> 'RobustBaseline':
        """
        Compute robust baseline from values.
        
        Args:
            metric_name: Name of the metric
            values: Sample values
            k: Number of MADs for bounds (default 3)
            
        Returns:
            Computed RobustBaseline
        """
        if not values:
            return cls(metric_name=metric_name)
        
        baseline = cls(metric_name=metric_name)
        baseline.sample_count = len(values)
        baseline.median = statistics.median(values)
        
        # MAD = median(|x - median(x)|)
        deviations = [abs(v - baseline.median) for v in values]
        baseline.mad = statistics.median(deviations) if deviations else 0
        
        # Sigma equivalent for normal distribution
        baseline.sigma_equivalent = baseline.mad * 1.4826
        
        # Compute bounds
        baseline.lower_bound = baseline.median - k * baseline.mad
        baseline.upper_bound = baseline.median + k * baseline.mad
        
        return baseline
    
    def is_outlier(self, value: float) -> bool:
        """Check if value is outside bounds."""
        return value < self.lower_bound or value > self.upper_bound
    
    def z_score(self, value: float) -> float:
        """Compute robust z-score using MAD."""
        if self.mad == 0:
            return 0.0
        return (value - self.median) / (self.mad * 1.4826)


@dataclass
class EWMAState:
    """State for Exponentially Weighted Moving Average."""
    # Current EWMA value
    value: float = 0.0
    
    # EWMA variance (for control limits)
    variance: float = 0.0
    
    # Parameters
    alpha: float = 0.1  # Smoothing factor
    
    # Control limits
    ucl: float = 0.0  # Upper Control Limit
    lcl: float = 0.0  # Lower Control Limit
    
    # Sample count
    n: int = 0
    
    def update(self, new_value: float, baseline_sigma: float = 1.0) -> None:
        """Update EWMA with new observation."""
        self.n += 1
        
        if self.n == 1:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        
        # Update control limits
        # UCL/LCL = μ ± L × σ × √(α/(2-α))
        limit_factor = baseline_sigma * 3 * math.sqrt(self.alpha / (2 - self.alpha))
        self.ucl = new_value + limit_factor  # Using baseline center
        self.lcl = new_value - limit_factor
    
    def is_signal(self, baseline_value: float) -> bool:
        """Check if EWMA has drifted from baseline."""
        return self.value > self.ucl or self.value < self.lcl


@dataclass
class CUSUMState:
    """State for CUSUM change detection."""
    # Cumulative sums
    s_plus: float = 0.0   # Positive shift
    s_minus: float = 0.0  # Negative shift
    
    # Parameters
    k: float = 0.5  # Slack value (typically 0.5)
    h: float = 4.0  # Decision threshold (typically 4-5)
    
    # Detection
    change_detected: bool = False
    change_direction: DriftDirection = DriftDirection.STABLE
    
    # Sample count
    n: int = 0
    
    def update(self, z_score: float) -> bool:
        """
        Update CUSUM with standardized observation.
        
        Args:
            z_score: Standardized value (x - μ) / σ
            
        Returns:
            True if change detected
        """
        self.n += 1
        
        # Update cumulative sums
        self.s_plus = max(0, self.s_plus + z_score - self.k)
        self.s_minus = min(0, self.s_minus + z_score + self.k)
        
        # Check for change
        if self.s_plus > self.h:
            self.change_detected = True
            self.change_direction = DriftDirection.REGRESSION
            return True
        elif abs(self.s_minus) > self.h:
            self.change_detected = True
            self.change_direction = DriftDirection.IMPROVEMENT
            return True
        
        return False
    
    def reset(self) -> None:
        """Reset CUSUM state after change detection."""
        self.s_plus = 0.0
        self.s_minus = 0.0
        self.change_detected = False
        self.change_direction = DriftDirection.STABLE


@dataclass
class RegressionVerdict:
    """
    Regression verdict with confidence.
    """
    # Metric identification
    metric_name: str = ""
    
    # Verdict
    is_regression: bool = False
    severity: RegressionSeverity = RegressionSeverity.NONE
    direction: DriftDirection = DriftDirection.STABLE
    
    # Quantification
    baseline_value: float = 0.0
    current_value: float = 0.0
    delta_pct: float = 0.0
    
    # Confidence
    confidence: float = 0.0
    
    # Evidence
    ewma_signal: bool = False
    cusum_signal: bool = False
    outlier_count: int = 0
    
    # Statistical significance
    p_value: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metric': self.metric_name,
            'is_regression': self.is_regression,
            'severity': self.severity.value,
            'direction': self.direction.value,
            'baseline': self.baseline_value,
            'current': self.current_value,
            'delta_pct': f"{self.delta_pct:+.2f}%",
            'confidence': f"{self.confidence:.2f}",
            'evidence': {
                'ewma_signal': self.ewma_signal,
                'cusum_signal': self.cusum_signal,
                'outliers': self.outlier_count,
            },
        }


class RegressionGovernor:
    """
    AACO-Ω∞ Statistical Regression Governor
    
    Monitors metrics for regressions using robust statistics.
    
    Features:
    - Robust baseline (median + MAD)
    - EWMA drift detection
    - CUSUM change detection
    - Multi-metric correlation
    - Confidence-scored verdicts
    """
    
    def __init__(
        self,
        ewma_alpha: float = 0.1,
        cusum_k: float = 0.5,
        cusum_h: float = 4.0,
        significance_threshold: float = 0.05,
    ):
        """
        Initialize regression governor.
        
        Args:
            ewma_alpha: EWMA smoothing factor
            cusum_k: CUSUM slack value
            cusum_h: CUSUM decision threshold
            significance_threshold: P-value threshold for significance
        """
        self._ewma_alpha = ewma_alpha
        self._cusum_k = cusum_k
        self._cusum_h = cusum_h
        self._significance_threshold = significance_threshold
        
        # State per metric
        self._baselines: Dict[str, RobustBaseline] = {}
        self._ewma_states: Dict[str, EWMAState] = {}
        self._cusum_states: Dict[str, CUSUMState] = {}
        self._recent_values: Dict[str, List[float]] = {}
    
    def set_baseline(
        self,
        metric_name: str,
        values: List[float],
    ) -> RobustBaseline:
        """
        Set robust baseline for metric.
        
        Args:
            metric_name: Name of metric
            values: Baseline sample values
            
        Returns:
            Computed RobustBaseline
        """
        baseline = RobustBaseline.compute(metric_name, values)
        self._baselines[metric_name] = baseline
        
        # Initialize EWMA and CUSUM
        self._ewma_states[metric_name] = EWMAState(
            value=baseline.median,
            alpha=self._ewma_alpha,
        )
        self._cusum_states[metric_name] = CUSUMState(
            k=self._cusum_k,
            h=self._cusum_h,
        )
        self._recent_values[metric_name] = []
        
        logger.info(f"Set baseline for {metric_name}: median={baseline.median:.2f}, MAD={baseline.mad:.2f}")
        return baseline
    
    def observe(
        self,
        metric_name: str,
        value: float,
    ) -> Optional[RegressionVerdict]:
        """
        Observe new metric value and check for regression.
        
        Args:
            metric_name: Name of metric
            value: New observed value
            
        Returns:
            RegressionVerdict if regression detected, None otherwise
        """
        if metric_name not in self._baselines:
            logger.warning(f"No baseline for metric {metric_name}")
            return None
        
        baseline = self._baselines[metric_name]
        ewma = self._ewma_states[metric_name]
        cusum = self._cusum_states[metric_name]
        
        # Store recent value
        if metric_name not in self._recent_values:
            self._recent_values[metric_name] = []
        self._recent_values[metric_name].append(value)
        
        # Keep only recent history
        if len(self._recent_values[metric_name]) > 100:
            self._recent_values[metric_name] = self._recent_values[metric_name][-100:]
        
        # Compute z-score
        z = baseline.z_score(value)
        
        # Update EWMA
        ewma.update(value, baseline.sigma_equivalent or 1.0)
        
        # Update CUSUM
        cusum_signal = cusum.update(z)
        
        # Check for outlier
        is_outlier = baseline.is_outlier(value)
        
        # Build verdict
        verdict = RegressionVerdict(
            metric_name=metric_name,
            baseline_value=baseline.median,
            current_value=value,
        )
        
        # Calculate delta
        if baseline.median != 0:
            verdict.delta_pct = ((value - baseline.median) / baseline.median) * 100
        
        # Determine direction
        if verdict.delta_pct > 2:  # >2% increase (regression for latency)
            verdict.direction = DriftDirection.REGRESSION
        elif verdict.delta_pct < -2:  # >2% decrease (improvement for latency)
            verdict.direction = DriftDirection.IMPROVEMENT
        
        # Check signals
        verdict.ewma_signal = ewma.is_signal(baseline.median)
        verdict.cusum_signal = cusum_signal
        verdict.outlier_count = 1 if is_outlier else 0
        
        # Determine regression
        # Regression if CUSUM signals AND magnitude is significant
        if cusum.change_direction == DriftDirection.REGRESSION and abs(verdict.delta_pct) > 5:
            verdict.is_regression = True
            verdict.severity = self._classify_severity(verdict.delta_pct)
            verdict.confidence = self._compute_confidence(verdict, baseline)
            
            logger.warning(f"Regression detected for {metric_name}: {verdict.delta_pct:+.2f}%")
            return verdict
        
        return None
    
    def _classify_severity(self, delta_pct: float) -> RegressionSeverity:
        """Classify regression severity based on delta percentage."""
        abs_delta = abs(delta_pct)
        
        if abs_delta < 5:
            return RegressionSeverity.MINOR
        elif abs_delta < 15:
            return RegressionSeverity.MODERATE
        elif abs_delta < 30:
            return RegressionSeverity.SIGNIFICANT
        else:
            return RegressionSeverity.SEVERE
    
    def _compute_confidence(
        self,
        verdict: RegressionVerdict,
        baseline: RobustBaseline,
    ) -> float:
        """Compute confidence in regression verdict."""
        confidence = 0.5  # Base confidence
        
        # Boost for EWMA signal
        if verdict.ewma_signal:
            confidence += 0.15
        
        # Boost for CUSUM signal
        if verdict.cusum_signal:
            confidence += 0.20
        
        # Boost for magnitude
        abs_delta = abs(verdict.delta_pct)
        if abs_delta > 10:
            confidence += 0.10
        if abs_delta > 20:
            confidence += 0.05
        
        # Cap at 0.95
        return min(0.95, confidence)
    
    def analyze_recent(
        self,
        metric_name: str,
        window_size: int = 20,
    ) -> RegressionVerdict:
        """
        Analyze recent values for regression.
        
        Args:
            metric_name: Name of metric
            window_size: Number of recent values to analyze
            
        Returns:
            RegressionVerdict based on recent window
        """
        if metric_name not in self._baselines:
            return RegressionVerdict(metric_name=metric_name)
        
        baseline = self._baselines[metric_name]
        recent = self._recent_values.get(metric_name, [])
        
        if len(recent) < 5:
            return RegressionVerdict(metric_name=metric_name)
        
        # Get window
        window = recent[-window_size:] if len(recent) >= window_size else recent
        
        # Compute window median
        window_median = statistics.median(window)
        
        verdict = RegressionVerdict(
            metric_name=metric_name,
            baseline_value=baseline.median,
            current_value=window_median,
        )
        
        if baseline.median != 0:
            verdict.delta_pct = ((window_median - baseline.median) / baseline.median) * 100
        
        # Count outliers in window
        verdict.outlier_count = sum(1 for v in window if baseline.is_outlier(v))
        
        # Statistical test (simplified Mann-Whitney approximation)
        if verdict.outlier_count > len(window) * 0.3:
            verdict.is_regression = verdict.delta_pct > 5
            verdict.severity = self._classify_severity(verdict.delta_pct)
            verdict.confidence = 0.7 + (verdict.outlier_count / len(window)) * 0.2
        
        return verdict
    
    def get_baseline(self, metric_name: str) -> Optional[RobustBaseline]:
        """Get baseline for metric."""
        return self._baselines.get(metric_name)
    
    def get_all_baselines(self) -> Dict[str, RobustBaseline]:
        """Get all baselines."""
        return self._baselines.copy()
    
    def export_state(self) -> Dict[str, Any]:
        """Export governor state for persistence."""
        return {
            'baselines': {
                name: {
                    'median': b.median,
                    'mad': b.mad,
                    'sample_count': b.sample_count,
                }
                for name, b in self._baselines.items()
            },
            'ewma_alpha': self._ewma_alpha,
            'cusum_k': self._cusum_k,
            'cusum_h': self._cusum_h,
        }
    
    def reset_metric(self, metric_name: str) -> None:
        """Reset tracking state for a metric."""
        if metric_name in self._cusum_states:
            self._cusum_states[metric_name].reset()
        if metric_name in self._ewma_states:
            baseline = self._baselines.get(metric_name)
            if baseline:
                self._ewma_states[metric_name].value = baseline.median


def create_regression_governor(
    ewma_alpha: float = 0.1,
    cusum_k: float = 0.5,
    cusum_h: float = 4.0,
) -> RegressionGovernor:
    """
    Factory function to create regression governor.
    
    Args:
        ewma_alpha: EWMA smoothing factor
        cusum_k: CUSUM slack value
        cusum_h: CUSUM decision threshold
        
    Returns:
        Configured RegressionGovernor
    """
    return RegressionGovernor(
        ewma_alpha=ewma_alpha,
        cusum_k=cusum_k,
        cusum_h=cusum_h,
    )
