"""
AACO-SIGMA Regression Detector

Statistical detection of performance regressions.
Uses multiple detection methods for robust regression identification.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum, auto
import statistics
import math


class RegressionSeverity(Enum):
    """Severity levels for detected regressions."""
    NONE = auto()       # No regression
    NOISE = auto()      # Within noise threshold
    MINOR = auto()      # Small regression (< 5%)
    MODERATE = auto()   # Noticeable regression (5-15%)
    MAJOR = auto()      # Significant regression (15-30%)
    CRITICAL = auto()   # Critical regression (> 30%)


class DetectionMethod(Enum):
    """Statistical methods for regression detection."""
    THRESHOLD = auto()       # Simple threshold comparison
    MOVING_AVERAGE = auto()  # Moving average comparison
    ZSCORE = auto()          # Z-score based detection
    MANN_WHITNEY = auto()    # Non-parametric test
    BOOTSTRAP = auto()       # Bootstrap confidence intervals


@dataclass
class RegressionConfig:
    """Configuration for regression detection."""
    
    # Thresholds (percentage)
    noise_threshold_pct: float = 2.0     # Below this = noise
    minor_threshold_pct: float = 5.0     # Minor regression
    moderate_threshold_pct: float = 15.0 # Moderate regression
    major_threshold_pct: float = 30.0    # Major regression
    
    # Statistical parameters
    confidence_level: float = 0.95
    min_samples: int = 5
    
    # Detection method
    method: DetectionMethod = DetectionMethod.ZSCORE
    
    # Moving average window
    ma_window: int = 10
    
    # Z-score threshold
    zscore_threshold: float = 2.0
    
    # Ignore improvements?
    track_improvements: bool = True


@dataclass
class RegressionResult:
    """Result of regression detection."""
    
    # Identification
    metric_name: str
    kernel_name: str = ""
    
    # Comparison
    baseline_value: float = 0.0
    current_value: float = 0.0
    delta_pct: float = 0.0
    
    # Classification
    is_regression: bool = False
    is_improvement: bool = False
    severity: RegressionSeverity = RegressionSeverity.NONE
    
    # Statistical confidence
    confidence: float = 0.0
    p_value: float = 1.0
    
    # Detection details
    method_used: DetectionMethod = DetectionMethod.THRESHOLD
    zscore: float = 0.0
    
    # Context
    baseline_samples: int = 0
    current_samples: int = 0
    baseline_stddev: float = 0.0
    current_stddev: float = 0.0


class RegressionDetector:
    """
    Detects performance regressions using statistical methods.
    
    Supports multiple detection strategies:
    - Simple threshold comparison
    - Z-score based detection
    - Mann-Whitney U test (non-parametric)
    - Bootstrap confidence intervals
    """
    
    def __init__(self, config: Optional[RegressionConfig] = None):
        self.config = config or RegressionConfig()
        self._history: Dict[str, List[float]] = {}
    
    def detect(self, 
               metric_name: str,
               baseline_values: List[float],
               current_values: List[float],
               kernel_name: str = "") -> RegressionResult:
        """
        Detect regression between baseline and current values.
        
        Args:
            metric_name: Name of the metric (e.g., "latency_ms")
            baseline_values: Historical baseline measurements
            current_values: Current measurements to compare
            kernel_name: Optional kernel identifier
            
        Returns:
            RegressionResult with detection details
        """
        result = RegressionResult(
            metric_name=metric_name,
            kernel_name=kernel_name,
            baseline_samples=len(baseline_values),
            current_samples=len(current_values),
        )
        
        # Need minimum samples
        if len(baseline_values) < self.config.min_samples:
            return result
        if len(current_values) < 1:
            return result
        
        # Calculate statistics
        baseline_mean = statistics.mean(baseline_values)
        current_mean = statistics.mean(current_values)
        
        result.baseline_value = baseline_mean
        result.current_value = current_mean
        
        if baseline_mean > 0:
            result.delta_pct = ((current_mean - baseline_mean) / baseline_mean) * 100
        
        if len(baseline_values) > 1:
            result.baseline_stddev = statistics.stdev(baseline_values)
        if len(current_values) > 1:
            result.current_stddev = statistics.stdev(current_values)
        
        # Apply detection method
        result.method_used = self.config.method
        
        if self.config.method == DetectionMethod.THRESHOLD:
            self._detect_threshold(result)
        elif self.config.method == DetectionMethod.ZSCORE:
            self._detect_zscore(result, baseline_values, current_values)
        elif self.config.method == DetectionMethod.MANN_WHITNEY:
            self._detect_mann_whitney(result, baseline_values, current_values)
        elif self.config.method == DetectionMethod.BOOTSTRAP:
            self._detect_bootstrap(result, baseline_values, current_values)
        else:
            self._detect_threshold(result)
        
        # Classify severity
        result.severity = self._classify_severity(result.delta_pct, result.is_regression)
        
        return result
    
    def _detect_threshold(self, result: RegressionResult) -> None:
        """Simple threshold-based detection."""
        if result.delta_pct > self.config.noise_threshold_pct:
            result.is_regression = True
            result.confidence = 0.8  # Lower confidence for simple threshold
        elif result.delta_pct < -self.config.noise_threshold_pct:
            result.is_improvement = True
            result.confidence = 0.8
    
    def _detect_zscore(self, result: RegressionResult,
                       baseline: List[float],
                       current: List[float]) -> None:
        """Z-score based detection."""
        baseline_mean = statistics.mean(baseline)
        baseline_std = statistics.stdev(baseline) if len(baseline) > 1 else 0
        
        if baseline_std == 0:
            self._detect_threshold(result)
            return
        
        current_mean = statistics.mean(current)
        
        # Z-score of current mean relative to baseline distribution
        zscore = (current_mean - baseline_mean) / baseline_std
        result.zscore = zscore
        
        if zscore > self.config.zscore_threshold:
            result.is_regression = True
            # Confidence from z-score cumulative probability
            result.confidence = self._zscore_confidence(zscore)
        elif zscore < -self.config.zscore_threshold:
            result.is_improvement = True
            result.confidence = self._zscore_confidence(-zscore)
    
    def _detect_mann_whitney(self, result: RegressionResult,
                             baseline: List[float],
                             current: List[float]) -> None:
        """
        Mann-Whitney U test for non-parametric regression detection.
        
        Does not assume normal distribution.
        """
        # Simple implementation of Mann-Whitney U
        all_values = baseline + current
        all_values_sorted = sorted(enumerate(all_values), key=lambda x: x[1])
        
        ranks = [0.0] * len(all_values)
        for rank, (idx, _) in enumerate(all_values_sorted, 1):
            ranks[idx] = rank
        
        # Sum of ranks for baseline
        r1 = sum(ranks[:len(baseline)])
        n1 = len(baseline)
        n2 = len(current)
        
        # U statistic
        u1 = r1 - n1 * (n1 + 1) / 2
        
        # Expected value and standard deviation under null
        mu = n1 * n2 / 2
        sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        
        if sigma > 0:
            z = (u1 - mu) / sigma
            result.zscore = z
            
            if z > self.config.zscore_threshold:
                result.is_regression = True
                result.confidence = self._zscore_confidence(z)
            elif z < -self.config.zscore_threshold:
                result.is_improvement = True
                result.confidence = self._zscore_confidence(-z)
    
    def _detect_bootstrap(self, result: RegressionResult,
                          baseline: List[float],
                          current: List[float],
                          n_bootstrap: int = 1000) -> None:
        """
        Bootstrap confidence interval detection.
        
        Resamples to estimate confidence interval of difference.
        """
        import random
        
        baseline_mean = statistics.mean(baseline)
        current_mean = statistics.mean(current)
        observed_diff = current_mean - baseline_mean
        
        # Bootstrap resampling
        boot_diffs = []
        combined = baseline + current
        n_baseline = len(baseline)
        
        for _ in range(n_bootstrap):
            # Resample under null hypothesis (combined)
            sample = random.choices(combined, k=len(combined))
            boot_baseline = sample[:n_baseline]
            boot_current = sample[n_baseline:]
            boot_diff = statistics.mean(boot_current) - statistics.mean(boot_baseline)
            boot_diffs.append(boot_diff)
        
        boot_diffs.sort()
        
        # Calculate p-value (one-tailed)
        # Proportion of bootstrap diffs >= observed diff
        extreme_count = sum(1 for d in boot_diffs if d >= observed_diff)
        p_value = extreme_count / n_bootstrap
        result.p_value = p_value
        
        # Confidence interval
        alpha = 1 - self.config.confidence_level
        lower_idx = int(alpha / 2 * n_bootstrap)
        upper_idx = int((1 - alpha / 2) * n_bootstrap)
        
        ci_lower = boot_diffs[lower_idx]
        ci_upper = boot_diffs[upper_idx]
        
        # Detect regression if CI doesn't include 0 on the positive side
        if ci_lower > 0:
            result.is_regression = True
            result.confidence = self.config.confidence_level
        elif ci_upper < 0:
            result.is_improvement = True
            result.confidence = self.config.confidence_level
    
    def _zscore_confidence(self, zscore: float) -> float:
        """Convert z-score to confidence level."""
        # Simplified normal CDF approximation
        if zscore > 3:
            return 0.999
        elif zscore > 2.5:
            return 0.99
        elif zscore > 2.0:
            return 0.95
        elif zscore > 1.5:
            return 0.90
        else:
            return 0.80
    
    def _classify_severity(self, delta_pct: float, is_regression: bool) -> RegressionSeverity:
        """Classify regression severity based on delta percentage."""
        if not is_regression:
            return RegressionSeverity.NONE
        
        abs_delta = abs(delta_pct)
        
        if abs_delta < self.config.noise_threshold_pct:
            return RegressionSeverity.NOISE
        elif abs_delta < self.config.minor_threshold_pct:
            return RegressionSeverity.MINOR
        elif abs_delta < self.config.moderate_threshold_pct:
            return RegressionSeverity.MODERATE
        elif abs_delta < self.config.major_threshold_pct:
            return RegressionSeverity.MAJOR
        else:
            return RegressionSeverity.CRITICAL
    
    def add_to_history(self, metric_name: str, value: float) -> None:
        """Add a measurement to history for moving average detection."""
        if metric_name not in self._history:
            self._history[metric_name] = []
        self._history[metric_name].append(value)
        
        # Keep only recent values
        max_history = self.config.ma_window * 5
        if len(self._history[metric_name]) > max_history:
            self._history[metric_name] = self._history[metric_name][-max_history:]
    
    def detect_from_history(self, metric_name: str,
                            current_values: List[float]) -> RegressionResult:
        """Detect regression using stored history as baseline."""
        history = self._history.get(metric_name, [])
        
        # Use recent history as baseline
        baseline = history[-self.config.ma_window:] if history else []
        
        return self.detect(metric_name, baseline, current_values)
    
    def batch_detect(self, 
                     metrics: Dict[str, Tuple[List[float], List[float]]]) -> List[RegressionResult]:
        """
        Detect regressions for multiple metrics.
        
        Args:
            metrics: Dict of metric_name -> (baseline_values, current_values)
        """
        results = []
        for metric_name, (baseline, current) in metrics.items():
            result = self.detect(metric_name, baseline, current)
            results.append(result)
        
        return results
