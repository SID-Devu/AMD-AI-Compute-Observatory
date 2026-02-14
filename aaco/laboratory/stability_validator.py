# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Stability Validator

Validates measurement stability and determines if results are scientifically valid.
"""

import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StabilityLevel(Enum):
    """Measurement stability classification."""
    EXCELLENT = "excellent"      # < 1% CoV
    GOOD = "good"               # 1-3% CoV
    ACCEPTABLE = "acceptable"   # 3-5% CoV
    MARGINAL = "marginal"       # 5-10% CoV
    UNSTABLE = "unstable"       # > 10% CoV


@dataclass
class StabilityReport:
    """Report on measurement stability."""
    level: StabilityLevel
    coefficient_of_variation: float
    mean: float
    std_dev: float
    median: float
    mad: float  # Median Absolute Deviation
    min_value: float
    max_value: float
    outlier_count: int
    sample_count: int
    valid: bool
    invalid_reason: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


class StabilityValidator:
    """
    Validates measurement stability for scientific rigor.
    
    Uses robust statistics (median, MAD) to handle outliers
    and provides actionable recommendations.
    """
    
    # Thresholds for stability levels (Coefficient of Variation)
    EXCELLENT_THRESHOLD = 0.01   # 1%
    GOOD_THRESHOLD = 0.03       # 3%
    ACCEPTABLE_THRESHOLD = 0.05  # 5%
    MARGINAL_THRESHOLD = 0.10   # 10%
    
    # Minimum samples for valid statistics
    MIN_SAMPLES = 10
    
    def __init__(
        self,
        min_samples: int = 10,
        max_cov_threshold: float = 0.10,
        outlier_threshold: float = 3.0  # MAD multiplier
    ):
        """Initialize validator."""
        self.min_samples = min_samples
        self.max_cov_threshold = max_cov_threshold
        self.outlier_threshold = outlier_threshold
    
    def validate(self, measurements: List[float]) -> StabilityReport:
        """
        Validate a series of measurements.
        
        Args:
            measurements: List of measurement values (e.g., latencies)
            
        Returns:
            StabilityReport with analysis and recommendations
        """
        # Check minimum samples
        if len(measurements) < self.min_samples:
            return StabilityReport(
                level=StabilityLevel.UNSTABLE,
                coefficient_of_variation=float('inf'),
                mean=0, std_dev=0, median=0, mad=0,
                min_value=0, max_value=0,
                outlier_count=0,
                sample_count=len(measurements),
                valid=False,
                invalid_reason=f"Insufficient samples: {len(measurements)} < {self.min_samples}",
                recommendations=["Increase measurement iterations"]
            )
        
        # Basic statistics
        mean_val = statistics.mean(measurements)
        std_dev = statistics.stdev(measurements)
        median_val = statistics.median(measurements)
        mad = self._median_absolute_deviation(measurements)
        
        # Coefficient of Variation
        cov = std_dev / mean_val if mean_val > 0 else float('inf')
        
        # Detect outliers using MAD
        outliers = self._detect_outliers(measurements, median_val, mad)
        
        # Determine stability level
        level = self._classify_stability(cov)
        
        # Generate report
        report = StabilityReport(
            level=level,
            coefficient_of_variation=cov,
            mean=mean_val,
            std_dev=std_dev,
            median=median_val,
            mad=mad,
            min_value=min(measurements),
            max_value=max(measurements),
            outlier_count=len(outliers),
            sample_count=len(measurements),
            valid=level != StabilityLevel.UNSTABLE,
        )
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(
            report, measurements, outliers
        )
        
        # Set invalid reason if applicable
        if not report.valid:
            report.invalid_reason = f"CoV {cov:.1%} exceeds threshold {self.max_cov_threshold:.1%}"
        
        return report
    
    def _median_absolute_deviation(self, data: List[float]) -> float:
        """Compute Median Absolute Deviation (robust measure of spread)."""
        median = statistics.median(data)
        deviations = [abs(x - median) for x in data]
        return statistics.median(deviations)
    
    def _detect_outliers(
        self,
        data: List[float],
        median: float,
        mad: float
    ) -> List[int]:
        """Detect outliers using MAD-based method."""
        if mad == 0:
            return []
        
        outlier_indices = []
        # Modified Z-score using MAD
        k = 1.4826  # Consistency constant for normal distribution
        
        for i, x in enumerate(data):
            modified_z = (x - median) / (k * mad)
            if abs(modified_z) > self.outlier_threshold:
                outlier_indices.append(i)
        
        return outlier_indices
    
    def _classify_stability(self, cov: float) -> StabilityLevel:
        """Classify stability level based on CoV."""
        if cov <= self.EXCELLENT_THRESHOLD:
            return StabilityLevel.EXCELLENT
        elif cov <= self.GOOD_THRESHOLD:
            return StabilityLevel.GOOD
        elif cov <= self.ACCEPTABLE_THRESHOLD:
            return StabilityLevel.ACCEPTABLE
        elif cov <= self.MARGINAL_THRESHOLD:
            return StabilityLevel.MARGINAL
        else:
            return StabilityLevel.UNSTABLE
    
    def _generate_recommendations(
        self,
        report: StabilityReport,
        measurements: List[float],
        outlier_indices: List[int]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []
        
        # High outlier count
        if len(outlier_indices) > len(measurements) * 0.1:
            recs.append(
                f"High outlier rate ({len(outlier_indices)}/{len(measurements)}). "
                "Consider increasing warmup iterations or enabling system isolation."
            )
        
        # Trend detection - are measurements increasing/decreasing?
        if len(measurements) > 20:
            first_half = statistics.mean(measurements[:len(measurements)//2])
            second_half = statistics.mean(measurements[len(measurements)//2:])
            drift = (second_half - first_half) / first_half if first_half > 0 else 0
            
            if abs(drift) > 0.05:
                direction = "increasing" if drift > 0 else "decreasing"
                recs.append(
                    f"Detected {direction} trend ({drift:+.1%}). "
                    "May indicate thermal throttling or resource contention."
                )
        
        # High variance recommendations
        if report.level in [StabilityLevel.MARGINAL, StabilityLevel.UNSTABLE]:
            recs.extend([
                "Enable CPU isolation with dedicated cores",
                "Set CPU governor to 'performance'",
                "Lock GPU clocks to fixed frequency",
                "Increase warmup iterations",
                "Check for background processes",
            ])
        
        # Good but could be better
        if report.level == StabilityLevel.ACCEPTABLE:
            recs.append(
                "For publication-quality results, consider additional "
                "isolation measures to reduce variability."
            )
        
        return recs
    
    def compare_distributions(
        self,
        baseline: List[float],
        current: List[float],
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Compare two measurement distributions for regression detection.
        
        Uses robust statistics and accounts for measurement noise.
        """
        # Validate both distributions
        baseline_report = self.validate(baseline)
        current_report = self.validate(current)
        
        # Compute robust difference
        diff_pct = (
            (current_report.median - baseline_report.median) / 
            baseline_report.median * 100
            if baseline_report.median > 0 else 0
        )
        
        # Compute noise-aware threshold
        # Use quadrature addition of MADs
        combined_uncertainty = (
            (baseline_report.mad ** 2 + current_report.mad ** 2) ** 0.5
        )
        
        # Normalize by baseline median
        relative_uncertainty = (
            combined_uncertainty / baseline_report.median * 100
            if baseline_report.median > 0 else float('inf')
        )
        
        # Is the difference significant?
        significant = abs(diff_pct) > relative_uncertainty * 2
        
        return {
            'baseline': {
                'median': baseline_report.median,
                'mad': baseline_report.mad,
                'stability': baseline_report.level.value,
            },
            'current': {
                'median': current_report.median,
                'mad': current_report.mad,
                'stability': current_report.level.value,
            },
            'difference_pct': diff_pct,
            'uncertainty_pct': relative_uncertainty,
            'significant': significant,
            'regression': significant and diff_pct > 0,
            'improvement': significant and diff_pct < 0,
            'confidence': confidence_level,
        }
