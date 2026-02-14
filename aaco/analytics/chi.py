"""
Compute Health Index (CHI)
Composite metric for overall system performance health.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .feature_store import SessionFeatures

logger = logging.getLogger(__name__)


# ============================================================================
# Health Rating
# ============================================================================


class HealthRating(str, Enum):
    """Overall health rating."""

    EXCELLENT = "excellent"  # CHI >= 90
    GOOD = "good"  # CHI >= 75
    FAIR = "fair"  # CHI >= 50
    POOR = "poor"  # CHI >= 25
    CRITICAL = "critical"  # CHI < 25


# ============================================================================
# CHI Components
# ============================================================================


@dataclass
class CHIComponent:
    """Individual component contributing to CHI."""

    name: str
    score: float  # [0-1] component score
    weight: float  # Weight in final calculation
    description: str
    contributing_metrics: List[str] = field(default_factory=list)

    @property
    def weighted_contribution(self) -> float:
        return self.score * self.weight

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "weight": self.weight,
            "weighted_contribution": self.weighted_contribution,
            "description": self.description,
            "contributing_metrics": self.contributing_metrics,
        }


@dataclass
class CHIReport:
    """Complete Compute Health Index report."""

    session_id: str
    chi_score: float  # [0-100] final score
    rating: HealthRating

    # Component breakdown
    components: List[CHIComponent] = field(default_factory=list)

    # Trend (if historical data available)
    chi_trend: Optional[float] = None  # Positive = improving
    trend_sessions: int = 0

    # Summary
    summary: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "chi_score": self.chi_score,
            "rating": self.rating.value,
            "components": [c.to_dict() for c in self.components],
            "chi_trend": self.chi_trend,
            "trend_sessions": self.trend_sessions,
            "summary": self.summary,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
        }


# ============================================================================
# CHI Calculator
# ============================================================================


class CHICalculator:
    """
    Calculates Compute Health Index from session features.

    CHI is a composite score [0-100] that summarizes overall system health:

    Components:
    - Stability (25%): Latency consistency and spike-free operation
    - Efficiency (25%): GPU utilization and kernel active ratio
    - Launch Tax (15%): Kernel dispatch overhead
    - Clock Stability (15%): GPU clock consistency
    - Thermal (10%): Temperature headroom
    - System (10%): OS-level interference

    Higher scores indicate better health:
    - 90-100: Excellent - optimal performance
    - 75-89: Good - minor issues
    - 50-74: Fair - notable issues
    - 25-49: Poor - significant problems
    - 0-24: Critical - severe issues
    """

    # Default component weights
    DEFAULT_WEIGHTS = {
        "stability": 0.25,
        "efficiency": 0.25,
        "launch_tax": 0.15,
        "clock_stability": 0.15,
        "thermal": 0.10,
        "system": 0.10,
    }

    # Thresholds for component scoring
    THRESHOLDS = {
        # Stability
        "cv_excellent": 5.0,  # CV% <= this is excellent
        "cv_good": 10.0,
        "cv_fair": 20.0,
        "spike_ratio_excellent": 0.01,
        "spike_ratio_good": 0.05,
        "spike_ratio_fair": 0.10,
        # Efficiency
        "kar_excellent": 0.9,
        "kar_good": 0.75,
        "kar_fair": 0.5,
        "gpu_util_excellent": 85,
        "gpu_util_good": 70,
        "gpu_util_fair": 50,
        # Thermal
        "temp_excellent": 65,  # Celsius
        "temp_good": 75,
        "temp_fair": 85,
        # Clock
        "clock_cv_excellent": 2,  # CV%
        "clock_cv_good": 5,
        "clock_cv_fair": 10,
        # System
        "ctx_excellent": 20,  # Per iteration
        "ctx_good": 50,
        "ctx_fair": 100,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def calculate(self, features: SessionFeatures) -> CHIReport:
        """
        Calculate CHI from session features.

        Args:
            features: Session features

        Returns:
            CHIReport with score and breakdown
        """
        report = CHIReport(
            session_id=features.session_id, chi_score=0, rating=HealthRating.CRITICAL
        )

        # Calculate each component
        components = [
            self._calc_stability(features),
            self._calc_efficiency(features),
            self._calc_launch_tax(features),
            self._calc_clock_stability(features),
            self._calc_thermal(features),
            self._calc_system(features),
        ]

        report.components = components

        # Calculate final CHI score
        chi = sum(c.weighted_contribution for c in components) * 100
        report.chi_score = round(chi, 1)

        # Determine rating
        report.rating = self._get_rating(report.chi_score)

        # Generate summary
        report.summary = self._generate_summary(report)
        report.strengths, report.weaknesses = self._identify_strengths_weaknesses(components)

        return report

    def _calc_stability(self, features: SessionFeatures) -> CHIComponent:
        """Calculate stability component score."""
        # CV-based score
        cv = features.cv_pct
        if cv <= self.THRESHOLDS["cv_excellent"]:
            cv_score = 1.0
        elif cv <= self.THRESHOLDS["cv_good"]:
            cv_score = 0.8
        elif cv <= self.THRESHOLDS["cv_fair"]:
            cv_score = 0.5
        else:
            cv_score = max(0.0, 1.0 - cv / 50)

        # Spike-based score
        spike = features.spike_ratio
        if spike <= self.THRESHOLDS["spike_ratio_excellent"]:
            spike_score = 1.0
        elif spike <= self.THRESHOLDS["spike_ratio_good"]:
            spike_score = 0.8
        elif spike <= self.THRESHOLDS["spike_ratio_fair"]:
            spike_score = 0.5
        else:
            spike_score = max(0.0, 1.0 - spike * 5)

        # Combine (weighted average)
        score = cv_score * 0.6 + spike_score * 0.4

        return CHIComponent(
            name="stability",
            score=score,
            weight=self.weights["stability"],
            description=f"CV={cv:.1f}%, spikes={spike:.1%}",
            contributing_metrics=["cv_pct", "spike_ratio", "noise_score"],
        )

    def _calc_efficiency(self, features: SessionFeatures) -> CHIComponent:
        """Calculate efficiency component score."""
        # KAR-based score
        kar = features.kar if features.kar > 0 else 0.5
        if kar >= self.THRESHOLDS["kar_excellent"]:
            kar_score = 1.0
        elif kar >= self.THRESHOLDS["kar_good"]:
            kar_score = 0.8
        elif kar >= self.THRESHOLDS["kar_fair"]:
            kar_score = 0.5
        else:
            kar_score = kar

        # GPU util-based score
        util = features.gpu_util_mean_pct if features.gpu_util_mean_pct > 0 else 50
        if util >= self.THRESHOLDS["gpu_util_excellent"]:
            util_score = 1.0
        elif util >= self.THRESHOLDS["gpu_util_good"]:
            util_score = 0.8
        elif util >= self.THRESHOLDS["gpu_util_fair"]:
            util_score = 0.5
        else:
            util_score = util / 100

        score = kar_score * 0.6 + util_score * 0.4

        return CHIComponent(
            name="efficiency",
            score=score,
            weight=self.weights["efficiency"],
            description=f"KAR={kar:.2f}, GPU util={util:.0f}%",
            contributing_metrics=["kar", "gpu_util_mean_pct"],
        )

    def _calc_launch_tax(self, features: SessionFeatures) -> CHIComponent:
        """Calculate launch tax component score."""
        # Lower launch tax = higher score
        tax = features.launch_tax_score
        score = 1.0 - tax

        return CHIComponent(
            name="launch_tax",
            score=score,
            weight=self.weights["launch_tax"],
            description=f"Launch tax={tax:.2f}",
            contributing_metrics=["launch_tax_score"],
        )

    def _calc_clock_stability(self, features: SessionFeatures) -> CHIComponent:
        """Calculate clock stability component score."""
        # Use provided stability score, or compute from std/mean
        if features.clock_stability_score > 0:
            score = features.clock_stability_score
        elif features.clock_mean_mhz > 0:
            clock_cv = (features.clock_std_mhz / features.clock_mean_mhz) * 100
            if clock_cv <= self.THRESHOLDS["clock_cv_excellent"]:
                score = 1.0
            elif clock_cv <= self.THRESHOLDS["clock_cv_good"]:
                score = 0.8
            elif clock_cv <= self.THRESHOLDS["clock_cv_fair"]:
                score = 0.5
            else:
                score = max(0.0, 1.0 - clock_cv / 20)
        else:
            score = 0.7  # Unknown, assume fair

        return CHIComponent(
            name="clock_stability",
            score=score,
            weight=self.weights["clock_stability"],
            description=f"Stability={score:.2f}",
            contributing_metrics=["clock_stability_score", "clock_std_mhz"],
        )

    def _calc_thermal(self, features: SessionFeatures) -> CHIComponent:
        """Calculate thermal component score."""
        temp = features.temp_max_c

        if temp <= 0:
            score = 0.8  # Unknown, assume decent
        elif temp <= self.THRESHOLDS["temp_excellent"]:
            score = 1.0
        elif temp <= self.THRESHOLDS["temp_good"]:
            score = 0.8
        elif temp <= self.THRESHOLDS["temp_fair"]:
            score = 0.5
        else:
            # Above 85C, score degrades quickly
            score = max(0.0, 1.0 - (temp - self.THRESHOLDS["temp_fair"]) / 20)

        return CHIComponent(
            name="thermal",
            score=score,
            weight=self.weights["thermal"],
            description=f"Max temp={temp:.0f}°C" if temp > 0 else "N/A",
            contributing_metrics=["temp_max_c", "temp_mean_c"],
        )

    def _calc_system(self, features: SessionFeatures) -> CHIComponent:
        """Calculate system overhead component score."""
        ctx = features.ctx_switches_per_iter

        if ctx <= self.THRESHOLDS["ctx_excellent"]:
            score = 1.0
        elif ctx <= self.THRESHOLDS["ctx_good"]:
            score = 0.8
        elif ctx <= self.THRESHOLDS["ctx_fair"]:
            score = 0.5
        else:
            score = max(0.0, 1.0 - ctx / 200)

        return CHIComponent(
            name="system",
            score=score,
            weight=self.weights["system"],
            description=f"Ctx switches/iter={ctx:.1f}",
            contributing_metrics=["ctx_switches_per_iter", "page_faults_per_iter"],
        )

    def _get_rating(self, chi: float) -> HealthRating:
        """Get health rating from CHI score."""
        if chi >= 90:
            return HealthRating.EXCELLENT
        elif chi >= 75:
            return HealthRating.GOOD
        elif chi >= 50:
            return HealthRating.FAIR
        elif chi >= 25:
            return HealthRating.POOR
        else:
            return HealthRating.CRITICAL

    def _generate_summary(self, report: CHIReport) -> str:
        """Generate summary text."""
        rating_text = {
            HealthRating.EXCELLENT: "System is performing optimally",
            HealthRating.GOOD: "System is performing well with minor issues",
            HealthRating.FAIR: "System has notable performance issues",
            HealthRating.POOR: "System has significant performance problems",
            HealthRating.CRITICAL: "System has critical performance issues",
        }

        return f"CHI Score: {report.chi_score:.0f}/100 ({report.rating.value}). {rating_text[report.rating]}."

    def _identify_strengths_weaknesses(self, components: List[CHIComponent]) -> tuple:
        """Identify top strengths and weaknesses."""
        sorted_by_score = sorted(components, key=lambda c: c.score, reverse=True)

        strengths = []
        weaknesses = []

        for c in sorted_by_score[:2]:
            if c.score >= 0.8:
                strengths.append(f"{c.name.replace('_', ' ').title()}: {c.description}")

        for c in sorted_by_score[-2:]:
            if c.score < 0.6:
                weaknesses.append(f"{c.name.replace('_', ' ').title()}: {c.description}")

        return strengths, weaknesses


# ============================================================================
# Trend Analysis
# ============================================================================


def compute_chi_trend(
    current_chi: float,
    historical_chis: List[float],
) -> Optional[float]:
    """
    Compute CHI trend (positive = improving).

    Uses simple linear regression on recent CHIs.
    """
    if len(historical_chis) < 3:
        return None

    import numpy as np

    # Recent history + current
    all_chis = historical_chis[-10:] + [current_chi]
    x = np.arange(len(all_chis))

    # Simple linear fit
    slope, _ = np.polyfit(x, all_chis, 1)

    return float(slope)


def get_chi_badge(chi: float) -> str:
    """Get a text badge for CHI score (for CI/CD output)."""
    if chi >= 90:
        return "🟢 EXCELLENT"
    elif chi >= 75:
        return "🟢 GOOD"
    elif chi >= 50:
        return "🟡 FAIR"
    elif chi >= 25:
        return "🟠 POOR"
    else:
        return "🔴 CRITICAL"
