# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Bayesian Root Cause Engine

Probabilistic root cause ranking with:
- Prior probabilities from historical data
- Likelihood computation from evidence
- Posterior ranking via Bayes' theorem
- Multi-cause disambiguation
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RootCauseCategory(Enum):
    """Categories of performance root causes."""

    # Kernel-level
    LAUNCH_OVERHEAD = "launch_overhead"
    KERNEL_INEFFICIENCY = "kernel_inefficiency"
    WAVEFRONT_UNDERUTIL = "wavefront_underutilization"

    # Memory-level
    MEMORY_BANDWIDTH = "memory_bandwidth"
    CACHE_THRASHING = "cache_thrashing"
    LDS_CONTENTION = "lds_contention"

    # Compute-level
    ALU_BOTTLENECK = "alu_bottleneck"
    REGISTER_SPILLING = "register_spilling"
    OCCUPANCY_LIMITED = "occupancy_limited"

    # System-level
    CPU_SCHEDULING = "cpu_scheduling"
    THERMAL_THROTTLING = "thermal_throttling"
    POWER_LIMIT = "power_limit"

    # Graph-level
    PARTITION_OVERHEAD = "partition_overhead"
    TRANSFER_BOUND = "transfer_bound"
    FUSION_MISS = "fusion_miss"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class Evidence:
    """Single piece of evidence for root cause analysis."""

    name: str = ""
    value: float = 0.0
    threshold: float = 0.0
    is_anomalous: bool = False
    weight: float = 1.0


@dataclass
class RootCausePrior:
    """Prior probability for a root cause category."""

    category: RootCauseCategory = RootCauseCategory.UNKNOWN
    probability: float = 0.1  # Base prior
    historical_frequency: float = 0.0
    last_occurrence_count: int = 0


@dataclass
class RootCausePosterior:
    """
    Posterior probability for root cause.

    P(Cause|Evidence) ∝ P(Evidence|Cause) × P(Cause)
    """

    category: RootCauseCategory = RootCauseCategory.UNKNOWN

    # Probabilities
    prior: float = 0.1
    likelihood: float = 0.5
    posterior: float = 0.0

    # Normalized rank
    rank: int = 0

    # Supporting evidence
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)

    # Confidence in posterior
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "posterior": f"{self.posterior:.3f}",
            "rank": self.rank,
            "prior": f"{self.prior:.3f}",
            "likelihood": f"{self.likelihood:.3f}",
            "confidence": f"{self.confidence:.2f}",
            "supporting_evidence": self.supporting_evidence,
        }


@dataclass
class RootCauseAnalysis:
    """Complete root cause analysis result."""

    # Top causes
    ranked_causes: List[RootCausePosterior] = field(default_factory=list)

    # Best guess
    primary_cause: RootCauseCategory = RootCauseCategory.UNKNOWN
    primary_confidence: float = 0.0

    # Multi-cause possibility
    is_multi_cause: bool = False
    secondary_causes: List[RootCauseCategory] = field(default_factory=list)

    # Evidence summary
    total_evidence_count: int = 0
    anomalous_evidence_count: int = 0

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class BayesianRootCauseEngine:
    """
    AACO-Ω∞ Bayesian Root Cause Engine

    Ranks root causes by posterior probability.

    P(Cause|Evidence) = P(Evidence|Cause) × P(Cause) / P(Evidence)

    Features:
    - Configurable priors from historical data
    - Likelihood models per evidence type
    - Multi-cause detection
    - Confidence scoring
    """

    # Default priors based on typical AMD GPU workloads
    DEFAULT_PRIORS = {
        RootCauseCategory.LAUNCH_OVERHEAD: 0.15,
        RootCauseCategory.MEMORY_BANDWIDTH: 0.20,
        RootCauseCategory.KERNEL_INEFFICIENCY: 0.10,
        RootCauseCategory.CACHE_THRASHING: 0.08,
        RootCauseCategory.ALU_BOTTLENECK: 0.12,
        RootCauseCategory.OCCUPANCY_LIMITED: 0.08,
        RootCauseCategory.PARTITION_OVERHEAD: 0.07,
        RootCauseCategory.THERMAL_THROTTLING: 0.05,
        RootCauseCategory.CPU_SCHEDULING: 0.05,
        RootCauseCategory.TRANSFER_BOUND: 0.05,
        RootCauseCategory.FUSION_MISS: 0.03,
        RootCauseCategory.UNKNOWN: 0.02,
    }

    # Evidence-to-cause likelihood mappings
    # P(Evidence|Cause) - how likely to see evidence if cause is true
    LIKELIHOOD_MAP = {
        "high_kar": {
            RootCauseCategory.LAUNCH_OVERHEAD: 0.9,
            RootCauseCategory.FUSION_MISS: 0.7,
            RootCauseCategory.KERNEL_INEFFICIENCY: 0.4,
        },
        "high_lts": {
            RootCauseCategory.LAUNCH_OVERHEAD: 0.85,
            RootCauseCategory.PARTITION_OVERHEAD: 0.5,
        },
        "low_occupancy": {
            RootCauseCategory.OCCUPANCY_LIMITED: 0.9,
            RootCauseCategory.REGISTER_SPILLING: 0.6,
            RootCauseCategory.LDS_CONTENTION: 0.5,
        },
        "high_memory_intensity": {
            RootCauseCategory.MEMORY_BANDWIDTH: 0.9,
            RootCauseCategory.CACHE_THRASHING: 0.6,
        },
        "low_cache_hit": {
            RootCauseCategory.CACHE_THRASHING: 0.9,
            RootCauseCategory.MEMORY_BANDWIDTH: 0.7,
        },
        "high_valu_util": {
            RootCauseCategory.ALU_BOTTLENECK: 0.9,
        },
        "thermal_spike": {
            RootCauseCategory.THERMAL_THROTTLING: 0.95,
        },
        "clock_drop": {
            RootCauseCategory.THERMAL_THROTTLING: 0.8,
            RootCauseCategory.POWER_LIMIT: 0.7,
        },
        "high_sii": {
            RootCauseCategory.CPU_SCHEDULING: 0.85,
        },
        "many_partitions": {
            RootCauseCategory.PARTITION_OVERHEAD: 0.8,
            RootCauseCategory.TRANSFER_BOUND: 0.6,
        },
        "high_transfer_time": {
            RootCauseCategory.TRANSFER_BOUND: 0.9,
            RootCauseCategory.PARTITION_OVERHEAD: 0.5,
        },
        "short_kernels": {
            RootCauseCategory.LAUNCH_OVERHEAD: 0.8,
            RootCauseCategory.FUSION_MISS: 0.6,
        },
        "kernel_variance": {
            RootCauseCategory.KERNEL_INEFFICIENCY: 0.7,
            RootCauseCategory.THERMAL_THROTTLING: 0.4,
        },
    }

    def __init__(self):
        """Initialize Bayesian root cause engine."""
        self._priors: Dict[RootCauseCategory, float] = dict(self.DEFAULT_PRIORS)
        self._historical_counts: Dict[RootCauseCategory, int] = {}

    def set_prior(
        self,
        category: RootCauseCategory,
        probability: float,
    ) -> None:
        """Set prior probability for a cause category."""
        self._priors[category] = probability

    def update_priors_from_history(
        self,
        historical_causes: List[RootCauseCategory],
    ) -> None:
        """
        Update priors based on historical root cause detections.

        Uses Bayesian updating with pseudo-counts.

        Args:
            historical_causes: List of historically detected causes
        """
        # Count occurrences
        counts: Dict[RootCauseCategory, int] = {}
        for cause in historical_causes:
            counts[cause] = counts.get(cause, 0) + 1

        total = len(historical_causes)
        if total == 0:
            return

        # Update priors with smoothing
        alpha = 1.0  # Laplace smoothing
        for category in RootCauseCategory:
            count = counts.get(category, 0)
            self._priors[category] = (count + alpha) / (total + alpha * len(RootCauseCategory))
            self._historical_counts[category] = count

        logger.info(f"Updated priors from {total} historical samples")

    def analyze(
        self,
        evidence: List[Evidence],
    ) -> RootCauseAnalysis:
        """
        Analyze evidence and rank root causes by posterior probability.

        Args:
            evidence: List of evidence observations

        Returns:
            RootCauseAnalysis with ranked causes
        """
        analysis = RootCauseAnalysis(
            total_evidence_count=len(evidence),
            anomalous_evidence_count=sum(1 for e in evidence if e.is_anomalous),
        )

        # Compute posteriors for all causes
        posteriors: Dict[RootCauseCategory, RootCausePosterior] = {}

        for category in RootCauseCategory:
            posterior = self._compute_posterior(category, evidence)
            posteriors[category] = posterior

        # Normalize posteriors (sum to 1)
        total_unnorm = sum(p.posterior for p in posteriors.values())
        if total_unnorm > 0:
            for posterior in posteriors.values():
                posterior.posterior /= total_unnorm

        # Rank by posterior
        ranked = sorted(
            posteriors.values(),
            key=lambda p: p.posterior,
            reverse=True,
        )

        for i, posterior in enumerate(ranked):
            posterior.rank = i + 1
            posterior.confidence = self._compute_confidence(posterior, ranked)

        analysis.ranked_causes = ranked

        # Primary cause
        if ranked:
            analysis.primary_cause = ranked[0].category
            analysis.primary_confidence = ranked[0].confidence

            # Check for multi-cause
            if len(ranked) > 1 and ranked[1].posterior > 0.25:
                analysis.is_multi_cause = True
                analysis.secondary_causes = [p.category for p in ranked[1:4] if p.posterior > 0.15]

        # Generate recommendations
        analysis.recommendations = self._generate_recommendations(analysis)

        return analysis

    def _compute_posterior(
        self,
        category: RootCauseCategory,
        evidence: List[Evidence],
    ) -> RootCausePosterior:
        """
        Compute posterior probability for a cause category.

        P(Cause|Evidence) ∝ P(Evidence|Cause) × P(Cause)
        """
        posterior = RootCausePosterior(
            category=category,
            prior=self._priors.get(category, 0.1),
        )

        # Compute likelihood as product of evidence likelihoods
        log_likelihood = 0.0

        for ev in evidence:
            if not ev.is_anomalous:
                continue

            # Get likelihood from map
            ev_likelihoods = self.LIKELIHOOD_MAP.get(ev.name, {})
            cause_likelihood = ev_likelihoods.get(category, 0.1)  # Default low

            # Apply evidence weight
            weighted_likelihood = cause_likelihood**ev.weight

            if weighted_likelihood > 0:
                log_likelihood += math.log(weighted_likelihood)

            # Track supporting evidence
            if cause_likelihood > 0.5:
                posterior.supporting_evidence.append(ev.name)
            elif cause_likelihood < 0.3:
                posterior.contradicting_evidence.append(ev.name)

        # Convert log-likelihood back
        posterior.likelihood = math.exp(log_likelihood) if log_likelihood != 0 else 0.5

        # Unnormalized posterior
        posterior.posterior = posterior.likelihood * posterior.prior

        return posterior

    def _compute_confidence(
        self,
        posterior: RootCausePosterior,
        all_posteriors: List[RootCausePosterior],
    ) -> float:
        """Compute confidence in posterior estimate."""
        # Base confidence from posterior value
        confidence = posterior.posterior

        # Boost if clearly dominant
        if len(all_posteriors) > 1:
            gap = posterior.posterior - all_posteriors[1].posterior
            if gap > 0.2:
                confidence += 0.1

        # Boost from supporting evidence
        confidence += len(posterior.supporting_evidence) * 0.05

        # Reduce for contradicting evidence
        confidence -= len(posterior.contradicting_evidence) * 0.1

        return max(0.1, min(0.95, confidence))

    def _generate_recommendations(
        self,
        analysis: RootCauseAnalysis,
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = {
            RootCauseCategory.LAUNCH_OVERHEAD: [
                "Enable kernel fusion to reduce launch overhead",
                "Check for excessive operator decomposition",
                "Consider increasing batch size to amortize launch costs",
            ],
            RootCauseCategory.MEMORY_BANDWIDTH: [
                "Optimize memory access patterns for coalescing",
                "Enable memory-efficient variants of operators",
                "Check tensor layouts for cache efficiency",
            ],
            RootCauseCategory.CACHE_THRASHING: [
                "Reduce working set size per kernel",
                "Enable cache-optimized kernel variants",
                "Consider tiling strategies",
            ],
            RootCauseCategory.OCCUPANCY_LIMITED: [
                "Reduce register usage in kernels",
                "Check for excessive LDS allocation",
                "Consider smaller workgroup sizes",
            ],
            RootCauseCategory.ALU_BOTTLENECK: [
                "Model is compute-bound - good efficiency",
                "Consider lower precision (FP16/INT8) if accuracy permits",
                "Check for redundant computations",
            ],
            RootCauseCategory.THERMAL_THROTTLING: [
                "Improve system cooling",
                "Add thermal breaks in workload",
                "Monitor junction temperature",
            ],
            RootCauseCategory.PARTITION_OVERHEAD: [
                "Reduce number of execution provider transitions",
                "Optimize graph partitioning",
                "Consider single-EP execution if possible",
            ],
            RootCauseCategory.TRANSFER_BOUND: [
                "Minimize host-device transfers",
                "Use pinned memory for transfers",
                "Pipeline transfers with computation",
            ],
            RootCauseCategory.CPU_SCHEDULING: [
                "Use CPU affinity for critical threads",
                "Reduce system background load",
                "Consider real-time scheduling priorities",
            ],
        }

        if analysis.primary_cause in recommendations:
            return recommendations[analysis.primary_cause]

        return ["Gather additional profiling data for root cause analysis"]

    def get_priors(self) -> Dict[str, float]:
        """Get current prior probabilities."""
        return {k.value: v for k, v in self._priors.items()}


def create_root_cause_evidence(
    kar: float = 1.0,
    lts: float = 0.0,
    pfi: float = 0.0,
    sii: float = 0.0,
    occupancy: float = 1.0,
    cache_hit_rate: float = 0.9,
    memory_intensity: float = 0.0,
    valu_util: float = 0.0,
    thermal_state: str = "normal",
) -> List[Evidence]:
    """
    Create evidence list from metric values.

    Args:
        kar: Kernel Amplification Ratio
        lts: Launch Tax Score
        pfi: Partition Fragmentation Index
        sii: Scheduler Interference Index
        occupancy: GPU occupancy (0-1)
        cache_hit_rate: L2 cache hit rate (0-1)
        memory_intensity: Memory bandwidth usage
        valu_util: VALU utilization (0-1)
        thermal_state: Thermal state (normal, warning, throttling)

    Returns:
        List of Evidence for root cause analysis
    """
    evidence = []

    # KAR evidence
    if kar > 3:
        evidence.append(
            Evidence(
                name="high_kar",
                value=kar,
                threshold=3.0,
                is_anomalous=True,
                weight=1.0,
            )
        )

    # LTS evidence
    if lts > 0.2:
        evidence.append(
            Evidence(
                name="high_lts",
                value=lts,
                threshold=0.2,
                is_anomalous=True,
                weight=1.0,
            )
        )

    # Occupancy evidence
    if occupancy < 0.5:
        evidence.append(
            Evidence(
                name="low_occupancy",
                value=occupancy,
                threshold=0.5,
                is_anomalous=True,
                weight=1.0,
            )
        )

    # Cache hit evidence
    if cache_hit_rate < 0.7:
        evidence.append(
            Evidence(
                name="low_cache_hit",
                value=cache_hit_rate,
                threshold=0.7,
                is_anomalous=True,
                weight=0.8,
            )
        )

    # Memory intensity evidence
    if memory_intensity > 10:
        evidence.append(
            Evidence(
                name="high_memory_intensity",
                value=memory_intensity,
                threshold=10.0,
                is_anomalous=True,
                weight=1.0,
            )
        )

    # VALU util evidence
    if valu_util > 0.8:
        evidence.append(
            Evidence(
                name="high_valu_util",
                value=valu_util,
                threshold=0.8,
                is_anomalous=True,
                weight=1.0,
            )
        )

    # Scheduler evidence
    if sii > 0.3:
        evidence.append(
            Evidence(
                name="high_sii",
                value=sii,
                threshold=0.3,
                is_anomalous=True,
                weight=0.7,
            )
        )

    # Partition evidence
    if pfi > 0.5:
        evidence.append(
            Evidence(
                name="many_partitions",
                value=pfi,
                threshold=0.5,
                is_anomalous=True,
                weight=0.8,
            )
        )

    # Thermal evidence
    if thermal_state == "throttling":
        evidence.append(
            Evidence(
                name="thermal_spike",
                value=1.0,
                threshold=0.5,
                is_anomalous=True,
                weight=1.2,
            )
        )

    return evidence


def create_bayesian_engine() -> BayesianRootCauseEngine:
    """Factory function to create Bayesian root cause engine."""
    return BayesianRootCauseEngine()
