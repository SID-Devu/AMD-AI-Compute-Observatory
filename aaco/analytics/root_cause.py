"""
Bayesian Root Cause Ranking System.

Ranks potential performance root causes using Bayesian inference
with probability mass, evidence scoring, and explainability.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


# ============================================================================
# Root Cause Categories
# ============================================================================


class RootCauseCategory(str, Enum):
    """High-level categories of performance root causes."""

    MEMORY_BOUND = "memory_bound"
    COMPUTE_BOUND = "compute_bound"
    LAUNCH_OVERHEAD = "launch_overhead"
    SYNC_OVERHEAD = "sync_overhead"
    DATA_TRANSFER = "data_transfer"
    KERNEL_CONFIG = "kernel_config"
    MEMORY_PATTERN = "memory_pattern"
    CACHE_MISS = "cache_miss"
    OCCUPANCY = "occupancy"
    THERMAL = "thermal"
    DRIVER = "driver"
    FRAGMENTATION = "memory_fragmentation"
    CONTENTION = "resource_contention"
    UNKNOWN = "unknown"


# ============================================================================
# Root Cause Data Models
# ============================================================================


@dataclass
class RootCauseSuspect:
    """
    A suspected root cause with probability and evidence.
    """

    suspect_id: str
    category: RootCauseCategory
    description: str

    # Bayesian scores
    prior_probability: float = 0.1  # Prior belief
    likelihood: float = 0.5  # P(evidence | cause)
    posterior_probability: float = 0.0  # Updated belief

    # Ranking
    rank: int = 0
    confidence: float = 0.0

    # Evidence
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Actionability
    recommended_actions: List[str] = field(default_factory=list)
    estimated_impact: str = ""  # "high", "medium", "low"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["category"] = self.category.value
        return d


@dataclass
class RootCauseRanking:
    """Result of root cause analysis."""

    suspects: List[RootCauseSuspect] = field(default_factory=list)

    # Summary
    primary_cause: Optional[RootCauseSuspect] = None
    total_probability_mass: float = 0.0
    analysis_confidence: float = 0.0

    # Context
    analysis_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suspects": [s.to_dict() for s in self.suspects[:10]],
            "primary_cause": self.primary_cause.to_dict() if self.primary_cause else None,
            "total_probability_mass": self.total_probability_mass,
            "analysis_confidence": self.analysis_confidence,
            "analysis_context": self.analysis_context,
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# Evidence
# ============================================================================


@dataclass
class CauseEvidence:
    """Evidence observation for root cause analysis."""

    name: str
    value: float
    threshold: float
    direction: str  # "above", "below", "equals"
    weight: float = 1.0

    @property
    def is_triggered(self) -> bool:
        if self.direction == "above":
            return self.value > self.threshold
        elif self.direction == "below":
            return self.value < self.threshold
        else:
            return abs(self.value - self.threshold) < 0.01


# ============================================================================
# Bayesian Root Cause Analyzer
# ============================================================================


class BayesianRootCauseAnalyzer:
    """
    Bayesian inference engine for root cause ranking.

    Uses performance metrics as evidence to update beliefs about
    potential root causes.

    The analysis follows Bayes' theorem:
        P(cause | evidence) ∝ P(evidence | cause) × P(cause)

    Where:
        - P(cause) is the prior probability
        - P(evidence | cause) is the likelihood
        - P(cause | evidence) is the posterior probability

    Usage:
        analyzer = BayesianRootCauseAnalyzer()

        # Add observations
        analyzer.add_observation("memory_bandwidth_utilization", 0.95)
        analyzer.add_observation("compute_utilization", 0.30)

        # Run analysis
        ranking = analyzer.analyze()

        print(f"Primary cause: {ranking.primary_cause.description}")
    """

    def __init__(self):
        self._observations: Dict[str, float] = {}
        self._suspects = self._initialize_suspects()
        self._evidence_rules = self._build_evidence_rules()

    def add_observation(self, metric: str, value: float) -> None:
        """Add a performance metric observation."""
        self._observations[metric] = value

    def add_observations(self, observations: Dict[str, float]) -> None:
        """Add multiple observations."""
        self._observations.update(observations)

    def clear_observations(self) -> None:
        """Clear all observations."""
        self._observations.clear()

    def analyze(self) -> RootCauseRanking:
        """
        Run Bayesian root cause analysis.

        Returns:
            Ranked list of suspected root causes
        """
        logger.info("Running Bayesian root cause analysis...")

        # Reset suspects
        suspects = self._initialize_suspects()

        # Compute likelihoods from evidence
        for suspect in suspects:
            likelihood, evidence = self._compute_likelihood(suspect)
            suspect.likelihood = likelihood
            suspect.supporting_evidence = evidence["supporting"]
            suspect.contradicting_evidence = evidence["contradicting"]

        # Apply Bayes' theorem
        # P(cause | evidence) ∝ P(evidence | cause) × P(cause)
        raw_posteriors = []
        for suspect in suspects:
            raw_posterior = suspect.likelihood * suspect.prior_probability
            raw_posteriors.append(raw_posterior)

        # Normalize to get proper probabilities
        total = sum(raw_posteriors)
        if total > 0:
            for i, suspect in enumerate(suspects):
                suspect.posterior_probability = raw_posteriors[i] / total

        # Rank by posterior probability
        suspects.sort(key=lambda s: s.posterior_probability, reverse=True)
        for i, suspect in enumerate(suspects):
            suspect.rank = i + 1
            suspect.confidence = self._compute_confidence(suspect)

        # Build result
        ranking = RootCauseRanking()
        ranking.suspects = suspects
        ranking.total_probability_mass = sum(s.posterior_probability for s in suspects)

        if suspects:
            ranking.primary_cause = suspects[0]
            ranking.analysis_confidence = self._compute_overall_confidence(suspects)

        ranking.analysis_context = {
            "observations": self._observations.copy(),
            "num_suspects": len(suspects),
        }

        logger.info(
            f"Analysis complete. Primary cause: {ranking.primary_cause.description if ranking.primary_cause else 'Unknown'}"
        )

        return ranking

    def _initialize_suspects(self) -> List[RootCauseSuspect]:
        """Initialize the suspect list with priors."""
        return [
            RootCauseSuspect(
                suspect_id="memory_bandwidth",
                category=RootCauseCategory.MEMORY_BOUND,
                description="Kernel is memory bandwidth limited",
                prior_probability=0.25,  # Common cause
                recommended_actions=[
                    "Improve memory access patterns (coalescing)",
                    "Reduce memory footprint",
                    "Use shared memory for data reuse",
                    "Consider data compression",
                ],
                estimated_impact="high",
            ),
            RootCauseSuspect(
                suspect_id="compute_bound",
                category=RootCauseCategory.COMPUTE_BOUND,
                description="Kernel is compute limited",
                prior_probability=0.20,
                recommended_actions=[
                    "Use lower precision (FP16, INT8)",
                    "Algorithm optimization",
                    "Fused kernel operations",
                ],
                estimated_impact="high",
            ),
            RootCauseSuspect(
                suspect_id="launch_overhead",
                category=RootCauseCategory.LAUNCH_OVERHEAD,
                description="Many small kernels with high launch overhead",
                prior_probability=0.15,
                recommended_actions=[
                    "Batch small kernels",
                    "Use kernel fusion",
                    "CUDA graphs / HIP graphs",
                ],
                estimated_impact="medium",
            ),
            RootCauseSuspect(
                suspect_id="sync_stalls",
                category=RootCauseCategory.SYNC_OVERHEAD,
                description="Excessive synchronization causing stalls",
                prior_probability=0.10,
                recommended_actions=[
                    "Reduce explicit synchronization",
                    "Use async operations",
                    "Pipeline data transfers",
                ],
                estimated_impact="medium",
            ),
            RootCauseSuspect(
                suspect_id="pcie_transfer",
                category=RootCauseCategory.DATA_TRANSFER,
                description="PCIe data transfers are bottleneck",
                prior_probability=0.10,
                recommended_actions=[
                    "Use pinned memory",
                    "Overlap transfers with compute",
                    "Reduce data movement",
                ],
                estimated_impact="high",
            ),
            RootCauseSuspect(
                suspect_id="low_occupancy",
                category=RootCauseCategory.OCCUPANCY,
                description="Low GPU occupancy limiting parallelism",
                prior_probability=0.08,
                recommended_actions=[
                    "Adjust block size",
                    "Reduce register usage",
                    "Reduce shared memory usage",
                ],
                estimated_impact="medium",
            ),
            RootCauseSuspect(
                suspect_id="cache_thrashing",
                category=RootCauseCategory.CACHE_MISS,
                description="Poor cache utilization / thrashing",
                prior_probability=0.05,
                recommended_actions=[
                    "Improve data locality",
                    "Use tiling / blocking",
                    "Prefetch data",
                ],
                estimated_impact="medium",
            ),
            RootCauseSuspect(
                suspect_id="thermal_throttle",
                category=RootCauseCategory.THERMAL,
                description="GPU thermal throttling",
                prior_probability=0.03,
                recommended_actions=[
                    "Improve cooling",
                    "Reduce sustained workload",
                    "Check fan speed / thermal paste",
                ],
                estimated_impact="high",
            ),
            RootCauseSuspect(
                suspect_id="memory_fragmentation",
                category=RootCauseCategory.FRAGMENTATION,
                description="GPU memory fragmentation",
                prior_probability=0.02,
                recommended_actions=[
                    "Use memory pools",
                    "Reduce allocation/free cycles",
                    "Preallocate memory",
                ],
                estimated_impact="low",
            ),
            RootCauseSuspect(
                suspect_id="resource_contention",
                category=RootCauseCategory.CONTENTION,
                description="Resource contention with other processes",
                prior_probability=0.02,
                recommended_actions=[
                    "Check for other GPU workloads",
                    "Use exclusive GPU mode",
                    "Isolate workload",
                ],
                estimated_impact="medium",
            ),
        ]

    def _build_evidence_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build evidence rules for each suspect."""
        return {
            "memory_bandwidth": [
                {
                    "metric": "memory_bandwidth_utilization",
                    "threshold": 0.8,
                    "direction": "above",
                    "weight": 2.0,
                },
                {
                    "metric": "compute_utilization",
                    "threshold": 0.5,
                    "direction": "below",
                    "weight": 1.5,
                },
                {
                    "metric": "l2_hit_rate",
                    "threshold": 0.5,
                    "direction": "below",
                    "weight": 1.0,
                },
            ],
            "compute_bound": [
                {
                    "metric": "compute_utilization",
                    "threshold": 0.8,
                    "direction": "above",
                    "weight": 2.0,
                },
                {
                    "metric": "memory_bandwidth_utilization",
                    "threshold": 0.5,
                    "direction": "below",
                    "weight": 1.5,
                },
            ],
            "launch_overhead": [
                {
                    "metric": "avg_kernel_duration_us",
                    "threshold": 10,
                    "direction": "below",
                    "weight": 2.0,
                },
                {
                    "metric": "kernel_count",
                    "threshold": 1000,
                    "direction": "above",
                    "weight": 1.5,
                },
                {
                    "metric": "launch_overhead_pct",
                    "threshold": 0.1,
                    "direction": "above",
                    "weight": 2.0,
                },
            ],
            "sync_stalls": [
                {
                    "metric": "sync_time_pct",
                    "threshold": 0.1,
                    "direction": "above",
                    "weight": 2.0,
                },
                {
                    "metric": "stall_rate",
                    "threshold": 0.2,
                    "direction": "above",
                    "weight": 1.5,
                },
            ],
            "pcie_transfer": [
                {
                    "metric": "transfer_time_pct",
                    "threshold": 0.2,
                    "direction": "above",
                    "weight": 2.0,
                },
                {
                    "metric": "transfer_bandwidth_utilization",
                    "threshold": 0.7,
                    "direction": "above",
                    "weight": 1.5,
                },
            ],
            "low_occupancy": [
                {
                    "metric": "occupancy",
                    "threshold": 0.5,
                    "direction": "below",
                    "weight": 2.0,
                },
                {
                    "metric": "active_warps",
                    "threshold": 0.3,
                    "direction": "below",
                    "weight": 1.5,
                },
            ],
            "cache_thrashing": [
                {
                    "metric": "l2_hit_rate",
                    "threshold": 0.3,
                    "direction": "below",
                    "weight": 2.0,
                },
                {
                    "metric": "l1_hit_rate",
                    "threshold": 0.5,
                    "direction": "below",
                    "weight": 1.5,
                },
            ],
            "thermal_throttle": [
                {
                    "metric": "gpu_temperature_c",
                    "threshold": 85,
                    "direction": "above",
                    "weight": 2.0,
                },
                {
                    "metric": "clock_throttle_detected",
                    "threshold": 0.5,
                    "direction": "above",
                    "weight": 2.5,
                },
            ],
            "memory_fragmentation": [
                {
                    "metric": "allocation_failure_rate",
                    "threshold": 0.01,
                    "direction": "above",
                    "weight": 2.0,
                },
                {
                    "metric": "memory_fragmentation_score",
                    "threshold": 0.3,
                    "direction": "above",
                    "weight": 1.5,
                },
            ],
            "resource_contention": [
                {
                    "metric": "gpu_utilization_variance",
                    "threshold": 0.3,
                    "direction": "above",
                    "weight": 1.5,
                },
                {
                    "metric": "other_processes_gpu_pct",
                    "threshold": 0.1,
                    "direction": "above",
                    "weight": 2.0,
                },
            ],
        }

    def _compute_likelihood(self, suspect: RootCauseSuspect) -> Tuple[float, Dict[str, List[str]]]:
        """
        Compute likelihood P(evidence | cause) for a suspect.

        Returns:
            Tuple of (likelihood, evidence_dict)
        """
        rules = self._evidence_rules.get(suspect.suspect_id, [])

        if not rules:
            return 0.5, {"supporting": [], "contradicting": []}

        supporting = []
        contradicting = []
        likelihood_scores = []

        for rule in rules:
            metric = rule["metric"]
            if metric not in self._observations:
                continue

            value = self._observations[metric]
            threshold = rule["threshold"]
            direction = rule["direction"]
            weight = rule["weight"]

            # Check if evidence supports or contradicts
            evidence = CauseEvidence(metric, value, threshold, direction, weight)

            if evidence.is_triggered:
                supporting.append(f"{metric}={value:.3f} ({direction} {threshold})")
                likelihood_scores.append((1.0, weight))
            else:
                contradicting.append(f"{metric}={value:.3f} (NOT {direction} {threshold})")
                likelihood_scores.append((0.2, weight))  # Low but not zero

        # Weighted average of likelihood scores
        if likelihood_scores:
            total_weight = sum(w for _, w in likelihood_scores)
            likelihood = sum(s * w for s, w in likelihood_scores) / total_weight
        else:
            likelihood = 0.5  # No evidence, neutral

        return likelihood, {"supporting": supporting, "contradicting": contradicting}

    def _compute_confidence(self, suspect: RootCauseSuspect) -> float:
        """Compute confidence in the suspect ranking."""
        # Confidence based on:
        # 1. Amount of evidence
        # 2. Posterior probability
        # 3. Evidence consistency

        evidence_count = len(suspect.supporting_evidence)
        contradict_count = len(suspect.contradicting_evidence)

        if evidence_count + contradict_count == 0:
            return 0.5  # No evidence

        # Evidence ratio
        evidence_ratio = evidence_count / (evidence_count + contradict_count)

        # Combine with posterior
        confidence = 0.5 * evidence_ratio + 0.5 * suspect.posterior_probability

        return min(confidence, 0.99)  # Cap at 99%

    def _compute_overall_confidence(self, suspects: List[RootCauseSuspect]) -> float:
        """Compute overall analysis confidence."""
        if not suspects:
            return 0.0

        # Higher confidence if:
        # 1. Clear winner (top suspect much higher than #2)
        # 2. Multiple supporting evidence

        top = suspects[0]
        second = suspects[1] if len(suspects) > 1 else None

        # Separation between top causes
        if second:
            separation = top.posterior_probability - second.posterior_probability
        else:
            separation = top.posterior_probability

        # Evidence count
        total_evidence = len(top.supporting_evidence)

        # Combined confidence
        confidence = 0.5 * min(separation * 2, 1.0) + 0.5 * min(total_evidence / 5, 1.0)

        return min(confidence, 0.95)


# ============================================================================
# Quick Analysis Functions
# ============================================================================


def quick_root_cause_analysis(metrics: Dict[str, float]) -> RootCauseRanking:
    """
    Quick root cause analysis from metrics dictionary.

    Args:
        metrics: Dictionary of metric name -> value

    Returns:
        Root cause ranking
    """
    analyzer = BayesianRootCauseAnalyzer()
    analyzer.add_observations(metrics)
    return analyzer.analyze()


def explain_root_cause(ranking: RootCauseRanking) -> str:
    """Generate human-readable explanation of root cause analysis."""
    if not ranking.primary_cause:
        return "Unable to determine root cause. Insufficient evidence."

    lines = []
    cause = ranking.primary_cause

    lines.append(f"PRIMARY ROOT CAUSE: {cause.description}")
    lines.append(f"Probability: {cause.posterior_probability:.1%}")
    lines.append(f"Confidence: {cause.confidence:.1%}")
    lines.append("")

    if cause.supporting_evidence:
        lines.append("Supporting Evidence:")
        for ev in cause.supporting_evidence[:5]:
            lines.append(f"  • {ev}")
        lines.append("")

    if cause.recommended_actions:
        lines.append("Recommended Actions:")
        for action in cause.recommended_actions[:5]:
            lines.append(f"  → {action}")
        lines.append("")

    # Secondary causes
    if len(ranking.suspects) > 1:
        lines.append("Other Possible Causes:")
        for suspect in ranking.suspects[1:4]:
            lines.append(
                f"  {suspect.rank}. {suspect.description} ({suspect.posterior_probability:.1%})"
            )

    return "\n".join(lines)
