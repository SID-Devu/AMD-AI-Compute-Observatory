"""
AACO-SIGMA Explanation Generator

Generates human-readable explanations of root causes.
Provides multi-level explanations from executive summary to deep technical.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum, auto

from .root_cause_analyzer import RootCause, CauseCategory
from .cause_ranker import RankingResult
from .evidence_collector import EvidenceChain


class ExplanationLevel(Enum):
    """Levels of explanation detail."""

    EXECUTIVE = auto()  # High-level, non-technical
    ENGINEER = auto()  # Technical, actionable
    DEEP_DIVE = auto()  # Full technical details
    DEBUG = auto()  # Maximum detail for debugging


@dataclass
class ActionableInsight:
    """An actionable insight derived from analysis."""

    title: str
    description: str

    # Action
    action: str
    action_complexity: str  # "quick", "moderate", "significant"

    # Impact
    expected_improvement: str
    confidence: float

    # Prerequisites
    prerequisites: List[str] = field(default_factory=list)

    # Related
    related_causes: List[str] = field(default_factory=list)


@dataclass
class Explanation:
    """A generated explanation."""

    # Level
    level: ExplanationLevel

    # Content sections
    summary: str = ""
    details: str = ""
    evidence_narrative: str = ""
    recommendations: List[str] = field(default_factory=list)

    # Insights
    insights: List[ActionableInsight] = field(default_factory=list)

    # Formatting
    format: str = "text"  # text, markdown, html

    # Metadata
    generated_for: str = ""  # Cause ID or "all"


class ExplanationGenerator:
    """
    Generates human-readable explanations for root causes.

    Supports multiple audiences:
    - Executive: Business impact focus
    - Engineer: Technical actionable
    - Deep Dive: Full technical context
    - Debug: Maximum verbosity
    """

    # Category descriptions for executives
    EXECUTIVE_CATEGORY_DESC = {
        CauseCategory.HARDWARE_MEMORY_BANDWIDTH: "Data transfer bottleneck limiting processing speed",
        CauseCategory.HARDWARE_COMPUTE_BOUND: "Compute resources fully utilized",
        CauseCategory.HARDWARE_CACHE_MISS: "Data access inefficiency",
        CauseCategory.HARDWARE_OCCUPANCY: "GPU underutilization",
        CauseCategory.SOFTWARE_KERNEL_INEFFICIENCY: "Suboptimal kernel implementation",
        CauseCategory.SOFTWARE_FUSION_MISSED: "Missed optimization opportunity",
        CauseCategory.CONFIG_BATCH_SIZE: "Suboptimal batch size configuration",
        CauseCategory.CONFIG_PRECISION: "Precision/accuracy trade-off opportunity",
        CauseCategory.MODEL_ARCHITECTURE: "Model design consideration",
    }

    # Technical descriptions for engineers
    ENGINEER_CATEGORY_DESC = {
        CauseCategory.HARDWARE_MEMORY_BANDWIDTH: "Memory-bound: HBM bandwidth saturated while compute units idle",
        CauseCategory.HARDWARE_COMPUTE_BOUND: "Compute-bound: VALU/MFMA utilization high, memory bandwidth available",
        CauseCategory.HARDWARE_CACHE_MISS: "Cache thrashing: High L1/L2 miss rates degrading memory latency",
        CauseCategory.HARDWARE_OCCUPANCY: "Low occupancy: Insufficient waves per CU due to resource constraints",
        CauseCategory.SOFTWARE_KERNEL_INEFFICIENCY: "Kernel inefficiency: Suboptimal instruction mix or vectorization",
        CauseCategory.SOFTWARE_FUSION_MISSED: "Fusion opportunity: Multiple small kernels could be combined",
        CauseCategory.CONFIG_BATCH_SIZE: "Batch size: Small batch underutilizing parallel resources",
        CauseCategory.CONFIG_PRECISION: "Precision: FP32 used where FP16/BF16 would suffice",
    }

    def __init__(self):
        self._templates: Dict[str, str] = {}

    def generate(
        self,
        cause: RootCause,
        level: ExplanationLevel = ExplanationLevel.ENGINEER,
        evidence_chain: Optional[EvidenceChain] = None,
        format: str = "text",
    ) -> Explanation:
        """
        Generate explanation for a single root cause.
        """
        explanation = Explanation(
            level=level,
            generated_for=cause.cause_id,
            format=format,
        )

        if level == ExplanationLevel.EXECUTIVE:
            self._generate_executive(cause, explanation)
        elif level == ExplanationLevel.ENGINEER:
            self._generate_engineer(cause, explanation)
        elif level == ExplanationLevel.DEEP_DIVE:
            self._generate_deep_dive(cause, explanation, evidence_chain)
        else:  # DEBUG
            self._generate_debug(cause, explanation, evidence_chain)

        # Generate insights
        explanation.insights = self._generate_insights(cause)

        # Format output
        if format == "markdown":
            explanation = self._format_markdown(explanation)

        return explanation

    def generate_summary(
        self,
        ranking_result: RankingResult,
        level: ExplanationLevel = ExplanationLevel.ENGINEER,
    ) -> Explanation:
        """
        Generate summary explanation for all ranked causes.
        """
        explanation = Explanation(
            level=level,
            generated_for="all",
        )

        if level == ExplanationLevel.EXECUTIVE:
            explanation.summary = self._executive_summary(ranking_result)
        else:
            explanation.summary = self._engineer_summary(ranking_result)

        # Top recommendations
        for ranked in ranking_result.ranked_causes[:5]:
            if ranked.recommended_action:
                explanation.recommendations.append(f"[P{ranked.rank}] {ranked.recommended_action}")

        return explanation

    def _generate_executive(self, cause: RootCause, explanation: Explanation) -> None:
        """Generate executive-level explanation."""
        category_desc = self.EXECUTIVE_CATEGORY_DESC.get(
            cause.category, "Performance issue detected"
        )

        explanation.summary = (
            f"Performance Issue: {category_desc}\n\n"
            f"Impact: This issue accounts for approximately {cause.impact_pct:.1f}% "
            f"of total processing time.\n\n"
            f"Confidence: {cause.confidence * 100:.0f}% confidence in this diagnosis."
        )

        explanation.recommendations = [fix for fix in cause.suggested_fixes[:2]]

    def _generate_engineer(self, cause: RootCause, explanation: Explanation) -> None:
        """Generate engineer-level explanation."""
        category_desc = self.ENGINEER_CATEGORY_DESC.get(cause.category, cause.category.name)

        explanation.summary = f"**{cause.title}**\n\n{cause.description}"

        explanation.details = (
            f"Category: {category_desc}\n"
            f"Impact: {cause.impact_pct:.1f}% of total latency\n"
            f"Confidence: {cause.confidence * 100:.0f}%\n"
        )

        if cause.kernel_name:
            explanation.details += f"Kernel: {cause.kernel_name}\n"
        if cause.layer_name:
            explanation.details += f"Layer: {cause.layer_name}\n"

        explanation.details += f"Fix Complexity: {cause.fix_complexity}\n"

        explanation.recommendations = cause.suggested_fixes.copy()

    def _generate_deep_dive(
        self,
        cause: RootCause,
        explanation: Explanation,
        evidence_chain: Optional[EvidenceChain],
    ) -> None:
        """Generate deep-dive explanation with full context."""
        self._generate_engineer(cause, explanation)

        # Add evidence narrative
        if evidence_chain:
            explanation.evidence_narrative = self._narrate_evidence(evidence_chain)

        # Add related causes
        if cause.related_causes:
            explanation.details += f"\nRelated Issues: {', '.join(cause.related_causes)}\n"

        # Add technical context based on category
        technical_context = self._get_technical_context(cause)
        if technical_context:
            explanation.details += f"\n{technical_context}"

    def _generate_debug(
        self,
        cause: RootCause,
        explanation: Explanation,
        evidence_chain: Optional[EvidenceChain],
    ) -> None:
        """Generate maximum verbosity debug explanation."""
        self._generate_deep_dive(cause, explanation, evidence_chain)

        # Add all evidence details
        if evidence_chain:
            explanation.details += "\n\n--- Evidence Details ---\n"
            for evidence in evidence_chain.evidence_list:
                explanation.details += (
                    f"\n[{evidence.evidence_id}] {evidence.evidence_type.name}\n"
                    f"  Description: {evidence.description}\n"
                    f"  Value: {evidence.value} {evidence.unit}\n"
                    f"  Source: {evidence.source}\n"
                    f"  Strength: {evidence.strength.name}\n"
                )

    def _narrate_evidence(self, chain: EvidenceChain) -> str:
        """Create narrative from evidence chain."""
        if not chain.evidence_list:
            return "No supporting evidence collected."

        narrative = "Evidence Trail:\n"

        for i, evidence in enumerate(chain.evidence_list, 1):
            narrative += f"{i}. {evidence.description}\n"
            if evidence.threshold_exceeded:
                narrative += "   ⚠️ Threshold exceeded\n"

        narrative += f"\nEvidence chain strength: {chain.chain_strength * 100:.0f}%"

        return narrative

    def _get_technical_context(self, cause: RootCause) -> str:
        """Get technical context for a cause category."""
        contexts = {
            CauseCategory.HARDWARE_MEMORY_BANDWIDTH: (
                "Technical Context:\n"
                "- AMD MI250X: 3.2 TB/s aggregate HBM bandwidth\n"
                "- AMD RX 7900 XTX: 960 GB/s memory bandwidth\n"
                "- Memory-bound kernels are limited by arithmetic intensity\n"
                "- Roofline analysis can quantify potential improvement"
            ),
            CauseCategory.HARDWARE_OCCUPANCY: (
                "Technical Context:\n"
                "- Max waves per CU: 32 (CDNA2/RDNA3)\n"
                "- Occupancy limited by: registers, shared memory, workgroup size\n"
                "- Use rocprof --stats for detailed occupancy analysis"
            ),
            CauseCategory.SOFTWARE_FUSION_MISSED: (
                "Technical Context:\n"
                "- Kernel launch overhead: ~5-10µs per launch\n"
                "- Fusion can reduce launches and intermediate memory\n"
                "- Consider graph mode execution for automatic fusion"
            ),
        }

        return contexts.get(cause.category, "")

    def _generate_insights(self, cause: RootCause) -> List[ActionableInsight]:
        """Generate actionable insights from a cause."""
        insights = []

        for i, fix in enumerate(cause.suggested_fixes):
            complexity_map = {0: "quick", 1: "moderate", 2: "significant"}

            insight = ActionableInsight(
                title=f"Fix Option {i + 1}",
                description=fix,
                action=fix,
                action_complexity=complexity_map.get(i, "moderate"),
                expected_improvement=f"Up to {cause.impact_pct * (0.5 - i * 0.1):.1f}% improvement",
                confidence=cause.confidence * (1 - i * 0.1),
                related_causes=[cause.cause_id],
            )
            insights.append(insight)

        return insights

    def _executive_summary(self, result: RankingResult) -> str:
        """Generate executive summary."""
        return (
            f"Performance Analysis Summary\n"
            f"============================\n\n"
            f"Identified {result.total_causes} performance issues.\n"
            f"- {result.critical_count} critical issues requiring immediate attention\n"
            f"- {result.high_count} high-priority issues\n\n"
            f"Addressable potential improvement: {result.addressable_impact_pct:.1f}% "
            f"of total processing time.\n\n"
            f"Top Actions:\n" + "\n".join(f"  • {action}" for action in result.top_actions[:3])
        )

    def _engineer_summary(self, result: RankingResult) -> str:
        """Generate engineer summary."""
        lines = [
            "Root Cause Analysis Summary",
            "=" * 40,
            "",
            f"Total issues found: {result.total_causes}",
            f"Critical: {result.critical_count}, High: {result.high_count}",
            f"Addressable impact: {result.addressable_impact_pct:.1f}%",
            "",
            "Top Ranked Issues:",
        ]

        for ranked in result.ranked_causes[:5]:
            lines.append(
                f"  #{ranked.rank} [{ranked.priority.upper()}] "
                f"{ranked.cause.title} ({ranked.cause.impact_pct:.1f}%)"
            )

        lines.extend(["", "Recommended Actions:"])
        for action in result.top_actions[:5]:
            lines.append(f"  • {action}")

        return "\n".join(lines)

    def _format_markdown(self, explanation: Explanation) -> Explanation:
        """Format explanation as markdown."""
        if explanation.summary:
            explanation.summary = explanation.summary.replace("\n", "\n\n")

        if explanation.recommendations:
            formatted_recs = []
            for rec in explanation.recommendations:
                formatted_recs.append(f"- {rec}")
            explanation.recommendations = formatted_recs

        return explanation
