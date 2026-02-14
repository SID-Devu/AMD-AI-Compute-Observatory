"""
AACO-SIGMA Root Cause Ranking RCR++

Automated root cause analysis and ranking for performance issues.
Identifies, prioritizes, and explains performance bottlenecks.
"""

from .root_cause_analyzer import (
    RootCauseAnalyzer,
    RootCause,
    CauseCategory,
    AnalysisContext,
)

from .cause_ranker import (
    CauseRanker,
    RankedCause,
    RankingCriteria,
    RankingResult,
)

from .evidence_collector import (
    EvidenceCollector,
    Evidence,
    EvidenceType,
    EvidenceChain,
)

from .explanation_generator import (
    ExplanationGenerator,
    Explanation,
    ExplanationLevel,
    ActionableInsight,
)

__all__ = [
    # Root cause analysis
    "RootCauseAnalyzer",
    "RootCause",
    "CauseCategory",
    "AnalysisContext",
    # Cause ranking
    "CauseRanker",
    "RankedCause",
    "RankingCriteria",
    "RankingResult",
    # Evidence collection
    "EvidenceCollector",
    "Evidence",
    "EvidenceType",
    "EvidenceChain",
    # Explanation generation
    "ExplanationGenerator",
    "Explanation",
    "ExplanationLevel",
    "ActionableInsight",
]
