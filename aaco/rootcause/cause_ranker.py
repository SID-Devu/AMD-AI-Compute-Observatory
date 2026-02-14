"""
AACO-SIGMA Cause Ranker

Ranks and prioritizes identified root causes.
Uses multi-criteria ranking for actionability.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum, auto

from .root_cause_analyzer import RootCause, CauseCategory


class RankingCriteria(Enum):
    """Criteria for ranking root causes."""
    IMPACT = auto()            # Performance impact percentage
    CONFIDENCE = auto()        # Detection confidence
    FIX_COMPLEXITY = auto()    # Ease of fixing
    FIX_ROI = auto()          # Return on investment (impact/complexity)
    ACTIONABILITY = auto()     # How actionable is the fix


@dataclass
class RankedCause:
    """A root cause with ranking information."""
    cause: RootCause
    
    # Scores (0-1)
    impact_score: float = 0.0
    confidence_score: float = 0.0
    complexity_score: float = 0.0  # Lower is better
    roi_score: float = 0.0
    actionability_score: float = 0.0
    
    # Aggregate
    overall_score: float = 0.0
    rank: int = 0
    
    # Recommendation
    priority: str = ""  # "critical", "high", "medium", "low"
    recommended_action: str = ""


@dataclass
class RankingResult:
    """Result of cause ranking."""
    ranked_causes: List[RankedCause] = field(default_factory=list)
    
    # Summary
    total_causes: int = 0
    critical_count: int = 0
    high_count: int = 0
    
    # Top recommendations
    top_actions: List[str] = field(default_factory=list)
    
    # Expected improvement
    addressable_impact_pct: float = 0.0


class CauseRanker:
    """
    Ranks root causes by priority and actionability.
    
    Considers:
    - Impact: How much time/performance is affected
    - Confidence: How certain we are of the diagnosis
    - Complexity: How hard is it to fix
    - ROI: Impact vs complexity ratio
    """
    
    # Complexity mappings
    COMPLEXITY_SCORES = {
        "low": 0.2,
        "medium": 0.5,
        "high": 0.8,
        "varies": 0.5,
        "": 0.5,
    }
    
    # Category actionability (how easy to address)
    CATEGORY_ACTIONABILITY = {
        CauseCategory.CONFIG_BATCH_SIZE: 0.9,
        CauseCategory.CONFIG_PRECISION: 0.85,
        CauseCategory.CONFIG_WORKGROUP_SIZE: 0.8,
        CauseCategory.SOFTWARE_FUSION_MISSED: 0.7,
        CauseCategory.SOFTWARE_MEMORY_PATTERN: 0.6,
        CauseCategory.HARDWARE_MEMORY_BANDWIDTH: 0.5,
        CauseCategory.HARDWARE_CACHE_MISS: 0.6,
        CauseCategory.HARDWARE_OCCUPANCY: 0.65,
        CauseCategory.SOFTWARE_SYNCHRONIZATION: 0.4,
        CauseCategory.MODEL_ARCHITECTURE: 0.2,
        CauseCategory.SYSTEM_DRIVER: 0.3,
    }
    
    # Default weights
    DEFAULT_WEIGHTS = {
        RankingCriteria.IMPACT: 0.35,
        RankingCriteria.CONFIDENCE: 0.2,
        RankingCriteria.FIX_COMPLEXITY: 0.15,
        RankingCriteria.ACTIONABILITY: 0.3,
    }
    
    def __init__(self, weights: Optional[Dict[RankingCriteria, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self._custom_scorers: Dict[RankingCriteria, Callable] = {}
    
    def rank(self, causes: List[RootCause]) -> RankingResult:
        """
        Rank a list of root causes.
        
        Returns ranked causes with scores and recommendations.
        """
        if not causes:
            return RankingResult()
        
        # Score each cause
        ranked = []
        for cause in causes:
            ranked_cause = self._score_cause(cause)
            ranked.append(ranked_cause)
        
        # Sort by overall score
        ranked.sort(key=lambda r: r.overall_score, reverse=True)
        
        # Assign ranks
        for i, rc in enumerate(ranked):
            rc.rank = i + 1
            rc.priority = self._compute_priority(rc)
            rc.recommended_action = self._recommend_action(rc)
        
        # Build result
        result = RankingResult(
            ranked_causes=ranked,
            total_causes=len(ranked),
            critical_count=sum(1 for r in ranked if r.priority == "critical"),
            high_count=sum(1 for r in ranked if r.priority == "high"),
        )
        
        # Top actions from highest-ranked causes
        result.top_actions = [
            rc.recommended_action for rc in ranked[:5] if rc.recommended_action
        ]
        
        # Sum addressable impact
        result.addressable_impact_pct = sum(
            rc.cause.impact_pct * rc.actionability_score 
            for rc in ranked
        )
        
        return result
    
    def _score_cause(self, cause: RootCause) -> RankedCause:
        """Score a single cause across all criteria."""
        ranked = RankedCause(cause=cause)
        
        # Impact score (normalized, assuming max 50% single-cause impact)
        ranked.impact_score = min(1.0, cause.impact_pct / 50.0)
        
        # Confidence score (already 0-1)
        ranked.confidence_score = cause.confidence
        
        # Complexity score (inverted - lower complexity is better)
        complexity = self.COMPLEXITY_SCORES.get(cause.fix_complexity, 0.5)
        ranked.complexity_score = 1.0 - complexity
        
        # Actionability score
        ranked.actionability_score = self.CATEGORY_ACTIONABILITY.get(
            cause.category, 0.5
        )
        
        # ROI score (impact / complexity)
        if complexity > 0:
            ranked.roi_score = ranked.impact_score / complexity
        else:
            ranked.roi_score = ranked.impact_score
        ranked.roi_score = min(1.0, ranked.roi_score)
        
        # Apply custom scorers
        for criteria, scorer in self._custom_scorers.items():
            score = scorer(cause)
            if criteria == RankingCriteria.IMPACT:
                ranked.impact_score = score
            elif criteria == RankingCriteria.CONFIDENCE:
                ranked.confidence_score = score
            elif criteria == RankingCriteria.FIX_COMPLEXITY:
                ranked.complexity_score = score
            elif criteria == RankingCriteria.ACTIONABILITY:
                ranked.actionability_score = score
        
        # Weighted aggregate
        ranked.overall_score = (
            self.weights.get(RankingCriteria.IMPACT, 0) * ranked.impact_score +
            self.weights.get(RankingCriteria.CONFIDENCE, 0) * ranked.confidence_score +
            self.weights.get(RankingCriteria.FIX_COMPLEXITY, 0) * ranked.complexity_score +
            self.weights.get(RankingCriteria.ACTIONABILITY, 0) * ranked.actionability_score
        )
        
        return ranked
    
    def _compute_priority(self, ranked: RankedCause) -> str:
        """Compute priority level from scores."""
        if ranked.overall_score > 0.8:
            return "critical"
        elif ranked.overall_score > 0.6:
            return "high"
        elif ranked.overall_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _recommend_action(self, ranked: RankedCause) -> str:
        """Generate a recommended action string."""
        cause = ranked.cause
        
        if cause.suggested_fixes:
            # Pick the best fix based on complexity and actionability
            if ranked.complexity_score > 0.7:  # Easy fixes preferred
                return f"[Quick Win] {cause.suggested_fixes[0]}"
            else:
                return f"[Optimize] {cause.suggested_fixes[0]}"
        
        # Default action by category
        category_actions = {
            CauseCategory.HARDWARE_MEMORY_BANDWIDTH: "Optimize memory access patterns",
            CauseCategory.HARDWARE_COMPUTE_BOUND: "Check for vectorization opportunities",
            CauseCategory.CONFIG_BATCH_SIZE: "Increase batch size",
            CauseCategory.SOFTWARE_FUSION_MISSED: "Enable operator fusion",
        }
        
        return category_actions.get(cause.category, "Investigate further")
    
    def register_scorer(self, criteria: RankingCriteria, 
                        scorer: Callable[[RootCause], float]) -> None:
        """Register a custom scoring function."""
        self._custom_scorers[criteria] = scorer
    
    def get_ranking_explanation(self, ranked: RankedCause) -> str:
        """Generate explanation of ranking."""
        return (
            f"Rank #{ranked.rank} (score: {ranked.overall_score:.2f})\n"
            f"  Impact: {ranked.impact_score:.2f} ({ranked.cause.impact_pct:.1f}%)\n"
            f"  Confidence: {ranked.confidence_score:.2f}\n"
            f"  Fix Ease: {ranked.complexity_score:.2f}\n"
            f"  Actionability: {ranked.actionability_score:.2f}\n"
            f"  Priority: {ranked.priority}\n"
            f"  Recommended: {ranked.recommended_action}"
        )
    
    def filter_actionable(self, result: RankingResult,
                          min_actionability: float = 0.5) -> List[RankedCause]:
        """Filter to only actionable causes."""
        return [
            rc for rc in result.ranked_causes
            if rc.actionability_score >= min_actionability
        ]
    
    def group_by_category(self, result: RankingResult) -> Dict[CauseCategory, List[RankedCause]]:
        """Group ranked causes by category."""
        groups: Dict[CauseCategory, List[RankedCause]] = {}
        
        for rc in result.ranked_causes:
            cat = rc.cause.category
            if cat not in groups:
                groups[cat] = []
            groups[cat].append(rc)
        
        return groups
