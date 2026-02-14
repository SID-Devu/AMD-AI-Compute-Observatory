"""
AACO-SIGMA Rule Engine

Engine for matching and applying optimization rules.
Coordinates rule evaluation, conflict resolution, and application ordering.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set
from enum import Enum, auto
import re

from .optimization_rule import (
    OptimizationRule,
    RuleCondition,
    RuleAction,
    RuleCategory,
    RulePriority,
    RULE_CATALOG,
)


class MatchResult(Enum):
    """Result of rule matching."""
    FULL_MATCH = auto()     # All conditions matched
    PARTIAL_MATCH = auto()  # Some conditions matched
    NO_MATCH = auto()       # No conditions matched
    CONFLICT = auto()       # Conflicts with higher priority rule


@dataclass
class RuleMatch:
    """A matched rule with context."""
    
    rule: OptimizationRule
    result: MatchResult
    
    # Match details
    matched_conditions: List[RuleCondition] = field(default_factory=list)
    failed_conditions: List[RuleCondition] = field(default_factory=list)
    match_confidence: float = 0.0
    
    # Context
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Target
    target_kernel: Optional[str] = None
    target_layer: Optional[str] = None
    
    @property
    def is_full_match(self) -> bool:
        return self.result == MatchResult.FULL_MATCH


@dataclass
class RuleResult:
    """Result of applying optimization rules."""
    
    # Matches
    matches: List[RuleMatch] = field(default_factory=list)
    
    # By category
    matches_by_category: Dict[RuleCategory, List[RuleMatch]] = field(default_factory=dict)
    
    # Conflicts
    conflicts: List[tuple] = field(default_factory=list)  # (rule1, rule2, reason)
    
    # Actions to take
    ordered_actions: List[tuple] = field(default_factory=list)  # (rule, action)
    
    # Stats
    total_rules_evaluated: int = 0
    full_matches: int = 0
    partial_matches: int = 0
    
    @property
    def has_matches(self) -> bool:
        return self.full_matches > 0


class RuleEngine:
    """
    Engine for matching and applying optimization rules.
    
    Features:
    - Rule matching against performance context
    - Conflict detection and resolution
    - Priority-based ordering
    - Safe action sequencing
    """
    
    def __init__(self, custom_rules: Optional[List[OptimizationRule]] = None):
        self._rules: Dict[str, OptimizationRule] = RULE_CATALOG.copy()
        
        if custom_rules:
            for rule in custom_rules:
                self._rules[rule.rule_id] = rule
        
        # Conflict registry
        self._conflicts: Dict[str, Set[str]] = {}  # rule_id -> conflicting rule_ids
        
        # Register known conflicts
        self._register_conflicts()
    
    def evaluate(self, context: Dict[str, Any],
                 categories: Optional[List[RuleCategory]] = None) -> RuleResult:
        """
        Evaluate all rules against context.
        
        Args:
            context: Performance metrics and metadata
            categories: Optional filter by category
            
        Returns:
            RuleResult with all matches and ordered actions
        """
        result = RuleResult()
        
        # Filter rules by category if specified
        rules = self._rules.values()
        if categories:
            rules = [r for r in rules if r.category in categories]
        
        # Evaluate each rule
        for rule in rules:
            result.total_rules_evaluated += 1
            
            # Check architecture compatibility
            if rule.applicable_archs:
                arch = context.get("gpu_arch", "")
                if arch and arch not in rule.applicable_archs:
                    continue
            
            # Match rule
            match = self._match_rule(rule, context)
            
            if match.result == MatchResult.FULL_MATCH:
                result.matches.append(match)
                result.full_matches += 1
                
                # Categorize
                if rule.category not in result.matches_by_category:
                    result.matches_by_category[rule.category] = []
                result.matches_by_category[rule.category].append(match)
                
            elif match.result == MatchResult.PARTIAL_MATCH:
                result.partial_matches += 1
        
        # Detect conflicts
        result.conflicts = self._detect_conflicts(result.matches)
        
        # Order actions
        result.ordered_actions = self._order_actions(result.matches)
        
        return result
    
    def _match_rule(self, rule: OptimizationRule, 
                    context: Dict[str, Any]) -> RuleMatch:
        """Match a single rule against context."""
        match = RuleMatch(
            rule=rule,
            context=context,
        )
        
        # Evaluate each condition
        for condition in rule.conditions:
            if self._evaluate_condition(condition, context):
                match.matched_conditions.append(condition)
            else:
                match.failed_conditions.append(condition)
        
        # Determine match result
        total = len(rule.conditions)
        matched = len(match.matched_conditions)
        
        if matched == total and total > 0:
            match.result = MatchResult.FULL_MATCH
            match.match_confidence = 1.0
        elif matched > 0:
            match.result = MatchResult.PARTIAL_MATCH
            match.match_confidence = matched / total
        else:
            match.result = MatchResult.NO_MATCH
            match.match_confidence = 0.0
        
        # Extract target info
        if "kernel_name" in context:
            match.target_kernel = context["kernel_name"]
        if "layer_name" in context:
            match.target_layer = context["layer_name"]
        
        return match
    
    def _evaluate_condition(self, condition: RuleCondition,
                           context: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        # Check kernel pattern filter
        if condition.kernel_pattern:
            kernel = context.get("kernel_name", "")
            if not re.search(condition.kernel_pattern, kernel, re.IGNORECASE):
                return False
        
        # Check layer pattern filter
        if condition.layer_pattern:
            layer = context.get("layer_name", "")
            if not re.search(condition.layer_pattern, layer, re.IGNORECASE):
                return False
        
        # Evaluate metric condition
        return condition.evaluate(context)
    
    def _detect_conflicts(self, matches: List[RuleMatch]) -> List[tuple]:
        """Detect conflicts between matched rules."""
        conflicts = []
        
        for i, match1 in enumerate(matches):
            for match2 in matches[i+1:]:
                conflict = self._check_conflict(match1.rule, match2.rule)
                if conflict:
                    conflicts.append((match1.rule, match2.rule, conflict))
        
        return conflicts
    
    def _check_conflict(self, rule1: OptimizationRule,
                        rule2: OptimizationRule) -> Optional[str]:
        """Check if two rules conflict."""
        # Check registered conflicts
        if rule2.rule_id in self._conflicts.get(rule1.rule_id, set()):
            return "registered_conflict"
        
        # Check action conflicts
        for action1 in rule1.actions:
            for action2 in rule2.actions:
                if self._actions_conflict(action1, action2):
                    return f"conflicting_actions_{action1.target}"
        
        return None
    
    def _actions_conflict(self, action1: RuleAction, action2: RuleAction) -> bool:
        """Check if two actions conflict."""
        # Same target with different values
        if action1.target == action2.target:
            if action1.action_type == "config_change" == action2.action_type:
                return action1.new_value != action2.new_value
        
        # Opposing transforms
        opposing_transforms = [
            ("increase_workgroup_size", "decrease_workgroup_size"),
            ("tile_for_cache", "unroll_completely"),
        ]
        
        t1 = action1.transform
        t2 = action2.transform
        for opp1, opp2 in opposing_transforms:
            if (t1 == opp1 and t2 == opp2) or (t1 == opp2 and t2 == opp1):
                return True
        
        return False
    
    def _order_actions(self, matches: List[RuleMatch]) -> List[tuple]:
        """Order actions by priority and dependencies."""
        # Sort matches by priority
        sorted_matches = sorted(
            matches,
            key=lambda m: m.rule.priority.value
        )
        
        ordered = []
        applied_targets: Set[str] = set()
        
        for match in sorted_matches:
            for action in match.rule.actions:
                # Skip if target already modified by higher priority
                if action.target in applied_targets:
                    continue
                
                ordered.append((match.rule, action))
                applied_targets.add(action.target)
        
        return ordered
    
    def _register_conflicts(self) -> None:
        """Register known rule conflicts."""
        # FP16 and BF16 conflict
        self._add_conflict("prec_001", "prec_002")
        
        # Register pressure vs occupancy via workgroup size
        self._add_conflict("occ_001", "occ_002")
    
    def _add_conflict(self, rule_id1: str, rule_id2: str) -> None:
        """Register bidirectional conflict."""
        if rule_id1 not in self._conflicts:
            self._conflicts[rule_id1] = set()
        if rule_id2 not in self._conflicts:
            self._conflicts[rule_id2] = set()
        
        self._conflicts[rule_id1].add(rule_id2)
        self._conflicts[rule_id2].add(rule_id1)
    
    def add_rule(self, rule: OptimizationRule) -> None:
        """Add a custom rule."""
        self._rules[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False
    
    def get_rules_by_category(self, category: RuleCategory) -> List[OptimizationRule]:
        """Get all rules in a category."""
        return [r for r in self._rules.values() if r.category == category]
    
    def get_safe_rules(self) -> List[OptimizationRule]:
        """Get rules that are safe to auto-apply."""
        return [r for r in self._rules.values() if r.is_safe]
