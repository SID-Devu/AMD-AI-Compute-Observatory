"""
AACO-SIGMA Auto-Optimizer

Main optimizer that coordinates rule matching, optimization planning,
and validation of applied optimizations.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum, auto
import time

from .optimization_rule import OptimizationRule, RuleAction, RulePriority
from .rule_engine import RuleEngine, RuleResult, RuleMatch


class OptimizationStatus(Enum):
    """Status of an optimization."""
    PLANNED = auto()       # Planned but not applied
    APPLIED = auto()       # Successfully applied
    VALIDATED = auto()     # Applied and validated
    FAILED = auto()        # Application failed
    REVERTED = auto()      # Applied but reverted
    SKIPPED = auto()       # Skipped due to constraints


@dataclass
class OptimizationItem:
    """A single optimization to be applied."""
    
    item_id: str
    rule: OptimizationRule
    action: RuleAction
    
    # Target
    target_kernel: Optional[str] = None
    target_layer: Optional[str] = None
    
    # Status
    status: OptimizationStatus = OptimizationStatus.PLANNED
    
    # Results
    applied_at: Optional[float] = None
    improvement_pct: Optional[float] = None
    validation_passed: Optional[bool] = None
    
    # Error handling
    error_message: Optional[str] = None


@dataclass
class OptimizationPlan:
    """Plan of optimizations to apply."""
    
    plan_id: str
    
    # Items
    items: List[OptimizationItem] = field(default_factory=list)
    
    # Ordering
    execution_order: List[str] = field(default_factory=list)  # item_ids in order
    
    # Constraints
    max_items: int = 10
    require_validation: bool = True
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_expected_improvement(self) -> float:
        """Total expected improvement from all items."""
        return sum(
            item.rule.expected_improvement_pct
            for item in self.items
            if item.status in [OptimizationStatus.PLANNED, OptimizationStatus.VALIDATED]
        )
    
    @property
    def safe_items(self) -> List[OptimizationItem]:
        """Items that are safe to auto-apply."""
        return [item for item in self.items if item.rule.is_safe]


@dataclass
class OptimizationResult:
    """Result of optimization execution."""
    
    plan: OptimizationPlan
    
    # Execution
    started_at: float = 0.0
    completed_at: float = 0.0
    
    # Results
    items_applied: int = 0
    items_validated: int = 0
    items_failed: int = 0
    items_reverted: int = 0
    
    # Impact
    baseline_latency_ms: float = 0.0
    optimized_latency_ms: float = 0.0
    
    @property
    def total_improvement_pct(self) -> float:
        """Calculate actual improvement percentage."""
        if self.baseline_latency_ms == 0:
            return 0.0
        return (1 - self.optimized_latency_ms / self.baseline_latency_ms) * 100
    
    @property
    def execution_time_s(self) -> float:
        return self.completed_at - self.started_at
    
    @property
    def success_rate(self) -> float:
        total = self.items_applied + self.items_failed
        if total == 0:
            return 0.0
        return self.items_validated / total


class AutoOptimizer:
    """
    Automated performance optimizer.
    
    Workflow:
    1. Analyze performance context
    2. Match optimization rules
    3. Create optimization plan
    4. Apply optimizations with validation
    5. Revert if validation fails
    """
    
    def __init__(self, rule_engine: Optional[RuleEngine] = None):
        self._engine = rule_engine or RuleEngine()
        self._plan_counter = 0
        
        # Hooks
        self._apply_hooks: List[Callable] = []
        self._validate_hooks: List[Callable] = []
        self._revert_hooks: List[Callable] = []
    
    def analyze(self, context: Dict[str, Any]) -> RuleResult:
        """
        Analyze performance context and find applicable rules.
        """
        return self._engine.evaluate(context)
    
    def create_plan(self, 
                    rule_result: RuleResult,
                    context: Dict[str, Any],
                    safe_only: bool = False,
                    max_items: int = 10) -> OptimizationPlan:
        """
        Create optimization plan from rule matches.
        
        Args:
            rule_result: Result from analyze()
            context: Performance context
            safe_only: Only include safe optimizations
            max_items: Maximum items in plan
        """
        self._plan_counter += 1
        plan = OptimizationPlan(
            plan_id=f"plan_{self._plan_counter:04d}",
            max_items=max_items,
            context_snapshot=context.copy(),
        )
        
        item_id = 0
        for rule, action in rule_result.ordered_actions:
            if safe_only and not rule.is_safe:
                continue
            
            item_id += 1
            item = OptimizationItem(
                item_id=f"opt_{item_id:03d}",
                rule=rule,
                action=action,
                target_kernel=context.get("kernel_name"),
                target_layer=context.get("layer_name"),
            )
            plan.items.append(item)
            plan.execution_order.append(item.item_id)
            
            if len(plan.items) >= max_items:
                break
        
        return plan
    
    def execute(self, 
                plan: OptimizationPlan,
                apply_fn: Optional[Callable] = None,
                validate_fn: Optional[Callable] = None) -> OptimizationResult:
        """
        Execute optimization plan.
        
        Args:
            plan: Plan to execute
            apply_fn: Function to apply optimization (item) -> bool
            validate_fn: Function to validate (item) -> bool
        """
        result = OptimizationResult(
            plan=plan,
            started_at=time.time(),
        )
        
        # Get baseline if validate function provided
        if validate_fn and plan.require_validation:
            result.baseline_latency_ms = self._get_baseline(plan.context_snapshot)
        
        # Execute in order
        for item_id in plan.execution_order:
            item = self._get_item(plan, item_id)
            if not item:
                continue
            
            # Apply
            success = self._apply_item(item, apply_fn)
            
            if success:
                result.items_applied += 1
                
                # Validate if required
                if plan.require_validation and item.action.requires_validation:
                    validated = self._validate_item(item, validate_fn)
                    
                    if validated:
                        item.status = OptimizationStatus.VALIDATED
                        result.items_validated += 1
                    else:
                        # Revert
                        self._revert_item(item)
                        item.status = OptimizationStatus.REVERTED
                        result.items_reverted += 1
                else:
                    item.status = OptimizationStatus.APPLIED
                    result.items_validated += 1
            else:
                result.items_failed += 1
        
        # Get final latency
        if validate_fn and plan.require_validation:
            result.optimized_latency_ms = self._get_baseline(plan.context_snapshot)
        
        result.completed_at = time.time()
        return result
    
    def _get_item(self, plan: OptimizationPlan, item_id: str) -> Optional[OptimizationItem]:
        """Find item by ID."""
        for item in plan.items:
            if item.item_id == item_id:
                return item
        return None
    
    def _apply_item(self, item: OptimizationItem,
                    apply_fn: Optional[Callable]) -> bool:
        """Apply a single optimization item."""
        try:
            # Run hooks
            for hook in self._apply_hooks:
                hook(item)
            
            # Apply
            if apply_fn:
                success = apply_fn(item)
            else:
                success = self._default_apply(item)
            
            if success:
                item.applied_at = time.time()
                item.status = OptimizationStatus.APPLIED
            else:
                item.status = OptimizationStatus.FAILED
            
            return success
            
        except Exception as e:
            item.status = OptimizationStatus.FAILED
            item.error_message = str(e)
            return False
    
    def _validate_item(self, item: OptimizationItem,
                       validate_fn: Optional[Callable]) -> bool:
        """Validate an applied optimization."""
        try:
            for hook in self._validate_hooks:
                hook(item)
            
            if validate_fn:
                validated = validate_fn(item)
            else:
                validated = self._default_validate(item)
            
            item.validation_passed = validated
            return validated
            
        except Exception as e:
            item.validation_passed = False
            item.error_message = f"Validation error: {e}"
            return False
    
    def _revert_item(self, item: OptimizationItem) -> bool:
        """Revert an applied optimization."""
        try:
            for hook in self._revert_hooks:
                hook(item)
            
            item.status = OptimizationStatus.REVERTED
            return True
            
        except Exception as e:
            item.error_message = f"Revert error: {e}"
            return False
    
    def _default_apply(self, item: OptimizationItem) -> bool:
        """Default apply implementation (simulation)."""
        # In production, this would call actual optimization APIs
        return True
    
    def _default_validate(self, item: OptimizationItem) -> bool:
        """Default validation (always passes in simulation)."""
        return True
    
    def _get_baseline(self, context: Dict[str, Any]) -> float:
        """Get baseline latency."""
        return context.get("latency_ms", 0.0)
    
    def suggest(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get suggestions without applying.
        
        Returns human-readable suggestions.
        """
        result = self.analyze(context)
        suggestions = []
        
        for match in result.matches:
            if not match.is_full_match:
                continue
            
            suggestion = {
                "rule_id": match.rule.rule_id,
                "title": match.rule.name,
                "description": match.rule.description,
                "expected_improvement": f"{match.rule.expected_improvement_pct:.1f}%",
                "priority": match.rule.priority.name,
                "category": match.rule.category.name,
                "is_safe": match.rule.is_safe,
                "actions": [
                    {"type": a.action_type, "description": a.description}
                    for a in match.rule.actions
                ],
            }
            suggestions.append(suggestion)
        
        # Sort by priority
        suggestions.sort(key=lambda s: RulePriority[s["priority"]].value)
        
        return suggestions
    
    def register_apply_hook(self, hook: Callable) -> None:
        """Register hook called before applying optimization."""
        self._apply_hooks.append(hook)
    
    def register_validate_hook(self, hook: Callable) -> None:
        """Register hook called before validation."""
        self._validate_hooks.append(hook)
    
    def register_revert_hook(self, hook: Callable) -> None:
        """Register hook called before reverting."""
        self._revert_hooks.append(hook)
