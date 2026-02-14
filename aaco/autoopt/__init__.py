"""
AACO-SIGMA Auto-Optimization Engine

Automated performance optimization recommendations and application.
Includes rule-based optimization, code generation, and validation.
"""

from .optimization_rule import (
    OptimizationRule,
    RuleCategory,
    RuleCondition,
    RuleAction,
    RulePriority,
)
from .rule_engine import (
    RuleEngine,
    RuleMatch,
    RuleResult,
)
from .optimizer import (
    AutoOptimizer,
    OptimizationResult,
    OptimizationPlan,
)
from .code_generator import (
    CodeGenerator,
    CodeVariant,
    GeneratedCode,
)

__all__ = [
    # Rules
    "OptimizationRule",
    "RuleCategory",
    "RuleCondition",
    "RuleAction",
    "RulePriority",
    # Engine
    "RuleEngine",
    "RuleMatch",
    "RuleResult",
    # Optimizer
    "AutoOptimizer",
    "OptimizationResult",
    "OptimizationPlan",
    # Code Gen
    "CodeGenerator",
    "CodeVariant",
    "GeneratedCode",
]
