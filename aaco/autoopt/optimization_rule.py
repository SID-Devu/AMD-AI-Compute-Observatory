"""
AACO-SIGMA Optimization Rules

Defines the structure and catalog of automated optimization rules.
Rules encode expert knowledge for performance optimization.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum, auto


class RuleCategory(Enum):
    """Categories of optimization rules."""
    MEMORY_OPTIMIZATION = auto()    # Memory access, layout, caching
    COMPUTE_OPTIMIZATION = auto()   # Compute utilization
    KERNEL_OPTIMIZATION = auto()    # Kernel-level optimizations
    FUSION_OPTIMIZATION = auto()    # Kernel fusion
    CONFIG_OPTIMIZATION = auto()    # Configuration tuning
    PRECISION_OPTIMIZATION = auto() # Precision/quantization
    LAUNCH_OPTIMIZATION = auto()    # Kernel launch configurations
    GRAPH_OPTIMIZATION = auto()     # Graph-level transformations


class RulePriority(Enum):
    """Priority levels for rule application."""
    CRITICAL = 1    # Always apply if matched
    HIGH = 2        # Apply unless conflicts
    MEDIUM = 3      # Apply conservatively
    LOW = 4         # Suggest only
    EXPERIMENTAL = 5 # Experimental, needs validation


@dataclass
class RuleCondition:
    """A condition that must be met for rule to match."""
    
    # Condition type
    metric_name: str  # e.g., "memory_bound_pct", "cache_miss_rate"
    operator: str  # "gt", "lt", "eq", "ge", "le", "in_range"
    threshold: Any  # Value or tuple for in_range
    
    # Optional
    kernel_pattern: Optional[str] = None  # Regex for kernel name
    layer_pattern: Optional[str] = None   # Regex for layer name
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate condition against context."""
        if self.metric_name not in context:
            return False
        
        value = context[self.metric_name]
        
        if self.operator == "gt":
            return value > self.threshold
        elif self.operator == "lt":
            return value < self.threshold
        elif self.operator == "ge":
            return value >= self.threshold
        elif self.operator == "le":
            return value <= self.threshold
        elif self.operator == "eq":
            return value == self.threshold
        elif self.operator == "in_range":
            low, high = self.threshold
            return low <= value <= high
        
        return False


@dataclass
class RuleAction:
    """Action to take when rule matches."""
    
    action_type: str  # "config_change", "code_transform", "suggestion"
    
    # What to change
    target: str  # Parameter or code element
    new_value: Any = None  # New value for config changes
    
    # For code transforms
    transform: Optional[str] = None  # Transform name
    transform_params: Dict[str, Any] = field(default_factory=dict)
    
    # Human-readable
    description: str = ""
    
    # Validation
    requires_validation: bool = True
    validation_threshold: float = 0.95  # Must maintain this % of correctness


@dataclass
class OptimizationRule:
    """
    A rule for automated performance optimization.
    
    Rules encode patterns like:
    - IF memory_bound > 80% AND cache_miss > 30% THEN suggest memory layout change
    - IF occupancy < 50% AND register_usage > 64 THEN suggest register reduction
    """
    
    # Identification
    rule_id: str
    name: str
    description: str
    
    # Categorization
    category: RuleCategory
    priority: RulePriority
    
    # Conditions (all must match)
    conditions: List[RuleCondition] = field(default_factory=list)
    
    # Actions to take
    actions: List[RuleAction] = field(default_factory=list)
    
    # Expected impact
    expected_improvement_pct: float = 0.0
    confidence: float = 0.8
    
    # Applicability
    applicable_ops: List[str] = field(default_factory=list)  # ["MatMul", "Conv2D"]
    applicable_archs: List[str] = field(default_factory=list)  # ["gfx90a", "gfx1100"]
    
    # Safety
    is_safe: bool = True  # Safe to auto-apply
    requires_profiling: bool = True  # Needs profiling validation
    
    # Metadata
    source: str = "expert"  # "expert", "learned", "autotuned"
    tags: List[str] = field(default_factory=list)


# ============ Built-in Rule Catalog ============

def create_builtin_rules() -> List[OptimizationRule]:
    """Create catalog of built-in optimization rules."""
    rules = []
    
    # === Memory Optimization Rules ===
    
    rules.append(OptimizationRule(
        rule_id="mem_001",
        name="Memory Coalescing Optimization",
        description="Enable memory coalescing for strided access patterns",
        category=RuleCategory.MEMORY_OPTIMIZATION,
        priority=RulePriority.HIGH,
        conditions=[
            RuleCondition("memory_bound_pct", "gt", 70),
            RuleCondition("coalesced_access_pct", "lt", 50),
        ],
        actions=[
            RuleAction(
                action_type="suggestion",
                target="memory_layout",
                description="Consider transposing tensors for coalesced access"
            ),
            RuleAction(
                action_type="code_transform",
                target="tensor",
                transform="transpose_for_coalescing",
                description="Transpose tensor to improve memory coalescing"
            ),
        ],
        expected_improvement_pct=15.0,
        applicable_ops=["MatMul", "Conv2D", "Attention"],
        tags=["memory", "coalescing"],
    ))
    
    rules.append(OptimizationRule(
        rule_id="mem_002",
        name="L2 Cache Tiling",
        description="Apply tiling to improve L2 cache hit rate",
        category=RuleCategory.MEMORY_OPTIMIZATION,
        priority=RulePriority.MEDIUM,
        conditions=[
            RuleCondition("l2_cache_miss_rate", "gt", 40),
            RuleCondition("working_set_mb", "gt", 32),
        ],
        actions=[
            RuleAction(
                action_type="code_transform",
                target="loop",
                transform="tile_for_cache",
                transform_params={"tile_size": "auto"},
                description="Tile loops to fit in L2 cache"
            ),
        ],
        expected_improvement_pct=20.0,
        tags=["memory", "cache", "tiling"],
    ))
    
    # === Compute Optimization Rules ===
    
    rules.append(OptimizationRule(
        rule_id="comp_001",
        name="MFMA Utilization",
        description="Utilize Matrix Fused Multiply-Add units",
        category=RuleCategory.COMPUTE_OPTIMIZATION,
        priority=RulePriority.HIGH,
        conditions=[
            RuleCondition("mfma_utilization", "lt", 50),
            RuleCondition("op_type", "eq", "MatMul"),
        ],
        actions=[
            RuleAction(
                action_type="code_transform",
                target="matmul",
                transform="use_mfma_intrinsics",
                description="Use MFMA intrinsics for matrix multiply"
            ),
        ],
        expected_improvement_pct=30.0,
        applicable_archs=["gfx90a", "gfx940", "gfx942"],
        tags=["compute", "mfma", "matrix"],
    ))
    
    rules.append(OptimizationRule(
        rule_id="comp_002",
        name="Vectorization Enhancement",
        description="Improve SIMD vectorization",
        category=RuleCategory.COMPUTE_OPTIMIZATION,
        priority=RulePriority.MEDIUM,
        conditions=[
            RuleCondition("vectorization_pct", "lt", 70),
            RuleCondition("compute_bound_pct", "gt", 60),
        ],
        actions=[
            RuleAction(
                action_type="suggestion",
                target="kernel",
                description="Review loop for vectorization blockers"
            ),
        ],
        expected_improvement_pct=15.0,
        tags=["compute", "vectorization", "simd"],
    ))
    
    # === Occupancy Optimization Rules ===
    
    rules.append(OptimizationRule(
        rule_id="occ_001",
        name="Register Pressure Reduction",
        description="Reduce register usage to improve occupancy",
        category=RuleCategory.LAUNCH_OPTIMIZATION,
        priority=RulePriority.HIGH,
        conditions=[
            RuleCondition("occupancy_pct", "lt", 50),
            RuleCondition("register_usage", "gt", 64),
        ],
        actions=[
            RuleAction(
                action_type="config_change",
                target="max_registers",
                new_value=64,
                description="Limit registers to 64 per thread"
            ),
            RuleAction(
                action_type="code_transform",
                target="kernel",
                transform="spill_to_memory",
                description="Spill excess registers to memory"
            ),
        ],
        expected_improvement_pct=25.0,
        tags=["occupancy", "registers"],
    ))
    
    rules.append(OptimizationRule(
        rule_id="occ_002",
        name="Workgroup Size Optimization",
        description="Optimize workgroup size for better occupancy",
        category=RuleCategory.LAUNCH_OPTIMIZATION,
        priority=RulePriority.MEDIUM,
        conditions=[
            RuleCondition("occupancy_pct", "lt", 60),
            RuleCondition("workgroup_size", "lt", 256),
        ],
        actions=[
            RuleAction(
                action_type="config_change",
                target="workgroup_size",
                new_value=256,
                description="Increase workgroup size to 256"
            ),
        ],
        expected_improvement_pct=10.0,
        tags=["occupancy", "workgroup"],
    ))
    
    # === Fusion Optimization Rules ===
    
    rules.append(OptimizationRule(
        rule_id="fuse_001",
        name="Elementwise Kernel Fusion",
        description="Fuse consecutive elementwise operations",
        category=RuleCategory.FUSION_OPTIMIZATION,
        priority=RulePriority.HIGH,
        conditions=[
            RuleCondition("consecutive_elementwise_ops", "gt", 2),
            RuleCondition("kernel_launch_overhead_pct", "gt", 10),
        ],
        actions=[
            RuleAction(
                action_type="code_transform",
                target="graph",
                transform="fuse_elementwise",
                description="Fuse consecutive elementwise operations"
            ),
        ],
        expected_improvement_pct=20.0,
        is_safe=True,
        tags=["fusion", "elementwise"],
    ))
    
    rules.append(OptimizationRule(
        rule_id="fuse_002",
        name="MatMul + Bias + Activation Fusion",
        description="Fuse matmul with subsequent bias and activation",
        category=RuleCategory.FUSION_OPTIMIZATION,
        priority=RulePriority.HIGH,
        conditions=[
            RuleCondition("has_matmul_bias_activation_pattern", "eq", True),
        ],
        actions=[
            RuleAction(
                action_type="code_transform",
                target="graph",
                transform="fuse_matmul_bias_activation",
                description="Fuse MatMul + BiasAdd + Activation"
            ),
        ],
        expected_improvement_pct=15.0,
        is_safe=True,
        tags=["fusion", "matmul", "activation"],
    ))
    
    # === Precision Optimization Rules ===
    
    rules.append(OptimizationRule(
        rule_id="prec_001",
        name="FP16 Mixed Precision",
        description="Convert FP32 to FP16 for compute-bound operations",
        category=RuleCategory.PRECISION_OPTIMIZATION,
        priority=RulePriority.MEDIUM,
        conditions=[
            RuleCondition("precision", "eq", "fp32"),
            RuleCondition("compute_bound_pct", "gt", 70),
            RuleCondition("loss_tolerance", "gt", 0.001),
        ],
        actions=[
            RuleAction(
                action_type="config_change",
                target="dtype",
                new_value="fp16",
                description="Convert to FP16 precision",
                requires_validation=True,
            ),
        ],
        expected_improvement_pct=40.0,
        is_safe=False,  # Needs accuracy validation
        tags=["precision", "fp16", "mixed"],
    ))
    
    rules.append(OptimizationRule(
        rule_id="prec_002",
        name="BF16 for Training",
        description="Use BF16 for better training stability than FP16",
        category=RuleCategory.PRECISION_OPTIMIZATION,
        priority=RulePriority.MEDIUM,
        conditions=[
            RuleCondition("precision", "eq", "fp32"),
            RuleCondition("is_training", "eq", True),
            RuleCondition("arch_supports_bf16", "eq", True),
        ],
        actions=[
            RuleAction(
                action_type="config_change",
                target="dtype",
                new_value="bf16",
                description="Convert to BF16 precision",
                requires_validation=True,
            ),
        ],
        expected_improvement_pct=30.0,
        applicable_archs=["gfx90a", "gfx940", "gfx942"],
        is_safe=False,
        tags=["precision", "bf16", "training"],
    ))
    
    # === Config Optimization Rules ===
    
    rules.append(OptimizationRule(
        rule_id="cfg_001",
        name="Batch Size Optimization",
        description="Increase batch size for better GPU utilization",
        category=RuleCategory.CONFIG_OPTIMIZATION,
        priority=RulePriority.LOW,
        conditions=[
            RuleCondition("gpu_utilization", "lt", 60),
            RuleCondition("batch_size", "lt", 32),
            RuleCondition("memory_usage_pct", "lt", 70),
        ],
        actions=[
            RuleAction(
                action_type="config_change",
                target="batch_size",
                new_value="double",
                description="Double the batch size"
            ),
        ],
        expected_improvement_pct=20.0,
        tags=["config", "batch_size"],
    ))
    
    rules.append(OptimizationRule(
        rule_id="cfg_002",
        name="Tensor Layout Optimization",
        description="Change tensor layout for better memory access",
        category=RuleCategory.CONFIG_OPTIMIZATION,
        priority=RulePriority.MEDIUM,
        conditions=[
            RuleCondition("memory_bound_pct", "gt", 60),
            RuleCondition("tensor_layout", "eq", "NHWC"),
        ],
        actions=[
            RuleAction(
                action_type="config_change",
                target="tensor_layout",
                new_value="NCHW",
                description="Change layout to NCHW for better vectorization"
            ),
        ],
        expected_improvement_pct=10.0,
        applicable_ops=["Conv2D"],
        tags=["config", "layout"],
    ))
    
    return rules


# Create global rule catalog
BUILTIN_RULES = create_builtin_rules()
RULE_CATALOG: Dict[str, OptimizationRule] = {r.rule_id: r for r in BUILTIN_RULES}
