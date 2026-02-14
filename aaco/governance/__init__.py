"""
AACO-SIGMA Regression Governance RG++

Automated performance regression detection, SLA enforcement,
and CI/CD integration for continuous performance governance.
"""

from .regression_detector import (
    RegressionDetector,
    RegressionConfig,
    RegressionResult,
    RegressionSeverity,
)

from .sla_engine import (
    SLAEngine,
    SLAPolicy,
    SLAViolation,
    SLACheck,
    SLAResult,
)

from .baseline_manager import (
    BaselineManager,
    PerformanceBaseline,
    BaselineVersion,
    BaselineComparison,
)

from .ci_integration import (
    CIIntegration,
    CIPipelineResult,
    GateDecision,
    PipelineConfig,
)

__all__ = [
    # Regression detection
    "RegressionDetector",
    "RegressionConfig",
    "RegressionResult",
    "RegressionSeverity",
    
    # SLA engine
    "SLAEngine",
    "SLAPolicy",
    "SLAViolation",
    "SLACheck",
    "SLAResult",
    
    # Baseline management
    "BaselineManager",
    "PerformanceBaseline",
    "BaselineVersion",
    "BaselineComparison",
    
    # CI/CD integration
    "CIIntegration",
    "CIPipelineResult",
    "GateDecision",
    "PipelineConfig",
]
