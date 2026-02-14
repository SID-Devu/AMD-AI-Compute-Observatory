"""
AACO-Ω∞ Governance Module

Statistical regression detection, Bayesian root cause analysis,
and performance governance for continuous optimization.
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

# AACO-Ω∞ Enhanced Governance
from .regression_governor import (
    DriftDirection,
    RobustBaseline,
    EWMAState,
    CUSUMState,
    RegressionVerdict,
    RegressionGovernor,
    create_regression_governor,
)

from .bayesian_engine import (
    RootCauseCategory,
    Evidence,
    RootCausePrior,
    RootCausePosterior,
    RootCauseAnalysis,
    BayesianRootCauseEngine,
    create_root_cause_evidence,
    create_bayesian_engine,
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
    
    # AACO-Ω∞ Regression Governor
    "DriftDirection",
    "RobustBaseline",
    "EWMAState",
    "CUSUMState",
    "RegressionVerdict",
    "RegressionGovernor",
    "create_regression_governor",
    
    # AACO-Ω∞ Bayesian Root Cause
    "RootCauseCategory",
    "Evidence",
    "RootCausePrior",
    "RootCausePosterior",
    "RootCauseAnalysis",
    "BayesianRootCauseEngine",
    "create_root_cause_evidence",
    "create_bayesian_engine",
]
