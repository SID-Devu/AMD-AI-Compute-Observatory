"""
AMD AI Compute Observatory (AACO)
Full-stack observability and performance intelligence for AMD AI workloads.

AACO-Ω∞: Model-to-Metal Performance Science & Governance Engine

A deterministic, self-calibrating, cross-layer AI performance laboratory that
models, predicts, diagnoses, and experimentally validates model-to-hardware
behavior on AMD GPUs.

10-Pillar Architecture:
1. Laboratory Mode - Deterministic execution control
2. eBPF Forensic Scheduler - Scheduler interference analysis
3. GPU Counter-Calibrated Intelligence - Advanced kernel fingerprinting
4. Probabilistic Attribution Engine - Graph→Partition→Kernel mapping
5. Hardware-Calibrated Digital Twin - HEU scoring with microbenchmarks
6. Unified Trace Lake - Perfetto-compatible cross-layer traces
7. Statistical Regression Governance - Robust baseline with drift detection
8. Bayesian Root Cause Engine - Evidence-based posterior ranking
9. Auto-Optimization Engine - Closed-loop hypothesis testing
10. Fleet-Level Performance Ops - Multi-session aggregation and trends
"""

__version__ = "4.0.0"  # AACO-Ω∞
__author__ = "Sudheer Devu"

from aaco.core.session import Session, SessionManager
from aaco.core.schema import (
    SessionMetadata,
    InferenceResult,
    KernelMetrics,
    BottleneckClassification,
    RegressionVerdict,
)

# High-level API exports
from aaco.analytics import (
    DerivedMetricsEngine,
    BottleneckClassifier,
    RegressionDetector,
    BatchScalingAnalyzer,
    TimelineCorrelator,
    LaunchTaxAnalyzer,
)

from aaco.runner import ORTRunner, LLMProfiler

from aaco.graph import ONNXGraphExtractor, OpKernelMapper

# AACO-X Advanced Analytics
from aaco.analytics.feature_store import (
    FeatureStore,
    FeatureExtractor,
    SessionFeatures,
    IterationFeatures,
)

from aaco.analytics.regression_guard import (
    RegressionGuard,
    RegressionReport,
    RegressionSeverity,
    BaselineModel,
)

from aaco.analytics.recommendation_engine import (
    RecommendationEngine,
    BottleneckAnalysis,
    Recommendation,
    BottleneckClass,
)

from aaco.analytics.chi import (
    CHICalculator,
    CHIReport,
    HealthRating,
)

from aaco.analytics.perfetto_export import (
    PerfettoTraceBuilder,
    AACOSessionToPerfetto,
)

from aaco.analytics.trace_events import (
    UnifiedTraceStore,
    UnifiedEvent,
    EventSource,
    EventType,
)

# AACO-Λ Isolation Layer
from aaco.isolation import (
    MeasurementCapsule,
    CapsulePolicy,
    CapsuleManifest,
    IsolationLevel,
    NoiseSentinel,
    NoiseReport,
    NoiseEvent,
    NoiseSource,
)

# AACO-Λ Hardware Calibration
from aaco.calibration import (
    HardwareEnvelope,
    HardwareEnvelopeCalibrator,
    BandwidthEnvelope,
    ComputeEnvelope,
    quick_calibrate,
    load_or_calibrate,
)

# AACO-Λ Kernel Fingerprinting
from aaco.analytics.kernel_fingerprint import (
    KernelFamilyFingerprinter,
    KernelFingerprint,
    KernelFamilyRegistry,
    KernelCategory,
    KernelEvent,
)

# AACO-Λ Probabilistic Attribution
from aaco.analytics.attribution import (
    ProbabilisticAttributionEngine,
    AttributionGraph,
    AttributionResult,
    AttributionLevel,
)

# AACO-Λ Bayesian Root Cause
from aaco.analytics.root_cause import (
    BayesianRootCauseAnalyzer,
    RootCauseRanking,
    RootCauseSuspect,
    RootCauseCategory,
    explain_root_cause,
)

# AACO-Λ Auto-Experiment
from aaco.experiment import (
    AutoExperimentEngine,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    Hypothesis,
    quick_ab_test,
)

# AACO-Λ Fleet Warehouse
from aaco.warehouse import (
    FleetWarehouse,
    SessionMetadata as WarehouseSessionMetadata,
    BenchmarkResult,
    TrendPoint,
    get_default_warehouse,
)

# ════════════════════════════════════════════════════════════════════════════
# AACO-Ω∞ Advanced Performance Science Modules
# ════════════════════════════════════════════════════════════════════════════

# AACO-Ω∞ Advanced Kernel Fingerprinting (Pillar 3)
from aaco.kff import (
    KernelFamily,
    DurationDistribution,
    GridSignature,
    CounterSignature,
    KernelFingerprint as AdvancedKernelFingerprint,
    KernelFamilyCluster,
    KernelFamilyClassifier,
)

# AACO-Ω∞ Hardware-Calibrated Digital Twin (Pillar 5)
from aaco.twin import (
    BenchmarkType,
    CalibrationSample,
    HardwareEnvelope as TwinHardwareEnvelope,
    CalibrationResult,
    MicrobenchmarkSuite,
    DigitalTwinCalibrator,
)

# AACO-Ω∞ Unified Trace Lake (Pillar 6)
from aaco.trace_lake import (
    TraceCategory,
    EventPhase,
    TraceEvent,
    AnomalyMarker,
    TraceLakeConfig,
    UnifiedTraceLake,
)

# AACO-Ω∞ Statistical Regression Governance (Pillar 7)
from aaco.governance import (
    DriftDirection,
    RobustBaseline,
    EWMAState,
    CUSUMState,
    RegressionVerdict as GovernanceRegressionVerdict,
    RegressionGovernor,
)

# AACO-Ω∞ Bayesian Root Cause Engine (Pillar 8)
from aaco.governance import (
    RootCauseCategory as BayesianRootCauseCategory,
    Evidence,
    RootCausePosterior,
    RootCauseAnalysis,
    BayesianRootCauseEngine,
)

# AACO-Ω∞ Auto-Optimization Engine (Pillar 9)
from aaco.optimizer import (
    OptimizationType,
    ExperimentStatus,
    OptimizationHypothesis,
    ExperimentConfig as OptExperimentConfig,
    ExperimentResult as OptExperimentResult,
    AutoOptimizationEngine,
)

# AACO-Ω∞ Fleet-Level Performance Ops (Pillar 10)
from aaco.fleet import (
    FleetHealthLevel,
    SessionRecord,
    TrendPoint as FleetTrendPoint,
    MetricTrend,
    RegressionHeatmapCell,
    FleetHealthReport,
    FleetPerformanceOps,
)

__all__ = [
    # Core
    "Session",
    "SessionManager",
    "SessionMetadata",
    "InferenceResult",
    "KernelMetrics",
    "BottleneckClassification",
    "RegressionVerdict",
    
    # Analytics (Base)
    "DerivedMetricsEngine",
    "BottleneckClassifier",
    "RegressionDetector",
    "BatchScalingAnalyzer",
    "TimelineCorrelator",
    "LaunchTaxAnalyzer",
    
    # AACO-X Feature Store
    "FeatureStore",
    "FeatureExtractor",
    "SessionFeatures",
    "IterationFeatures",
    
    # AACO-X Regression Guard
    "RegressionGuard",
    "RegressionReport",
    "RegressionSeverity",
    "BaselineModel",
    
    # AACO-X Recommendation Engine
    "RecommendationEngine",
    "BottleneckAnalysis",
    "Recommendation",
    "BottleneckClass",
    
    # AACO-X CHI
    "CHICalculator",
    "CHIReport",
    "HealthRating",
    
    # AACO-X Perfetto
    "PerfettoTraceBuilder",
    "AACOSessionToPerfetto",
    
    # AACO-X Unified Trace
    "UnifiedTraceStore",
    "UnifiedEvent",
    "EventSource",
    "EventType",
    
    # Runner
    "ORTRunner",
    "LLMProfiler",
    
    # Graph
    "ONNXGraphExtractor",
    "OpKernelMapper",
    
    # AACO-Λ Isolation
    "MeasurementCapsule",
    "CapsulePolicy",
    "CapsuleManifest",
    "IsolationLevel",
    "NoiseSentinel",
    "NoiseReport",
    "NoiseEvent",
    "NoiseSource",
    
    # AACO-Λ Calibration
    "HardwareEnvelope",
    "HardwareEnvelopeCalibrator",
    "BandwidthEnvelope",
    "ComputeEnvelope",
    "quick_calibrate",
    "load_or_calibrate",
    
    # AACO-Λ Kernel Fingerprinting
    "KernelFamilyFingerprinter",
    "KernelFingerprint",
    "KernelFamilyRegistry",
    "KernelCategory",
    "KernelEvent",
    
    # AACO-Λ Attribution
    "ProbabilisticAttributionEngine",
    "AttributionGraph",
    "AttributionResult",
    "AttributionLevel",
    
    # AACO-Λ Root Cause
    "BayesianRootCauseAnalyzer",
    "RootCauseRanking",
    "RootCauseSuspect",
    "RootCauseCategory",
    "explain_root_cause",
    
    # AACO-Λ Experiment
    "AutoExperimentEngine",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "Hypothesis",
    "quick_ab_test",
    
    # AACO-Λ Warehouse
    "FleetWarehouse",
    "WarehouseSessionMetadata",
    "BenchmarkResult",
    "TrendPoint",
    "get_default_warehouse",
    
    # ════════════════════════════════════════════════════════════════════════
    # AACO-Ω∞ Advanced Performance Science
    # ════════════════════════════════════════════════════════════════════════
    
    # AACO-Ω∞ Advanced Kernel Fingerprinting (Pillar 3)
    "KernelFamily",
    "DurationDistribution",
    "GridSignature",
    "CounterSignature",
    "AdvancedKernelFingerprint",
    "KernelFamilyCluster",
    "KernelFamilyClassifier",
    
    # AACO-Ω∞ Hardware-Calibrated Digital Twin (Pillar 5)
    "BenchmarkType",
    "CalibrationSample",
    "TwinHardwareEnvelope",
    "CalibrationResult",
    "MicrobenchmarkSuite",
    "DigitalTwinCalibrator",
    
    # AACO-Ω∞ Unified Trace Lake (Pillar 6)
    "TraceCategory",
    "EventPhase",
    "TraceEvent",
    "AnomalyMarker",
    "TraceLakeConfig",
    "UnifiedTraceLake",
    
    # AACO-Ω∞ Statistical Regression Governance (Pillar 7)
    "DriftDirection",
    "RobustBaseline",
    "EWMAState",
    "CUSUMState",
    "GovernanceRegressionVerdict",
    "RegressionGovernor",
    
    # AACO-Ω∞ Bayesian Root Cause Engine (Pillar 8)
    "BayesianRootCauseCategory",
    "Evidence",
    "RootCausePosterior",
    "RootCauseAnalysis",
    "BayesianRootCauseEngine",
    
    # AACO-Ω∞ Auto-Optimization Engine (Pillar 9)
    "OptimizationType",
    "ExperimentStatus",
    "OptimizationHypothesis",
    "OptExperimentConfig",
    "OptExperimentResult",
    "AutoOptimizationEngine",
    
    # AACO-Ω∞ Fleet-Level Performance Ops (Pillar 10)
    "FleetHealthLevel",
    "SessionRecord",
    "FleetTrendPoint",
    "MetricTrend",
    "RegressionHeatmapCell",
    "FleetHealthReport",
    "FleetPerformanceOps",
    
    # Version
    "__version__",
]
