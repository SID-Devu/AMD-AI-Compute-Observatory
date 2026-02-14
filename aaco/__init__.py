"""
AMD AI Compute Observatory (AACO)
Full-stack observability and performance intelligence for AMD AI workloads.

AACO-Λ: Model-to-Metal Performance Engineering Platform
"""

__version__ = "3.0.0"  # AACO-Λ
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
    
    # Version
    "__version__",
]
