"""
AMD AI Compute Observatory (AACO)
Full-stack observability and performance intelligence for AMD AI workloads.
"""

__version__ = "2.0.0"  # AACO-X
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
    
    # Version
    "__version__",
]
