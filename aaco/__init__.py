"""
AMD AI Compute Observatory (AACO)
Full-stack observability and performance intelligence for AMD AI workloads.
"""

__version__ = "1.0.0"
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

__all__ = [
    # Core
    "Session",
    "SessionManager",
    "SessionMetadata",
    "InferenceResult",
    "KernelMetrics",
    "BottleneckClassification",
    "RegressionVerdict",
    
    # Analytics
    "DerivedMetricsEngine",
    "BottleneckClassifier",
    "RegressionDetector",
    "BatchScalingAnalyzer",
    "TimelineCorrelator",
    "LaunchTaxAnalyzer",
    
    # Runner
    "ORTRunner",
    "LLMProfiler",
    
    # Graph
    "ONNXGraphExtractor",
    "OpKernelMapper",
    
    # Version
    "__version__",
]
