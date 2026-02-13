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

__all__ = [
    "Session",
    "SessionManager",
    "SessionMetadata",
    "InferenceResult",
    "KernelMetrics",
    "BottleneckClassification",
    "RegressionVerdict",
    "__version__",
]
