"""
AACO Analytics Module - Metrics computation, bottleneck analysis, regression detection.
"""

from aaco.analytics.metrics import DerivedMetricsEngine
from aaco.analytics.classify import BottleneckClassifier
from aaco.analytics.diff import RegressionDetector

__all__ = ["DerivedMetricsEngine", "BottleneckClassifier", "RegressionDetector"]
