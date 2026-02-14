"""
AACO Analytics Module - Metrics computation, bottleneck analysis, regression detection.
"""

from aaco.analytics.metrics import DerivedMetricsEngine
from aaco.analytics.classify import BottleneckClassifier
from aaco.analytics.diff import RegressionDetector
from aaco.analytics.batch_scaler import (
    BatchScalingAnalyzer,
    BatchPoint,
    ScalingAnalysis,
)
from aaco.analytics.timeline import TimelineCorrelator, TimelineEvent
from aaco.analytics.launch_tax import LaunchTaxAnalyzer, LaunchTaxReport

__all__ = [
    "DerivedMetricsEngine",
    "BottleneckClassifier",
    "RegressionDetector",
    "BatchScalingAnalyzer",
    "BatchPoint",
    "ScalingAnalysis",
    "TimelineCorrelator",
    "TimelineEvent",
    "LaunchTaxAnalyzer",
    "LaunchTaxReport",
]
