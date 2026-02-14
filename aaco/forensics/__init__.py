"""
AACO-SIGMA Forensic Bundle System

Complete forensic capture for reproducible performance analysis.
Bundles all data needed to investigate and reproduce issues.
"""

from .bundle import (
    ForensicBundle,
    BundleMetadata,
    BundleSection,
    BundleVersion,
    EnvironmentInfo,
    TraceData,
    CounterData,
    MetricsData,
)
from .collector import (
    ForensicCollector,
    CollectorConfig,
    CollectorMode,
)
from .exporter import (
    BundleExporter,
    ExportFormat,
    ExportResult,
)
from .analyzer import (
    BundleAnalyzer,
    AnalysisReport,
    AnalysisType,
    ComparisonResult,
    PostMortemResult,
    Finding,
)

__all__ = [
    # Bundle
    "ForensicBundle",
    "BundleMetadata",
    "BundleSection",
    "BundleVersion",
    "EnvironmentInfo",
    "TraceData",
    "CounterData",
    "MetricsData",
    # Collector
    "ForensicCollector",
    "CollectorConfig",
    "CollectorMode",
    # Exporter
    "BundleExporter",
    "ExportFormat",
    "ExportResult",
    # Analyzer
    "BundleAnalyzer",
    "AnalysisReport",
    "AnalysisType",
    "ComparisonResult",
    "PostMortemResult",
    "Finding",
]
