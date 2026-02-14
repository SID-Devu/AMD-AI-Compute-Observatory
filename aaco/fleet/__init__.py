"""
AACO-Ω∞ Fleet Mode Platform

Manages performance profiling across GPU fleets with:
- Cross-system analysis and comparison
- Fleet-wide trend analytics
- Regression heatmaps
- Hardware health scoring
"""

from .fleet_manager import (
    FleetManager,
    GPUNode,
    NodeStatus,
    FleetConfig,
)
from .job_scheduler import (
    JobScheduler,
    ProfileJob,
    JobStatus,
    JobResult,
)
from .aggregator import (
    FleetAggregator,
    AggregatedResult,
    StatisticalSummary,
)
from .health_monitor import (
    HealthMonitor,
    NodeHealth,
    HealthStatus,
    HealthAlert,
)

# AACO-Ω∞ Fleet Performance Ops
from .fleet_ops import (
    FleetHealthLevel,
    SessionRecord,
    TrendPoint,
    MetricTrend,
    RegressionHeatmapCell,
    FleetHealthReport,
    FleetPerformanceOps,
    create_fleet_ops,
)

__all__ = [
    # Fleet Manager
    "FleetManager",
    "GPUNode",
    "NodeStatus",
    "FleetConfig",
    # Scheduler
    "JobScheduler",
    "ProfileJob",
    "JobStatus",
    "JobResult",
    # Aggregator
    "FleetAggregator",
    "AggregatedResult",
    "StatisticalSummary",
    # Health
    "HealthMonitor",
    "NodeHealth",
    "HealthStatus",
    "HealthAlert",
    # AACO-Ω∞ Fleet Ops
    "FleetHealthLevel",
    "SessionRecord",
    "TrendPoint",
    "MetricTrend",
    "RegressionHeatmapCell",
    "FleetHealthReport",
    "FleetPerformanceOps",
    "create_fleet_ops",
]
