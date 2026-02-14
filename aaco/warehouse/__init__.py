"""
AACO-Λ Fleet Warehouse Module.

Provides fleet-scale benchmark result storage and analytics.
"""

from .store import (
    FleetWarehouse,
    SessionMetadata,
    BenchmarkResult,
    TrendPoint,
    create_warehouse,
    get_default_warehouse,
)

__all__ = [
    "FleetWarehouse",
    "SessionMetadata",
    "BenchmarkResult",
    "TrendPoint",
    "create_warehouse",
    "get_default_warehouse",
]
