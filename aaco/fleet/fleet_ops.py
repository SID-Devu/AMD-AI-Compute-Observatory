# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Fleet-Level Performance Ops

Multi-session aggregation and fleet-wide analytics with:
- Session ingestion and correlation
- Trend analysis dashboards
- Regression heatmaps
- Hardware fleet health scoring
"""

import json
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class FleetHealthLevel(Enum):
    """Health levels for fleet components."""

    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SessionRecord:
    """Record of a profiling session."""

    # Identity
    session_id: str = ""
    timestamp: float = 0.0

    # Environment
    hostname: str = ""
    device_name: str = ""
    device_id: int = 0
    architecture: str = ""
    driver_version: str = ""

    # Model/workload
    model_name: str = ""
    model_hash: str = ""
    batch_size: int = 1

    # Metrics
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    throughput_qps: float = 0.0

    # AACO-Ω∞ metrics
    kar: float = 1.0
    pfi: float = 0.0
    lts: float = 0.0
    heu: float = 0.0
    chi: float = 1.0  # Compute Health Index

    # Root cause (if any)
    detected_root_cause: str = ""
    root_cause_confidence: float = 0.0

    # Regression status
    is_regression: bool = False
    regression_delta_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "hostname": self.hostname,
            "device": self.device_name,
            "model": self.model_name,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "throughput_qps": self.throughput_qps,
            "kar": self.kar,
            "pfi": self.pfi,
            "heu": self.heu,
            "chi": self.chi,
            "root_cause": self.detected_root_cause,
            "is_regression": self.is_regression,
        }


@dataclass
class TrendPoint:
    """Single point in a metric trend."""

    timestamp: float = 0.0
    value: float = 0.0
    session_count: int = 1
    std_dev: float = 0.0


@dataclass
class MetricTrend:
    """Trend data for a metric over time."""

    metric_name: str = ""
    data_points: List[TrendPoint] = field(default_factory=list)

    # Statistics
    overall_mean: float = 0.0
    overall_std: float = 0.0
    trend_direction: str = "stable"  # improving, degrading, stable
    trend_slope: float = 0.0


@dataclass
class RegressionHeatmapCell:
    """Cell in regression heatmap."""

    model_name: str = ""
    device_name: str = ""
    regression_count: int = 0
    improvement_count: int = 0
    stable_count: int = 0
    avg_delta_pct: float = 0.0
    health_level: FleetHealthLevel = FleetHealthLevel.UNKNOWN


@dataclass
class FleetHealthReport:
    """Fleet-wide health report."""

    # Summary
    total_devices: int = 0
    total_sessions: int = 0
    total_regressions: int = 0
    regression_rate: float = 0.0

    # Health levels
    excellent_count: int = 0
    good_count: int = 0
    degraded_count: int = 0
    critical_count: int = 0

    # Top issues
    top_root_causes: List[Dict[str, Any]] = field(default_factory=list)
    top_regression_models: List[str] = field(default_factory=list)
    top_regression_devices: List[str] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class FleetPerformanceOps:
    """
    AACO-Ω∞ Fleet-Level Performance Ops

    Manages multi-session analytics for fleet-wide monitoring.

    Features:
    - Session ingestion and storage
    - Trend analysis
    - Regression heatmaps
    - Hardware fleet health scoring
    - Anomaly correlation across fleet
    """

    def __init__(
        self,
        storage_dir: str = ".aaco/fleet",
    ):
        """
        Initialize fleet performance ops.

        Args:
            storage_dir: Directory for fleet data storage
        """
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        self._sessions: List[SessionRecord] = []
        self._sessions_by_model: Dict[str, List[SessionRecord]] = defaultdict(list)
        self._sessions_by_device: Dict[str, List[SessionRecord]] = defaultdict(list)

    def ingest_session(
        self,
        session: SessionRecord,
    ) -> None:
        """
        Ingest a profiling session record.

        Args:
            session: Session record to ingest
        """
        self._sessions.append(session)
        self._sessions_by_model[session.model_name].append(session)
        self._sessions_by_device[session.device_name].append(session)

        logger.debug(f"Ingested session {session.session_id}")

    def ingest_from_json(
        self,
        data: Dict[str, Any],
    ) -> SessionRecord:
        """
        Ingest session from JSON data.

        Args:
            data: JSON session data

        Returns:
            Created SessionRecord
        """
        session = SessionRecord(
            session_id=data.get("session_id", ""),
            timestamp=data.get("timestamp", time.time()),
            hostname=data.get("hostname", ""),
            device_name=data.get("device_name", data.get("device", "")),
            device_id=data.get("device_id", 0),
            architecture=data.get("architecture", ""),
            model_name=data.get("model_name", data.get("model", "")),
            model_hash=data.get("model_hash", ""),
            batch_size=data.get("batch_size", 1),
            latency_p50_ms=data.get("latency_p50_ms", 0),
            latency_p99_ms=data.get("latency_p99_ms", 0),
            throughput_qps=data.get("throughput_qps", 0),
            kar=data.get("kar", 1.0),
            pfi=data.get("pfi", 0.0),
            lts=data.get("lts", 0.0),
            heu=data.get("heu", 0.0),
            chi=data.get("chi", 1.0),
            detected_root_cause=data.get("root_cause", ""),
            root_cause_confidence=data.get("root_cause_confidence", 0),
            is_regression=data.get("is_regression", False),
            regression_delta_pct=data.get("regression_delta_pct", 0),
        )

        self.ingest_session(session)
        return session

    def get_metric_trend(
        self,
        metric_name: str,
        model_filter: Optional[str] = None,
        device_filter: Optional[str] = None,
        time_window_hours: float = 168,  # 1 week
    ) -> MetricTrend:
        """
        Get trend for a specific metric.

        Args:
            metric_name: Name of metric (latency_p50_ms, kar, etc.)
            model_filter: Filter by model name
            device_filter: Filter by device name
            time_window_hours: Time window in hours

        Returns:
            MetricTrend with data points
        """
        trend = MetricTrend(metric_name=metric_name)

        # Filter sessions
        sessions = self._filter_sessions(
            model_filter=model_filter,
            device_filter=device_filter,
            time_window_hours=time_window_hours,
        )

        if not sessions:
            return trend

        # Group by hour
        hourly_values: Dict[int, List[float]] = defaultdict(list)

        for session in sessions:
            value = getattr(session, metric_name, None)
            if value is not None:
                hour_bucket = int(session.timestamp / 3600)
                hourly_values[hour_bucket].append(value)

        # Create trend points
        for hour, values in sorted(hourly_values.items()):
            point = TrendPoint(
                timestamp=hour * 3600,
                value=statistics.mean(values),
                session_count=len(values),
                std_dev=statistics.stdev(values) if len(values) > 1 else 0,
            )
            trend.data_points.append(point)

        # Compute overall statistics
        all_values = [p.value for p in trend.data_points]
        if all_values:
            trend.overall_mean = statistics.mean(all_values)
            trend.overall_std = statistics.stdev(all_values) if len(all_values) > 1 else 0

            # Compute trend direction using simple linear regression
            if len(all_values) >= 3:
                n = len(all_values)
                x_mean = (n - 1) / 2
                y_mean = trend.overall_mean

                numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(all_values))
                denominator = sum((i - x_mean) ** 2 for i in range(n))

                if denominator > 0:
                    trend.trend_slope = numerator / denominator

                    if trend.trend_slope > 0.1:
                        trend.trend_direction = "degrading"  # Higher latency = worse
                    elif trend.trend_slope < -0.1:
                        trend.trend_direction = "improving"
                    else:
                        trend.trend_direction = "stable"

        return trend

    def get_regression_heatmap(
        self,
        time_window_hours: float = 168,
    ) -> List[RegressionHeatmapCell]:
        """
        Generate regression heatmap (model x device matrix).

        Args:
            time_window_hours: Time window in hours

        Returns:
            List of heatmap cells
        """
        sessions = self._filter_sessions(time_window_hours=time_window_hours)

        # Group by (model, device)
        cells: Dict[Tuple[str, str], RegressionHeatmapCell] = {}

        for session in sessions:
            key = (session.model_name, session.device_name)

            if key not in cells:
                cells[key] = RegressionHeatmapCell(
                    model_name=session.model_name,
                    device_name=session.device_name,
                )

            cell = cells[key]

            if session.is_regression:
                cell.regression_count += 1
            elif session.regression_delta_pct < -2:
                cell.improvement_count += 1
            else:
                cell.stable_count += 1

        # Compute health levels
        for cell in cells.values():
            total = cell.regression_count + cell.improvement_count + cell.stable_count
            if total > 0:
                regression_rate = cell.regression_count / total

                if regression_rate < 0.05:
                    cell.health_level = FleetHealthLevel.EXCELLENT
                elif regression_rate < 0.15:
                    cell.health_level = FleetHealthLevel.GOOD
                elif regression_rate < 0.30:
                    cell.health_level = FleetHealthLevel.DEGRADED
                else:
                    cell.health_level = FleetHealthLevel.CRITICAL

        return list(cells.values())

    def get_fleet_health_report(
        self,
        time_window_hours: float = 168,
    ) -> FleetHealthReport:
        """
        Generate fleet-wide health report.

        Args:
            time_window_hours: Time window in hours

        Returns:
            FleetHealthReport with summary and recommendations
        """
        sessions = self._filter_sessions(time_window_hours=time_window_hours)
        report = FleetHealthReport()

        if not sessions:
            return report

        report.total_sessions = len(sessions)
        report.total_devices = len(set(s.device_name for s in sessions))
        report.total_regressions = sum(1 for s in sessions if s.is_regression)
        report.regression_rate = report.total_regressions / report.total_sessions

        # Count root causes
        root_cause_counts: Dict[str, int] = defaultdict(int)
        for session in sessions:
            if session.detected_root_cause:
                root_cause_counts[session.detected_root_cause] += 1

        report.top_root_causes = [
            {"cause": cause, "count": count}
            for cause, count in sorted(
                root_cause_counts.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
        ]

        # Count regressions by model
        regression_by_model: Dict[str, int] = defaultdict(int)
        for session in sessions:
            if session.is_regression:
                regression_by_model[session.model_name] += 1

        report.top_regression_models = [
            model
            for model, _ in sorted(
                regression_by_model.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
        ]

        # Count regressions by device
        regression_by_device: Dict[str, int] = defaultdict(int)
        for session in sessions:
            if session.is_regression:
                regression_by_device[session.device_name] += 1

        report.top_regression_devices = [
            device
            for device, _ in sorted(
                regression_by_device.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
        ]

        # Health levels from heatmap
        heatmap = self.get_regression_heatmap(time_window_hours)
        for cell in heatmap:
            if cell.health_level == FleetHealthLevel.EXCELLENT:
                report.excellent_count += 1
            elif cell.health_level == FleetHealthLevel.GOOD:
                report.good_count += 1
            elif cell.health_level == FleetHealthLevel.DEGRADED:
                report.degraded_count += 1
            elif cell.health_level == FleetHealthLevel.CRITICAL:
                report.critical_count += 1

        # Generate recommendations
        report.recommendations = self._generate_fleet_recommendations(report)

        return report

    def _filter_sessions(
        self,
        model_filter: Optional[str] = None,
        device_filter: Optional[str] = None,
        time_window_hours: float = 168,
    ) -> List[SessionRecord]:
        """Filter sessions by criteria."""
        cutoff = time.time() - (time_window_hours * 3600)

        sessions = []
        for session in self._sessions:
            if session.timestamp < cutoff:
                continue
            if model_filter and session.model_name != model_filter:
                continue
            if device_filter and session.device_name != device_filter:
                continue
            sessions.append(session)

        return sessions

    def _generate_fleet_recommendations(
        self,
        report: FleetHealthReport,
    ) -> List[str]:
        """Generate recommendations based on fleet report."""
        recommendations = []

        if report.regression_rate > 0.20:
            recommendations.append("High regression rate detected - investigate CI/CD gates")

        if report.top_root_causes:
            top_cause = report.top_root_causes[0]["cause"]
            recommendations.append(f"Most common root cause: {top_cause} - prioritize optimization")

        if report.critical_count > 0:
            recommendations.append(
                f"{report.critical_count} model-device pairs in critical state - "
                "immediate investigation required"
            )

        if report.degraded_count > report.excellent_count:
            recommendations.append("Fleet health degrading - review recent changes")

        return recommendations

    def export_dashboard_data(self) -> Dict[str, Any]:
        """Export data for dashboard visualization."""
        return {
            "total_sessions": len(self._sessions),
            "unique_models": len(self._sessions_by_model),
            "unique_devices": len(self._sessions_by_device),
            "recent_sessions": [
                s.to_dict()
                for s in sorted(
                    self._sessions,
                    key=lambda s: s.timestamp,
                    reverse=True,
                )[:50]
            ],
            "health_report": {
                "total_regressions": sum(1 for s in self._sessions if s.is_regression),
            },
        }

    def save_state(self) -> None:
        """Save fleet state to storage."""
        filepath = self._storage_dir / "fleet_state.json"

        data = {
            "sessions": [s.to_dict() for s in self._sessions],
            "saved_at": time.time(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved fleet state to {filepath}")

    def load_state(self) -> None:
        """Load fleet state from storage."""
        filepath = self._storage_dir / "fleet_state.json"

        if not filepath.exists():
            return

        try:
            with open(filepath) as f:
                data = json.load(f)

            for session_data in data.get("sessions", []):
                self.ingest_from_json(session_data)

            logger.info(f"Loaded {len(self._sessions)} sessions from {filepath}")
        except Exception as e:
            logger.warning(f"Failed to load fleet state: {e}")


def create_fleet_ops(
    storage_dir: str = ".aaco/fleet",
) -> FleetPerformanceOps:
    """
    Factory function to create fleet performance ops.

    Args:
        storage_dir: Directory for fleet data storage

    Returns:
        Configured FleetPerformanceOps
    """
    ops = FleetPerformanceOps(storage_dir=storage_dir)
    ops.load_state()
    return ops
