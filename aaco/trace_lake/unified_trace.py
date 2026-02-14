# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Unified Trace Lake

Perfetto-compatible trace integration with:
- CPU scheduler events
- GPU kernel execution
- Power/thermal markers
- Anomaly annotations
- Cross-layer correlation
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TraceCategory(Enum):
    """Categories of trace events."""

    CPU_SCHEDULER = "cpu_scheduler"
    GPU_KERNEL = "gpu_kernel"
    GPU_MEMORY = "gpu_memory"
    POWER = "power"
    THERMAL = "thermal"
    SYSTEM = "system"
    ONNX_OPERATOR = "onnx_operator"
    ANOMALY = "anomaly"


class EventPhase(Enum):
    """Trace event phases (Chrome Trace format)."""

    DURATION_BEGIN = "B"
    DURATION_END = "E"
    COMPLETE = "X"
    INSTANT = "i"
    COUNTER = "C"
    ASYNC_BEGIN = "b"
    ASYNC_END = "e"
    FLOW_START = "s"
    FLOW_END = "f"
    METADATA = "M"


@dataclass
class TraceEvent:
    """
    Single trace event in Chrome Trace format.

    Compatible with Perfetto UI and chrome://tracing.
    """

    # Required fields
    name: str = ""
    category: str = ""
    phase: str = "X"  # EventPhase
    timestamp_us: int = 0  # Microseconds since trace start

    # Duration events
    duration_us: int = 0

    # Process/thread identification
    pid: int = 0
    tid: int = 0

    # Additional data
    args: Dict[str, Any] = field(default_factory=dict)

    # Flow events
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Chrome Trace JSON format."""
        event = {
            "name": self.name,
            "cat": self.category,
            "ph": self.phase,
            "ts": self.timestamp_us,
            "pid": self.pid,
            "tid": self.tid,
        }

        if self.phase == "X" and self.duration_us > 0:
            event["dur"] = self.duration_us

        if self.args:
            event["args"] = self.args

        if self.id is not None:
            event["id"] = self.id

        return event


@dataclass
class AnomalyMarker:
    """Marker for detected anomaly in trace."""

    timestamp_us: int = 0
    category: str = ""
    severity: str = "warning"  # info, warning, error, critical
    description: str = ""
    related_events: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TraceLakeConfig:
    """Configuration for trace lake."""

    # Output settings
    output_format: str = "json"  # json, perfetto, chrome
    compression: bool = True

    # Category filters
    enabled_categories: List[TraceCategory] = field(default_factory=lambda: list(TraceCategory))

    # Sampling settings
    sample_rate: float = 1.0  # 1.0 = all events

    # Metadata
    include_metadata: bool = True
    include_process_names: bool = True


class UnifiedTraceLake:
    """
    AACO-Ω∞ Unified Trace Lake

    Aggregates traces from multiple sources into Perfetto-compatible format.

    Sources:
    - CPU scheduler (from eBPF or /proc)
    - GPU kernels (from rocprof)
    - Power/thermal (from rocm-smi)
    - ONNX operators (from profiler)
    """

    # Process IDs for different sources
    PROCESS_IDS = {
        TraceCategory.CPU_SCHEDULER: 1,
        TraceCategory.GPU_KERNEL: 2,
        TraceCategory.GPU_MEMORY: 3,
        TraceCategory.POWER: 4,
        TraceCategory.THERMAL: 5,
        TraceCategory.ONNX_OPERATOR: 6,
        TraceCategory.ANOMALY: 7,
    }

    def __init__(self, config: Optional[TraceLakeConfig] = None):
        """Initialize trace lake."""
        self._config = config or TraceLakeConfig()
        self._events: List[TraceEvent] = []
        self._anomalies: List[AnomalyMarker] = []
        self._metadata: Dict[str, Any] = {}
        self._start_time_ns: int = 0
        self._next_id: int = 1

    def start_session(self) -> None:
        """Start new trace session."""
        self._events.clear()
        self._anomalies.clear()
        self._start_time_ns = time.time_ns()

        # Add metadata event
        if self._config.include_metadata:
            self._add_metadata_events()

    def _add_metadata_events(self) -> None:
        """Add metadata events for Perfetto UI."""
        # Process names
        for category, pid in self.PROCESS_IDS.items():
            self._events.append(
                TraceEvent(
                    name="process_name",
                    category="__metadata",
                    phase="M",
                    pid=pid,
                    args={"name": category.value},
                )
            )

    def add_cpu_event(
        self,
        name: str,
        timestamp_ns: int,
        duration_ns: int,
        cpu_id: int = 0,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add CPU scheduler event."""
        if TraceCategory.CPU_SCHEDULER not in self._config.enabled_categories:
            return

        self._events.append(
            TraceEvent(
                name=name,
                category=TraceCategory.CPU_SCHEDULER.value,
                phase="X",
                timestamp_us=self._ns_to_us(timestamp_ns),
                duration_us=duration_ns // 1000,
                pid=self.PROCESS_IDS[TraceCategory.CPU_SCHEDULER],
                tid=cpu_id,
                args=args or {},
            )
        )

    def add_gpu_kernel_event(
        self,
        kernel_name: str,
        start_ns: int,
        duration_ns: int,
        device_id: int = 0,
        queue_id: int = 0,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add GPU kernel execution event."""
        if TraceCategory.GPU_KERNEL not in self._config.enabled_categories:
            return

        event_args = args or {}
        event_args["device"] = device_id
        event_args["queue"] = queue_id

        self._events.append(
            TraceEvent(
                name=kernel_name,
                category=TraceCategory.GPU_KERNEL.value,
                phase="X",
                timestamp_us=self._ns_to_us(start_ns),
                duration_us=duration_ns // 1000,
                pid=self.PROCESS_IDS[TraceCategory.GPU_KERNEL],
                tid=queue_id,
                args=event_args,
            )
        )

    def add_operator_event(
        self,
        op_name: str,
        op_type: str,
        start_ns: int,
        duration_ns: int,
        partition_id: str = "",
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add ONNX operator event."""
        if TraceCategory.ONNX_OPERATOR not in self._config.enabled_categories:
            return

        event_args = args or {}
        event_args["op_type"] = op_type
        event_args["partition"] = partition_id

        self._events.append(
            TraceEvent(
                name=op_name,
                category=TraceCategory.ONNX_OPERATOR.value,
                phase="X",
                timestamp_us=self._ns_to_us(start_ns),
                duration_us=duration_ns // 1000,
                pid=self.PROCESS_IDS[TraceCategory.ONNX_OPERATOR],
                tid=0,
                args=event_args,
            )
        )

    def add_power_sample(
        self,
        timestamp_ns: int,
        power_watts: float,
        device_id: int = 0,
    ) -> None:
        """Add power measurement sample."""
        if TraceCategory.POWER not in self._config.enabled_categories:
            return

        self._events.append(
            TraceEvent(
                name="GPU Power",
                category=TraceCategory.POWER.value,
                phase="C",  # Counter
                timestamp_us=self._ns_to_us(timestamp_ns),
                pid=self.PROCESS_IDS[TraceCategory.POWER],
                tid=0,
                args={"watts": power_watts, "device": device_id},
            )
        )

    def add_thermal_sample(
        self,
        timestamp_ns: int,
        temperature_c: float,
        device_id: int = 0,
    ) -> None:
        """Add thermal measurement sample."""
        if TraceCategory.THERMAL not in self._config.enabled_categories:
            return

        self._events.append(
            TraceEvent(
                name="GPU Temperature",
                category=TraceCategory.THERMAL.value,
                phase="C",  # Counter
                timestamp_us=self._ns_to_us(timestamp_ns),
                pid=self.PROCESS_IDS[TraceCategory.THERMAL],
                tid=0,
                args={"celsius": temperature_c, "device": device_id},
            )
        )

    def add_anomaly(
        self,
        timestamp_ns: int,
        category: str,
        description: str,
        severity: str = "warning",
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add anomaly marker to trace."""
        marker = AnomalyMarker(
            timestamp_us=self._ns_to_us(timestamp_ns),
            category=category,
            severity=severity,
            description=description,
            metrics=metrics or {},
        )
        self._anomalies.append(marker)

        # Also add as instant event
        if TraceCategory.ANOMALY in self._config.enabled_categories:
            self._events.append(
                TraceEvent(
                    name=f"ANOMALY: {description}",
                    category=TraceCategory.ANOMALY.value,
                    phase="i",  # Instant
                    timestamp_us=marker.timestamp_us,
                    pid=self.PROCESS_IDS[TraceCategory.ANOMALY],
                    tid=0,
                    args={
                        "category": category,
                        "severity": severity,
                        "metrics": metrics or {},
                    },
                )
            )

    def add_flow_link(
        self,
        from_name: str,
        to_name: str,
        from_timestamp_ns: int,
        to_timestamp_ns: int,
        flow_category: str = "data_flow",
    ) -> None:
        """Add flow link between events (e.g., operator to kernel)."""
        flow_id = self._next_id
        self._next_id += 1

        # Flow start
        self._events.append(
            TraceEvent(
                name=from_name,
                category=flow_category,
                phase="s",  # Flow start
                timestamp_us=self._ns_to_us(from_timestamp_ns),
                pid=self.PROCESS_IDS[TraceCategory.ONNX_OPERATOR],
                tid=0,
                id=flow_id,
            )
        )

        # Flow end
        self._events.append(
            TraceEvent(
                name=to_name,
                category=flow_category,
                phase="f",  # Flow end
                timestamp_us=self._ns_to_us(to_timestamp_ns),
                pid=self.PROCESS_IDS[TraceCategory.GPU_KERNEL],
                tid=0,
                id=flow_id,
            )
        )

    def _ns_to_us(self, timestamp_ns: int) -> int:
        """Convert absolute nanosecond timestamp to relative microseconds."""
        if self._start_time_ns > 0:
            return (timestamp_ns - self._start_time_ns) // 1000
        return timestamp_ns // 1000

    def import_rocprof_trace(
        self,
        trace_data: List[Dict[str, Any]],
    ) -> int:
        """
        Import trace data from rocprof output.

        Args:
            trace_data: List of kernel trace records

        Returns:
            Number of events imported
        """
        count = 0
        for trace in trace_data:
            kernel_name = trace.get("name", trace.get("kernel_name", ""))
            if not kernel_name:
                continue

            self.add_gpu_kernel_event(
                kernel_name=kernel_name,
                start_ns=trace.get("start_ns", trace.get("BeginNs", 0)),
                duration_ns=trace.get("duration_ns", trace.get("DurationNs", 0)),
                device_id=trace.get("device_id", trace.get("gpu-id", 0)),
                queue_id=trace.get("queue_id", 0),
            )
            count += 1

        return count

    def import_cpu_trace(
        self,
        trace_data: List[Dict[str, Any]],
    ) -> int:
        """
        Import CPU scheduler trace data.

        Args:
            trace_data: List of CPU events

        Returns:
            Number of events imported
        """
        count = 0
        for trace in trace_data:
            self.add_cpu_event(
                name=trace.get("name", "cpu_event"),
                timestamp_ns=trace.get("timestamp_ns", 0),
                duration_ns=trace.get("duration_ns", 0),
                cpu_id=trace.get("cpu_id", 0),
            )
            count += 1

        return count

    def export_chrome_trace(self, filepath: str) -> None:
        """
        Export trace in Chrome Trace JSON format.

        Compatible with chrome://tracing and Perfetto UI.

        Args:
            filepath: Output file path
        """
        # Sort events by timestamp
        sorted_events = sorted(
            self._events,
            key=lambda e: e.timestamp_us,
        )

        trace_data = {
            "traceEvents": [e.to_dict() for e in sorted_events],
            "metadata": {
                "source": "AACO-Ω∞ Unified Trace Lake",
                "version": "1.0",
                "event_count": len(sorted_events),
                "anomaly_count": len(self._anomalies),
            },
        }

        with open(filepath, "w") as f:
            json.dump(trace_data, f)

        logger.info(f"Exported {len(sorted_events)} events to {filepath}")

    def export_perfetto_proto(self, filepath: str) -> None:
        """
        Export trace in Perfetto protobuf format.

        Note: Requires perfetto library. Falls back to JSON.

        Args:
            filepath: Output file path
        """
        # For now, export as Chrome JSON which Perfetto can read
        json_path = filepath.replace(".perfetto", ".json")
        self.export_chrome_trace(json_path)
        logger.info(f"Exported Perfetto-compatible trace to {json_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get trace lake statistics."""
        cat_counts = {}
        for event in self._events:
            cat = event.category
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        return {
            "total_events": len(self._events),
            "anomaly_count": len(self._anomalies),
            "events_by_category": cat_counts,
            "session_duration_us": (
                max(e.timestamp_us for e in self._events)
                - min(e.timestamp_us for e in self._events)
                if self._events
                else 0
            ),
        }

    def get_anomalies(self) -> List[AnomalyMarker]:
        """Get all detected anomalies."""
        return self._anomalies


def create_trace_lake(
    config: Optional[TraceLakeConfig] = None,
) -> UnifiedTraceLake:
    """
    Factory function to create unified trace lake.

    Args:
        config: Trace lake configuration

    Returns:
        Configured UnifiedTraceLake
    """
    lake = UnifiedTraceLake(config=config)
    lake.start_session()
    return lake
