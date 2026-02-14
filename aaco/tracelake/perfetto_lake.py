"""
AACO-SIGMA Perfetto Trace Lake

Primary trace format implementation with Perfetto-compatible output.
Provides timeline visualization and causality analysis.
"""

import json
import gzip
import time
import uuid
from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from collections import defaultdict


class TrackType(Enum):
    """Types of tracks in the trace."""

    PROCESS = auto()  # Process-level track
    THREAD = auto()  # Thread-level track
    COUNTER = auto()  # Counter track (metrics)
    ASYNC = auto()  # Async events
    FLOW = auto()  # Flow events (causality)
    MARKER = auto()  # Instant markers
    GPU = auto()  # GPU events
    KERNEL = auto()  # Kernel (driver) events


@dataclass
class PerfettoTrack:
    """A track in the Perfetto trace."""

    track_id: int
    track_type: TrackType
    name: str

    # Optional hierarchy
    parent_track_id: Optional[int] = None
    process_id: Optional[int] = None
    thread_id: Optional[int] = None

    # Track properties
    unit: str = ""  # For counter tracks
    description: str = ""

    # Track metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerfettoEvent:
    """Base class for Perfetto events."""

    track_id: int
    timestamp_ns: int
    name: str

    # Event properties
    category: str = ""
    color: str = ""

    # Debug/flow
    debug_annotations: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InstantEvent(PerfettoEvent):
    """Instant event (marker)."""

    scope: str = "t"  # t=thread, p=process, g=global


@dataclass
class DurationEvent(PerfettoEvent):
    """Duration event (slice)."""

    duration_ns: int = 0
    end_timestamp_ns: int = 0

    # Nesting
    depth: int = 0

    # Arguments (shown in details)
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterEvent(PerfettoEvent):
    """Counter event (metric sample)."""

    value: float = 0.0
    delta: Optional[float] = None


@dataclass
class FlowEvent(PerfettoEvent):
    """Flow event for causality visualization."""

    flow_id: int = 0
    flow_direction: str = "s"  # s=start, t=step, f=finish

    # Connected events
    bind_id: Optional[int] = None


class PerfettoTraceLake:
    """
    Unified trace lake with Perfetto-compatible output.

    Collects events from multiple sources and produces
    a trace file that can be opened in Perfetto UI.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]

        self._tracks: Dict[int, PerfettoTrack] = {}
        self._events: List[Union[PerfettoEvent, DurationEvent, CounterEvent, FlowEvent]] = []

        self._track_counter = 0
        self._flow_counter = 0

        # Metadata
        self._start_time_ns = time.time_ns()
        self._metadata: Dict[str, Any] = {}

        # Category colors
        self._category_colors: Dict[str, str] = {
            "gpu": "yellow",
            "cpu": "blue",
            "memory": "green",
            "io": "orange",
            "inference": "purple",
            "kernel": "red",
            "compile": "cyan",
        }

        # Standard tracks
        self._create_standard_tracks()

    def _create_standard_tracks(self) -> None:
        """Create standard tracks for common event types."""
        # Process track for the main process
        self.create_track(TrackType.PROCESS, "AACO Session", process_id=0)

        # Standard counter tracks
        self.create_track(TrackType.COUNTER, "GPU Utilization", unit="%")
        self.create_track(TrackType.COUNTER, "GPU Memory", unit="MB")
        self.create_track(TrackType.COUNTER, "GPU Clock", unit="MHz")
        self.create_track(TrackType.COUNTER, "GPU Temperature", unit="°C")
        self.create_track(TrackType.COUNTER, "GPU Power", unit="W")
        self.create_track(TrackType.COUNTER, "CPU Utilization", unit="%")
        self.create_track(TrackType.COUNTER, "Memory Pressure", unit="%")

        # Event tracks
        self.create_track(TrackType.GPU, "GPU Kernels")
        self.create_track(TrackType.THREAD, "Inference")
        self.create_track(TrackType.MARKER, "Phase Markers")
        self.create_track(TrackType.KERNEL, "Driver Events")

    def create_track(
        self,
        track_type: TrackType,
        name: str,
        parent_id: Optional[int] = None,
        process_id: Optional[int] = None,
        thread_id: Optional[int] = None,
        unit: str = "",
        **metadata,
    ) -> int:
        """Create a new track and return its ID."""
        self._track_counter += 1
        track_id = self._track_counter

        track = PerfettoTrack(
            track_id=track_id,
            track_type=track_type,
            name=name,
            parent_track_id=parent_id,
            process_id=process_id,
            thread_id=thread_id,
            unit=unit,
            metadata=metadata,
        )

        self._tracks[track_id] = track
        return track_id

    def get_track_id(self, name: str) -> Optional[int]:
        """Get track ID by name."""
        for track_id, track in self._tracks.items():
            if track.name == name:
                return track_id
        return None

    def add_instant(
        self,
        track_id: int,
        name: str,
        timestamp_ns: int,
        category: str = "",
        **annotations,
    ) -> None:
        """Add an instant event (marker)."""
        event = InstantEvent(
            track_id=track_id,
            timestamp_ns=timestamp_ns,
            name=name,
            category=category,
            color=self._category_colors.get(category, ""),
            debug_annotations=annotations,
        )
        self._events.append(event)

    def add_duration(
        self,
        track_id: int,
        name: str,
        start_ns: int,
        duration_ns: int,
        category: str = "",
        depth: int = 0,
        **args,
    ) -> None:
        """Add a duration event (slice)."""
        event = DurationEvent(
            track_id=track_id,
            timestamp_ns=start_ns,
            name=name,
            duration_ns=duration_ns,
            end_timestamp_ns=start_ns + duration_ns,
            category=category,
            color=self._category_colors.get(category, ""),
            depth=depth,
            args=args,
        )
        self._events.append(event)

    def add_counter(
        self,
        track_id: int,
        name: str,
        timestamp_ns: int,
        value: float,
        category: str = "",
    ) -> None:
        """Add a counter sample."""
        event = CounterEvent(
            track_id=track_id,
            timestamp_ns=timestamp_ns,
            name=name,
            value=value,
            category=category,
        )
        self._events.append(event)

    def add_flow_start(
        self, track_id: int, name: str, timestamp_ns: int, category: str = ""
    ) -> int:
        """Start a flow event and return flow ID."""
        self._flow_counter += 1
        flow_id = self._flow_counter

        event = FlowEvent(
            track_id=track_id,
            timestamp_ns=timestamp_ns,
            name=name,
            category=category,
            flow_id=flow_id,
            flow_direction="s",
        )
        self._events.append(event)
        return flow_id

    def add_flow_step(
        self,
        track_id: int,
        name: str,
        timestamp_ns: int,
        flow_id: int,
        category: str = "",
    ) -> None:
        """Add an intermediate flow step."""
        event = FlowEvent(
            track_id=track_id,
            timestamp_ns=timestamp_ns,
            name=name,
            category=category,
            flow_id=flow_id,
            flow_direction="t",
        )
        self._events.append(event)

    def add_flow_end(
        self,
        track_id: int,
        name: str,
        timestamp_ns: int,
        flow_id: int,
        category: str = "",
    ) -> None:
        """End a flow event."""
        event = FlowEvent(
            track_id=track_id,
            timestamp_ns=timestamp_ns,
            name=name,
            category=category,
            flow_id=flow_id,
            flow_direction="f",
        )
        self._events.append(event)

    def add_gpu_kernel(
        self,
        kernel_name: str,
        start_ns: int,
        duration_ns: int,
        grid: str = "",
        block: str = "",
        **metrics,
    ) -> None:
        """Add a GPU kernel event."""
        track_id = self.get_track_id("GPU Kernels") or 1

        self.add_duration(
            track_id=track_id,
            name=kernel_name,
            start_ns=start_ns,
            duration_ns=duration_ns,
            category="gpu",
            grid=grid,
            block=block,
            **metrics,
        )

    def add_inference_iteration(
        self, iteration: int, start_ns: int, duration_ns: int, **metrics
    ) -> None:
        """Add an inference iteration event."""
        track_id = self.get_track_id("Inference") or 2

        self.add_duration(
            track_id=track_id,
            name=f"Iteration {iteration}",
            start_ns=start_ns,
            duration_ns=duration_ns,
            category="inference",
            iteration=iteration,
            **metrics,
        )

    def add_phase_marker(self, phase: str, timestamp_ns: int, **annotations) -> None:
        """Add a phase marker (warmup, measure, etc.)."""
        track_id = self.get_track_id("Phase Markers") or 3

        self.add_instant(
            track_id=track_id,
            name=phase,
            timestamp_ns=timestamp_ns,
            category="inference",
            **annotations,
        )

    def set_metadata(self, **kwargs) -> None:
        """Set trace metadata."""
        self._metadata.update(kwargs)

    def to_perfetto_json(self) -> Dict[str, Any]:
        """
        Convert to Perfetto JSON format.

        Returns a dictionary that can be serialized to JSON
        and opened in Perfetto UI (ui.perfetto.dev).
        """
        trace_events = []

        # Add metadata event
        trace_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": 0,
                "args": {"name": f"AACO Session {self.session_id}"},
            }
        )

        # Add track metadata
        for track_id, track in self._tracks.items():
            if track.track_type == TrackType.PROCESS:
                trace_events.append(
                    {
                        "name": "process_name",
                        "ph": "M",
                        "pid": track.process_id or track_id,
                        "args": {"name": track.name},
                    }
                )
            elif track.track_type == TrackType.THREAD:
                trace_events.append(
                    {
                        "name": "thread_name",
                        "ph": "M",
                        "pid": track.process_id or 0,
                        "tid": track.thread_id or track_id,
                        "args": {"name": track.name},
                    }
                )

        # Add events
        for event in self._events:
            if isinstance(event, InstantEvent):
                trace_events.append(
                    {
                        "name": event.name,
                        "cat": event.category,
                        "ph": "i",  # Instant event
                        "ts": event.timestamp_ns / 1000,  # Convert to microseconds
                        "pid": self._get_pid_for_track(event.track_id),
                        "tid": self._get_tid_for_track(event.track_id),
                        "s": event.scope,
                        "args": event.debug_annotations,
                    }
                )

            elif isinstance(event, DurationEvent):
                trace_events.append(
                    {
                        "name": event.name,
                        "cat": event.category,
                        "ph": "X",  # Complete event
                        "ts": event.timestamp_ns / 1000,
                        "dur": event.duration_ns / 1000,
                        "pid": self._get_pid_for_track(event.track_id),
                        "tid": self._get_tid_for_track(event.track_id),
                        "args": event.args,
                    }
                )

            elif isinstance(event, CounterEvent):
                track = self._tracks.get(event.track_id)
                counter_name = track.name if track else event.name

                trace_events.append(
                    {
                        "name": counter_name,
                        "cat": event.category or "counter",
                        "ph": "C",  # Counter event
                        "ts": event.timestamp_ns / 1000,
                        "pid": self._get_pid_for_track(event.track_id),
                        "args": {event.name: event.value},
                    }
                )

            elif isinstance(event, FlowEvent):
                trace_events.append(
                    {
                        "name": event.name,
                        "cat": event.category,
                        "ph": event.flow_direction,
                        "ts": event.timestamp_ns / 1000,
                        "pid": self._get_pid_for_track(event.track_id),
                        "tid": self._get_tid_for_track(event.track_id),
                        "id": event.flow_id,
                    }
                )

        return {"traceEvents": trace_events}

    def _get_pid_for_track(self, track_id: int) -> int:
        """Get process ID for a track."""
        track = self._tracks.get(track_id)
        if track and track.process_id is not None:
            return track.process_id
        return 0

    def _get_tid_for_track(self, track_id: int) -> int:
        """Get thread ID for a track."""
        track = self._tracks.get(track_id)
        if track and track.thread_id is not None:
            return track.thread_id
        return track_id

    def save_json(self, path: Path, compress: bool = False) -> None:
        """Save trace as JSON file."""
        trace_data = self.to_perfetto_json()

        if compress:
            with gzip.open(path.with_suffix(".json.gz"), "wt") as f:
                json.dump(trace_data, f)
        else:
            with open(path, "w") as f:
                json.dump(trace_data, f)

    def get_statistics(self) -> Dict[str, Any]:
        """Get trace statistics."""
        stats = {
            "session_id": self.session_id,
            "track_count": len(self._tracks),
            "event_count": len(self._events),
            "tracks": {},
        }

        # Count events per track
        events_per_track: Dict[int, int] = defaultdict(int)
        for event in self._events:
            events_per_track[event.track_id] += 1

        for track_id, track in self._tracks.items():
            stats["tracks"][track.name] = {
                "type": track.track_type.name,
                "event_count": events_per_track.get(track_id, 0),
            }

        # Time range
        if self._events:
            timestamps = [e.timestamp_ns for e in self._events]
            stats["time_range_ns"] = {
                "start": min(timestamps),
                "end": max(timestamps),
                "duration": max(timestamps) - min(timestamps),
            }

        return stats


class TraceLakeBuilder:
    """
    Builder for constructing traces from multiple sources.
    """

    def __init__(self, session_id: Optional[str] = None):
        self.lake = PerfettoTraceLake(session_id)
        self._current_iteration: int = 0
        self._iteration_start_ns: int = 0

    def start_iteration(self, iteration: int) -> None:
        """Mark start of an inference iteration."""
        self._current_iteration = iteration
        self._iteration_start_ns = time.time_ns()

        self.lake.add_phase_marker(
            f"iter_{iteration}_start", self._iteration_start_ns, iteration=iteration
        )

    def end_iteration(self, **metrics) -> None:
        """Mark end of an inference iteration."""
        end_ns = time.time_ns()
        duration_ns = end_ns - self._iteration_start_ns

        self.lake.add_inference_iteration(
            self._current_iteration, self._iteration_start_ns, duration_ns, **metrics
        )

    def add_gpu_kernels(self, kernels: List[Dict[str, Any]]) -> None:
        """Add GPU kernel events from rocprof output."""
        for kernel in kernels:
            name = kernel.get("name", kernel.get("kernel_name", "unknown"))
            start_ns = kernel.get("start_ns", 0)
            duration_ns = kernel.get("duration_ns", kernel.get("dur", 0))

            self.lake.add_gpu_kernel(
                kernel_name=name,
                start_ns=start_ns,
                duration_ns=duration_ns,
                grid=kernel.get("grid", ""),
                block=kernel.get("block", ""),
                grd=kernel.get("grd", ""),
                wgr=kernel.get("wgr", ""),
            )

    def add_counter_samples(self, samples: List[Dict[str, Any]], track_name: str) -> None:
        """Add counter samples to a track."""
        track_id = self.lake.get_track_id(track_name)
        if track_id is None:
            track_id = self.lake.create_track(TrackType.COUNTER, track_name)

        for sample in samples:
            self.lake.add_counter(
                track_id=track_id,
                name=track_name,
                timestamp_ns=sample.get("timestamp_ns", 0),
                value=sample.get("value", 0),
            )

    def add_rocm_smi_samples(self, samples: List[Dict[str, Any]]) -> None:
        """Add rocm-smi samples as counter events."""
        for sample in samples:
            ts = sample.get("timestamp_ns", 0)

            if "gpu_util" in sample:
                self.lake.add_counter(
                    self.lake.get_track_id("GPU Utilization") or 0,
                    "gpu_util",
                    ts,
                    sample["gpu_util"],
                )

            if "memory_used_mb" in sample:
                self.lake.add_counter(
                    self.lake.get_track_id("GPU Memory") or 0,
                    "memory",
                    ts,
                    sample["memory_used_mb"],
                )

            if "clock_mhz" in sample or "sclk" in sample:
                self.lake.add_counter(
                    self.lake.get_track_id("GPU Clock") or 0,
                    "clock",
                    ts,
                    sample.get("clock_mhz", sample.get("sclk", 0)),
                )

            if "temp_c" in sample or "temperature" in sample:
                self.lake.add_counter(
                    self.lake.get_track_id("GPU Temperature") or 0,
                    "temp",
                    ts,
                    sample.get("temp_c", sample.get("temperature", 0)),
                )

            if "power_w" in sample or "power" in sample:
                self.lake.add_counter(
                    self.lake.get_track_id("GPU Power") or 0,
                    "power",
                    ts,
                    sample.get("power_w", sample.get("power", 0)),
                )

    def finalize(self) -> PerfettoTraceLake:
        """Finalize and return the trace lake."""
        return self.lake


def create_trace_lake(session_id: Optional[str] = None) -> PerfettoTraceLake:
    """Create a new trace lake instance."""
    return PerfettoTraceLake(session_id)


def builder(session_id: Optional[str] = None) -> TraceLakeBuilder:
    """Create a trace lake builder."""
    return TraceLakeBuilder(session_id)
