"""
AACO-SIGMA Trace Query Engine

Query interface for exploring unified traces.
Supports time-range filtering, event selection, and aggregation.
"""

import re
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Callable, Iterator
from pathlib import Path
from enum import Enum, auto
import statistics


class AggregationType(Enum):
    """Aggregation operations for trace queries."""

    COUNT = auto()
    SUM = auto()
    MEAN = auto()
    MIN = auto()
    MAX = auto()
    PERCENTILE = auto()
    HISTOGRAM = auto()


@dataclass
class TimeRange:
    """Time range specification."""

    start_ns: int = 0
    end_ns: int = 0

    @classmethod
    def from_relative(cls, duration_ns: int, offset_ns: int = 0) -> "TimeRange":
        """Create from relative duration."""
        return cls(start_ns=offset_ns, end_ns=offset_ns + duration_ns)

    @classmethod
    def from_ms(cls, start_ms: float, end_ms: float) -> "TimeRange":
        """Create from milliseconds."""
        return cls(start_ns=int(start_ms * 1_000_000), end_ns=int(end_ms * 1_000_000))

    @classmethod
    def from_us(cls, start_us: float, end_us: float) -> "TimeRange":
        """Create from microseconds."""
        return cls(start_ns=int(start_us * 1_000), end_ns=int(end_us * 1_000))

    def contains(self, timestamp_ns: int) -> bool:
        """Check if timestamp is within range."""
        return self.start_ns <= timestamp_ns <= self.end_ns

    def overlaps(self, start_ns: int, duration_ns: int) -> bool:
        """Check if an event overlaps this range."""
        end_ns = start_ns + duration_ns
        return not (end_ns < self.start_ns or start_ns > self.end_ns)

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass
class TraceFilter:
    """Filter specification for trace queries."""

    # Time filtering
    time_range: Optional[TimeRange] = None

    # Name filtering
    name_pattern: Optional[str] = None  # Regex pattern
    name_exact: Optional[str] = None
    name_prefix: Optional[str] = None

    # Category filtering
    categories: Optional[List[str]] = None
    exclude_categories: Optional[List[str]] = None

    # Event type filtering
    event_phases: Optional[List[str]] = None  # B, E, X, C, etc.

    # Duration filtering
    min_duration_ns: int = 0
    max_duration_ns: int = 0

    # Source filtering
    sources: Optional[List[str]] = None

    # Custom predicate
    predicate: Optional[Callable[[Dict[str, Any]], bool]] = None


@dataclass
class QueryResult:
    """Result of a trace query."""

    events: List[Dict[str, Any]]
    total_count: int
    matched_count: int
    time_range: Tuple[int, int]
    query_time_us: float = 0.0


@dataclass
class AggregationResult:
    """Result of trace aggregation."""

    metric: str
    aggregation: AggregationType
    value: float
    count: int
    breakdown: Optional[Dict[str, Any]] = None


class TraceQuery:
    """
    Query engine for exploring unified traces.

    Supports:
    - Time range filtering
    - Event name filtering (exact, prefix, regex)
    - Category filtering
    - Duration filtering
    - Aggregation operations
    - Event iteration
    """

    def __init__(self, trace_data: Optional[Dict[str, Any]] = None):
        self._events: List[Dict[str, Any]] = []
        self._metadata: Dict[str, Any] = {}

        if trace_data:
            self.load_trace(trace_data)

    def load_trace(self, trace_data: Dict[str, Any]) -> None:
        """Load trace data."""
        self._events = trace_data.get("traceEvents", [])
        self._metadata = trace_data.get("metadata", {})

    def load_file(self, path: Path) -> None:
        """Load trace from file."""
        with open(path) as f:
            self.load_trace(json.load(f))

    def _get_event_time_ns(self, event: Dict[str, Any]) -> int:
        """Get event timestamp in nanoseconds."""
        # Handle both Perfetto (us) and raw (ns) formats
        if "ts" in event:
            return int(event["ts"] * 1000)  # us to ns
        return event.get("timestamp_ns", 0)

    def _get_event_duration_ns(self, event: Dict[str, Any]) -> int:
        """Get event duration in nanoseconds."""
        if "dur" in event:
            return int(event["dur"] * 1000)  # us to ns
        return event.get("duration_ns", 0)

    def _matches_filter(self, event: Dict[str, Any], filter: TraceFilter) -> bool:
        """Check if event matches filter."""
        # Time range
        if filter.time_range:
            ts_ns = self._get_event_time_ns(event)
            dur_ns = self._get_event_duration_ns(event)
            if not filter.time_range.overlaps(ts_ns, dur_ns):
                return False

        # Name filtering
        name = event.get("name", "")

        if filter.name_exact and name != filter.name_exact:
            return False

        if filter.name_prefix and not name.startswith(filter.name_prefix):
            return False

        if filter.name_pattern:
            if not re.search(filter.name_pattern, name):
                return False

        # Category filtering
        cat = event.get("cat", "")

        if filter.categories and cat not in filter.categories:
            return False

        if filter.exclude_categories and cat in filter.exclude_categories:
            return False

        # Event phase filtering
        if filter.event_phases:
            phase = event.get("ph", "")
            if phase not in filter.event_phases:
                return False

        # Duration filtering
        dur_ns = self._get_event_duration_ns(event)

        if filter.min_duration_ns > 0 and dur_ns < filter.min_duration_ns:
            return False

        if filter.max_duration_ns > 0 and dur_ns > filter.max_duration_ns:
            return False

        # Source filtering
        if filter.sources:
            source = event.get("_source", "")
            if source not in filter.sources:
                return False

        # Custom predicate
        if filter.predicate and not filter.predicate(event):
            return False

        return True

    def query(
        self, filter: Optional[TraceFilter] = None, limit: Optional[int] = None
    ) -> QueryResult:
        """Query trace with filter."""
        import time

        start = time.perf_counter_ns()

        matched = []
        total = len(self._events)

        for event in self._events:
            if filter is None or self._matches_filter(event, filter):
                matched.append(event)
                if limit and len(matched) >= limit:
                    break

        # Compute time range
        if matched:
            ts = [self._get_event_time_ns(e) for e in matched]
            time_range = (min(ts), max(ts))
        else:
            time_range = (0, 0)

        query_time_us = (time.perf_counter_ns() - start) / 1000

        return QueryResult(
            events=matched,
            total_count=total,
            matched_count=len(matched),
            time_range=time_range,
            query_time_us=query_time_us,
        )

    def query_kernels(
        self, name_pattern: Optional[str] = None, min_duration_ns: int = 0
    ) -> List[Dict[str, Any]]:
        """Query GPU kernel events."""
        filter = TraceFilter(
            categories=["gpu_kernel", "gpu", "kernel"],
            name_pattern=name_pattern,
            min_duration_ns=min_duration_ns,
        )
        return self.query(filter).events

    def query_counters(self, counter_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query counter events."""
        filter = TraceFilter(event_phases=["C"], name_exact=counter_name)
        return self.query(filter).events

    def query_time_range(self, time_range: TimeRange) -> QueryResult:
        """Query events within time range."""
        filter = TraceFilter(time_range=time_range)
        return self.query(filter)

    def iterate(self, filter: Optional[TraceFilter] = None) -> Iterator[Dict[str, Any]]:
        """Iterate over matching events."""
        for event in self._events:
            if filter is None or self._matches_filter(event, filter):
                yield event

    def aggregate_duration(
        self,
        filter: Optional[TraceFilter] = None,
        aggregation: AggregationType = AggregationType.MEAN,
        percentile: float = 50.0,
    ) -> AggregationResult:
        """Aggregate event durations."""
        durations = []

        for event in self.iterate(filter):
            dur = self._get_event_duration_ns(event)
            if dur > 0:
                durations.append(dur)

        if not durations:
            return AggregationResult(
                metric="duration_ns", aggregation=aggregation, value=0, count=0
            )

        value = 0.0
        if aggregation == AggregationType.COUNT:
            value = len(durations)
        elif aggregation == AggregationType.SUM:
            value = sum(durations)
        elif aggregation == AggregationType.MEAN:
            value = statistics.mean(durations)
        elif aggregation == AggregationType.MIN:
            value = min(durations)
        elif aggregation == AggregationType.MAX:
            value = max(durations)
        elif aggregation == AggregationType.PERCENTILE:
            sorted_dur = sorted(durations)
            idx = int(len(sorted_dur) * percentile / 100)
            value = sorted_dur[min(idx, len(sorted_dur) - 1)]

        return AggregationResult(
            metric="duration_ns",
            aggregation=aggregation,
            value=value,
            count=len(durations),
        )

    def group_by_name(
        self, filter: Optional[TraceFilter] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group events by name."""
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for event in self.iterate(filter):
            name = event.get("name", "unknown")
            if name not in groups:
                groups[name] = []
            groups[name].append(event)

        return groups

    def summarize_kernels(self, filter: Optional[TraceFilter] = None) -> List[Dict[str, Any]]:
        """Summarize kernel statistics grouped by name."""
        # Default to kernel categories
        if filter is None:
            filter = TraceFilter(categories=["gpu_kernel", "gpu", "kernel"])

        groups = self.group_by_name(filter)
        summaries = []

        for name, events in groups.items():
            durations = [self._get_event_duration_ns(e) for e in events]
            durations = [d for d in durations if d > 0]

            if not durations:
                continue

            sorted_dur = sorted(durations)

            summary = {
                "name": name,
                "count": len(events),
                "total_ns": sum(durations),
                "mean_ns": statistics.mean(durations),
                "std_ns": statistics.stdev(durations) if len(durations) > 1 else 0,
                "min_ns": min(durations),
                "max_ns": max(durations),
                "p50_ns": sorted_dur[len(sorted_dur) // 2],
                "p90_ns": sorted_dur[int(len(sorted_dur) * 0.9)],
                "p99_ns": sorted_dur[int(len(sorted_dur) * 0.99)]
                if len(sorted_dur) >= 100
                else sorted_dur[-1],
            }
            summaries.append(summary)

        # Sort by total time descending
        summaries.sort(key=lambda x: x["total_ns"], reverse=True)
        return summaries

    def get_counter_series(self, counter_name: str) -> List[Tuple[int, float]]:
        """Get time series for a counter."""
        series = []

        for event in self._events:
            if event.get("ph") == "C" and event.get("name") == counter_name:
                ts = self._get_event_time_ns(event)
                args = event.get("args", {})
                # Counter value is usually the first (only) arg
                for key, value in args.items():
                    if isinstance(value, (int, float)):
                        series.append((ts, value))
                        break

        return series

    def get_event_count(self) -> int:
        """Get total event count."""
        return len(self._events)

    def get_time_bounds(self) -> Tuple[int, int]:
        """Get trace time bounds."""
        if not self._events:
            return (0, 0)

        timestamps = [self._get_event_time_ns(e) for e in self._events]
        return (min(timestamps), max(timestamps))

    def get_categories(self) -> List[str]:
        """Get unique categories in trace."""
        return list(set(e.get("cat", "") for e in self._events if e.get("cat")))

    def get_unique_event_names(self) -> List[str]:
        """Get unique event names."""
        return list(set(e.get("name", "") for e in self._events if e.get("name")))


class TraceAnalyzer:
    """High-level trace analysis utilities."""

    def __init__(self, query: TraceQuery):
        self.query = query

    def get_top_kernels(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get top N kernels by total time."""
        return self.query.summarize_kernels()[:n]

    def get_kernel_amplification_ratio(self, iteration_count: int) -> float:
        """Calculate kernel amplification ratio."""
        kernel_summaries = self.query.summarize_kernels()
        total_kernel_calls = sum(k["count"] for k in kernel_summaries)

        if iteration_count == 0:
            return 0.0
        return total_kernel_calls / iteration_count

    def identify_microkernels(self, threshold_us: float = 10.0) -> List[Dict[str, Any]]:
        """Identify microkernels (short duration)."""
        threshold_ns = int(threshold_us * 1000)
        return [k for k in self.query.summarize_kernels() if k["mean_ns"] < threshold_ns]

    def calculate_gpu_time_breakdown(self) -> Dict[str, float]:
        """Calculate GPU time breakdown by kernel category."""
        summaries = self.query.summarize_kernels()
        total_gpu_time = sum(k["total_ns"] for k in summaries)

        if total_gpu_time == 0:
            return {}

        breakdown = {}
        for k in summaries:
            name = k["name"]
            pct = k["total_ns"] / total_gpu_time * 100
            breakdown[name] = pct

        return breakdown

    def detect_outliers(self, threshold_sigma: float = 3.0) -> List[Dict[str, Any]]:
        """Detect duration outliers in kernels."""
        outliers = []

        for summary in self.query.summarize_kernels():
            if summary["std_ns"] == 0:
                continue

            mean = summary["mean_ns"]
            std = summary["std_ns"]
            threshold = mean + threshold_sigma * std

            if summary["max_ns"] > threshold:
                outliers.append(
                    {
                        "kernel": summary["name"],
                        "max_ns": summary["max_ns"],
                        "mean_ns": mean,
                        "std_ns": std,
                        "sigma": (summary["max_ns"] - mean) / std,
                    }
                )

        outliers.sort(key=lambda x: x["sigma"], reverse=True)
        return outliers
