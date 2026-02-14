"""
AACO-SIGMA Unified Trace Lake

Trace-first design where everything becomes a trace event.
Primary: Perfetto trace (openable timeline)
Secondary: Parquet feature store
"""

from aaco.tracelake.perfetto_lake import (
    PerfettoTraceLake,
    PerfettoTrack,
    TrackType,
    PerfettoEvent,
    InstantEvent,
    DurationEvent,
    CounterEvent,
    FlowEvent,
)

from aaco.tracelake.parquet_store import (
    ParquetFeatureStore,
    SessionFeatureTable,
    KernelFeatureTable,
    IterationFeatureTable,
)

from aaco.tracelake.track_registry import (
    TrackRegistry,
    TrackDefinition,
    StandardTracks,
    register_track,
    get_track,
)

from aaco.tracelake.trace_merger import (
    TraceMerger,
    TraceSource,
    MergedTrace,
    merge_traces,
)

from aaco.tracelake.trace_query import (
    TraceQuery,
    TimeRange,
    TraceFilter,
    query_trace,
)

__all__ = [
    # Perfetto Lake
    "PerfettoTraceLake",
    "PerfettoTrack",
    "TrackType",
    "PerfettoEvent",
    "InstantEvent",
    "DurationEvent",
    "CounterEvent",
    "FlowEvent",
    # Parquet Store
    "ParquetFeatureStore",
    "SessionFeatureTable",
    "KernelFeatureTable",
    "IterationFeatureTable",
    # Track Registry
    "TrackRegistry",
    "TrackDefinition",
    "StandardTracks",
    "register_track",
    "get_track",
    # Trace Merger
    "TraceMerger",
    "TraceSource",
    "MergedTrace",
    "merge_traces",
    # Trace Query
    "TraceQuery",
    "TimeRange",
    "TraceFilter",
    "query_trace",
]
