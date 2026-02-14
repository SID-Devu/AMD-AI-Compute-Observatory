"""
AACO-SIGMA Trace Merger

Multi-source trace merging with time alignment.
Handles clock skew, concurrent sources, and event ordering.
"""

import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path
from enum import Enum, auto


class TraceSourceType(Enum):
    """Types of trace sources."""
    ROCPROF = auto()
    ROCM_SMI = auto()
    CAPSULE = auto()
    NOISE_SENTINEL = auto()
    PERFETTO = auto()
    CUSTOM = auto()


@dataclass
class TraceSource:
    """Represents a trace data source."""
    source_type: TraceSourceType
    source_id: str
    events: List[Dict[str, Any]] = field(default_factory=list)
    time_offset_ns: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Time bounds
    start_time_ns: int = 0
    end_time_ns: int = 0
    
    def add_event(self, event: Dict[str, Any]) -> None:
        """Add an event to this source."""
        self.events.append(event)
        
        # Update time bounds
        ts = event.get("timestamp_ns", event.get("ts", 0))
        dur = event.get("duration_ns", event.get("dur", 0))
        
        if ts > 0:
            if self.start_time_ns == 0 or ts < self.start_time_ns:
                self.start_time_ns = ts
            end = ts + dur if dur else ts
            if end > self.end_time_ns:
                self.end_time_ns = end


@dataclass
class MergeOptions:
    """Options for trace merging."""
    # Time alignment
    align_to_first_event: bool = True
    reference_source: Optional[str] = None
    
    # Clock skew handling
    detect_clock_skew: bool = True
    max_clock_skew_us: int = 100  # Max acceptable skew
    
    # Event ordering
    sort_by_timestamp: bool = True
    deduplicate_events: bool = False
    
    # Filtering
    min_duration_ns: int = 0
    max_events: Optional[int] = None
    
    # Output
    preserve_source_metadata: bool = True


@dataclass
class MergeResult:
    """Result of trace merge operation."""
    events: List[Dict[str, Any]]
    sources: List[str]
    time_range: Tuple[int, int]  # (start_ns, end_ns)
    event_count: int
    clock_skew_detected: bool = False
    clock_skew_ns: int = 0
    warnings: List[str] = field(default_factory=list)


class TraceMerger:
    """
    Merges traces from multiple sources into a unified trace.
    
    Features:
    - Multi-source trace merging
    - Automatic time alignment
    - Clock skew detection and correction
    - Event deduplication
    - Hierarchical event ordering
    """
    
    def __init__(self, options: Optional[MergeOptions] = None):
        self.options = options or MergeOptions()
        self._sources: Dict[str, TraceSource] = {}
        self._reference_time_ns: int = 0
    
    def add_source(self, source: TraceSource) -> None:
        """Add a trace source."""
        self._sources[source.source_id] = source
    
    def add_rocprof_trace(self, events: List[Dict[str, Any]], 
                         source_id: str = "rocprof") -> None:
        """Add rocprof trace events."""
        source = TraceSource(
            source_type=TraceSourceType.ROCPROF,
            source_id=source_id,
            metadata={"profiler": "rocprof"}
        )
        for event in events:
            source.add_event(self._normalize_rocprof_event(event))
        self._sources[source_id] = source
    
    def add_rocm_smi_samples(self, samples: List[Dict[str, Any]],
                            source_id: str = "rocm_smi") -> None:
        """Add rocm-smi counter samples."""
        source = TraceSource(
            source_type=TraceSourceType.ROCM_SMI,
            source_id=source_id,
            metadata={"type": "counters"}
        )
        for sample in samples:
            source.add_event(self._normalize_counter_sample(sample))
        self._sources[source_id] = source
    
    def add_capsule_events(self, events: List[Dict[str, Any]],
                          source_id: str = "capsule") -> None:
        """Add capsule lifecycle events."""
        source = TraceSource(
            source_type=TraceSourceType.CAPSULE,
            source_id=source_id,
            metadata={"type": "capsule"}
        )
        for event in events:
            source.add_event(event)
        self._sources[source_id] = source
    
    def add_noise_events(self, events: List[Dict[str, Any]],
                        source_id: str = "noise") -> None:
        """Add noise/interference events."""
        source = TraceSource(
            source_type=TraceSourceType.NOISE_SENTINEL,
            source_id=source_id,
            metadata={"type": "noise"}
        )
        for event in events:
            source.add_event(event)
        self._sources[source_id] = source
    
    def add_perfetto_trace(self, trace_data: Dict[str, Any],
                          source_id: str = "perfetto") -> None:
        """Add existing Perfetto trace."""
        source = TraceSource(
            source_type=TraceSourceType.PERFETTO,
            source_id=source_id,
            metadata=trace_data.get("metadata", {})
        )
        
        events = trace_data.get("traceEvents", [])
        for event in events:
            source.add_event(event)
        
        self._sources[source_id] = source
    
    def _normalize_rocprof_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize rocprof event to Perfetto format."""
        normalized = {
            "name": event.get("kernel_name", event.get("name", "unknown")),
            "cat": "gpu_kernel",
            "ph": "X",  # Duration event
            "ts": event.get("start_ns", event.get("timestamp_ns", 0)) / 1000,  # ns to us
            "dur": event.get("duration_ns", event.get("dur_ns", 0)) / 1000,
            "pid": 1,
            "tid": event.get("queue_id", event.get("stream_id", 1)),
            "args": {
                "gpu_id": event.get("gpu_id", 0),
                "queue_id": event.get("queue_id", 0),
            }
        }
        
        # Copy additional fields
        for key in ["grid_size", "block_size", "occupancy"]:
            if key in event:
                normalized["args"][key] = event[key]
        
        return normalized
    
    def _normalize_counter_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize counter sample to Perfetto counter format."""
        return {
            "name": sample.get("counter_name", "counter"),
            "cat": "counter",
            "ph": "C",  # Counter event
            "ts": sample.get("timestamp_ns", 0) / 1000,
            "pid": 1,
            "tid": 0,
            "args": {key: val for key, val in sample.items() 
                    if key not in ["timestamp_ns", "counter_name"]}
        }
    
    def _detect_clock_skew(self) -> Tuple[bool, int]:
        """Detect clock skew between sources."""
        if len(self._sources) < 2:
            return False, 0
        
        # Get start times from each source
        start_times = []
        for source in self._sources.values():
            if source.start_time_ns > 0:
                start_times.append((source.source_id, source.start_time_ns))
        
        if len(start_times) < 2:
            return False, 0
        
        # Check for significant differences
        start_times.sort(key=lambda x: x[1])
        min_start = start_times[0][1]
        max_start = start_times[-1][1]
        
        skew_ns = max_start - min_start
        skew_us = skew_ns / 1000
        
        if skew_us > self.options.max_clock_skew_us:
            return True, skew_ns
        
        return False, skew_ns
    
    def _compute_time_offset(self, source: TraceSource) -> int:
        """Compute time offset to align source with reference."""
        if self._reference_time_ns == 0:
            return 0
        
        if source.start_time_ns == 0:
            return 0
        
        return self._reference_time_ns - source.start_time_ns
    
    def merge(self) -> MergeResult:
        """Merge all sources into unified trace."""
        if not self._sources:
            return MergeResult(
                events=[],
                sources=[],
                time_range=(0, 0),
                event_count=0
            )
        
        warnings: List[str] = []
        
        # Detect clock skew
        clock_skew_detected, clock_skew_ns = False, 0
        if self.options.detect_clock_skew:
            clock_skew_detected, clock_skew_ns = self._detect_clock_skew()
            if clock_skew_detected:
                warnings.append(
                    f"Clock skew detected: {clock_skew_ns/1000:.1f} µs"
                )
        
        # Determine reference time
        if self.options.reference_source and self.options.reference_source in self._sources:
            ref_source = self._sources[self.options.reference_source]
            self._reference_time_ns = ref_source.start_time_ns
        elif self.options.align_to_first_event:
            # Use earliest event as reference
            min_time = min(
                s.start_time_ns for s in self._sources.values() if s.start_time_ns > 0
            )
            self._reference_time_ns = min_time
        
        # Collect all events with time adjustment
        all_events: List[Dict[str, Any]] = []
        
        for source in self._sources.values():
            offset = self._compute_time_offset(source)
            
            for event in source.events:
                merged_event = event.copy()
                
                # Adjust timestamp
                if "ts" in merged_event:
                    merged_event["ts"] += offset / 1000  # ns to us
                if "timestamp_ns" in merged_event:
                    merged_event["timestamp_ns"] += offset
                
                # Add source metadata
                if self.options.preserve_source_metadata:
                    merged_event["_source"] = source.source_id
                    merged_event["_source_type"] = source.source_type.name
                
                # Apply duration filter
                dur = merged_event.get("dur", merged_event.get("duration_ns", 0))
                if dur >= self.options.min_duration_ns / 1000:  # Compare in us
                    all_events.append(merged_event)
        
        # Sort by timestamp
        if self.options.sort_by_timestamp:
            all_events.sort(key=lambda e: e.get("ts", e.get("timestamp_ns", 0)))
        
        # Deduplicate if requested
        if self.options.deduplicate_events:
            all_events = self._deduplicate_events(all_events)
        
        # Apply max events limit
        if self.options.max_events and len(all_events) > self.options.max_events:
            all_events = all_events[:self.options.max_events]
            warnings.append(f"Truncated to {self.options.max_events} events")
        
        # Compute time range
        if all_events:
            start_time = int(all_events[0].get("ts", 0) * 1000)  # us to ns
            end_time = int(all_events[-1].get("ts", 0) * 1000)
            end_dur = int(all_events[-1].get("dur", 0) * 1000)
            time_range = (start_time, end_time + end_dur)
        else:
            time_range = (0, 0)
        
        return MergeResult(
            events=all_events,
            sources=list(self._sources.keys()),
            time_range=time_range,
            event_count=len(all_events),
            clock_skew_detected=clock_skew_detected,
            clock_skew_ns=clock_skew_ns,
            warnings=warnings
        )
    
    def _deduplicate_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate events based on name/timestamp."""
        seen = set()
        deduped = []
        
        for event in events:
            # Create dedup key
            key = (
                event.get("name", ""),
                int(event.get("ts", 0)),
                event.get("cat", ""),
            )
            
            if key not in seen:
                seen.add(key)
                deduped.append(event)
        
        return deduped
    
    def to_perfetto_json(self) -> Dict[str, Any]:
        """Merge and export as Perfetto JSON."""
        result = self.merge()
        
        return {
            "traceEvents": result.events,
            "displayTimeUnit": "ns",
            "metadata": {
                "sources": result.sources,
                "event_count": result.event_count,
                "time_range_ns": result.time_range,
                "clock_skew_detected": result.clock_skew_detected,
                "warnings": result.warnings,
            }
        }
    
    def save(self, path: Path) -> None:
        """Save merged trace to file."""
        trace = self.to_perfetto_json()
        with open(path, 'w') as f:
            json.dump(trace, f)


class IncrementalMerger:
    """
    Streaming/incremental trace merger for large traces.
    
    Processes events in chunks to avoid memory issues.
    """
    
    def __init__(self, output_path: Path, options: Optional[MergeOptions] = None):
        self.output_path = output_path
        self.options = options or MergeOptions()
        self._event_count = 0
        self._time_range = (float('inf'), 0)
        self._sources: List[str] = []
        self._file = None
    
    def __enter__(self):
        """Start incremental merge."""
        self._file = open(self.output_path, 'w')
        self._file.write('{"traceEvents":[\n')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalize incremental merge."""
        if self._file:
            # Remove trailing comma if any events
            if self._event_count > 0:
                self._file.seek(self._file.tell() - 2)  # Remove ",\n"
                self._file.write('\n')
            
            # Write metadata
            self._file.write('],\n')
            self._file.write('"displayTimeUnit":"ns",\n')
            self._file.write(f'"metadata":{{\n')
            self._file.write(f'  "sources":{json.dumps(self._sources)},\n')
            self._file.write(f'  "event_count":{self._event_count}\n')
            self._file.write('}}\n')
            self._file.write('}\n')
            self._file.close()
    
    def add_events(self, events: List[Dict[str, Any]], source: str) -> int:
        """Add a batch of events."""
        if source not in self._sources:
            self._sources.append(source)
        
        count = 0
        for event in events:
            self._file.write(json.dumps(event))
            self._file.write(',\n')
            count += 1
        
        self._event_count += count
        return count
