"""
Unified Trace Events Schema
Cross-layer event model for AACO-X trace correlation.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================================
# Event Source Taxonomy
# ============================================================================

class EventSource(str, Enum):
    """Source layer for cross-layer telemetry."""
    INFERENCE = "inference"  # Model inference framework
    GPU_KERNEL = "gpu_kernel"  # rocprof kernel traces
    GPU_METRICS = "gpu_metrics"  # rocm-smi power/clock/temp
    CPU_SCHED = "cpu_sched"  # eBPF/procfs scheduler
    DRIVER = "driver"  # /dev/aaco kernel module
    SYSTEM = "system"  # System-level events


class EventType(str, Enum):
    """Event type within each source."""
    # Inference events
    ITER_START = "iter_start"
    ITER_END = "iter_end"
    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    PREFILL_START = "prefill_start"
    PREFILL_END = "prefill_end"
    DECODE_START = "decode_start"
    DECODE_END = "decode_end"
    
    # GPU kernel events
    KERNEL_DISPATCH = "kernel_dispatch"
    KERNEL_START = "kernel_start"
    KERNEL_END = "kernel_end"
    
    # GPU metrics events
    GPU_SAMPLE = "gpu_sample"
    CLOCK_CHANGE = "clock_change"
    THROTTLE_START = "throttle_start"
    THROTTLE_END = "throttle_end"
    
    # CPU scheduling events
    CTX_SWITCH = "ctx_switch"
    WAKEUP = "wakeup"
    PAGE_FAULT = "page_fault"
    PREEMPTION = "preemption"
    
    # Driver events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    MARKER = "marker"
    CPU_TIME = "cpu_time"
    
    # System events
    SPIKE_DETECTED = "spike_detected"
    REGRESSION_DETECTED = "regression_detected"
    ANOMALY = "anomaly"


# ============================================================================
# Unified Event Record
# ============================================================================

@dataclass
class UnifiedEvent:
    """
    Cross-layer event record.
    
    All events from all sources are normalized to this format for
    unified storage, querying, and correlation.
    """
    # Core fields (always present)
    t_ns: int  # Timestamp in nanoseconds since session start
    source: EventSource
    event_type: EventType
    
    # Process/thread context
    pid: int = 0
    tid: int = 0
    
    # Event-specific payload (as JSON-compatible dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    
    # Optional correlation fields
    iteration: Optional[int] = None  # Inference iteration number
    kernel_name: Optional[str] = None  # GPU kernel name
    duration_ns: Optional[int] = None  # Event duration (for span events)
    
    # Telemetry values (for counter events)
    value_f64: Optional[float] = None  # Float value
    value_i64: Optional[int] = None  # Int value
    
    # Debugging
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# Arrow/Parquet Schema
# ============================================================================

UNIFIED_EVENT_SCHEMA = pa.schema([
    pa.field("t_ns", pa.int64(), nullable=False),
    pa.field("source", pa.string(), nullable=False),
    pa.field("event_type", pa.string(), nullable=False),
    pa.field("pid", pa.int32()),
    pa.field("tid", pa.int32()),
    pa.field("iteration", pa.int32()),
    pa.field("kernel_name", pa.string()),
    pa.field("duration_ns", pa.int64()),
    pa.field("value_f64", pa.float64()),
    pa.field("value_i64", pa.int64()),
    pa.field("payload_json", pa.string()),  # JSON-encoded payload
    pa.field("session_id", pa.string()),
])


# ============================================================================
# Unified Trace Store
# ============================================================================

class UnifiedTraceStore:
    """
    In-memory and persistent store for unified events.
    
    Features:
    - Append events from any source
    - Query by time range, source, type
    - Export to Parquet for analysis
    - Import from multiple sources
    """
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self._events: List[UnifiedEvent] = []
        self._base_time_ns: Optional[int] = None
    
    @property
    def event_count(self) -> int:
        return len(self._events)
    
    def set_base_time(self, time_ns: int) -> None:
        """Set base time for relative timestamps."""
        self._base_time_ns = time_ns
    
    def _normalize_time(self, time_ns: int) -> int:
        """Normalize timestamp relative to base time."""
        if self._base_time_ns is None:
            self._base_time_ns = time_ns
        return time_ns - self._base_time_ns
    
    def append(self, event: UnifiedEvent) -> None:
        """Append a single event."""
        if event.session_id is None:
            event.session_id = self.session_id
        self._events.append(event)
    
    def append_batch(self, events: List[UnifiedEvent]) -> None:
        """Append multiple events."""
        for event in events:
            self.append(event)
    
    # ==========================================================================
    # Source-Specific Importers
    # ==========================================================================
    
    def import_inference_results(self, results: List[Dict[str, Any]],
                                  warmup_count: int = 10) -> int:
        """Import inference iteration results."""
        count = 0
        
        for i, result in enumerate(results):
            latency_ms = result.get("latency_ms", 0)
            start_ns = result.get("start_ns", 0)
            end_ns = result.get("end_ns", start_ns + int(latency_ms * 1_000_000))
            is_warmup = result.get("is_warmup", i < warmup_count)
            
            # Iteration start
            self.append(UnifiedEvent(
                t_ns=self._normalize_time(start_ns),
                source=EventSource.INFERENCE,
                event_type=EventType.ITER_START,
                iteration=i,
                payload={"is_warmup": is_warmup},
            ))
            
            # Iteration end
            self.append(UnifiedEvent(
                t_ns=self._normalize_time(end_ns),
                source=EventSource.INFERENCE,
                event_type=EventType.ITER_END,
                iteration=i,
                duration_ns=end_ns - start_ns,
                value_f64=latency_ms,
                payload={
                    "latency_ms": latency_ms,
                    "is_warmup": is_warmup,
                },
            ))
            count += 2
        
        return count
    
    def import_kernel_traces(self, kernels: List[Dict[str, Any]]) -> int:
        """Import GPU kernel trace data."""
        count = 0
        current_time = 0
        
        for kernel in kernels:
            name = kernel.get("kernel_name", kernel.get("name", "unknown"))
            total_us = kernel.get("total_time_ms", 0) * 1000
            calls = kernel.get("calls", 1)
            
            if calls == 0:
                continue
            
            avg_ns = int((total_us / calls) * 1000)
            
            # Create kernel dispatch/end events
            self.append(UnifiedEvent(
                t_ns=current_time,
                source=EventSource.GPU_KERNEL,
                event_type=EventType.KERNEL_DISPATCH,
                kernel_name=name,
                duration_ns=avg_ns,
                value_i64=calls,
                payload={
                    "total_calls": calls,
                    "avg_duration_us": total_us / calls if calls > 0 else 0,
                },
            ))
            
            current_time += avg_ns
            count += 1
        
        return count
    
    def import_gpu_samples(self, samples: List[Dict[str, Any]]) -> int:
        """Import GPU telemetry samples."""
        count = 0
        
        for sample in samples:
            t_ns = int(sample.get("t_ms", 0) * 1_000_000)
            t_ns = self._normalize_time(t_ns)
            
            # Create a single GPU_SAMPLE event with all metrics
            payload = {}
            value_f64 = None
            
            if "power_w" in sample:
                payload["power_w"] = sample["power_w"]
                value_f64 = sample["power_w"]
            
            if "gfx_clock_mhz" in sample:
                payload["gfx_clock_mhz"] = sample["gfx_clock_mhz"]
            
            if "mem_clock_mhz" in sample:
                payload["mem_clock_mhz"] = sample["mem_clock_mhz"]
            
            if "temp_c" in sample:
                payload["temp_c"] = sample["temp_c"]
            
            if "vram_used_mb" in sample:
                payload["vram_used_mb"] = sample["vram_used_mb"]
            
            if "gpu_util_pct" in sample:
                payload["gpu_util_pct"] = sample["gpu_util_pct"]
            
            self.append(UnifiedEvent(
                t_ns=t_ns,
                source=EventSource.GPU_METRICS,
                event_type=EventType.GPU_SAMPLE,
                value_f64=value_f64,
                payload=payload,
            ))
            count += 1
        
        return count
    
    def import_sched_events(self, events: List[Dict[str, Any]]) -> int:
        """Import CPU scheduling events."""
        count = 0
        
        for event in events:
            t_ns = event.get("t_ns", 0)
            event_type_str = event.get("event_type", "ctx_switch")
            
            # Map to EventType
            if event_type_str in ("ctx_switch", "context_switch"):
                event_type = EventType.CTX_SWITCH
            elif event_type_str == "wakeup":
                event_type = EventType.WAKEUP
            elif event_type_str in ("page_fault", "fault"):
                event_type = EventType.PAGE_FAULT
            else:
                event_type = EventType.CTX_SWITCH
            
            self.append(UnifiedEvent(
                t_ns=self._normalize_time(t_ns),
                source=EventSource.CPU_SCHED,
                event_type=event_type,
                pid=event.get("pid", 0),
                tid=event.get("tid", 0),
                payload=event,
            ))
            count += 1
        
        return count
    
    def import_driver_events(self, events: List[Dict[str, Any]]) -> int:
        """Import driver telemetry events."""
        count = 0
        
        for event in events:
            t_ns = event.get("t_ns", 0)
            event_type_code = event.get("event_type", 0)
            
            # Map driver event type code to EventType
            if event_type_code == 1:
                event_type = EventType.SESSION_START
            elif event_type_code == 2:
                event_type = EventType.SESSION_END
            elif event_type_code == 5:
                event_type = EventType.CPU_TIME
            elif event_type_code == 10:
                event_type = EventType.MARKER
            else:
                event_type = EventType.MARKER
            
            self.append(UnifiedEvent(
                t_ns=self._normalize_time(t_ns),
                source=EventSource.DRIVER,
                event_type=event_type,
                pid=event.get("pid", 0),
                tid=event.get("tid", 0),
                value_i64=event.get("value1", 0),
                payload=event,
            ))
            count += 1
        
        return count
    
    # ==========================================================================
    # Query Interface
    # ==========================================================================
    
    def query(self, source: Optional[EventSource] = None,
              event_type: Optional[EventType] = None,
              t_start_ns: Optional[int] = None,
              t_end_ns: Optional[int] = None,
              iteration: Optional[int] = None) -> List[UnifiedEvent]:
        """Query events with filters."""
        result = []
        
        for event in self._events:
            if source is not None and event.source != source:
                continue
            if event_type is not None and event.event_type != event_type:
                continue
            if t_start_ns is not None and event.t_ns < t_start_ns:
                continue
            if t_end_ns is not None and event.t_ns > t_end_ns:
                continue
            if iteration is not None and event.iteration != iteration:
                continue
            result.append(event)
        
        return result
    
    def get_time_range(self) -> tuple:
        """Get (min_t_ns, max_t_ns) of all events."""
        if not self._events:
            return (0, 0)
        
        min_t = min(e.t_ns for e in self._events)
        max_t = max(e.t_ns for e in self._events)
        return (min_t, max_t)
    
    def get_source_counts(self) -> Dict[str, int]:
        """Get event count by source."""
        counts: Dict[str, int] = {}
        for event in self._events:
            key = event.source.value
            counts[key] = counts.get(key, 0) + 1
        return counts
    
    # ==========================================================================
    # Persistence
    # ==========================================================================
    
    def to_arrow_table(self) -> pa.Table:
        """Convert to Arrow table."""
        data = {
            "t_ns": [],
            "source": [],
            "event_type": [],
            "pid": [],
            "tid": [],
            "iteration": [],
            "kernel_name": [],
            "duration_ns": [],
            "value_f64": [],
            "value_i64": [],
            "payload_json": [],
            "session_id": [],
        }
        
        for event in self._events:
            data["t_ns"].append(event.t_ns)
            data["source"].append(event.source.value)
            data["event_type"].append(event.event_type.value)
            data["pid"].append(event.pid)
            data["tid"].append(event.tid)
            data["iteration"].append(event.iteration)
            data["kernel_name"].append(event.kernel_name)
            data["duration_ns"].append(event.duration_ns)
            data["value_f64"].append(event.value_f64)
            data["value_i64"].append(event.value_i64)
            data["payload_json"].append(json.dumps(event.payload) if event.payload else None)
            data["session_id"].append(event.session_id)
        
        return pa.table(data, schema=UNIFIED_EVENT_SCHEMA)
    
    def save_parquet(self, path: Union[str, Path]) -> None:
        """Save to Parquet file."""
        table = self.to_arrow_table()
        pq.write_table(table, str(path), compression="zstd")
    
    @classmethod
    def load_parquet(cls, path: Union[str, Path]) -> "UnifiedTraceStore":
        """Load from Parquet file."""
        table = pq.read_table(str(path))
        
        store = cls()
        
        for i in range(table.num_rows):
            row = table.slice(i, 1)
            
            payload_json = row.column("payload_json")[0].as_py()
            payload = json.loads(payload_json) if payload_json else {}
            
            event = UnifiedEvent(
                t_ns=row.column("t_ns")[0].as_py(),
                source=EventSource(row.column("source")[0].as_py()),
                event_type=EventType(row.column("event_type")[0].as_py()),
                pid=row.column("pid")[0].as_py() or 0,
                tid=row.column("tid")[0].as_py() or 0,
                iteration=row.column("iteration")[0].as_py(),
                kernel_name=row.column("kernel_name")[0].as_py(),
                duration_ns=row.column("duration_ns")[0].as_py(),
                value_f64=row.column("value_f64")[0].as_py(),
                value_i64=row.column("value_i64")[0].as_py(),
                payload=payload,
                session_id=row.column("session_id")[0].as_py(),
            )
            store.append(event)
        
        return store
    
    def save_json(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        events_data = [e.to_dict() for e in self._events]
        
        # Convert enums to strings
        for e in events_data:
            e["source"] = e["source"].value if hasattr(e["source"], "value") else e["source"]
            e["event_type"] = e["event_type"].value if hasattr(e["event_type"], "value") else e["event_type"]
        
        with open(path, "w") as f:
            json.dump({
                "session_id": self.session_id,
                "event_count": len(events_data),
                "events": events_data,
            }, f, indent=2)


# ============================================================================
# Session Importer
# ============================================================================

def import_session_to_unified(session_path: Path) -> UnifiedTraceStore:
    """
    Import all data from an AACO session into a unified trace store.
    
    Args:
        session_path: Path to session directory
        
    Returns:
        UnifiedTraceStore with all imported events
    """
    session_path = Path(session_path)
    store = UnifiedTraceStore(session_id=session_path.name)
    
    total_imported = 0
    
    # Import inference results
    results_file = session_path / "inference_results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        count = store.import_inference_results(results)
        total_imported += count
    
    # Import kernel traces
    kernel_file = session_path / "kernel_summary.json"
    if kernel_file.exists():
        with open(kernel_file) as f:
            kernels = json.load(f)
        count = store.import_kernel_traces(kernels)
        total_imported += count
    
    # Import GPU samples
    gpu_file = session_path / "gpu_samples.json"
    if gpu_file.exists():
        with open(gpu_file) as f:
            samples = json.load(f)
        count = store.import_gpu_samples(samples)
        total_imported += count
    
    # Import driver events
    driver_file = session_path / "kernel_driver_events.json"
    if driver_file.exists():
        with open(driver_file) as f:
            events = json.load(f)
        count = store.import_driver_events(events)
        total_imported += count
    
    return store
