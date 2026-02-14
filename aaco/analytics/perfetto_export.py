"""
Perfetto Trace Exporter
Exports AACO events to Perfetto trace format for unified timeline visualization.
"""

import json
import logging
import gzip
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Perfetto Trace Format
# ============================================================================

class PerfettoEventPhase(str, Enum):
    """Perfetto trace event phases."""
    BEGIN = "B"
    END = "E"
    COMPLETE = "X"
    INSTANT = "i"
    COUNTER = "C"
    ASYNC_BEGIN = "b"
    ASYNC_END = "e"
    ASYNC_INSTANT = "n"
    FLOW_START = "s"
    FLOW_END = "f"
    METADATA = "M"


@dataclass
class PerfettoEvent:
    """Single Perfetto trace event."""
    name: str
    cat: str  # Category
    ph: PerfettoEventPhase  # Phase
    ts: float  # Timestamp in microseconds
    pid: int
    tid: int
    args: Dict[str, Any] = field(default_factory=dict)
    
    # Optional fields
    dur: Optional[float] = None  # Duration (for complete events)
    id: Optional[Union[int, str]] = None  # Async event ID
    bp: Optional[str] = None  # Binding point for instant events
    cname: Optional[str] = None  # Color name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to Perfetto JSON format."""
        d = {
            "name": self.name,
            "cat": self.cat,
            "ph": self.ph.value if isinstance(self.ph, PerfettoEventPhase) else self.ph,
            "ts": self.ts,
            "pid": self.pid,
            "tid": self.tid,
        }
        
        if self.args:
            d["args"] = self.args
        
        if self.dur is not None:
            d["dur"] = self.dur
        
        if self.id is not None:
            d["id"] = self.id
        
        if self.bp:
            d["bp"] = self.bp
        
        if self.cname:
            d["cname"] = self.cname
        
        return d


@dataclass
class PerfettoMetadata:
    """Process/thread metadata for Perfetto."""
    pid: int
    tid: int
    name: str
    sort_index: int = 0
    
    def to_events(self) -> List[Dict[str, Any]]:
        """Generate metadata events."""
        return [
            {
                "name": "process_name",
                "ph": "M",
                "pid": self.pid,
                "args": {"name": self.name}
            },
            {
                "name": "thread_name", 
                "ph": "M",
                "pid": self.pid,
                "tid": self.tid,
                "args": {"name": self.name}
            },
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": self.pid,
                "args": {"sort_index": self.sort_index}
            }
        ]


# ============================================================================
# Track Definitions for AACO
# ============================================================================

class AACOTrack:
    """Track identifiers for AACO visualization."""
    # Process IDs (virtual, for grouping)
    PID_INFERENCE = 1000
    PID_GPU_KERNELS = 2000
    PID_GPU_METRICS = 3000
    PID_CPU_SCHED = 4000
    PID_DRIVER = 5000
    PID_SYSTEM = 6000
    
    # Thread IDs within processes
    TID_LATENCY = 1001
    TID_ITERATIONS = 1002
    TID_PHASES = 1003
    
    TID_KERNEL_TIMELINE = 2001
    TID_KERNEL_LAUNCHES = 2002
    
    TID_POWER = 3001
    TID_CLOCKS = 3002
    TID_TEMP = 3003
    TID_VRAM = 3004
    
    TID_CTX_SWITCHES = 4001
    TID_WAKEUPS = 4002
    TID_FAULTS = 4003
    
    TID_DRIVER_EVENTS = 5001
    TID_MARKERS = 5002


# ============================================================================
# Perfetto Trace Builder
# ============================================================================

class PerfettoTraceBuilder:
    """
    Builds Perfetto-compatible trace from AACO events.
    
    Supports:
    - Inference iteration spans
    - GPU kernel executions
    - GPU metrics (power, clocks, temp)
    - CPU scheduling events
    - Driver telemetry
    - Phase markers
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._events: List[Dict[str, Any]] = []
        self._metadata: List[PerfettoMetadata] = []
        self._base_time_us: Optional[float] = None  # For time normalization
        
        # Initialize default tracks
        self._init_tracks()
    
    def _init_tracks(self) -> None:
        """Initialize track metadata."""
        tracks = [
            PerfettoMetadata(AACOTrack.PID_INFERENCE, AACOTrack.TID_LATENCY, 
                            "Inference Latency", 0),
            PerfettoMetadata(AACOTrack.PID_INFERENCE, AACOTrack.TID_ITERATIONS,
                            "Iterations", 1),
            PerfettoMetadata(AACOTrack.PID_GPU_KERNELS, AACOTrack.TID_KERNEL_TIMELINE,
                            "GPU Kernels", 2),
            PerfettoMetadata(AACOTrack.PID_GPU_METRICS, AACOTrack.TID_POWER,
                            "GPU Power (W)", 3),
            PerfettoMetadata(AACOTrack.PID_GPU_METRICS, AACOTrack.TID_CLOCKS,
                            "GPU Clock (MHz)", 4),
            PerfettoMetadata(AACOTrack.PID_GPU_METRICS, AACOTrack.TID_TEMP,
                            "GPU Temp (°C)", 5),
            PerfettoMetadata(AACOTrack.PID_CPU_SCHED, AACOTrack.TID_CTX_SWITCHES,
                            "Context Switches", 6),
            PerfettoMetadata(AACOTrack.PID_DRIVER, AACOTrack.TID_DRIVER_EVENTS,
                            "Driver Events", 7),
        ]
        self._metadata = tracks
    
    def _normalize_time(self, time_ns: int) -> float:
        """Convert nanoseconds to microseconds, with base time normalization."""
        time_us = time_ns / 1000.0
        
        if self._base_time_us is None:
            self._base_time_us = time_us
        
        return time_us - self._base_time_us
    
    def add_inference_iteration(self, iteration: int, start_ns: int, 
                                 end_ns: int, latency_ms: float,
                                 is_warmup: bool = False) -> None:
        """Add an inference iteration span."""
        start_us = self._normalize_time(start_ns)
        dur_us = (end_ns - start_ns) / 1000.0
        
        category = "warmup" if is_warmup else "measure"
        color = "grey" if is_warmup else "good"
        
        self._events.append(PerfettoEvent(
            name=f"Iter {iteration}",
            cat="inference",
            ph=PerfettoEventPhase.COMPLETE,
            ts=start_us,
            dur=dur_us,
            pid=AACOTrack.PID_INFERENCE,
            tid=AACOTrack.TID_ITERATIONS,
            args={
                "iteration": iteration,
                "latency_ms": latency_ms,
                "phase": category,
            },
            cname=color,
        ).to_dict())
    
    def add_gpu_kernel(self, kernel_name: str, start_ns: int, 
                       duration_ns: int, **kwargs) -> None:
        """Add a GPU kernel execution span."""
        start_us = self._normalize_time(start_ns)
        dur_us = duration_ns / 1000.0
        
        # Truncate long kernel names
        display_name = kernel_name[-40:] if len(kernel_name) > 40 else kernel_name
        
        self._events.append(PerfettoEvent(
            name=display_name,
            cat="gpu_kernel",
            ph=PerfettoEventPhase.COMPLETE,
            ts=start_us,
            dur=dur_us,
            pid=AACOTrack.PID_GPU_KERNELS,
            tid=AACOTrack.TID_KERNEL_TIMELINE,
            args={
                "full_name": kernel_name,
                "duration_us": dur_us,
                **kwargs
            },
            cname="thread_state_running" if dur_us > 10 else "terrible",  # Color by duration
        ).to_dict())
    
    def add_gpu_counter(self, name: str, time_ns: int, value: float,
                        track_id: int = AACOTrack.TID_POWER) -> None:
        """Add a GPU metric counter sample."""
        ts = self._normalize_time(time_ns)
        
        self._events.append({
            "name": name,
            "cat": "gpu_metrics",
            "ph": "C",
            "ts": ts,
            "pid": AACOTrack.PID_GPU_METRICS,
            "tid": track_id,
            "args": {name: value}
        })
    
    def add_sched_event(self, event_type: str, time_ns: int,
                        value: int = 0, **kwargs) -> None:
        """Add a CPU scheduling event."""
        ts = self._normalize_time(time_ns)
        
        # Instant event with process scope
        self._events.append({
            "name": event_type,
            "cat": "sched",
            "ph": "i",
            "ts": ts,
            "pid": AACOTrack.PID_CPU_SCHED,
            "tid": AACOTrack.TID_CTX_SWITCHES,
            "s": "p",  # Process scope
            "args": {"value": value, **kwargs}
        })
    
    def add_driver_event(self, event_type: str, time_ns: int,
                         value1: int, value2: int, comm: str = "") -> None:
        """Add a driver telemetry event."""
        ts = self._normalize_time(time_ns)
        
        self._events.append({
            "name": event_type,
            "cat": "driver",
            "ph": "i",
            "ts": ts,
            "pid": AACOTrack.PID_DRIVER,
            "tid": AACOTrack.TID_DRIVER_EVENTS,
            "s": "p",
            "args": {
                "value1": value1,
                "value2": value2,
                "comm": comm,
            }
        })
    
    def add_phase_marker(self, phase_name: str, time_ns: int,
                         phase_type: str = "begin") -> None:
        """Add a phase marker (warmup, measure, prefill, decode)."""
        ts = self._normalize_time(time_ns)
        
        ph = PerfettoEventPhase.BEGIN if phase_type == "begin" else PerfettoEventPhase.END
        
        self._events.append({
            "name": f"Phase: {phase_name}",
            "cat": "phase",
            "ph": ph.value,
            "ts": ts,
            "pid": AACOTrack.PID_INFERENCE,
            "tid": AACOTrack.TID_PHASES,
        })
    
    def add_latency_spike(self, time_ns: int, latency_ms: float,
                          cause: str = "unknown") -> None:
        """Add a latency spike marker."""
        ts = self._normalize_time(time_ns)
        
        self._events.append({
            "name": f"Spike: {latency_ms:.1f}ms",
            "cat": "anomaly",
            "ph": "i",
            "ts": ts,
            "pid": AACOTrack.PID_INFERENCE,
            "tid": AACOTrack.TID_LATENCY,
            "s": "g",  # Global scope
            "args": {
                "latency_ms": latency_ms,
                "cause": cause,
            },
            "cname": "terrible",
        })
    
    def add_custom_event(self, event: PerfettoEvent) -> None:
        """Add a custom Perfetto event."""
        self._events.append(event.to_dict())
    
    def build(self) -> Dict[str, Any]:
        """Build the complete Perfetto trace."""
        # Generate metadata events
        metadata_events = []
        for meta in self._metadata:
            metadata_events.extend(meta.to_events())
        
        # Combine all events
        all_events = metadata_events + self._events
        
        return {
            "traceEvents": all_events,
            "metadata": {
                "aaco_session": self.session_id,
                "format_version": 1,
            },
            "displayTimeUnit": "ms",
        }
    
    def save_json(self, path: Union[str, Path], compress: bool = False) -> None:
        """Save trace to JSON file."""
        trace = self.build()
        path = Path(path)
        
        if compress:
            with gzip.open(path.with_suffix('.json.gz'), 'wt') as f:
                json.dump(trace, f, indent=2)
        else:
            with open(path, 'w') as f:
                json.dump(trace, f, indent=2)
        
        logger.info(f"Perfetto trace saved: {path} ({len(self._events)} events)")


# ============================================================================
# Session to Perfetto Converter
# ============================================================================

class AACOSessionToPerfetto:
    """
    Converts AACO session data to Perfetto trace format.
    
    Integrates:
    - Inference results
    - Kernel traces
    - GPU telemetry
    - Driver events
    - Scheduling events
    """
    
    def __init__(self, session_path: Path):
        self.session_path = Path(session_path)
        self.builder: Optional[PerfettoTraceBuilder] = None
    
    def convert(self) -> PerfettoTraceBuilder:
        """Convert session to Perfetto trace."""
        # Get session ID from path
        session_id = self.session_path.name
        self.builder = PerfettoTraceBuilder(session_id)
        
        # Load and convert each data source
        self._load_inference_results()
        self._load_kernel_traces()
        self._load_gpu_samples()
        self._load_driver_events()
        
        return self.builder
    
    def _load_inference_results(self) -> None:
        """Load and convert inference results."""
        results_file = self.session_path / "inference_results.json"
        if not results_file.exists():
            return
        
        with open(results_file) as f:
            results = json.load(f)
        
        # Convert to iterations
        for i, result in enumerate(results):
            # Estimate timestamps if not present
            latency_ms = result.get("latency_ms", 0)
            is_warmup = result.get("is_warmup", i < 10)
            
            # Use actual timestamps if available, otherwise estimate
            start_ns = result.get("start_ns", i * int(latency_ms * 1_000_000))
            end_ns = result.get("end_ns", start_ns + int(latency_ms * 1_000_000))
            
            self.builder.add_inference_iteration(
                iteration=i,
                start_ns=start_ns,
                end_ns=end_ns,
                latency_ms=latency_ms,
                is_warmup=is_warmup,
            )
    
    def _load_kernel_traces(self) -> None:
        """Load and convert kernel traces."""
        kernel_file = self.session_path / "kernel_summary.json"
        if not kernel_file.exists():
            return
        
        with open(kernel_file) as f:
            kernels = json.load(f)
        
        # Group kernels into timeline (simplified - assumes sequential)
        current_time_ns = 0
        
        for kernel in kernels:
            name = kernel.get("kernel_name", "unknown")
            total_time_us = kernel.get("total_time_ms", 0) * 1000
            calls = kernel.get("calls", 1)
            avg_time_us = total_time_us / calls if calls > 0 else 0
            
            # Add individual kernel spans
            for _ in range(min(calls, 100)):  # Limit to avoid huge traces
                self.builder.add_gpu_kernel(
                    kernel_name=name,
                    start_ns=current_time_ns,
                    duration_ns=int(avg_time_us * 1000),
                    calls=calls,
                )
                current_time_ns += int(avg_time_us * 1000) + 100  # Small gap
    
    def _load_gpu_samples(self) -> None:
        """Load and convert GPU telemetry samples."""
        gpu_file = self.session_path / "gpu_samples.json"
        if not gpu_file.exists():
            return
        
        with open(gpu_file) as f:
            samples = json.load(f)
        
        for sample in samples:
            t_ns = int(sample.get("t_ms", 0) * 1_000_000)
            
            # Power
            if "power_w" in sample:
                self.builder.add_gpu_counter("power_w", t_ns, sample["power_w"],
                                            AACOTrack.TID_POWER)
            
            # Clock
            if "gfx_clock_mhz" in sample:
                self.builder.add_gpu_counter("gfx_clock_mhz", t_ns, 
                                            sample["gfx_clock_mhz"],
                                            AACOTrack.TID_CLOCKS)
            
            # Temperature
            if "temp_c" in sample:
                self.builder.add_gpu_counter("temp_c", t_ns, sample["temp_c"],
                                            AACOTrack.TID_TEMP)
    
    def _load_driver_events(self) -> None:
        """Load and convert driver events."""
        driver_file = self.session_path / "kernel_driver_events.json"
        if not driver_file.exists():
            return
        
        with open(driver_file) as f:
            events = json.load(f)
        
        for event in events:
            self.builder.add_driver_event(
                event_type=event.get("event_type_name", "unknown"),
                time_ns=event.get("t_ns", 0),
                value1=event.get("value1", 0),
                value2=event.get("value2", 0),
                comm=event.get("comm", ""),
            )
    
    def save(self, output_path: Optional[Path] = None, 
             compress: bool = False) -> Path:
        """Save the converted trace."""
        if self.builder is None:
            self.convert()
        
        if output_path is None:
            output_path = self.session_path / "trace.perfetto.json"
        
        self.builder.save_json(output_path, compress=compress)
        return output_path


# ============================================================================
# Utility Functions
# ============================================================================

def open_in_perfetto(trace_path: Union[str, Path]) -> None:
    """
    Open trace in Perfetto UI (browser).
    
    Note: Requires trace to be accessible via file:// or served via HTTP.
    """
    import webbrowser
    
    trace_path = Path(trace_path).resolve()
    
    # Perfetto UI URL
    perfetto_url = "https://ui.perfetto.dev"
    
    # For local files, user needs to drag-drop or use --enable-file-access-from-files
    print(f"Open {perfetto_url} and drag-drop: {trace_path}")
    webbrowser.open(perfetto_url)


def create_trace_from_session(session_path: Union[str, Path],
                              output_path: Optional[Path] = None,
                              compress: bool = False) -> Path:
    """
    Create Perfetto trace from AACO session.
    
    Args:
        session_path: Path to AACO session directory
        output_path: Output trace path (default: session/trace.perfetto.json)
        compress: Whether to gzip the output
        
    Returns:
        Path to created trace file
    """
    converter = AACOSessionToPerfetto(Path(session_path))
    return converter.save(output_path, compress=compress)
