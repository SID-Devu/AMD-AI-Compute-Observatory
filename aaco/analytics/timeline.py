"""
Unified Timeline Correlator
Aligns and correlates events from all planes (kernel, GPU, system, inference) to a single timeline.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import bisect

import numpy as np

logger = logging.getLogger(__name__)


class EventSource(Enum):
    """Source plane for an event."""
    INFERENCE = "inference"
    KERNEL = "kernel"
    SYSTEM = "system"
    GPU = "gpu"
    MEMORY = "memory"
    POWER = "power"


class EventType(Enum):
    """Type of event."""
    # Inference events
    INFERENCE_START = "inference_start"
    INFERENCE_END = "inference_end"
    TOKEN_GENERATED = "token_generated"
    
    # Kernel events
    KERNEL_LAUNCH = "kernel_launch"
    KERNEL_COMPLETE = "kernel_complete"
    
    # System events
    CPU_SAMPLE = "cpu_sample"
    MEMORY_SAMPLE = "memory_sample"
    CONTEXT_SWITCH = "context_switch"
    PAGE_FAULT = "page_fault"
    
    # GPU events
    GPU_SAMPLE = "gpu_sample"
    CLOCK_CHANGE = "clock_change"
    THROTTLE = "throttle"
    
    # Power events
    POWER_SAMPLE = "power_sample"


@dataclass
class TimelineEvent:
    """Single event on the unified timeline."""
    t_ns: int  # Nanoseconds from session start
    source: EventSource
    event_type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    duration_ns: Optional[int] = None
    correlation_id: Optional[str] = None  # For grouping related events
    
    @property
    def t_ms(self) -> float:
        return self.t_ns / 1e6
    
    @property
    def t_us(self) -> float:
        return self.t_ns / 1e3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_ns": self.t_ns,
            "t_ms": self.t_ms,
            "source": self.source.value,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "duration_ns": self.duration_ns,
            "correlation_id": self.correlation_id,
        }


@dataclass
class CorrelationWindow:
    """Window of correlated events."""
    start_ns: int
    end_ns: int
    inference_events: List[TimelineEvent]
    kernel_events: List[TimelineEvent]
    system_events: List[TimelineEvent]
    gpu_events: List[TimelineEvent]
    
    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6
    
    def get_summary(self) -> Dict[str, Any]:
        kernel_time = sum(
            e.duration_ns or 0 for e in self.kernel_events
        ) / 1e6  # ms
        
        return {
            "duration_ms": self.duration_ms,
            "kernel_count": len(self.kernel_events),
            "kernel_time_ms": kernel_time,
            "gpu_active_ratio": kernel_time / self.duration_ms if self.duration_ms > 0 else 0,
            "cpu_samples": len(self.system_events),
            "gpu_samples": len(self.gpu_events),
        }


@dataclass
class CorrelationInsight:
    """Insight derived from correlation analysis."""
    category: str
    severity: str  # "info", "warning", "critical"
    description: str
    evidence: Dict[str, Any]
    timestamp_ns: Optional[int] = None


class TimelineCorrelator:
    """
    Correlates events from all observability planes to unified timeline.
    
    Key capabilities:
    - Align all events to common t0
    - Window-based correlation for inference iterations
    - Spike detection and attribution
    - Cross-plane causality inference
    """
    
    def __init__(self, t0_ns: int = 0):
        """
        Args:
            t0_ns: Session start time in nanoseconds (monotonic)
        """
        self.t0_ns = t0_ns
        self.events: List[TimelineEvent] = []
        self._sorted = False
    
    def set_t0(self, t0_ns: int) -> None:
        """Set the reference time for all events."""
        self.t0_ns = t0_ns
    
    def add_event(
        self,
        t_ns: int,
        source: EventSource,
        event_type: EventType,
        payload: Optional[Dict[str, Any]] = None,
        duration_ns: Optional[int] = None,
        correlation_id: Optional[str] = None,
    ) -> TimelineEvent:
        """Add an event to the timeline."""
        # Normalize to relative time
        relative_t = t_ns - self.t0_ns if self.t0_ns > 0 else t_ns
        
        event = TimelineEvent(
            t_ns=relative_t,
            source=source,
            event_type=event_type,
            payload=payload or {},
            duration_ns=duration_ns,
            correlation_id=correlation_id,
        )
        self.events.append(event)
        self._sorted = False
        return event
    
    def add_inference_iteration(
        self, iter_idx: int, t_start_ns: int, t_end_ns: int, latency_ms: float
    ) -> None:
        """Add inference iteration boundary events."""
        corr_id = f"iter_{iter_idx}"
        
        self.add_event(
            t_ns=t_start_ns,
            source=EventSource.INFERENCE,
            event_type=EventType.INFERENCE_START,
            payload={"iter_idx": iter_idx},
            correlation_id=corr_id,
        )
        
        self.add_event(
            t_ns=t_end_ns,
            source=EventSource.INFERENCE,
            event_type=EventType.INFERENCE_END,
            payload={"iter_idx": iter_idx, "latency_ms": latency_ms},
            duration_ns=t_end_ns - t_start_ns,
            correlation_id=corr_id,
        )
    
    def add_kernel_execution(
        self, kernel_name: str, t_start_ns: int, t_end_ns: int
    ) -> None:
        """Add GPU kernel execution event."""
        self.add_event(
            t_ns=t_start_ns,
            source=EventSource.KERNEL,
            event_type=EventType.KERNEL_LAUNCH,
            payload={
                "kernel_name": kernel_name,
            },
            duration_ns=t_end_ns - t_start_ns,
        )
    
    def add_system_sample(
        self, t_ns: int, cpu_pct: float, rss_mb: float, ctx_switches: int
    ) -> None:
        """Add system telemetry sample."""
        self.add_event(
            t_ns=t_ns,
            source=EventSource.SYSTEM,
            event_type=EventType.CPU_SAMPLE,
            payload={
                "cpu_pct": cpu_pct,
                "rss_mb": rss_mb,
                "ctx_switches": ctx_switches,
            },
        )
    
    def add_gpu_sample(
        self, t_ns: int, gfx_clock: float, power_w: float, temp_c: float, util_pct: float
    ) -> None:
        """Add GPU telemetry sample."""
        self.add_event(
            t_ns=t_ns,
            source=EventSource.GPU,
            event_type=EventType.GPU_SAMPLE,
            payload={
                "gfx_clock_mhz": gfx_clock,
                "power_w": power_w,
                "temp_c": temp_c,
                "gpu_util_pct": util_pct,
            },
        )
    
    def _ensure_sorted(self) -> None:
        """Ensure events are sorted by time."""
        if not self._sorted:
            self.events.sort(key=lambda e: e.t_ns)
            self._sorted = True
    
    def get_events_in_window(
        self, start_ns: int, end_ns: int, source: Optional[EventSource] = None
    ) -> List[TimelineEvent]:
        """Get all events within a time window."""
        self._ensure_sorted()
        
        result = []
        for event in self.events:
            if event.t_ns < start_ns:
                continue
            if event.t_ns > end_ns:
                break
            if source is None or event.source == source:
                result.append(event)
        
        return result
    
    def correlate_inference_iterations(self) -> List[CorrelationWindow]:
        """
        Create correlation windows for each inference iteration.
        Each window contains all events during that iteration.
        """
        self._ensure_sorted()
        
        # Find inference boundaries
        start_events = [
            e for e in self.events 
            if e.event_type == EventType.INFERENCE_START
        ]
        end_events = {
            e.correlation_id: e for e in self.events 
            if e.event_type == EventType.INFERENCE_END
        }
        
        windows = []
        for start in start_events:
            end = end_events.get(start.correlation_id)
            if not end:
                continue
            
            window_events = self.get_events_in_window(start.t_ns, end.t_ns)
            
            window = CorrelationWindow(
                start_ns=start.t_ns,
                end_ns=end.t_ns,
                inference_events=[e for e in window_events if e.source == EventSource.INFERENCE],
                kernel_events=[e for e in window_events if e.source == EventSource.KERNEL],
                system_events=[e for e in window_events if e.source == EventSource.SYSTEM],
                gpu_events=[e for e in window_events if e.source == EventSource.GPU],
            )
            windows.append(window)
        
        return windows
    
    def detect_latency_spikes(
        self, threshold_pct: float = 50.0
    ) -> List[Dict[str, Any]]:
        """
        Detect inference iterations with abnormal latency.
        
        Args:
            threshold_pct: % above median to flag as spike
            
        Returns:
            List of spike events with correlation evidence.
        """
        windows = self.correlate_inference_iterations()
        if len(windows) < 5:
            return []
        
        # Compute median latency
        latencies = [w.duration_ms for w in windows]
        median = np.median(latencies)
        threshold = median * (1 + threshold_pct / 100)
        
        spikes = []
        for i, window in enumerate(windows):
            if window.duration_ms > threshold:
                # Analyze what happened during spike
                evidence = self._analyze_spike(window, windows, median)
                
                spikes.append({
                    "iteration": i,
                    "latency_ms": window.duration_ms,
                    "median_ms": median,
                    "spike_pct": ((window.duration_ms - median) / median * 100),
                    "evidence": evidence,
                    "t_start_ns": window.start_ns,
                })
        
        return spikes
    
    def _analyze_spike(
        self, 
        spike_window: CorrelationWindow,
        all_windows: List[CorrelationWindow],
        median_latency: float
    ) -> Dict[str, Any]:
        """Analyze potential causes of a latency spike."""
        evidence = {"likely_causes": []}
        
        # Get typical values from other windows
        typical_kernel_counts = [
            len(w.kernel_events) for w in all_windows 
            if w.duration_ms < median_latency * 1.2
        ]
        
        # Check kernel count anomaly
        if typical_kernel_counts:
            median_kernels = np.median(typical_kernel_counts)
            spike_kernels = len(spike_window.kernel_events)
            if spike_kernels > median_kernels * 1.5:
                evidence["likely_causes"].append({
                    "cause": "kernel_explosion",
                    "details": f"Kernel count: {spike_kernels} vs median {median_kernels:.0f}",
                })
        
        # Check CPU jitter
        cpu_samples = [
            e.payload.get("cpu_pct", 0) for e in spike_window.system_events
        ]
        if cpu_samples and max(cpu_samples) > 90:
            evidence["likely_causes"].append({
                "cause": "cpu_contention",
                "details": f"CPU peaked at {max(cpu_samples):.1f}%",
            })
        
        # Check context switches
        ctx_samples = [
            e.payload.get("ctx_switches", 0) for e in spike_window.system_events
        ]
        if ctx_samples and np.std(ctx_samples) > np.mean(ctx_samples):
            evidence["likely_causes"].append({
                "cause": "scheduling_jitter",
                "details": f"High context switch variance",
            })
        
        # Check clock drops
        clock_samples = [
            e.payload.get("gfx_clock_mhz", 0) for e in spike_window.gpu_events
        ]
        if clock_samples:
            clock_range = max(clock_samples) - min(clock_samples)
            if clock_range > max(clock_samples) * 0.1:
                evidence["likely_causes"].append({
                    "cause": "clock_fluctuation",
                    "details": f"Clock varied by {clock_range:.0f} MHz",
                })
        
        if not evidence["likely_causes"]:
            evidence["likely_causes"].append({
                "cause": "unknown",
                "details": "No obvious cause detected",
            })
        
        return evidence
    
    def compute_gpu_active_timeline(
        self, resolution_ms: float = 10.0
    ) -> Dict[str, List[float]]:
        """
        Compute GPU active ratio over time at given resolution.
        
        Returns:
            Dict with 't_ms' and 'gpu_active_pct' time series.
        """
        self._ensure_sorted()
        
        if not self.events:
            return {"t_ms": [], "gpu_active_pct": []}
        
        max_t = max(e.t_ns for e in self.events)
        resolution_ns = int(resolution_ms * 1e6)
        
        t_points = []
        active_pcts = []
        
        t = 0
        while t < max_t:
            window_end = t + resolution_ns
            
            # Get kernel events in this window
            kernel_events = self.get_events_in_window(t, window_end, EventSource.KERNEL)
            
            # Sum kernel durations
            kernel_time = sum(e.duration_ns or 0 for e in kernel_events)
            active_pct = (kernel_time / resolution_ns * 100) if resolution_ns > 0 else 0
            
            t_points.append(t / 1e6)  # Convert to ms
            active_pcts.append(min(100, active_pct))
            
            t += resolution_ns
        
        return {"t_ms": t_points, "gpu_active_pct": active_pcts}
    
    def generate_insights(self) -> List[CorrelationInsight]:
        """Generate insights from correlation analysis."""
        insights = []
        
        # Check for latency spikes
        spikes = self.detect_latency_spikes()
        if spikes:
            insights.append(CorrelationInsight(
                category="latency_spike",
                severity="warning" if len(spikes) < 3 else "critical",
                description=f"Detected {len(spikes)} latency spike(s)",
                evidence={"spike_count": len(spikes), "spikes": spikes[:5]},
            ))
        
        # Compute overall GPU activity
        windows = self.correlate_inference_iterations()
        if windows:
            avg_gpu_active = np.mean([
                w.get_summary()["gpu_active_ratio"] for w in windows
            ])
            
            if avg_gpu_active < 0.5:
                insights.append(CorrelationInsight(
                    category="low_gpu_utilization",
                    severity="warning",
                    description=f"GPU active only {avg_gpu_active:.0%} during inference",
                    evidence={"gpu_active_ratio": avg_gpu_active},
                ))
        
        # Check for system noise correlation
        system_events = [e for e in self.events if e.source == EventSource.SYSTEM]
        if system_events:
            ctx_switches = [e.payload.get("ctx_switches", 0) for e in system_events]
            if np.std(ctx_switches) > np.mean(ctx_switches) * 0.5:
                insights.append(CorrelationInsight(
                    category="system_noise",
                    severity="info",
                    description="High variance in context switches detected",
                    evidence={
                        "ctx_switch_mean": float(np.mean(ctx_switches)),
                        "ctx_switch_std": float(np.std(ctx_switches)),
                    },
                ))
        
        return insights
    
    def export_timeline(self) -> List[Dict[str, Any]]:
        """Export full timeline as list of dicts."""
        self._ensure_sorted()
        return [e.to_dict() for e in self.events]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get timeline summary statistics."""
        self._ensure_sorted()
        
        if not self.events:
            return {"total_events": 0}
        
        by_source = {}
        for source in EventSource:
            count = sum(1 for e in self.events if e.source == source)
            if count > 0:
                by_source[source.value] = count
        
        duration_ns = self.events[-1].t_ns - self.events[0].t_ns
        
        return {
            "total_events": len(self.events),
            "by_source": by_source,
            "duration_ms": duration_ns / 1e6,
            "time_range": {
                "start_ns": self.events[0].t_ns,
                "end_ns": self.events[-1].t_ns,
            },
        }
