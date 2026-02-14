"""
Noise Sentinel - Runtime Interference Detection System.

Monitors for events that could contaminate measurement accuracy:
- IRQ storms
- Memory reclaim storms
- Thermal throttling
- Other process contention
"""

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ============================================================================
# Noise Event Types
# ============================================================================

class NoiseSource(str, Enum):
    """Types of noise sources that can affect measurements."""
    IRQ_STORM = "irq_storm"  # High interrupt rate
    RECLAIM_STORM = "reclaim_storm"  # Memory pressure / page reclaim
    THERMAL_THROTTLE = "thermal_throttle"  # CPU/GPU thermal limiting
    CPU_MIGRATION = "cpu_migration"  # Task moved between CPUs
    CONTEXT_SWITCH = "context_switch"  # Excessive context switches
    GPU_RESET = "gpu_reset"  # GPU reset event
    PAGE_FAULT = "page_fault"  # Major page faults
    IO_STALL = "io_stall"  # IO wait spikes
    OTHER_PROCESS = "other_process"  # Competing workload detected


@dataclass
class NoiseEvent:
    """A detected noise event."""
    timestamp: float
    source: NoiseSource
    severity: float  # 0.0 = minor, 1.0 = severe
    description: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "source": self.source.value,
            "severity": self.severity,
            "description": self.description,
            "metrics": self.metrics,
        }


@dataclass
class NoiseReport:
    """Summary of noise detected during a measurement window."""
    start_time: float
    end_time: float
    events: List[NoiseEvent] = field(default_factory=list)
    contamination_score: float = 0.0  # 0 = clean, 1 = severely contaminated
    recommended_discard: bool = False
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "events": [e.to_dict() for e in self.events],
            "contamination_score": self.contamination_score,
            "recommended_discard": self.recommended_discard,
            "summary": self.summary,
        }
    
    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# Threshold Configuration
# ============================================================================

@dataclass
class NoiseThresholds:
    """Thresholds for noise detection."""
    # IRQ rate (per second)
    irq_rate_warn: float = 10000
    irq_rate_severe: float = 50000
    
    # Memory reclaim (kswapd activity)
    reclaim_pages_warn: int = 1000
    reclaim_pages_severe: int = 10000
    
    # Thermal (degrees C from throttle point)
    thermal_margin_warn: int = 5
    thermal_margin_severe: int = 0
    
    # Context switches (per second)
    ctx_switch_rate_warn: float = 1000
    ctx_switch_rate_severe: float = 5000
    
    # Page faults (major faults)
    page_fault_rate_warn: float = 100
    page_fault_rate_severe: float = 500


# ============================================================================
# Noise Sentinel
# ============================================================================

class NoiseSentinel:
    """
    Runtime interference detection system.
    
    Monitors for noise events that could contaminate measurements.
    Runs in background thread during measurement window.
    
    Usage:
        sentinel = NoiseSentinel()
        sentinel.start()
        
        # Run measurement
        run_benchmark()
        
        sentinel.stop()
        report = sentinel.get_report()
        
        if report.recommended_discard:
            print("Measurement contaminated, should retry")
    """
    
    def __init__(self, 
                 thresholds: Optional[NoiseThresholds] = None,
                 poll_interval: float = 0.1):
        self.thresholds = thresholds or NoiseThresholds()
        self.poll_interval = poll_interval
        
        self._events: List[NoiseEvent] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0
        self._end_time: float = 0
        
        # Baseline stats
        self._baseline_irq_count: int = 0
        self._baseline_ctx_switches: int = 0
        self._baseline_pgfault: int = 0
        self._baseline_pswpin: int = 0
        
        # Previous poll stats
        self._prev_irq_count: int = 0
        self._prev_ctx_switches: int = 0
        self._prev_poll_time: float = 0
    
    def start(self) -> None:
        """Start noise monitoring."""
        if self._running:
            return
        
        self._events.clear()
        self._start_time = time.time()
        self._running = True
        
        # Capture baseline
        self._capture_baseline()
        
        # Start monitoring thread
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Noise sentinel started")
    
    def stop(self) -> NoiseReport:
        """Stop monitoring and return report."""
        self._running = False
        self._end_time = time.time()
        
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        
        logger.info(f"Noise sentinel stopped, detected {len(self._events)} events")
        return self.get_report()
    
    def get_report(self) -> NoiseReport:
        """Generate noise report from collected events."""
        report = NoiseReport(
            start_time=self._start_time,
            end_time=self._end_time or time.time(),
            events=list(self._events),
        )
        
        # Calculate contamination score
        if self._events:
            max_severity = max(e.severity for e in self._events)
            avg_severity = sum(e.severity for e in self._events) / len(self._events)
            
            # Weight max more heavily
            report.contamination_score = 0.7 * max_severity + 0.3 * avg_severity
            
            # Recommend discard if score > 0.5
            report.recommended_discard = report.contamination_score > 0.5
            
            # Generate summary
            sources = set(e.source.value for e in self._events)
            report.summary = f"Detected {len(self._events)} noise events from: {', '.join(sources)}"
        else:
            report.summary = "No significant noise detected"
        
        return report
    
    def add_event(self, event: NoiseEvent) -> None:
        """Manually add a noise event."""
        self._events.append(event)
        logger.debug(f"Noise event: {event.source.value} - {event.description}")
    
    # ==========================================================================
    # Baseline Capture
    # ==========================================================================
    
    def _capture_baseline(self) -> None:
        """Capture baseline system stats."""
        self._baseline_irq_count = self._get_total_irq_count()
        self._baseline_ctx_switches = self._get_context_switches()
        self._baseline_pgfault = self._get_page_faults()
        self._baseline_pswpin = self._get_swap_activity()
        
        self._prev_irq_count = self._baseline_irq_count
        self._prev_ctx_switches = self._baseline_ctx_switches
        self._prev_poll_time = time.time()
    
    def _get_total_irq_count(self) -> int:
        """Get total interrupt count from /proc/stat."""
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if line.startswith("intr"):
                        parts = line.split()
                        return int(parts[1])  # Total interrupt count
        except:
            pass
        return 0
    
    def _get_context_switches(self) -> int:
        """Get total context switches from /proc/stat."""
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if line.startswith("ctxt"):
                        return int(line.split()[1])
        except:
            pass
        return 0
    
    def _get_page_faults(self) -> int:
        """Get major page faults from /proc/vmstat."""
        try:
            with open("/proc/vmstat") as f:
                for line in f:
                    if line.startswith("pgmajfault"):
                        return int(line.split()[1])
        except:
            pass
        return 0
    
    def _get_swap_activity(self) -> int:
        """Get swap-in pages from /proc/vmstat."""
        try:
            with open("/proc/vmstat") as f:
                for line in f:
                    if line.startswith("pswpin"):
                        return int(line.split()[1])
        except:
            pass
        return 0
    
    # ==========================================================================
    # Monitoring Loop
    # ==========================================================================
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                self._check_all_sources()
            except Exception as e:
                logger.debug(f"Monitor error: {e}")
            
            time.sleep(self.poll_interval)
    
    def _check_all_sources(self) -> None:
        """Check all noise sources."""
        now = time.time()
        dt = now - self._prev_poll_time
        if dt < 0.01:
            return
        
        # IRQ rate
        self._check_irq_rate(dt)
        
        # Context switches
        self._check_ctx_switches(dt)
        
        # Memory pressure
        self._check_memory_pressure()
        
        # Thermal
        self._check_thermal()
        
        self._prev_poll_time = now
    
    def _check_irq_rate(self, dt: float) -> None:
        """Check interrupt rate."""
        current = self._get_total_irq_count()
        delta = current - self._prev_irq_count
        rate = delta / dt
        
        if rate > self.thresholds.irq_rate_severe:
            self.add_event(NoiseEvent(
                timestamp=time.time(),
                source=NoiseSource.IRQ_STORM,
                severity=1.0,
                description=f"Severe IRQ storm: {rate:.0f}/s",
                metrics={"rate": rate},
            ))
        elif rate > self.thresholds.irq_rate_warn:
            self.add_event(NoiseEvent(
                timestamp=time.time(),
                source=NoiseSource.IRQ_STORM,
                severity=0.5,
                description=f"High IRQ rate: {rate:.0f}/s",
                metrics={"rate": rate},
            ))
        
        self._prev_irq_count = current
    
    def _check_ctx_switches(self, dt: float) -> None:
        """Check context switch rate."""
        current = self._get_context_switches()
        delta = current - self._prev_ctx_switches
        rate = delta / dt
        
        if rate > self.thresholds.ctx_switch_rate_severe:
            self.add_event(NoiseEvent(
                timestamp=time.time(),
                source=NoiseSource.CONTEXT_SWITCH,
                severity=1.0,
                description=f"Severe context switch rate: {rate:.0f}/s",
                metrics={"rate": rate},
            ))
        elif rate > self.thresholds.ctx_switch_rate_warn:
            self.add_event(NoiseEvent(
                timestamp=time.time(),
                source=NoiseSource.CONTEXT_SWITCH,
                severity=0.4,
                description=f"High context switch rate: {rate:.0f}/s",
                metrics={"rate": rate},
            ))
        
        self._prev_ctx_switches = current
    
    def _check_memory_pressure(self) -> None:
        """Check for memory pressure / reclaim activity."""
        try:
            # Check pswpin (pages swapped in) as proxy for reclaim
            pswpin = self._get_swap_activity()
            delta = pswpin - self._baseline_pswpin
            
            if delta > 100:  # Significant swap activity
                self.add_event(NoiseEvent(
                    timestamp=time.time(),
                    source=NoiseSource.RECLAIM_STORM,
                    severity=0.8 if delta > 1000 else 0.4,
                    description=f"Swap activity detected: {delta} pages",
                    metrics={"pages_swapped": delta},
                ))
                self._baseline_pswpin = pswpin  # Reset to avoid repeated events
            
            # Check page faults
            pgfault = self._get_page_faults()
            fault_delta = pgfault - self._baseline_pgfault
            
            if fault_delta > self.thresholds.page_fault_rate_severe:
                self.add_event(NoiseEvent(
                    timestamp=time.time(),
                    source=NoiseSource.PAGE_FAULT,
                    severity=0.7,
                    description=f"High major page faults: {fault_delta}",
                    metrics={"faults": fault_delta},
                ))
                self._baseline_pgfault = pgfault
                
        except Exception as e:
            logger.debug(f"Memory pressure check failed: {e}")
    
    def _check_thermal(self) -> None:
        """Check for thermal throttling."""
        try:
            # CPU thermal
            for zone_dir in Path("/sys/class/thermal").glob("thermal_zone*"):
                temp_path = zone_dir / "temp"
                type_path = zone_dir / "type"
                
                if not temp_path.exists():
                    continue
                
                temp_mc = int(temp_path.read_text().strip())
                temp_c = temp_mc / 1000
                zone_type = type_path.read_text().strip() if type_path.exists() else "unknown"
                
                # Check against trip points
                for trip_dir in zone_dir.glob("trip_point_*_temp"):
                    trip_temp_mc = int(trip_dir.read_text().strip())
                    trip_temp_c = trip_temp_mc / 1000
                    
                    margin = trip_temp_c - temp_c
                    
                    if margin <= self.thresholds.thermal_margin_severe:
                        self.add_event(NoiseEvent(
                            timestamp=time.time(),
                            source=NoiseSource.THERMAL_THROTTLE,
                            severity=1.0,
                            description=f"Thermal throttle: {zone_type} at {temp_c:.1f}C",
                            metrics={"temp_c": temp_c, "zone": zone_type},
                        ))
                        return  # Only one thermal event per check
                    elif margin <= self.thresholds.thermal_margin_warn:
                        self.add_event(NoiseEvent(
                            timestamp=time.time(),
                            source=NoiseSource.THERMAL_THROTTLE,
                            severity=0.5,
                            description=f"Near thermal limit: {zone_type} at {temp_c:.1f}C",
                            metrics={"temp_c": temp_c, "zone": zone_type},
                        ))
                        return
                    
        except Exception as e:
            logger.debug(f"Thermal check failed: {e}")
    
    # ==========================================================================
    # GPU-specific Checks
    # ==========================================================================
    
    def check_gpu_reset(self) -> bool:
        """Check if GPU reset occurred (call at end of measurement)."""
        try:
            # Check dmesg for amdgpu reset messages
            import subprocess
            result = subprocess.run(
                ["dmesg", "-T", "--since", f"{int(self._start_time)}"],
                capture_output=True, text=True, timeout=5
            )
            
            if "amdgpu" in result.stdout.lower() and "reset" in result.stdout.lower():
                self.add_event(NoiseEvent(
                    timestamp=time.time(),
                    source=NoiseSource.GPU_RESET,
                    severity=1.0,
                    description="GPU reset detected during measurement",
                    metrics={},
                ))
                return True
        except:
            pass
        return False


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_noise_check(duration: float = 1.0) -> NoiseReport:
    """Run a quick noise check for specified duration."""
    sentinel = NoiseSentinel()
    sentinel.start()
    time.sleep(duration)
    return sentinel.stop()


class NoiseSentinelContext:
    """Context manager for noise monitoring."""
    
    def __init__(self, thresholds: Optional[NoiseThresholds] = None):
        self.sentinel = NoiseSentinel(thresholds)
        self.report: Optional[NoiseReport] = None
    
    def __enter__(self) -> "NoiseSentinelContext":
        self.sentinel.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.report = self.sentinel.stop()
