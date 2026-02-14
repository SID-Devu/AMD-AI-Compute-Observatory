"""
AACO eBPF Scheduler Collectors
High-fidelity kernel-level tracing for scheduling events, wakeups, and latency.
"""

import ctypes
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Check BCC availability
try:
    from bcc import BPF
    BCC_AVAILABLE = True
except ImportError:
    BCC_AVAILABLE = False
    BPF = None


# ============================================================================
# eBPF Event Types
# ============================================================================

class EBPFEventType(IntEnum):
    """eBPF event types."""
    SCHED_SWITCH = 1
    SCHED_WAKEUP = 2
    SCHED_WAKEUP_NEW = 3
    PAGE_FAULT = 4
    SYSCALL_ENTER = 5
    SYSCALL_EXIT = 6
    BLOCK_RQ_ISSUE = 7
    BLOCK_RQ_COMPLETE = 8


# ============================================================================
# eBPF Programs
# ============================================================================

EBPF_SCHED_PROGRAM = """
#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

// Event structure passed to user-space
struct sched_event_t {
    u64 timestamp_ns;
    u32 pid;
    u32 tgid;
    u32 prev_pid;
    u32 next_pid;
    u8 event_type;
    u8 cpu;
    u16 flags;
    s64 delta_ns;
    char prev_comm[16];
    char next_comm[16];
};

BPF_PERF_OUTPUT(sched_events);
BPF_HASH(start_times, u32, u64);  // pid -> start time for latency calculation

// Filter by PID set (up to 64 PIDs)
BPF_HASH(target_pids, u32, u8);

static inline int should_trace(u32 pid) {
    u8 *val = target_pids.lookup(&pid);
    if (val)
        return 1;
    // If no PIDs configured, trace everything
    if (target_pids.lookup(&pid) == NULL && target_pids.delete(&pid) != 0)
        return 1;  // Empty map = trace all
    return 0;
}

// sched:sched_switch tracepoint
TRACEPOINT_PROBE(sched, sched_switch) {
    u32 prev_pid = args->prev_pid;
    u32 next_pid = args->next_pid;
    
    // Filter by target PIDs
    if (!should_trace(prev_pid) && !should_trace(next_pid))
        return 0;
    
    struct sched_event_t event = {};
    
    event.timestamp_ns = bpf_ktime_get_ns();
    event.event_type = 1;  // SCHED_SWITCH
    event.cpu = bpf_get_smp_processor_id();
    event.prev_pid = prev_pid;
    event.next_pid = next_pid;
    event.pid = bpf_get_current_pid_tgid() >> 32;
    event.tgid = bpf_get_current_pid_tgid();
    
    bpf_probe_read_kernel_str(&event.prev_comm, sizeof(event.prev_comm), args->prev_comm);
    bpf_probe_read_kernel_str(&event.next_comm, sizeof(event.next_comm), args->next_comm);
    
    // Calculate off-CPU time for prev_pid
    u64 *start = start_times.lookup(&prev_pid);
    if (start) {
        event.delta_ns = event.timestamp_ns - *start;
    }
    
    // Record start time for next_pid going on-CPU
    u64 ts = event.timestamp_ns;
    start_times.update(&next_pid, &ts);
    
    sched_events.perf_submit(args, &event, sizeof(event));
    return 0;
}

// sched:sched_wakeup tracepoint
TRACEPOINT_PROBE(sched, sched_wakeup) {
    u32 pid = args->pid;
    
    if (!should_trace(pid))
        return 0;
    
    struct sched_event_t event = {};
    
    event.timestamp_ns = bpf_ktime_get_ns();
    event.event_type = 2;  // SCHED_WAKEUP
    event.cpu = bpf_get_smp_processor_id();
    event.pid = pid;
    event.next_pid = pid;
    
    bpf_probe_read_kernel_str(&event.next_comm, sizeof(event.next_comm), args->comm);
    
    // Record wakeup time for latency calculation
    u64 ts = event.timestamp_ns;
    start_times.update(&pid, &ts);
    
    sched_events.perf_submit(args, &event, sizeof(event));
    return 0;
}

// sched:sched_wakeup_new tracepoint
TRACEPOINT_PROBE(sched, sched_wakeup_new) {
    u32 pid = args->pid;
    
    struct sched_event_t event = {};
    
    event.timestamp_ns = bpf_ktime_get_ns();
    event.event_type = 3;  // SCHED_WAKEUP_NEW
    event.cpu = bpf_get_smp_processor_id();
    event.pid = pid;
    event.next_pid = pid;
    
    bpf_probe_read_kernel_str(&event.next_comm, sizeof(event.next_comm), args->comm);
    
    sched_events.perf_submit(args, &event, sizeof(event));
    return 0;
}
"""

EBPF_FAULT_PROGRAM = """
#include <uapi/linux/ptrace.h>

struct fault_event_t {
    u64 timestamp_ns;
    u32 pid;
    u32 tgid;
    u64 address;
    u8 event_type;
    u8 cpu;
    u16 flags;
    char comm[16];
};

BPF_PERF_OUTPUT(fault_events);
BPF_HASH(target_pids, u32, u8);

int trace_page_fault(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    
    // Check if we should trace this PID
    u8 *val = target_pids.lookup(&pid);
    if (val == NULL)
        return 0;
    
    struct fault_event_t event = {};
    
    event.timestamp_ns = bpf_ktime_get_ns();
    event.event_type = 4;  // PAGE_FAULT
    event.cpu = bpf_get_smp_processor_id();
    event.pid = pid;
    event.tgid = bpf_get_current_pid_tgid();
    event.address = PT_REGS_PARM1(ctx);
    
    bpf_get_current_comm(&event.comm, sizeof(event.comm));
    
    fault_events.perf_submit(ctx, &event, sizeof(event));
    return 0;
}
"""


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SchedEvent:
    """Scheduler event from eBPF."""
    timestamp_ns: int
    event_type: EBPFEventType
    cpu: int
    pid: int
    tgid: int
    prev_pid: int = 0
    next_pid: int = 0
    delta_ns: int = 0
    prev_comm: str = ""
    next_comm: str = ""
    
    @property
    def timestamp_ms(self) -> float:
        return self.timestamp_ns / 1_000_000
    
    @property
    def delta_us(self) -> float:
        return self.delta_ns / 1_000


@dataclass
class FaultEvent:
    """Page fault event from eBPF."""
    timestamp_ns: int
    event_type: EBPFEventType
    cpu: int
    pid: int
    tgid: int
    address: int
    comm: str = ""
    
    @property
    def timestamp_ms(self) -> float:
        return self.timestamp_ns / 1_000_000


@dataclass
class EBPFCollectorConfig:
    """Configuration for eBPF collection."""
    trace_sched: bool = True
    trace_wakeups: bool = True
    trace_faults: bool = False
    trace_syscalls: bool = False
    target_pids: Set[int] = field(default_factory=set)
    buffer_size: int = 64  # Pages


# ============================================================================
# eBPF Collector
# ============================================================================

class EBPFSchedCollector:
    """
    eBPF-based scheduler event collector.
    
    Provides high-fidelity kernel-level tracing for:
    - Context switches (sched_switch)
    - Wakeup events (sched_wakeup)
    - Run-queue latency
    - Page faults (optional)
    """
    
    def __init__(self, config: Optional[EBPFCollectorConfig] = None):
        self.config = config or EBPFCollectorConfig()
        self._bpf: Optional[Any] = None
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Event buffers
        self._sched_events: List[SchedEvent] = []
        self._fault_events: List[FaultEvent] = []
        self._lock = threading.Lock()
        
        # Statistics
        self._events_received = 0
        self._events_dropped = 0
    
    @staticmethod
    def is_available() -> bool:
        """Check if eBPF collection is available."""
        if not BCC_AVAILABLE:
            return False
        
        # Check if we have permissions
        return os.geteuid() == 0 or os.path.exists("/sys/kernel/debug/tracing")
    
    def start(self, target_pids: Optional[Set[int]] = None) -> bool:
        """
        Start eBPF collection.
        
        Args:
            target_pids: Set of PIDs to trace (None = trace all)
            
        Returns:
            True if started successfully
        """
        if not self.is_available():
            logger.warning("eBPF not available - requires root or CAP_BPF")
            return False
        
        if self._running:
            return True
        
        try:
            # Compile eBPF program
            self._bpf = BPF(text=EBPF_SCHED_PROGRAM)
            
            # Set target PIDs
            pids = target_pids or self.config.target_pids
            if pids:
                pid_map = self._bpf["target_pids"]
                for pid in pids:
                    pid_map[ctypes.c_uint32(pid)] = ctypes.c_uint8(1)
            
            # Open perf buffer
            self._bpf["sched_events"].open_perf_buffer(
                self._handle_sched_event,
                page_cnt=self.config.buffer_size
            )
            
            # Start polling thread
            self._running = True
            self._stop_event.clear()
            self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._poll_thread.start()
            
            logger.info("eBPF scheduler collector started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start eBPF collector: {e}")
            self._cleanup()
            return False
    
    def stop(self) -> None:
        """Stop eBPF collection."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._poll_thread:
            self._poll_thread.join(timeout=2.0)
            self._poll_thread = None
        
        self._cleanup()
        
        logger.info(f"eBPF collector stopped: {self._events_received} events, "
                   f"{self._events_dropped} dropped")
    
    def add_pid(self, pid: int) -> None:
        """Add a PID to trace."""
        if self._bpf:
            pid_map = self._bpf["target_pids"]
            pid_map[ctypes.c_uint32(pid)] = ctypes.c_uint8(1)
    
    def remove_pid(self, pid: int) -> None:
        """Remove a PID from tracing."""
        if self._bpf:
            pid_map = self._bpf["target_pids"]
            try:
                del pid_map[ctypes.c_uint32(pid)]
            except KeyError:
                pass
    
    def get_sched_events(self) -> List[SchedEvent]:
        """Get collected scheduler events."""
        with self._lock:
            return list(self._sched_events)
    
    def get_fault_events(self) -> List[FaultEvent]:
        """Get collected fault events."""
        with self._lock:
            return list(self._fault_events)
    
    def clear_events(self) -> None:
        """Clear collected events."""
        with self._lock:
            self._sched_events.clear()
            self._fault_events.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collector statistics."""
        with self._lock:
            return {
                "events_received": self._events_received,
                "events_dropped": self._events_dropped,
                "sched_events_buffered": len(self._sched_events),
                "fault_events_buffered": len(self._fault_events),
                "running": self._running,
            }
    
    def export_to_dicts(self) -> List[Dict[str, Any]]:
        """Export events as list of dictionaries."""
        events = []
        
        for e in self.get_sched_events():
            events.append({
                "source": "ebpf",
                "t_ns": e.timestamp_ns,
                "t_ms": e.timestamp_ms,
                "event_type": e.event_type.name,
                "cpu": e.cpu,
                "pid": e.pid,
                "tgid": e.tgid,
                "prev_pid": e.prev_pid,
                "next_pid": e.next_pid,
                "delta_ns": e.delta_ns,
                "delta_us": e.delta_us,
                "prev_comm": e.prev_comm,
                "next_comm": e.next_comm,
            })
        
        for e in self.get_fault_events():
            events.append({
                "source": "ebpf",
                "t_ns": e.timestamp_ns,
                "t_ms": e.timestamp_ms,
                "event_type": e.event_type.name,
                "cpu": e.cpu,
                "pid": e.pid,
                "tgid": e.tgid,
                "address": e.address,
                "comm": e.comm,
            })
        
        return sorted(events, key=lambda x: x["t_ns"])
    
    def _poll_loop(self) -> None:
        """Background polling loop."""
        while not self._stop_event.is_set():
            try:
                self._bpf.perf_buffer_poll(timeout=100)
            except Exception as e:
                if self._running:
                    logger.error(f"eBPF poll error: {e}")
                    time.sleep(0.1)
    
    def _handle_sched_event(self, cpu: int, data: ctypes.c_void_p, size: int) -> None:
        """Handle scheduler event from eBPF."""
        # Parse event struct
        event_data = ctypes.string_at(data, size)
        
        # Struct: timestamp(Q) pid(I) tgid(I) prev_pid(I) next_pid(I) 
        #         event_type(B) cpu(B) flags(H) delta_ns(q) prev_comm(16s) next_comm(16s)
        try:
            (timestamp_ns, pid, tgid, prev_pid, next_pid, event_type, cpu_id, 
             flags, delta_ns, prev_comm, next_comm) = struct.unpack(
                "<QIIIIBBHq16s16s", event_data[:64]
            )
            
            event = SchedEvent(
                timestamp_ns=timestamp_ns,
                event_type=EBPFEventType(event_type),
                cpu=cpu_id,
                pid=pid,
                tgid=tgid,
                prev_pid=prev_pid,
                next_pid=next_pid,
                delta_ns=delta_ns,
                prev_comm=prev_comm.rstrip(b'\x00').decode('utf-8', errors='replace'),
                next_comm=next_comm.rstrip(b'\x00').decode('utf-8', errors='replace'),
            )
            
            with self._lock:
                self._sched_events.append(event)
                self._events_received += 1
                
        except Exception as e:
            logger.error(f"Failed to parse sched event: {e}")
            self._events_dropped += 1
    
    def _cleanup(self) -> None:
        """Cleanup BPF resources."""
        if self._bpf:
            try:
                self._bpf.cleanup()
            except:
                pass
            self._bpf = None


# ============================================================================
# Scheduler Metrics Calculator
# ============================================================================

@dataclass
class SchedMetrics:
    """Computed scheduler metrics."""
    total_switches: int = 0
    voluntary_switches: int = 0
    involuntary_switches: int = 0
    wakeups: int = 0
    
    # Latency statistics
    avg_runq_latency_us: float = 0
    p50_runq_latency_us: float = 0
    p99_runq_latency_us: float = 0
    max_runq_latency_us: float = 0
    
    # Rates
    switch_rate_per_sec: float = 0
    wakeup_rate_per_sec: float = 0
    
    # Distribution
    cpu_distribution: Dict[int, int] = field(default_factory=dict)
    
    # Noise score (0-100)
    noise_score: float = 0


def compute_sched_metrics(events: List[SchedEvent], 
                          duration_ns: Optional[int] = None) -> SchedMetrics:
    """
    Compute scheduler metrics from events.
    
    Args:
        events: List of scheduler events
        duration_ns: Total duration in nanoseconds
    """
    if not events:
        return SchedMetrics()
    
    metrics = SchedMetrics()
    
    # Count events by type
    switches = [e for e in events if e.event_type == EBPFEventType.SCHED_SWITCH]
    wakeups = [e for e in events if e.event_type in 
               (EBPFEventType.SCHED_WAKEUP, EBPFEventType.SCHED_WAKEUP_NEW)]
    
    metrics.total_switches = len(switches)
    metrics.wakeups = len(wakeups)
    
    # Compute latencies
    latencies_us = [e.delta_us for e in switches if e.delta_ns > 0]
    
    if latencies_us:
        import numpy as np
        arr = np.array(latencies_us)
        
        metrics.avg_runq_latency_us = float(np.mean(arr))
        metrics.p50_runq_latency_us = float(np.percentile(arr, 50))
        metrics.p99_runq_latency_us = float(np.percentile(arr, 99))
        metrics.max_runq_latency_us = float(np.max(arr))
    
    # CPU distribution
    for e in switches:
        metrics.cpu_distribution[e.cpu] = metrics.cpu_distribution.get(e.cpu, 0) + 1
    
    # Compute rates
    if duration_ns:
        duration_sec = duration_ns / 1e9
        metrics.switch_rate_per_sec = metrics.total_switches / duration_sec
        metrics.wakeup_rate_per_sec = metrics.wakeups / duration_sec
    elif events:
        # Estimate from event timestamps
        duration_sec = (events[-1].timestamp_ns - events[0].timestamp_ns) / 1e9
        if duration_sec > 0:
            metrics.switch_rate_per_sec = metrics.total_switches / duration_sec
            metrics.wakeup_rate_per_sec = metrics.wakeups / duration_sec
    
    # Compute noise score (heuristic)
    # High switch rate + high latency variance = high noise
    noise_factors = []
    
    if metrics.switch_rate_per_sec > 100:
        noise_factors.append(min(100, metrics.switch_rate_per_sec / 10))
    
    if metrics.p99_runq_latency_us > 1000:  # > 1ms
        noise_factors.append(min(100, metrics.p99_runq_latency_us / 100))
    
    if len(metrics.cpu_distribution) > 4:
        # Many CPUs involved = potential cache thrashing
        noise_factors.append(len(metrics.cpu_distribution) * 5)
    
    if noise_factors:
        metrics.noise_score = min(100, sum(noise_factors) / len(noise_factors))
    
    return metrics


# ============================================================================
# Fallback: procfs-based scheduler stats
# ============================================================================

def read_procfs_sched_stats(pid: int) -> Optional[Dict[str, Any]]:
    """
    Read scheduler statistics from /proc/[pid]/sched.
    
    Fallback when eBPF is not available.
    """
    sched_path = f"/proc/{pid}/sched"
    
    if not os.path.exists(sched_path):
        return None
    
    try:
        with open(sched_path) as f:
            content = f.read()
        
        stats = {}
        for line in content.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                try:
                    # Try float first (handles scientific notation)
                    stats[key] = float(value)
                except ValueError:
                    stats[key] = value
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to read /proc/{pid}/sched: {e}")
        return None


def read_procfs_stat(pid: int) -> Optional[Dict[str, Any]]:
    """
    Read process stats from /proc/[pid]/stat.
    """
    stat_path = f"/proc/{pid}/stat"
    
    if not os.path.exists(stat_path):
        return None
    
    try:
        with open(stat_path) as f:
            content = f.read()
        
        # Parse stat file (fields are space-separated, comm in parens)
        # pid (comm) state ppid pgrp session tty_nr tpgid flags ...
        parts = content.split()
        
        return {
            "pid": int(parts[0]),
            "comm": parts[1].strip("()"),
            "state": parts[2],
            "ppid": int(parts[3]),
            "utime": int(parts[13]),
            "stime": int(parts[14]),
            "num_threads": int(parts[19]),
            "vsize": int(parts[22]),
            "rss": int(parts[23]),
            "voluntary_ctxt_switches": int(parts[42]) if len(parts) > 42 else 0,
            "nonvoluntary_ctxt_switches": int(parts[43]) if len(parts) > 43 else 0,
        }
        
    except Exception as e:
        logger.error(f"Failed to read /proc/{pid}/stat: {e}")
        return None
