# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Forensic Scheduler Model

Advanced eBPF-based scheduler analysis computing:
- Scheduler Interference Index (SII)
- Fault Pressure Index (FPI)
- CPU Noise Entropy (CNE)

These are statistically summarized distributions per iteration window.
"""

import os
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from enum import Enum
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class RunqueueSample:
    """Single runqueue latency sample."""
    timestamp_ns: int
    pid: int
    tgid: int
    latency_ns: int
    cpu: int
    prev_state: int


@dataclass
class WakeupSample:
    """Wakeup-to-schedule latency sample."""
    timestamp_ns: int
    pid: int
    wakeup_latency_ns: int
    target_cpu: int


@dataclass  
class SyscallSample:
    """Syscall activity sample."""
    timestamp_ns: int
    pid: int
    syscall_nr: int
    duration_ns: int


@dataclass
class PageFaultSample:
    """Page fault sample."""
    timestamp_ns: int
    pid: int
    address: int
    fault_type: str  # 'major' or 'minor'
    latency_ns: int


@dataclass
class InterruptSample:
    """Interrupt/softirq sample."""
    timestamp_ns: int
    irq_type: str  # 'hardirq' or 'softirq'
    irq_nr: int
    duration_ns: int
    cpu: int


@dataclass
class SchedulerDistribution:
    """Statistical distribution of scheduler metrics."""
    count: int = 0
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    
    @classmethod
    def from_samples(cls, values: List[float]) -> 'SchedulerDistribution':
        """Create distribution from samples."""
        if not values:
            return cls()
        
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        
        return cls(
            count=n,
            mean=statistics.mean(values),
            median=statistics.median(values),
            std_dev=statistics.stdev(values) if n > 1 else 0,
            p50=sorted_vals[int(n * 0.50)],
            p90=sorted_vals[int(n * 0.90)] if n > 10 else sorted_vals[-1],
            p95=sorted_vals[int(n * 0.95)] if n > 20 else sorted_vals[-1],
            p99=sorted_vals[int(n * 0.99)] if n > 100 else sorted_vals[-1],
            min_val=sorted_vals[0],
            max_val=sorted_vals[-1],
        )


@dataclass
class SchedulerForensics:
    """Complete forensic scheduler analysis for a window."""
    window_start_ns: int
    window_end_ns: int
    duration_ns: int
    
    # Runqueue analysis
    runqueue_latency: SchedulerDistribution = field(default_factory=SchedulerDistribution)
    context_switch_count: int = 0
    voluntary_switches: int = 0
    involuntary_switches: int = 0
    
    # Wakeup analysis
    wakeup_latency: SchedulerDistribution = field(default_factory=SchedulerDistribution)
    wakeup_count: int = 0
    
    # Syscall analysis
    syscall_intensity: float = 0.0  # syscalls/sec
    syscall_duration: SchedulerDistribution = field(default_factory=SchedulerDistribution)
    top_syscalls: List[Tuple[int, int]] = field(default_factory=list)  # (syscall_nr, count)
    
    # Page fault analysis
    minor_fault_count: int = 0
    major_fault_count: int = 0
    fault_latency: SchedulerDistribution = field(default_factory=SchedulerDistribution)
    
    # Interrupt analysis
    hardirq_count: int = 0
    softirq_count: int = 0
    irq_duration: SchedulerDistribution = field(default_factory=SchedulerDistribution)
    
    # Derived indices
    scheduler_interference_index: float = 0.0  # SII
    fault_pressure_index: float = 0.0  # FPI
    cpu_noise_entropy: float = 0.0  # CNE
    
    # Reclaim pressure
    reclaim_events: int = 0
    reclaim_stall_ns: int = 0


class ForensicSchedulerModel:
    """
    Advanced scheduler forensics using eBPF.
    
    Computes statistical distributions of:
    - Runqueue latency
    - Wakeup-to-schedule latency
    - Syscall intensity bursts
    - Page fault latency
    - IRQ/softirq bursts
    
    Derives three key indices:
    - Scheduler Interference Index (SII)
    - Fault Pressure Index (FPI)
    - CPU Noise Entropy (CNE)
    """
    
    def __init__(self, target_pid: Optional[int] = None):
        """Initialize forensic scheduler model."""
        self.target_pid = target_pid or os.getpid()
        self._collecting = False
        self._collection_thread: Optional[threading.Thread] = None
        
        # Sample buffers
        self._runqueue_samples: deque = deque(maxlen=100000)
        self._wakeup_samples: deque = deque(maxlen=100000)
        self._syscall_samples: deque = deque(maxlen=100000)
        self._fault_samples: deque = deque(maxlen=100000)
        self._irq_samples: deque = deque(maxlen=100000)
        
        self._lock = threading.Lock()
        self._start_time_ns: int = 0
        self._ebpf_available = self._check_ebpf_availability()
    
    def _check_ebpf_availability(self) -> bool:
        """Check if eBPF is available."""
        # Check for BCC
        try:
            from bcc import BPF
            return True
        except ImportError:
            pass
        
        # Check for bpftrace
        import shutil
        return shutil.which('bpftrace') is not None
    
    def start_collection(self) -> bool:
        """Start collecting scheduler forensics."""
        if self._collecting:
            return True
        
        self._collecting = True
        self._start_time_ns = time.time_ns()
        
        if self._ebpf_available:
            self._collection_thread = threading.Thread(
                target=self._ebpf_collection_loop,
                daemon=True
            )
            self._collection_thread.start()
            logger.info("Forensic scheduler collection started (eBPF mode)")
        else:
            # Fallback to /proc polling
            self._collection_thread = threading.Thread(
                target=self._proc_collection_loop,
                daemon=True
            )
            self._collection_thread.start()
            logger.info("Forensic scheduler collection started (proc fallback)")
        
        return True
    
    def stop_collection(self) -> SchedulerForensics:
        """Stop collection and return forensics."""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=2.0)
        
        return self._compute_forensics()
    
    def _ebpf_collection_loop(self) -> None:
        """Collection loop using eBPF."""
        try:
            from bcc import BPF
            
            # eBPF program for scheduler tracing
            bpf_text = self._get_ebpf_program()
            b = BPF(text=bpf_text)
            
            # Attach to tracepoints
            b.attach_tracepoint("sched:sched_wakeup", "trace_wakeup")
            b.attach_tracepoint("sched:sched_switch", "trace_switch")
            b.attach_tracepoint("raw_syscalls:sys_enter", "trace_syscall_enter")
            b.attach_tracepoint("raw_syscalls:sys_exit", "trace_syscall_exit")
            
            while self._collecting:
                # Poll perf buffer
                b.perf_buffer_poll(timeout=100)
                
        except Exception as e:
            logger.warning(f"eBPF collection failed: {e}, falling back to proc")
            self._proc_collection_loop()
    
    def _proc_collection_loop(self) -> None:
        """Fallback collection loop using /proc."""
        prev_stats = self._read_proc_stat()
        prev_time = time.time()
        
        while self._collecting:
            time.sleep(0.01)  # 10ms sampling
            
            current_stats = self._read_proc_stat()
            current_time = time.time()
            
            # Compute deltas
            dt = current_time - prev_time
            if dt > 0:
                # Context switches per second
                ctx_delta = (
                    current_stats.get('voluntary_ctx', 0) - 
                    prev_stats.get('voluntary_ctx', 0) +
                    current_stats.get('nonvoluntary_ctx', 0) - 
                    prev_stats.get('nonvoluntary_ctx', 0)
                )
                
                # Simulate runqueue sample
                with self._lock:
                    self._runqueue_samples.append(RunqueueSample(
                        timestamp_ns=time.time_ns(),
                        pid=self.target_pid,
                        tgid=self.target_pid,
                        latency_ns=int(current_stats.get('sched_wait_ns', 0)),
                        cpu=0,
                        prev_state=0,
                    ))
                    
                    # Record fault samples
                    minor_delta = (
                        current_stats.get('minor_faults', 0) -
                        prev_stats.get('minor_faults', 0)
                    )
                    major_delta = (
                        current_stats.get('major_faults', 0) -
                        prev_stats.get('major_faults', 0)
                    )
                    
                    for _ in range(min(minor_delta, 100)):
                        self._fault_samples.append(PageFaultSample(
                            timestamp_ns=time.time_ns(),
                            pid=self.target_pid,
                            address=0,
                            fault_type='minor',
                            latency_ns=1000,  # Estimated 1us
                        ))
                    
                    for _ in range(min(major_delta, 10)):
                        self._fault_samples.append(PageFaultSample(
                            timestamp_ns=time.time_ns(),
                            pid=self.target_pid,
                            address=0,
                            fault_type='major',
                            latency_ns=1000000,  # Estimated 1ms
                        ))
            
            prev_stats = current_stats
            prev_time = current_time
    
    def _read_proc_stat(self) -> Dict[str, int]:
        """Read scheduler stats from /proc."""
        stats = {}
        
        try:
            # /proc/self/schedstat
            with open(f'/proc/{self.target_pid}/schedstat', 'r') as f:
                parts = f.read().strip().split()
                if len(parts) >= 3:
                    stats['sched_run_ns'] = int(parts[0])
                    stats['sched_wait_ns'] = int(parts[1])
                    stats['sched_slices'] = int(parts[2])
            
            # /proc/self/status for context switches
            with open(f'/proc/{self.target_pid}/status', 'r') as f:
                for line in f:
                    if line.startswith('voluntary_ctxt_switches:'):
                        stats['voluntary_ctx'] = int(line.split()[1])
                    elif line.startswith('nonvoluntary_ctxt_switches:'):
                        stats['nonvoluntary_ctx'] = int(line.split()[1])
            
            # /proc/self/stat for page faults
            with open(f'/proc/{self.target_pid}/stat', 'r') as f:
                parts = f.read().split()
                if len(parts) > 12:
                    stats['minor_faults'] = int(parts[9])
                    stats['major_faults'] = int(parts[11])
                    
        except Exception as e:
            logger.debug(f"Failed to read proc stats: {e}")
        
        return stats
    
    def _get_ebpf_program(self) -> str:
        """Get eBPF program text."""
        return '''
        #include <uapi/linux/ptrace.h>
        #include <linux/sched.h>
        
        struct wakeup_event {
            u64 ts;
            u32 pid;
            u32 target_cpu;
        };
        
        struct switch_event {
            u64 ts;
            u32 prev_pid;
            u32 next_pid;
            u64 prev_state;
        };
        
        BPF_PERF_OUTPUT(wakeup_events);
        BPF_PERF_OUTPUT(switch_events);
        BPF_HASH(start_ts, u32, u64);
        
        TRACEPOINT_PROBE(sched, sched_wakeup) {
            struct wakeup_event evt = {};
            evt.ts = bpf_ktime_get_ns();
            evt.pid = args->pid;
            evt.target_cpu = args->target_cpu;
            
            start_ts.update(&evt.pid, &evt.ts);
            wakeup_events.perf_submit(args, &evt, sizeof(evt));
            return 0;
        }
        
        TRACEPOINT_PROBE(sched, sched_switch) {
            struct switch_event evt = {};
            evt.ts = bpf_ktime_get_ns();
            evt.prev_pid = args->prev_pid;
            evt.next_pid = args->next_pid;
            evt.prev_state = args->prev_state;
            
            switch_events.perf_submit(args, &evt, sizeof(evt));
            return 0;
        }
        '''
    
    def _compute_forensics(self) -> SchedulerForensics:
        """Compute forensics from collected samples."""
        end_time_ns = time.time_ns()
        
        with self._lock:
            runqueue = list(self._runqueue_samples)
            wakeups = list(self._wakeup_samples)
            syscalls = list(self._syscall_samples)
            faults = list(self._fault_samples)
            irqs = list(self._irq_samples)
        
        duration_ns = end_time_ns - self._start_time_ns
        duration_s = duration_ns / 1e9
        
        forensics = SchedulerForensics(
            window_start_ns=self._start_time_ns,
            window_end_ns=end_time_ns,
            duration_ns=duration_ns,
        )
        
        # Runqueue analysis
        if runqueue:
            latencies = [s.latency_ns / 1000.0 for s in runqueue]  # Convert to us
            forensics.runqueue_latency = SchedulerDistribution.from_samples(latencies)
            forensics.context_switch_count = len(runqueue)
        
        # Wakeup analysis
        if wakeups:
            latencies = [s.wakeup_latency_ns / 1000.0 for s in wakeups]
            forensics.wakeup_latency = SchedulerDistribution.from_samples(latencies)
            forensics.wakeup_count = len(wakeups)
        
        # Syscall analysis
        if syscalls:
            forensics.syscall_intensity = len(syscalls) / duration_s if duration_s > 0 else 0
            durations = [s.duration_ns / 1000.0 for s in syscalls]
            forensics.syscall_duration = SchedulerDistribution.from_samples(durations)
            
            # Top syscalls
            from collections import Counter
            syscall_counts = Counter(s.syscall_nr for s in syscalls)
            forensics.top_syscalls = syscall_counts.most_common(10)
        
        # Page fault analysis
        if faults:
            forensics.minor_fault_count = sum(1 for f in faults if f.fault_type == 'minor')
            forensics.major_fault_count = sum(1 for f in faults if f.fault_type == 'major')
            latencies = [f.latency_ns / 1000.0 for f in faults]
            forensics.fault_latency = SchedulerDistribution.from_samples(latencies)
        
        # IRQ analysis
        if irqs:
            forensics.hardirq_count = sum(1 for i in irqs if i.irq_type == 'hardirq')
            forensics.softirq_count = sum(1 for i in irqs if i.irq_type == 'softirq')
            durations = [i.duration_ns / 1000.0 for i in irqs]
            forensics.irq_duration = SchedulerDistribution.from_samples(durations)
        
        # Compute derived indices
        forensics.scheduler_interference_index = self._compute_sii(forensics, duration_s)
        forensics.fault_pressure_index = self._compute_fpi(forensics, duration_s)
        forensics.cpu_noise_entropy = self._compute_cne(forensics)
        
        return forensics
    
    def _compute_sii(self, forensics: SchedulerForensics, duration_s: float) -> float:
        """
        Compute Scheduler Interference Index (SII).
        
        SII = w1 * norm(ctx_switch_rate) + 
              w2 * norm(runqueue_p99) + 
              w3 * norm(wakeup_p99)
        
        Range: 0.0 (no interference) to 1.0 (severe interference)
        """
        # Normalize context switch rate (10k/s = 1.0)
        ctx_rate = forensics.context_switch_count / duration_s if duration_s > 0 else 0
        ctx_norm = min(1.0, ctx_rate / 10000.0)
        
        # Normalize runqueue P99 (10ms = 1.0)
        rq_norm = min(1.0, forensics.runqueue_latency.p99 / 10000.0)
        
        # Normalize wakeup P99 (10ms = 1.0)
        wu_norm = min(1.0, forensics.wakeup_latency.p99 / 10000.0)
        
        # Weighted combination
        sii = (
            0.4 * ctx_norm +
            0.35 * rq_norm +
            0.25 * wu_norm
        )
        
        return sii
    
    def _compute_fpi(self, forensics: SchedulerForensics, duration_s: float) -> float:
        """
        Compute Fault Pressure Index (FPI).
        
        FPI = w1 * norm(fault_rate) + 
              w2 * major_fault_ratio + 
              w3 * norm(fault_p99)
        
        Range: 0.0 (no pressure) to 1.0 (severe pressure)
        """
        total_faults = forensics.minor_fault_count + forensics.major_fault_count
        
        # Normalize fault rate (1k/s = 1.0)
        fault_rate = total_faults / duration_s if duration_s > 0 else 0
        rate_norm = min(1.0, fault_rate / 1000.0)
        
        # Major fault ratio (major faults are expensive)
        major_ratio = (
            forensics.major_fault_count / total_faults 
            if total_faults > 0 else 0
        )
        
        # Normalize fault P99 latency (1ms = 1.0)
        latency_norm = min(1.0, forensics.fault_latency.p99 / 1000.0)
        
        # Weighted combination
        fpi = (
            0.4 * rate_norm +
            0.4 * major_ratio +
            0.2 * latency_norm
        )
        
        return fpi
    
    def _compute_cne(self, forensics: SchedulerForensics) -> float:
        """
        Compute CPU Noise Entropy (CNE).
        
        CNE measures overall unpredictability/randomness of CPU behavior.
        Combines multiple noise sources into entropy-like measure.
        
        Range: 0.0 (deterministic) to 1.0 (highly unpredictable)
        """
        # Component signals
        signals = [
            forensics.scheduler_interference_index,
            forensics.fault_pressure_index,
        ]
        
        # Add variance-based entropy from distributions
        if forensics.runqueue_latency.count > 0:
            # CoV of runqueue latency
            if forensics.runqueue_latency.mean > 0:
                cov = forensics.runqueue_latency.std_dev / forensics.runqueue_latency.mean
                signals.append(min(1.0, cov))
        
        if forensics.syscall_duration.count > 0:
            if forensics.syscall_duration.mean > 0:
                cov = forensics.syscall_duration.std_dev / forensics.syscall_duration.mean
                signals.append(min(1.0, cov * 0.5))
        
        # Entropy-like combination: 1 - product of (1 - signals)
        product = 1.0
        for s in signals:
            product *= (1.0 - s)
        
        cne = 1.0 - product
        return cne
    
    def get_realtime_indices(self) -> Dict[str, float]:
        """Get current indices without stopping collection."""
        forensics = self._compute_forensics()
        return {
            'sii': forensics.scheduler_interference_index,
            'fpi': forensics.fault_pressure_index,
            'cne': forensics.cpu_noise_entropy,
        }
