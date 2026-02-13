"""
System Telemetry Sampler
Collects CPU, memory, scheduling, and page fault metrics from /proc filesystem.
"""

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
import os

from aaco.core.schema import SystemEvent
from aaco.core.utils import (
    get_monotonic_ns,
    read_proc_stat,
    read_proc_vmstat,
    read_loadavg,
    read_process_status,
    compute_cpu_percent,
)

logger = logging.getLogger(__name__)


@dataclass
class SystemSample:
    """Single system telemetry sample."""
    t_ns: int
    cpu_pct: float
    rss_mb: float
    ctx_switches: int
    ctx_switches_delta: int
    majfaults: int
    majfault_delta: int
    load1: float
    load5: float
    procs_running: int


class SystemSampler:
    """
    Samples system telemetry from /proc at configurable intervals.
    Runs in a background thread during workload execution.
    """
    
    def __init__(
        self,
        interval_ms: int = 200,
        pid: Optional[int] = None,
        t0_ns: int = 0,
    ):
        self.interval_ms = interval_ms
        self.interval_s = interval_ms / 1000.0
        self.pid = pid or os.getpid()
        self.t0_ns = t0_ns
        
        self.samples: List[SystemSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Previous values for delta computation
        self._prev_stat: Dict = {}
        self._prev_vmstat: Dict = {}
        self._prev_ctx_switches = 0
        self._prev_majfaults = 0
    
    def start(self) -> None:
        """Start background sampling thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        logger.debug("System sampler started")
    
    def stop(self) -> List[SystemSample]:
        """Stop sampling and return collected samples."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        logger.debug(f"System sampler stopped, collected {len(self.samples)} samples")
        return self.samples
    
    def _sample_loop(self) -> None:
        """Main sampling loop."""
        while self._running:
            try:
                sample = self._collect_sample()
                if sample:
                    self.samples.append(sample)
            except Exception as e:
                logger.debug(f"Sample collection error: {e}")
            
            time.sleep(self.interval_s)
    
    def _collect_sample(self) -> Optional[SystemSample]:
        """Collect a single sample from /proc."""
        t_ns = get_monotonic_ns() - self.t0_ns
        
        # Read /proc/stat for CPU
        curr_stat = read_proc_stat()
        cpu_pct = compute_cpu_percent(self._prev_stat, curr_stat)
        self._prev_stat = curr_stat
        
        # Read /proc/vmstat for page faults
        vmstat = read_proc_vmstat()
        majfaults = vmstat.get("pgmajfault", 0)
        majfault_delta = majfaults - self._prev_majfaults
        self._prev_majfaults = majfaults
        
        # Context switches from /proc/stat
        ctx_switches = curr_stat.get("context_switches", 0)
        ctx_switches_delta = ctx_switches - self._prev_ctx_switches
        self._prev_ctx_switches = ctx_switches
        
        # Load average
        loadavg = read_loadavg()
        
        # Process RSS
        status = read_process_status(self.pid)
        rss_kb = status.get("VmRSS", 0)
        rss_mb = rss_kb / 1024.0
        
        return SystemSample(
            t_ns=t_ns,
            cpu_pct=cpu_pct,
            rss_mb=rss_mb,
            ctx_switches=ctx_switches,
            ctx_switches_delta=max(0, ctx_switches_delta),
            majfaults=majfaults,
            majfault_delta=max(0, majfault_delta),
            load1=loadavg.get("load1", 0.0),
            load5=loadavg.get("load5", 0.0),
            procs_running=curr_stat.get("procs_running", 0),
        )
    
    def to_events(self) -> List[SystemEvent]:
        """Convert samples to SystemEvent schema."""
        return [
            SystemEvent(
                t_ns=s.t_ns,
                cpu_pct=s.cpu_pct,
                rss_mb=s.rss_mb,
                ctx_switches_delta=s.ctx_switches_delta,
                majfault_delta=s.majfault_delta,
                runq_len=float(s.procs_running),
                load1=s.load1,
                pid=self.pid,
            )
            for s in self.samples
        ]
    
    def to_dataframe(self):
        """Convert samples to pandas DataFrame."""
        import pandas as pd
        
        if not self.samples:
            return pd.DataFrame()
        
        return pd.DataFrame([
            {
                "t_ns": s.t_ns,
                "t_ms": s.t_ns / 1_000_000,
                "cpu_pct": s.cpu_pct,
                "rss_mb": s.rss_mb,
                "ctx_switches_delta": s.ctx_switches_delta,
                "majfault_delta": s.majfault_delta,
                "load1": s.load1,
                "procs_running": s.procs_running,
            }
            for s in self.samples
        ])
    
    def get_summary(self) -> Dict:
        """Get summary statistics of collected samples."""
        if not self.samples:
            return {}
        
        import numpy as np
        
        cpu_vals = [s.cpu_pct for s in self.samples]
        ctx_deltas = [s.ctx_switches_delta for s in self.samples]
        majfault_deltas = [s.majfault_delta for s in self.samples]
        
        duration_s = (self.samples[-1].t_ns - self.samples[0].t_ns) / 1e9 if len(self.samples) > 1 else 1
        
        return {
            "sample_count": len(self.samples),
            "duration_s": duration_s,
            "cpu_pct_mean": float(np.mean(cpu_vals)),
            "cpu_pct_max": float(np.max(cpu_vals)),
            "cpu_pct_std": float(np.std(cpu_vals)),
            "ctx_switch_rate": sum(ctx_deltas) / duration_s if duration_s > 0 else 0,
            "majfault_rate": sum(majfault_deltas) / duration_s if duration_s > 0 else 0,
            "rss_mb_max": max(s.rss_mb for s in self.samples),
            "load1_mean": float(np.mean([s.load1 for s in self.samples])),
        }
