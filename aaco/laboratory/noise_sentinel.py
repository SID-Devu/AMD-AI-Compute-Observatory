# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Noise Sentinel

Advanced noise detection and measurement validation.
Detects system noise that would invalidate measurements.
"""

import time
import statistics
import os
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class NoiseSource(Enum):
    """Types of system noise sources."""
    SCHEDULER_JITTER = "scheduler_jitter"
    CONTEXT_SWITCHES = "context_switches"
    PAGE_FAULTS = "page_faults"
    INTERRUPTS = "interrupts"
    THERMAL_THROTTLE = "thermal_throttle"
    CLOCK_DRIFT = "clock_drift"
    MEMORY_PRESSURE = "memory_pressure"
    IO_CONTENTION = "io_contention"
    BACKGROUND_LOAD = "background_load"


@dataclass
class NoiseSignature:
    """Signature of detected noise."""
    source: NoiseSource
    severity: float  # 0.0 - 1.0
    timestamp: float
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NoiseProfile:
    """Complete noise profile for a measurement window."""
    window_start: float
    window_end: float
    
    # Scheduler noise
    scheduler_jitter_us: float = 0.0
    context_switch_rate: float = 0.0
    runqueue_latency_p99_us: float = 0.0
    
    # Memory noise
    page_fault_rate: float = 0.0
    minor_faults: int = 0
    major_faults: int = 0
    
    # Interrupt noise
    interrupt_rate: float = 0.0
    softirq_time_pct: float = 0.0
    
    # Thermal noise
    thermal_events: int = 0
    clock_variance_pct: float = 0.0
    
    # Background load
    system_load_avg: float = 0.0
    other_process_cpu_pct: float = 0.0
    
    # Composite scores
    scheduler_interference_index: float = 0.0  # SII
    fault_pressure_index: float = 0.0  # FPI
    cpu_noise_entropy: float = 0.0  # CNE
    
    # Overall
    stability_score: float = 1.0
    noise_signatures: List[NoiseSignature] = field(default_factory=list)


@dataclass
class StabilityThresholds:
    """Thresholds for measurement stability."""
    max_scheduler_jitter_us: float = 100.0
    max_context_switch_rate: float = 1000.0
    max_page_fault_rate: float = 10.0
    max_interrupt_rate: float = 10000.0
    max_clock_variance_pct: float = 2.0
    max_background_cpu_pct: float = 5.0
    min_stability_score: float = 0.8


class NoiseSentinel:
    """
    Advanced noise detection and measurement validation.
    
    Monitors system state to detect noise that would invalidate
    scientific measurements. Computes derived indices:
    - Scheduler Interference Index (SII)
    - Fault Pressure Index (FPI)
    - CPU Noise Entropy (CNE)
    """
    
    def __init__(self, thresholds: Optional[StabilityThresholds] = None):
        """Initialize noise sentinel."""
        self.thresholds = thresholds or StabilityThresholds()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._samples: deque = deque(maxlen=10000)
        self._noise_events: List[NoiseSignature] = []
        self._lock = threading.Lock()
    
    def start_monitoring(self, interval_ms: float = 10.0) -> None:
        """Start background noise monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_ms,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Noise sentinel started (interval={interval_ms}ms)")
    
    def stop_monitoring(self) -> NoiseProfile:
        """Stop monitoring and return noise profile."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        return self._compute_noise_profile()
    
    def _monitor_loop(self, interval_ms: float) -> None:
        """Background monitoring loop."""
        interval_s = interval_ms / 1000.0
        
        while self._monitoring:
            sample = self._collect_sample()
            with self._lock:
                self._samples.append(sample)
            
            # Check for noise events
            self._check_for_noise_events(sample)
            
            time.sleep(interval_s)
    
    def _collect_sample(self) -> Dict[str, Any]:
        """Collect a single noise sample."""
        sample = {
            'timestamp': time.time(),
            'timestamp_ns': time.time_ns(),
        }
        
        # Collect scheduler stats
        sample.update(self._collect_scheduler_stats())
        
        # Collect memory stats
        sample.update(self._collect_memory_stats())
        
        # Collect interrupt stats
        sample.update(self._collect_interrupt_stats())
        
        # Collect load stats
        sample.update(self._collect_load_stats())
        
        return sample
    
    def _collect_scheduler_stats(self) -> Dict[str, Any]:
        """Collect scheduler-related statistics."""
        stats = {}
        
        try:
            # Read from /proc/schedstat
            sched_path = f"/proc/{os.getpid()}/schedstat"
            if os.path.exists(sched_path):
                with open(sched_path, 'r') as f:
                    parts = f.read().strip().split()
                    if len(parts) >= 3:
                        stats['sched_run_time_ns'] = int(parts[0])
                        stats['sched_wait_time_ns'] = int(parts[1])
                        stats['sched_slices'] = int(parts[2])
            
            # Read context switches from /proc/self/status
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('voluntary_ctxt_switches:'):
                        stats['voluntary_ctx'] = int(line.split()[1])
                    elif line.startswith('nonvoluntary_ctxt_switches:'):
                        stats['nonvoluntary_ctx'] = int(line.split()[1])
        except Exception:
            pass
        
        return stats
    
    def _collect_memory_stats(self) -> Dict[str, Any]:
        """Collect memory-related statistics."""
        stats = {}
        
        try:
            # Read from /proc/self/stat
            with open('/proc/self/stat', 'r') as f:
                parts = f.read().split()
                if len(parts) > 12:
                    stats['minor_faults'] = int(parts[9])
                    stats['major_faults'] = int(parts[11])
        except Exception:
            pass
        
        return stats
    
    def _collect_interrupt_stats(self) -> Dict[str, Any]:
        """Collect interrupt statistics."""
        stats = {}
        
        try:
            # Read from /proc/interrupts
            with open('/proc/stat', 'r') as f:
                for line in f:
                    if line.startswith('intr '):
                        parts = line.split()
                        stats['total_interrupts'] = int(parts[1])
                        break
                    elif line.startswith('softirq '):
                        parts = line.split()
                        stats['total_softirq'] = int(parts[1])
        except Exception:
            pass
        
        return stats
    
    def _collect_load_stats(self) -> Dict[str, Any]:
        """Collect system load statistics."""
        stats = {}
        
        try:
            # Load average
            with open('/proc/loadavg', 'r') as f:
                parts = f.read().split()
                stats['load_1min'] = float(parts[0])
                stats['load_5min'] = float(parts[1])
                stats['load_15min'] = float(parts[2])
        except Exception:
            pass
        
        return stats
    
    def _check_for_noise_events(self, sample: Dict[str, Any]) -> None:
        """Check sample for noise events."""
        # Check load
        load = sample.get('load_1min', 0)
        if load > os.cpu_count() * 0.5:
            self._record_noise_event(
                NoiseSource.BACKGROUND_LOAD,
                severity=min(1.0, load / os.cpu_count()),
                details={'load_1min': load}
            )
    
    def _record_noise_event(
        self,
        source: NoiseSource,
        severity: float,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a noise event."""
        event = NoiseSignature(
            source=source,
            severity=severity,
            timestamp=time.time(),
            duration_ms=0,  # Point event
            details=details or {}
        )
        with self._lock:
            self._noise_events.append(event)
    
    def _compute_noise_profile(self) -> NoiseProfile:
        """Compute comprehensive noise profile from samples."""
        with self._lock:
            samples = list(self._samples)
            events = list(self._noise_events)
        
        if not samples:
            return NoiseProfile(
                window_start=time.time(),
                window_end=time.time()
            )
        
        profile = NoiseProfile(
            window_start=samples[0]['timestamp'],
            window_end=samples[-1]['timestamp'],
            noise_signatures=events,
        )
        
        duration = profile.window_end - profile.window_start
        if duration <= 0:
            duration = 1.0
        
        # Compute scheduler metrics
        profile.scheduler_jitter_us = self._compute_scheduler_jitter(samples)
        profile.context_switch_rate = self._compute_context_switch_rate(samples, duration)
        
        # Compute memory metrics
        profile.page_fault_rate = self._compute_page_fault_rate(samples, duration)
        
        # Compute load metrics
        loads = [s.get('load_1min', 0) for s in samples if 'load_1min' in s]
        if loads:
            profile.system_load_avg = statistics.mean(loads)
        
        # Compute derived indices
        profile.scheduler_interference_index = self._compute_sii(profile)
        profile.fault_pressure_index = self._compute_fpi(profile)
        profile.cpu_noise_entropy = self._compute_cne(profile)
        
        # Compute overall stability score
        profile.stability_score = self._compute_stability_score(profile)
        
        return profile
    
    def _compute_scheduler_jitter(self, samples: List[Dict[str, Any]]) -> float:
        """Compute scheduler jitter in microseconds."""
        if len(samples) < 2:
            return 0.0
        
        # Compute inter-sample timing variance
        timestamps = [s['timestamp_ns'] for s in samples if 'timestamp_ns' in s]
        if len(timestamps) < 2:
            return 0.0
        
        deltas = [
            (timestamps[i+1] - timestamps[i]) / 1000.0  # Convert to us
            for i in range(len(timestamps) - 1)
        ]
        
        if not deltas:
            return 0.0
        
        # Jitter is the standard deviation of deltas
        if len(deltas) > 1:
            return statistics.stdev(deltas)
        return 0.0
    
    def _compute_context_switch_rate(
        self,
        samples: List[Dict[str, Any]],
        duration: float
    ) -> float:
        """Compute context switch rate per second."""
        if len(samples) < 2:
            return 0.0
        
        start_ctx = samples[0].get('voluntary_ctx', 0) + samples[0].get('nonvoluntary_ctx', 0)
        end_ctx = samples[-1].get('voluntary_ctx', 0) + samples[-1].get('nonvoluntary_ctx', 0)
        
        return (end_ctx - start_ctx) / duration if duration > 0 else 0.0
    
    def _compute_page_fault_rate(
        self,
        samples: List[Dict[str, Any]],
        duration: float
    ) -> float:
        """Compute page fault rate per second."""
        if len(samples) < 2:
            return 0.0
        
        start_faults = samples[0].get('minor_faults', 0) + samples[0].get('major_faults', 0)
        end_faults = samples[-1].get('minor_faults', 0) + samples[-1].get('major_faults', 0)
        
        return (end_faults - start_faults) / duration if duration > 0 else 0.0
    
    def _compute_sii(self, profile: NoiseProfile) -> float:
        """
        Compute Scheduler Interference Index (SII).
        
        SII = weighted combination of:
        - Context switch rate (normalized)
        - Scheduler jitter (normalized)
        - Runqueue latency (normalized)
        """
        # Normalize components to 0-1 scale
        ctx_norm = min(1.0, profile.context_switch_rate / 10000.0)
        jitter_norm = min(1.0, profile.scheduler_jitter_us / 1000.0)
        
        # Weighted combination
        sii = (
            0.4 * ctx_norm +
            0.4 * jitter_norm +
            0.2 * min(1.0, profile.runqueue_latency_p99_us / 10000.0)
        )
        
        return sii
    
    def _compute_fpi(self, profile: NoiseProfile) -> float:
        """
        Compute Fault Pressure Index (FPI).
        
        FPI = weighted combination of:
        - Page fault rate
        - Major vs minor fault ratio
        - Memory pressure indicators
        """
        fault_norm = min(1.0, profile.page_fault_rate / 1000.0)
        
        major_ratio = 0.0
        total_faults = profile.minor_faults + profile.major_faults
        if total_faults > 0:
            major_ratio = profile.major_faults / total_faults
        
        fpi = (
            0.6 * fault_norm +
            0.4 * major_ratio
        )
        
        return fpi
    
    def _compute_cne(self, profile: NoiseProfile) -> float:
        """
        Compute CPU Noise Entropy (CNE).
        
        Measures the unpredictability/randomness of CPU behavior.
        Higher values indicate more noise.
        """
        # Combine multiple noise sources
        noise_sources = [
            profile.scheduler_interference_index,
            profile.fault_pressure_index,
            min(1.0, profile.interrupt_rate / 100000.0) if profile.interrupt_rate else 0,
            min(1.0, profile.system_load_avg / os.cpu_count()) if profile.system_load_avg else 0,
        ]
        
        # Entropy-like combination
        cne = 1.0 - (1.0 - noise_sources[0]) * (1.0 - noise_sources[1]) * \
              (1.0 - noise_sources[2]) * (1.0 - noise_sources[3])
        
        return cne
    
    def _compute_stability_score(self, profile: NoiseProfile) -> float:
        """
        Compute overall stability score (0-1, higher is better).
        """
        penalties = []
        
        # Scheduler jitter penalty
        if profile.scheduler_jitter_us > self.thresholds.max_scheduler_jitter_us:
            penalty = min(0.3, (profile.scheduler_jitter_us - self.thresholds.max_scheduler_jitter_us) / 1000.0)
            penalties.append(penalty)
        
        # Context switch penalty
        if profile.context_switch_rate > self.thresholds.max_context_switch_rate:
            penalty = min(0.2, (profile.context_switch_rate - self.thresholds.max_context_switch_rate) / 10000.0)
            penalties.append(penalty)
        
        # Background load penalty
        if profile.other_process_cpu_pct > self.thresholds.max_background_cpu_pct:
            penalty = min(0.3, (profile.other_process_cpu_pct - self.thresholds.max_background_cpu_pct) / 100.0)
            penalties.append(penalty)
        
        # Noise events penalty
        for event in profile.noise_signatures:
            penalties.append(event.severity * 0.1)
        
        stability = 1.0 - sum(penalties)
        return max(0.0, min(1.0, stability))
    
    def validate_measurement(self, profile: NoiseProfile) -> Dict[str, Any]:
        """
        Validate if measurements taken during this profile are scientifically valid.
        
        Returns:
            Dict with validation result and reasoning
        """
        result = {
            'valid': True,
            'stability_score': profile.stability_score,
            'threshold': self.thresholds.min_stability_score,
            'violations': [],
            'warnings': [],
        }
        
        # Check stability threshold
        if profile.stability_score < self.thresholds.min_stability_score:
            result['valid'] = False
            result['violations'].append({
                'metric': 'stability_score',
                'value': profile.stability_score,
                'threshold': self.thresholds.min_stability_score,
                'message': 'Overall stability below threshold'
            })
        
        # Check individual thresholds
        if profile.scheduler_jitter_us > self.thresholds.max_scheduler_jitter_us:
            result['warnings'].append({
                'metric': 'scheduler_jitter_us',
                'value': profile.scheduler_jitter_us,
                'threshold': self.thresholds.max_scheduler_jitter_us,
            })
        
        if profile.context_switch_rate > self.thresholds.max_context_switch_rate:
            result['warnings'].append({
                'metric': 'context_switch_rate',
                'value': profile.context_switch_rate,
                'threshold': self.thresholds.max_context_switch_rate,
            })
        
        # Classify noise level
        if profile.cpu_noise_entropy < 0.1:
            result['noise_level'] = 'excellent'
        elif profile.cpu_noise_entropy < 0.3:
            result['noise_level'] = 'good'
        elif profile.cpu_noise_entropy < 0.5:
            result['noise_level'] = 'acceptable'
        else:
            result['noise_level'] = 'high'
            if result['valid']:
                result['warnings'].append({
                    'metric': 'cpu_noise_entropy',
                    'value': profile.cpu_noise_entropy,
                    'message': 'High noise entropy detected'
                })
        
        return result
    
    def get_noise_summary(self) -> Dict[str, Any]:
        """Get current noise summary without stopping monitoring."""
        profile = self._compute_noise_profile()
        validation = self.validate_measurement(profile)
        
        return {
            'profile': {
                'scheduler_interference_index': profile.scheduler_interference_index,
                'fault_pressure_index': profile.fault_pressure_index,
                'cpu_noise_entropy': profile.cpu_noise_entropy,
                'stability_score': profile.stability_score,
            },
            'validation': validation,
            'sample_count': len(self._samples),
            'event_count': len(self._noise_events),
        }
