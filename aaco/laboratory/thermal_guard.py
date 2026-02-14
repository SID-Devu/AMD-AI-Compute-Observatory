# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Thermal Guard

Monitors thermal state and detects throttling that would invalidate measurements.
"""

import subprocess
import json
import time
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ThermalState(Enum):
    """GPU thermal state."""
    STABLE = "stable"
    WARMING = "warming"
    THROTTLING = "throttling"
    CRITICAL = "critical"


@dataclass
class ThermalSample:
    """Single thermal sample."""
    timestamp: float
    gpu_id: int
    temperature_c: float
    clock_mhz: float
    power_watts: float
    throttle_status: bool = False


@dataclass
class ThermalProfile:
    """Complete thermal profile for a measurement window."""
    gpu_id: int
    state: ThermalState
    
    # Temperature
    temp_min_c: float = 0.0
    temp_max_c: float = 0.0
    temp_mean_c: float = 0.0
    temp_delta_c: float = 0.0
    
    # Clocks
    clock_min_mhz: float = 0.0
    clock_max_mhz: float = 0.0
    clock_mean_mhz: float = 0.0
    clock_variance_pct: float = 0.0
    
    # Power
    power_min_watts: float = 0.0
    power_max_watts: float = 0.0
    power_mean_watts: float = 0.0
    
    # Throttling
    throttle_events: int = 0
    throttle_duration_pct: float = 0.0
    
    # Stability assessment
    clock_instability_score: float = 0.0  # CIS
    thermal_factor: float = 1.0  # Multiplier for expected perf


class ThermalGuard:
    """
    Monitors GPU thermal state during measurements.
    
    Detects thermal throttling that would invalidate measurements
    and computes Clock Instability Score (CIS).
    """
    
    # Thermal thresholds
    THROTTLE_TEMP_C = 90.0
    CRITICAL_TEMP_C = 100.0
    STABLE_VARIANCE_PCT = 3.0
    
    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        sample_interval_ms: float = 100.0
    ):
        """Initialize thermal guard."""
        self.gpu_ids = gpu_ids or [0]
        self.sample_interval_ms = sample_interval_ms
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._samples: Dict[int, deque] = {
            gpu_id: deque(maxlen=10000) for gpu_id in self.gpu_ids
        }
        self._lock = threading.Lock()
    
    def start_monitoring(self) -> None:
        """Start thermal monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info(f"Thermal guard started for GPUs: {self.gpu_ids}")
    
    def stop_monitoring(self) -> Dict[int, ThermalProfile]:
        """Stop monitoring and return thermal profiles."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        
        return {
            gpu_id: self._compute_thermal_profile(gpu_id)
            for gpu_id in self.gpu_ids
        }
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        interval_s = self.sample_interval_ms / 1000.0
        
        while self._monitoring:
            samples = self._collect_thermal_samples()
            
            with self._lock:
                for sample in samples:
                    if sample.gpu_id in self._samples:
                        self._samples[sample.gpu_id].append(sample)
            
            time.sleep(interval_s)
    
    def _collect_thermal_samples(self) -> List[ThermalSample]:
        """Collect thermal samples from all GPUs."""
        samples = []
        timestamp = time.time()
        
        try:
            result = subprocess.run(
                ['rocm-smi', '--showtemp', '--showclocks', '--showpower', '--json'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                
                for gpu_id in self.gpu_ids:
                    card_key = f'card{gpu_id}'
                    if card_key in data:
                        card = data[card_key]
                        
                        # Parse temperature
                        temp = card.get('Temperature (Sensor edge) (C)', 0)
                        if isinstance(temp, str):
                            temp = float(temp.split()[0]) if temp else 0
                        
                        # Parse clock
                        clock = card.get('GPU Clock Level', '0 MHz')
                        if isinstance(clock, str):
                            clock = float(clock.split()[0]) if clock else 0
                        
                        # Parse power
                        power = card.get('Average Graphics Package Power (W)', 0)
                        if isinstance(power, str):
                            power = float(power.split()[0]) if power else 0
                        
                        sample = ThermalSample(
                            timestamp=timestamp,
                            gpu_id=gpu_id,
                            temperature_c=float(temp),
                            clock_mhz=float(clock),
                            power_watts=float(power),
                            throttle_status=float(temp) >= self.THROTTLE_TEMP_C
                        )
                        samples.append(sample)
                        
        except Exception as e:
            logger.warning(f"Failed to collect thermal samples: {e}")
        
        return samples
    
    def _compute_thermal_profile(self, gpu_id: int) -> ThermalProfile:
        """Compute thermal profile from samples."""
        with self._lock:
            samples = list(self._samples.get(gpu_id, []))
        
        if not samples:
            return ThermalProfile(gpu_id=gpu_id, state=ThermalState.STABLE)
        
        temps = [s.temperature_c for s in samples]
        clocks = [s.clock_mhz for s in samples if s.clock_mhz > 0]
        powers = [s.power_watts for s in samples if s.power_watts > 0]
        
        profile = ThermalProfile(
            gpu_id=gpu_id,
            state=ThermalState.STABLE,
        )
        
        # Temperature stats
        if temps:
            profile.temp_min_c = min(temps)
            profile.temp_max_c = max(temps)
            profile.temp_mean_c = sum(temps) / len(temps)
            profile.temp_delta_c = profile.temp_max_c - profile.temp_min_c
        
        # Clock stats
        if clocks:
            profile.clock_min_mhz = min(clocks)
            profile.clock_max_mhz = max(clocks)
            profile.clock_mean_mhz = sum(clocks) / len(clocks)
            
            # Clock variance
            if profile.clock_mean_mhz > 0:
                variance = sum((c - profile.clock_mean_mhz) ** 2 for c in clocks) / len(clocks)
                std_dev = variance ** 0.5
                profile.clock_variance_pct = (std_dev / profile.clock_mean_mhz) * 100
        
        # Power stats
        if powers:
            profile.power_min_watts = min(powers)
            profile.power_max_watts = max(powers)
            profile.power_mean_watts = sum(powers) / len(powers)
        
        # Throttle events
        throttle_samples = [s for s in samples if s.throttle_status]
        profile.throttle_events = len(throttle_samples)
        profile.throttle_duration_pct = (len(throttle_samples) / len(samples)) * 100
        
        # Compute state
        profile.state = self._determine_thermal_state(profile)
        
        # Compute Clock Instability Score (CIS)
        profile.clock_instability_score = self._compute_cis(profile, clocks)
        
        # Compute thermal factor
        profile.thermal_factor = self._compute_thermal_factor(profile)
        
        return profile
    
    def _determine_thermal_state(self, profile: ThermalProfile) -> ThermalState:
        """Determine overall thermal state."""
        if profile.temp_max_c >= self.CRITICAL_TEMP_C:
            return ThermalState.CRITICAL
        
        if profile.throttle_duration_pct > 5:
            return ThermalState.THROTTLING
        
        if profile.temp_delta_c > 10 or profile.clock_variance_pct > 5:
            return ThermalState.WARMING
        
        return ThermalState.STABLE
    
    def _compute_cis(
        self,
        profile: ThermalProfile,
        clocks: List[float]
    ) -> float:
        """
        Compute Clock Instability Score (CIS).
        
        CIS ranges from 0 (perfectly stable) to 1 (highly unstable).
        """
        if not clocks or len(clocks) < 2:
            return 0.0
        
        # Component 1: Clock variance contribution
        variance_score = min(1.0, profile.clock_variance_pct / 10.0)
        
        # Component 2: Clock drop events (sudden decreases)
        drop_count = 0
        for i in range(1, len(clocks)):
            if clocks[i] < clocks[i-1] * 0.95:  # 5% drop
                drop_count += 1
        drop_score = min(1.0, drop_count / (len(clocks) * 0.1))
        
        # Component 3: Range score
        if profile.clock_max_mhz > 0:
            range_pct = (profile.clock_max_mhz - profile.clock_min_mhz) / profile.clock_max_mhz
            range_score = min(1.0, range_pct / 0.2)
        else:
            range_score = 0.0
        
        # Weighted combination
        cis = (
            0.5 * variance_score +
            0.3 * drop_score +
            0.2 * range_score
        )
        
        return cis
    
    def _compute_thermal_factor(self, profile: ThermalProfile) -> float:
        """
        Compute thermal factor - expected performance multiplier.
        
        Returns a value <= 1.0 indicating expected performance
        relative to steady-state.
        """
        factor = 1.0
        
        # Penalize for throttling
        if profile.throttle_duration_pct > 0:
            factor *= (1.0 - profile.throttle_duration_pct / 200.0)
        
        # Penalize for clock instability
        factor *= (1.0 - profile.clock_instability_score * 0.2)
        
        return max(0.5, factor)
    
    def validate_thermal_stability(
        self,
        profiles: Dict[int, ThermalProfile]
    ) -> Dict[str, Any]:
        """
        Validate if thermal state allows valid measurements.
        
        Returns validation result with recommendations.
        """
        result = {
            'valid': True,
            'gpu_states': {},
            'violations': [],
            'warnings': [],
            'recommendations': [],
        }
        
        for gpu_id, profile in profiles.items():
            result['gpu_states'][gpu_id] = {
                'state': profile.state.value,
                'cis': profile.clock_instability_score,
                'thermal_factor': profile.thermal_factor,
            }
            
            # Check for critical conditions
            if profile.state == ThermalState.CRITICAL:
                result['valid'] = False
                result['violations'].append({
                    'gpu_id': gpu_id,
                    'reason': f'Critical temperature: {profile.temp_max_c:.1f}C'
                })
            
            # Check for significant throttling
            if profile.state == ThermalState.THROTTLING:
                result['valid'] = False
                result['violations'].append({
                    'gpu_id': gpu_id,
                    'reason': f'Throttling detected: {profile.throttle_duration_pct:.1f}% of time'
                })
            
            # Warnings for warming state
            if profile.state == ThermalState.WARMING:
                result['warnings'].append({
                    'gpu_id': gpu_id,
                    'reason': 'GPU temperature unstable',
                    'temp_delta': profile.temp_delta_c,
                })
            
            # High CIS warning
            if profile.clock_instability_score > 0.3:
                result['warnings'].append({
                    'gpu_id': gpu_id,
                    'reason': f'High clock instability: CIS={profile.clock_instability_score:.2f}'
                })
        
        # Generate recommendations
        if not result['valid']:
            result['recommendations'].extend([
                "Wait for GPU to cool down before measurement",
                "Improve cooling (increase fan speed if manual control available)",
                "Reduce sustained load or add cooldown periods",
                "Consider undervolting to reduce thermal load",
            ])
        
        if result['warnings']:
            result['recommendations'].extend([
                "Increase warmup time to reach thermal steady-state",
                "Lock GPU clocks to reduce variance",
            ])
        
        return result
    
    def get_current_state(self) -> Dict[int, Dict[str, Any]]:
        """Get current thermal state without stopping monitoring."""
        state = {}
        
        for gpu_id in self.gpu_ids:
            profile = self._compute_thermal_profile(gpu_id)
            state[gpu_id] = {
                'state': profile.state.value,
                'temp_c': profile.temp_mean_c,
                'clock_mhz': profile.clock_mean_mhz,
                'power_w': profile.power_mean_watts,
                'cis': profile.clock_instability_score,
            }
        
        return state
