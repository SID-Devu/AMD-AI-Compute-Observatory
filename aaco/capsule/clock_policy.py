"""
AACO-SIGMA Clock Policy Management

Captures and enforces CPU governor and GPU clock policies
for deterministic performance measurements.
"""

import os
import re
import json
import subprocess
import threading
import time
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager


class CPUGovernor(Enum):
    """CPU frequency governors."""
    PERFORMANCE = "performance"      # Max frequency
    POWERSAVE = "powersave"          # Min frequency
    USERSPACE = "userspace"          # User-controlled
    ONDEMAND = "ondemand"            # Dynamic scaling
    CONSERVATIVE = "conservative"    # Gradual scaling
    SCHEDUTIL = "schedutil"          # Scheduler-based


class GPUPowerProfile(Enum):
    """AMD GPU power profiles."""
    AUTO = "auto"
    LOW = "low"
    HIGH = "high"
    VR = "vr"
    COMPUTE = "compute"
    CUSTOM = "custom"


@dataclass
class CPUGovernorPolicy:
    """CPU governor policy configuration."""
    governor: CPUGovernor = CPUGovernor.PERFORMANCE
    min_freq_khz: Optional[int] = None
    max_freq_khz: Optional[int] = None
    
    # Energy performance preference (Intel/AMD)
    energy_perf_pref: str = "performance"  # performance, balance_performance, balance_power, power
    
    # Boost control
    boost_enabled: bool = True
    
    # Per-core settings (if different)
    per_core_settings: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['governor'] = self.governor.value
        return d


@dataclass
class GPUClockPolicy:
    """GPU clock policy configuration."""
    # Target clocks
    target_sclk_mhz: Optional[int] = None   # GPU clock
    target_mclk_mhz: Optional[int] = None   # Memory clock
    
    # Clock range (locked if min == max)
    min_sclk_mhz: Optional[int] = None
    max_sclk_mhz: Optional[int] = None
    min_mclk_mhz: Optional[int] = None
    max_mclk_mhz: Optional[int] = None
    
    # Power profile
    power_profile: GPUPowerProfile = GPUPowerProfile.COMPUTE
    
    # Power cap
    power_cap_w: Optional[float] = None
    
    # Fan control (if supported)
    fan_speed_pct: Optional[int] = None  # Manual fan speed
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['power_profile'] = self.power_profile.value
        return d


@dataclass
class ClockState:
    """Current clock state snapshot."""
    timestamp_ns: int = 0
    
    # CPU state
    cpu_governors: Dict[int, str] = field(default_factory=dict)  # core_id -> governor
    cpu_frequencies_khz: Dict[int, int] = field(default_factory=dict)  # core_id -> freq
    cpu_boost_enabled: bool = True
    
    # GPU state
    gpu_sclk_mhz: int = 0
    gpu_mclk_mhz: int = 0
    gpu_power_profile: str = ""
    gpu_power_draw_w: float = 0.0
    gpu_power_cap_w: float = 0.0
    gpu_temp_c: float = 0.0
    gpu_fan_speed_pct: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClockPolicy:
    """Combined CPU and GPU clock policy."""
    cpu_policy: CPUGovernorPolicy = field(default_factory=CPUGovernorPolicy)
    gpu_policy: GPUClockPolicy = field(default_factory=GPUClockPolicy)
    
    # Enforcement options
    enforce_cpu: bool = True
    enforce_gpu: bool = False  # Requires root/capabilities
    
    # Monitoring
    monitor_interval_ms: int = 100
    detect_throttling: bool = True
    max_clock_variance_pct: float = 5.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_policy': self.cpu_policy.to_dict(),
            'gpu_policy': self.gpu_policy.to_dict(),
            'enforce_cpu': self.enforce_cpu,
            'enforce_gpu': self.enforce_gpu,
            'monitor_interval_ms': self.monitor_interval_ms,
            'detect_throttling': self.detect_throttling,
            'max_clock_variance_pct': self.max_clock_variance_pct,
        }
    
    @classmethod
    def performance(cls) -> 'ClockPolicy':
        """Create a performance-optimized policy."""
        return cls(
            cpu_policy=CPUGovernorPolicy(
                governor=CPUGovernor.PERFORMANCE,
                boost_enabled=True,
                energy_perf_pref="performance"
            ),
            gpu_policy=GPUClockPolicy(
                power_profile=GPUPowerProfile.COMPUTE
            ),
            enforce_cpu=True,
            enforce_gpu=True
        )
    
    @classmethod
    def stable(cls) -> 'ClockPolicy':
        """Create a stability-focused policy with locked clocks."""
        return cls(
            cpu_policy=CPUGovernorPolicy(
                governor=CPUGovernor.PERFORMANCE,
                boost_enabled=False,  # Disable turbo for stability
                energy_perf_pref="performance"
            ),
            gpu_policy=GPUClockPolicy(
                power_profile=GPUPowerProfile.COMPUTE
            ),
            enforce_cpu=True,
            enforce_gpu=True,
            max_clock_variance_pct=2.0
        )


class ClockEnforcer:
    """
    Enforces clock policies and monitors clock state.
    
    Provides:
    - CPU governor enforcement
    - GPU clock enforcement (via rocm-smi)
    - Clock stability monitoring
    - Throttling detection
    """
    
    CPU_FREQ_BASE = Path("/sys/devices/system/cpu")
    
    def __init__(self, policy: Optional[ClockPolicy] = None):
        self.policy = policy or ClockPolicy()
        
        # State tracking
        self._original_cpu_state: Dict[int, Dict[str, Any]] = {}
        self._original_gpu_state: Dict[str, Any] = {}
        self._clock_samples: List[ClockState] = []
        
        # Monitoring
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._throttle_events: List[Dict[str, Any]] = []
        
        # Available clocks (discovered)
        self._available_sclk: List[int] = []
        self._available_mclk: List[int] = []
    
    def capture_state(self) -> ClockState:
        """Capture current clock state."""
        state = ClockState(timestamp_ns=time.time_ns())
        
        # Capture CPU state
        state.cpu_governors = self._get_cpu_governors()
        state.cpu_frequencies_khz = self._get_cpu_frequencies()
        state.cpu_boost_enabled = self._get_boost_enabled()
        
        # Capture GPU state
        gpu_state = self._get_gpu_state()
        state.gpu_sclk_mhz = gpu_state.get('sclk_mhz', 0)
        state.gpu_mclk_mhz = gpu_state.get('mclk_mhz', 0)
        state.gpu_power_profile = gpu_state.get('power_profile', '')
        state.gpu_power_draw_w = gpu_state.get('power_draw_w', 0.0)
        state.gpu_power_cap_w = gpu_state.get('power_cap_w', 0.0)
        state.gpu_temp_c = gpu_state.get('temp_c', 0.0)
        state.gpu_fan_speed_pct = gpu_state.get('fan_speed_pct', 0)
        
        return state
    
    def _get_cpu_governors(self) -> Dict[int, str]:
        """Get current CPU governors for all cores."""
        governors = {}
        
        try:
            for cpu_dir in self.CPU_FREQ_BASE.glob("cpu[0-9]*"):
                cpu_id = int(cpu_dir.name[3:])
                gov_file = cpu_dir / "cpufreq" / "scaling_governor"
                if gov_file.exists():
                    governors[cpu_id] = gov_file.read_text().strip()
        except:
            pass
        
        return governors
    
    def _get_cpu_frequencies(self) -> Dict[int, int]:
        """Get current CPU frequencies for all cores."""
        frequencies = {}
        
        try:
            for cpu_dir in self.CPU_FREQ_BASE.glob("cpu[0-9]*"):
                cpu_id = int(cpu_dir.name[3:])
                freq_file = cpu_dir / "cpufreq" / "scaling_cur_freq"
                if freq_file.exists():
                    frequencies[cpu_id] = int(freq_file.read_text().strip())
        except:
            pass
        
        return frequencies
    
    def _get_boost_enabled(self) -> bool:
        """Check if CPU boost/turbo is enabled."""
        # AMD: /sys/devices/system/cpu/cpufreq/boost
        # Intel: /sys/devices/system/cpu/intel_pstate/no_turbo
        
        boost_file = self.CPU_FREQ_BASE / "cpufreq" / "boost"
        if boost_file.exists():
            try:
                return boost_file.read_text().strip() == "1"
            except:
                pass
        
        no_turbo_file = self.CPU_FREQ_BASE / "intel_pstate" / "no_turbo"
        if no_turbo_file.exists():
            try:
                return no_turbo_file.read_text().strip() == "0"
            except:
                pass
        
        return True
    
    def _get_gpu_state(self) -> Dict[str, Any]:
        """Get current GPU state via rocm-smi."""
        state = {}
        
        try:
            # Get clocks
            result = subprocess.run(
                ["rocm-smi", "--showclocks"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    line_lower = line.lower()
                    if 'sclk' in line_lower or 'gpu' in line_lower:
                        match = re.search(r'(\d+)\s*mhz', line_lower)
                        if match:
                            state['sclk_mhz'] = int(match.group(1))
                    elif 'mclk' in line_lower or 'mem' in line_lower:
                        match = re.search(r'(\d+)\s*mhz', line_lower)
                        if match:
                            state['mclk_mhz'] = int(match.group(1))
            
            # Get power
            result = subprocess.run(
                ["rocm-smi", "--showpower"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    line_lower = line.lower()
                    if 'power' in line_lower and 'cap' not in line_lower:
                        match = re.search(r'(\d+\.?\d*)\s*w', line_lower)
                        if match:
                            state['power_draw_w'] = float(match.group(1))
                    elif 'cap' in line_lower:
                        match = re.search(r'(\d+\.?\d*)\s*w', line_lower)
                        if match:
                            state['power_cap_w'] = float(match.group(1))
            
            # Get temperature
            result = subprocess.run(
                ["rocm-smi", "--showtemp"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                match = re.search(r'(\d+\.?\d*)\s*c', result.stdout.lower())
                if match:
                    state['temp_c'] = float(match.group(1))
            
            # Get fan
            result = subprocess.run(
                ["rocm-smi", "--showfan"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                match = re.search(r'(\d+)\s*%', result.stdout)
                if match:
                    state['fan_speed_pct'] = int(match.group(1))
            
            # Get power profile
            result = subprocess.run(
                ["rocm-smi", "--showprofile"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if '*' in line:  # Active profile marked with *
                        state['power_profile'] = line.strip().replace('*', '').strip()
                        break
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return state
    
    def _discover_gpu_clocks(self) -> None:
        """Discover available GPU clock levels."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showclkfrq"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    # Parse available frequencies
                    match = re.findall(r'(\d+)\s*Mhz', line, re.IGNORECASE)
                    if match:
                        if 'sclk' in line.lower() or 'gpu' in line.lower():
                            self._available_sclk = [int(m) for m in match]
                        elif 'mclk' in line.lower() or 'mem' in line.lower():
                            self._available_mclk = [int(m) for m in match]
        except:
            pass
    
    def enforce(self) -> Tuple[bool, List[str]]:
        """
        Enforce the clock policy.
        
        Returns:
            Tuple of (success, list of error messages)
        """
        errors = []
        
        # Save original state
        self._original_cpu_state = {}
        for cpu_id, gov in self._get_cpu_governors().items():
            self._original_cpu_state[cpu_id] = {
                'governor': gov,
                'frequency': self._get_cpu_frequencies().get(cpu_id, 0)
            }
        self._original_gpu_state = self._get_gpu_state()
        
        # Enforce CPU policy
        if self.policy.enforce_cpu:
            cpu_errors = self._enforce_cpu_policy()
            errors.extend(cpu_errors)
        
        # Enforce GPU policy
        if self.policy.enforce_gpu:
            gpu_errors = self._enforce_gpu_policy()
            errors.extend(gpu_errors)
        
        return len(errors) == 0, errors
    
    def _enforce_cpu_policy(self) -> List[str]:
        """Enforce CPU governor policy."""
        errors = []
        policy = self.policy.cpu_policy
        
        try:
            # Set governor for all cores
            target_gov = policy.governor.value
            
            for cpu_dir in self.CPU_FREQ_BASE.glob("cpu[0-9]*"):
                cpu_id = int(cpu_dir.name[3:])
                
                # Check per-core override
                if cpu_id in policy.per_core_settings:
                    per_core = policy.per_core_settings[cpu_id]
                    target_gov = per_core.get('governor', target_gov)
                
                # Set governor
                gov_file = cpu_dir / "cpufreq" / "scaling_governor"
                if gov_file.exists():
                    try:
                        gov_file.write_text(target_gov)
                    except PermissionError:
                        errors.append(f"Permission denied setting governor for cpu{cpu_id}")
                
                # Set min/max freq if specified
                if policy.min_freq_khz:
                    min_file = cpu_dir / "cpufreq" / "scaling_min_freq"
                    if min_file.exists():
                        try:
                            min_file.write_text(str(policy.min_freq_khz))
                        except:
                            pass
                
                if policy.max_freq_khz:
                    max_file = cpu_dir / "cpufreq" / "scaling_max_freq"
                    if max_file.exists():
                        try:
                            max_file.write_text(str(policy.max_freq_khz))
                        except:
                            pass
                
                # Set energy perf preference
                epp_file = cpu_dir / "cpufreq" / "energy_performance_preference"
                if epp_file.exists():
                    try:
                        epp_file.write_text(policy.energy_perf_pref)
                    except:
                        pass
            
            # Set boost
            boost_file = self.CPU_FREQ_BASE / "cpufreq" / "boost"
            if boost_file.exists():
                try:
                    boost_file.write_text("1" if policy.boost_enabled else "0")
                except PermissionError:
                    errors.append("Permission denied setting CPU boost")
            
            no_turbo_file = self.CPU_FREQ_BASE / "intel_pstate" / "no_turbo"
            if no_turbo_file.exists():
                try:
                    no_turbo_file.write_text("0" if policy.boost_enabled else "1")
                except PermissionError:
                    errors.append("Permission denied setting Intel turbo")
        
        except Exception as e:
            errors.append(f"CPU policy enforcement error: {e}")
        
        return errors
    
    def _enforce_gpu_policy(self) -> List[str]:
        """Enforce GPU clock policy via rocm-smi."""
        errors = []
        policy = self.policy.gpu_policy
        
        try:
            # Set power profile
            if policy.power_profile != GPUPowerProfile.AUTO:
                result = subprocess.run(
                    ["rocm-smi", "--setprofile", policy.power_profile.value],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    errors.append(f"Failed to set GPU power profile: {result.stderr}")
            
            # Set power cap
            if policy.power_cap_w is not None:
                result = subprocess.run(
                    ["rocm-smi", "--setpoweroverdrive", str(int(policy.power_cap_w))],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    errors.append(f"Failed to set GPU power cap: {result.stderr}")
            
            # Set GPU clock level (if we want to lock clocks)
            if policy.target_sclk_mhz is not None:
                # Find closest available clock level
                self._discover_gpu_clocks()
                if self._available_sclk:
                    closest_level = min(
                        range(len(self._available_sclk)),
                        key=lambda i: abs(self._available_sclk[i] - policy.target_sclk_mhz)
                    )
                    result = subprocess.run(
                        ["rocm-smi", "--setsclk", str(closest_level)],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode != 0:
                        errors.append(f"Failed to set GPU clock: {result.stderr}")
            
            # Set memory clock level
            if policy.target_mclk_mhz is not None:
                if self._available_mclk:
                    closest_level = min(
                        range(len(self._available_mclk)),
                        key=lambda i: abs(self._available_mclk[i] - policy.target_mclk_mhz)
                    )
                    result = subprocess.run(
                        ["rocm-smi", "--setmclk", str(closest_level)],
                        capture_output=True, text=True, timeout=10
                    )
                    if result.returncode != 0:
                        errors.append(f"Failed to set memory clock: {result.stderr}")
            
            # Set fan speed
            if policy.fan_speed_pct is not None:
                result = subprocess.run(
                    ["rocm-smi", "--setfan", str(policy.fan_speed_pct)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode != 0:
                    errors.append(f"Failed to set fan speed: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            errors.append("GPU policy enforcement timed out")
        except FileNotFoundError:
            errors.append("rocm-smi not found")
        except Exception as e:
            errors.append(f"GPU policy enforcement error: {e}")
        
        return errors
    
    def restore(self) -> None:
        """Restore original clock state."""
        # Restore CPU governors
        for cpu_id, state in self._original_cpu_state.items():
            cpu_dir = self.CPU_FREQ_BASE / f"cpu{cpu_id}"
            gov_file = cpu_dir / "cpufreq" / "scaling_governor"
            if gov_file.exists():
                try:
                    gov_file.write_text(state['governor'])
                except:
                    pass
        
        # Restore GPU (reset to auto)
        try:
            subprocess.run(
                ["rocm-smi", "--resetclocks"],
                capture_output=True, timeout=10
            )
            subprocess.run(
                ["rocm-smi", "--setprofile", "auto"],
                capture_output=True, timeout=10
            )
            subprocess.run(
                ["rocm-smi", "--resetfans"],
                capture_output=True, timeout=10
            )
        except:
            pass
    
    def start_monitoring(self) -> None:
        """Start clock monitoring thread."""
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop clock monitoring thread."""
        self._stop_monitoring.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        interval = self.policy.monitor_interval_ms / 1000.0
        
        while not self._stop_monitoring.is_set():
            state = self.capture_state()
            self._clock_samples.append(state)
            
            # Check for throttling
            if self.policy.detect_throttling:
                self._check_throttling(state)
            
            self._stop_monitoring.wait(interval)
    
    def _check_throttling(self, state: ClockState) -> None:
        """Check for signs of throttling."""
        if not self._clock_samples:
            return
        
        # Compare to first sample (baseline)
        baseline = self._clock_samples[0]
        
        # Check GPU clock drop
        if baseline.gpu_sclk_mhz > 0 and state.gpu_sclk_mhz > 0:
            drop_pct = (baseline.gpu_sclk_mhz - state.gpu_sclk_mhz) / baseline.gpu_sclk_mhz * 100
            if drop_pct > self.policy.max_clock_variance_pct:
                self._throttle_events.append({
                    "type": "gpu_clock_drop",
                    "timestamp_ns": state.timestamp_ns,
                    "baseline_mhz": baseline.gpu_sclk_mhz,
                    "current_mhz": state.gpu_sclk_mhz,
                    "drop_pct": drop_pct,
                    "temp_c": state.gpu_temp_c
                })
        
        # Check CPU frequency drop
        for cpu_id, baseline_freq in baseline.cpu_frequencies_khz.items():
            if baseline_freq > 0 and cpu_id in state.cpu_frequencies_khz:
                current_freq = state.cpu_frequencies_khz[cpu_id]
                drop_pct = (baseline_freq - current_freq) / baseline_freq * 100
                if drop_pct > self.policy.max_clock_variance_pct:
                    self._throttle_events.append({
                        "type": "cpu_freq_drop",
                        "timestamp_ns": state.timestamp_ns,
                        "cpu_id": cpu_id,
                        "baseline_khz": baseline_freq,
                        "current_khz": current_freq,
                        "drop_pct": drop_pct
                    })
    
    def get_clock_statistics(self) -> Dict[str, Any]:
        """Get statistics from clock monitoring."""
        if not self._clock_samples:
            return {}
        
        gpu_clocks = [s.gpu_sclk_mhz for s in self._clock_samples if s.gpu_sclk_mhz > 0]
        gpu_temps = [s.gpu_temp_c for s in self._clock_samples if s.gpu_temp_c > 0]
        
        stats = {
            "sample_count": len(self._clock_samples),
            "duration_ns": self._clock_samples[-1].timestamp_ns - self._clock_samples[0].timestamp_ns,
            "throttle_events": len(self._throttle_events),
        }
        
        if gpu_clocks:
            stats["gpu_clock"] = {
                "min_mhz": min(gpu_clocks),
                "max_mhz": max(gpu_clocks),
                "mean_mhz": sum(gpu_clocks) / len(gpu_clocks),
                "variance_pct": (max(gpu_clocks) - min(gpu_clocks)) / max(gpu_clocks) * 100 if max(gpu_clocks) > 0 else 0
            }
        
        if gpu_temps:
            stats["gpu_temp"] = {
                "min_c": min(gpu_temps),
                "max_c": max(gpu_temps),
                "mean_c": sum(gpu_temps) / len(gpu_temps)
            }
        
        return stats
    
    def get_throttle_events(self) -> List[Dict[str, Any]]:
        """Get list of throttling events."""
        return self._throttle_events.copy()


def capture_clock_state() -> ClockState:
    """Capture current clock state."""
    enforcer = ClockEnforcer()
    return enforcer.capture_state()


def enforce_clock_policy(policy: ClockPolicy) -> Tuple[bool, List[str]]:
    """Enforce a clock policy."""
    enforcer = ClockEnforcer(policy)
    return enforcer.enforce()


@contextmanager
def locked_clocks(policy: Optional[ClockPolicy] = None):
    """
    Context manager for running with locked clocks.
    
    Usage:
        with locked_clocks(policy):
            # Run benchmark with stable clocks
            run_benchmark()
    """
    if policy is None:
        policy = ClockPolicy.stable()
    
    enforcer = ClockEnforcer(policy)
    success, errors = enforcer.enforce()
    
    if not success:
        import warnings
        for error in errors:
            warnings.warn(f"Clock enforcement warning: {error}")
    
    enforcer.start_monitoring()
    
    try:
        yield enforcer
    finally:
        enforcer.stop_monitoring()
        enforcer.restore()
