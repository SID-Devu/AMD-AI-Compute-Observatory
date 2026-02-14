"""
AACO-SIGMA Thermal and Power Guardrails

Monitors thermal and power stability during measurements,
detecting throttling and flagging runs with stability violations.
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
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime


class ThrottleType(Enum):
    """Types of throttling detected."""
    THERMAL_GPU = auto()       # GPU thermal throttling
    THERMAL_CPU = auto()       # CPU thermal throttling
    POWER_GPU = auto()         # GPU power throttling
    POWER_CPU = auto()         # CPU power throttling
    CURRENT_GPU = auto()       # GPU current limit
    RELIABILITY = auto()       # Reliability voltage limit
    UNKNOWN = auto()


class ViolationSeverity(Enum):
    """Severity of guardrail violations."""
    WARNING = auto()      # Approaching limit
    VIOLATION = auto()    # Limit exceeded briefly
    CRITICAL = auto()     # Sustained limit breach


@dataclass
class ThrottleEvent:
    """A detected throttling event."""
    timestamp_ns: int
    throttle_type: ThrottleType
    severity: ViolationSeverity
    
    # Trigger values
    trigger_value: float = 0.0
    threshold: float = 0.0
    
    # Duration
    duration_ns: int = 0
    
    # Context
    clock_before_mhz: int = 0
    clock_after_mhz: int = 0
    temp_c: float = 0.0
    power_w: float = 0.0
    
    # Description
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['throttle_type'] = self.throttle_type.name
        d['severity'] = self.severity.name
        return d


@dataclass
class GuardrailViolation:
    """A guardrail violation event."""
    timestamp_ns: int
    guardrail: str
    severity: ViolationSeverity
    
    measured_value: float
    limit_value: float
    
    duration_ns: int = 0
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['severity'] = self.severity.name
        return d


@dataclass
class GuardrailConfig:
    """Configuration for thermal/power guardrails."""
    # GPU thermal limits
    gpu_temp_warning_c: float = 75.0
    gpu_temp_violation_c: float = 85.0
    gpu_temp_critical_c: float = 95.0
    
    # CPU thermal limits
    cpu_temp_warning_c: float = 80.0
    cpu_temp_violation_c: float = 90.0
    cpu_temp_critical_c: float = 100.0
    
    # GPU power limits (percentage of cap)
    gpu_power_warning_pct: float = 90.0
    gpu_power_violation_pct: float = 100.0
    
    # Clock stability (variance threshold)
    clock_variance_warning_pct: float = 3.0
    clock_variance_violation_pct: float = 5.0
    
    # Monitoring
    sample_interval_ms: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StabilityReport:
    """
    Comprehensive stability report for a measurement session.
    """
    session_id: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_ns: int = 0
    
    # Overall stability
    is_stable: bool = True
    stability_score: float = 100.0  # 0-100
    
    # Statistics
    sample_count: int = 0
    
    # GPU stats
    gpu_temp_min_c: float = 0.0
    gpu_temp_max_c: float = 0.0
    gpu_temp_mean_c: float = 0.0
    gpu_clock_min_mhz: int = 0
    gpu_clock_max_mhz: int = 0
    gpu_clock_mean_mhz: float = 0.0
    gpu_power_min_w: float = 0.0
    gpu_power_max_w: float = 0.0
    gpu_power_mean_w: float = 0.0
    
    # CPU stats
    cpu_temp_max_c: float = 0.0
    
    # Violations
    throttle_events: List[Dict[str, Any]] = field(default_factory=list)
    guardrail_violations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Save report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class ThermalGuard:
    """
    GPU thermal monitoring and guardrails.
    
    Detects thermal throttling and flags runs with temperature instability.
    """
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        
        self._samples: List[Dict[str, Any]] = []
        self._throttle_events: List[ThrottleEvent] = []
        self._violations: List[GuardrailViolation] = []
        
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        self._start_time_ns: int = 0
        self._in_throttle: bool = False
        self._throttle_start_ns: int = 0
    
    def start(self) -> None:
        """Start thermal monitoring."""
        self._start_time_ns = time.time_ns()
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop(self) -> StabilityReport:
        """Stop monitoring and generate report."""
        self._stop_monitoring.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        
        return self._generate_report()
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        interval = self.config.sample_interval_ms / 1000.0
        
        while not self._stop_monitoring.is_set():
            sample = self._collect_sample()
            if sample:
                self._samples.append(sample)
                self._check_guardrails(sample)
            
            self._stop_monitoring.wait(interval)
    
    def _collect_sample(self) -> Optional[Dict[str, Any]]:
        """Collect a thermal/power sample."""
        sample = {
            "timestamp_ns": time.time_ns()
        }
        
        # Get GPU data via rocm-smi
        try:
            # Temperature
            result = subprocess.run(
                ["rocm-smi", "--showtemp"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                match = re.search(r'(\d+\.?\d*)\s*c', result.stdout.lower())
                if match:
                    sample["gpu_temp_c"] = float(match.group(1))
            
            # Clock
            result = subprocess.run(
                ["rocm-smi", "--showclocks"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'sclk' in line.lower() or ('gpu' in line.lower() and 'mhz' in line.lower()):
                        match = re.search(r'(\d+)\s*mhz', line.lower())
                        if match:
                            sample["gpu_clock_mhz"] = int(match.group(1))
                            break
            
            # Power
            result = subprocess.run(
                ["rocm-smi", "--showpower"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    line_lower = line.lower()
                    if 'power' in line_lower:
                        match = re.search(r'(\d+\.?\d*)\s*w', line_lower)
                        if match:
                            if 'cap' not in line_lower:
                                sample["gpu_power_w"] = float(match.group(1))
                            else:
                                sample["gpu_power_cap_w"] = float(match.group(1))
            
            # Check throttle reasons (AMD specific)
            result = subprocess.run(
                ["rocm-smi", "--showxgmierr", "--showvoltage"],
                capture_output=True, text=True, timeout=2
            )
            # Parse throttle indicators if available
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Get CPU temperature
        try:
            hwmon_base = Path("/sys/class/hwmon")
            for hwmon_dir in hwmon_base.glob("hwmon*"):
                name_file = hwmon_dir / "name"
                if name_file.exists():
                    name = name_file.read_text().strip()
                    if name in ["coretemp", "k10temp", "zenpower"]:
                        # Find temp1_input (package temp)
                        temp_file = hwmon_dir / "temp1_input"
                        if temp_file.exists():
                            temp_mc = int(temp_file.read_text().strip())
                            sample["cpu_temp_c"] = temp_mc / 1000.0
                            break
        except:
            pass
        
        return sample if len(sample) > 1 else None
    
    def _check_guardrails(self, sample: Dict[str, Any]) -> None:
        """Check sample against guardrails."""
        config = self.config
        timestamp = sample["timestamp_ns"]
        
        # Check GPU temperature
        if "gpu_temp_c" in sample:
            temp = sample["gpu_temp_c"]
            
            if temp >= config.gpu_temp_critical_c:
                self._add_violation(
                    timestamp, "gpu_temp", ViolationSeverity.CRITICAL,
                    temp, config.gpu_temp_critical_c,
                    f"GPU temperature critical: {temp:.1f}°C"
                )
                self._detect_thermal_throttle(sample, ViolationSeverity.CRITICAL)
            
            elif temp >= config.gpu_temp_violation_c:
                self._add_violation(
                    timestamp, "gpu_temp", ViolationSeverity.VIOLATION,
                    temp, config.gpu_temp_violation_c,
                    f"GPU temperature exceeded limit: {temp:.1f}°C"
                )
                self._detect_thermal_throttle(sample, ViolationSeverity.VIOLATION)
            
            elif temp >= config.gpu_temp_warning_c:
                self._add_violation(
                    timestamp, "gpu_temp", ViolationSeverity.WARNING,
                    temp, config.gpu_temp_warning_c,
                    f"GPU temperature warning: {temp:.1f}°C"
                )
        
        # Check CPU temperature
        if "cpu_temp_c" in sample:
            temp = sample["cpu_temp_c"]
            
            if temp >= config.cpu_temp_critical_c:
                self._add_violation(
                    timestamp, "cpu_temp", ViolationSeverity.CRITICAL,
                    temp, config.cpu_temp_critical_c,
                    f"CPU temperature critical: {temp:.1f}°C"
                )
            elif temp >= config.cpu_temp_violation_c:
                self._add_violation(
                    timestamp, "cpu_temp", ViolationSeverity.VIOLATION,
                    temp, config.cpu_temp_violation_c,
                    f"CPU temperature exceeded limit: {temp:.1f}°C"
                )
        
        # Check GPU power
        if "gpu_power_w" in sample and "gpu_power_cap_w" in sample:
            power_pct = sample["gpu_power_w"] / sample["gpu_power_cap_w"] * 100
            
            if power_pct >= config.gpu_power_violation_pct:
                self._add_violation(
                    timestamp, "gpu_power", ViolationSeverity.VIOLATION,
                    power_pct, config.gpu_power_violation_pct,
                    f"GPU power at limit: {power_pct:.1f}%"
                )
            elif power_pct >= config.gpu_power_warning_pct:
                self._add_violation(
                    timestamp, "gpu_power", ViolationSeverity.WARNING,
                    power_pct, config.gpu_power_warning_pct,
                    f"GPU power high: {power_pct:.1f}%"
                )
        
        # Check clock variance
        if len(self._samples) >= 2 and "gpu_clock_mhz" in sample:
            first_clock = self._samples[0].get("gpu_clock_mhz", 0)
            if first_clock > 0:
                variance_pct = abs(sample["gpu_clock_mhz"] - first_clock) / first_clock * 100
                
                if variance_pct >= config.clock_variance_violation_pct:
                    self._add_violation(
                        timestamp, "clock_variance", ViolationSeverity.VIOLATION,
                        variance_pct, config.clock_variance_violation_pct,
                        f"GPU clock variance: {variance_pct:.1f}%"
                    )
    
    def _add_violation(self, timestamp: int, guardrail: str,
                       severity: ViolationSeverity, measured: float,
                       limit: float, description: str) -> None:
        """Add a guardrail violation."""
        self._violations.append(GuardrailViolation(
            timestamp_ns=timestamp,
            guardrail=guardrail,
            severity=severity,
            measured_value=measured,
            limit_value=limit,
            description=description
        ))
    
    def _detect_thermal_throttle(self, sample: Dict[str, Any],
                                  severity: ViolationSeverity) -> None:
        """Detect and track thermal throttling."""
        if not self._in_throttle:
            self._in_throttle = True
            self._throttle_start_ns = sample["timestamp_ns"]
            self._throttle_start_clock = sample.get("gpu_clock_mhz", 0)
        
        # If clocks dropped, record throttle event
        current_clock = sample.get("gpu_clock_mhz", 0)
        if self._throttle_start_clock > 0 and current_clock > 0:
            if current_clock < self._throttle_start_clock:
                event = ThrottleEvent(
                    timestamp_ns=sample["timestamp_ns"],
                    throttle_type=ThrottleType.THERMAL_GPU,
                    severity=severity,
                    trigger_value=sample.get("gpu_temp_c", 0),
                    threshold=self.config.gpu_temp_violation_c,
                    clock_before_mhz=self._throttle_start_clock,
                    clock_after_mhz=current_clock,
                    temp_c=sample.get("gpu_temp_c", 0),
                    power_w=sample.get("gpu_power_w", 0),
                    description=f"Thermal throttle: {self._throttle_start_clock}→{current_clock} MHz"
                )
                self._throttle_events.append(event)
    
    def _generate_report(self) -> StabilityReport:
        """Generate comprehensive stability report."""
        report = StabilityReport(
            start_time=datetime.fromtimestamp(self._start_time_ns / 1e9).isoformat(),
            end_time=datetime.now().isoformat(),
            sample_count=len(self._samples)
        )
        
        if self._samples:
            report.duration_ns = self._samples[-1]["timestamp_ns"] - self._samples[0]["timestamp_ns"]
            
            # Calculate GPU stats
            gpu_temps = [s["gpu_temp_c"] for s in self._samples if "gpu_temp_c" in s]
            if gpu_temps:
                report.gpu_temp_min_c = min(gpu_temps)
                report.gpu_temp_max_c = max(gpu_temps)
                report.gpu_temp_mean_c = sum(gpu_temps) / len(gpu_temps)
            
            gpu_clocks = [s["gpu_clock_mhz"] for s in self._samples if "gpu_clock_mhz" in s]
            if gpu_clocks:
                report.gpu_clock_min_mhz = min(gpu_clocks)
                report.gpu_clock_max_mhz = max(gpu_clocks)
                report.gpu_clock_mean_mhz = sum(gpu_clocks) / len(gpu_clocks)
            
            gpu_powers = [s["gpu_power_w"] for s in self._samples if "gpu_power_w" in s]
            if gpu_powers:
                report.gpu_power_min_w = min(gpu_powers)
                report.gpu_power_max_w = max(gpu_powers)
                report.gpu_power_mean_w = sum(gpu_powers) / len(gpu_powers)
            
            cpu_temps = [s["cpu_temp_c"] for s in self._samples if "cpu_temp_c" in s]
            if cpu_temps:
                report.cpu_temp_max_c = max(cpu_temps)
        
        # Add events
        report.throttle_events = [e.to_dict() for e in self._throttle_events]
        report.guardrail_violations = [v.to_dict() for v in self._violations]
        
        # Calculate stability score
        score = 100.0
        
        # Penalize for violations
        critical_count = sum(1 for v in self._violations if v.severity == ViolationSeverity.CRITICAL)
        violation_count = sum(1 for v in self._violations if v.severity == ViolationSeverity.VIOLATION)
        warning_count = sum(1 for v in self._violations if v.severity == ViolationSeverity.WARNING)
        
        score -= critical_count * 20
        score -= violation_count * 10
        score -= warning_count * 2
        score -= len(self._throttle_events) * 15
        
        report.stability_score = max(0, score)
        report.is_stable = score >= 70
        
        # Add recommendations
        if report.gpu_temp_max_c > self.config.gpu_temp_warning_c:
            report.recommendations.append("Consider improving GPU cooling or reducing workload intensity")
        
        if report.gpu_clock_max_mhz > 0 and report.gpu_clock_min_mhz > 0:
            variance = (report.gpu_clock_max_mhz - report.gpu_clock_min_mhz) / report.gpu_clock_max_mhz * 100
            if variance > 5:
                report.recommendations.append("Lock GPU clocks for more stable measurements")
        
        if self._throttle_events:
            report.recommendations.append("Thermal throttling detected - allow GPU to cool before next run")
        
        return report


class PowerGuard:
    """
    GPU power monitoring and guardrails.
    
    Detects power throttling and monitors power stability.
    """
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        self._samples: List[Dict[str, Any]] = []
        self._throttle_events: List[ThrottleEvent] = []
    
    def add_sample(self, power_w: float, power_cap_w: float,
                   clock_mhz: int, temp_c: float) -> Optional[ThrottleEvent]:
        """Add a power sample and check for throttling."""
        timestamp = time.time_ns()
        
        sample = {
            "timestamp_ns": timestamp,
            "power_w": power_w,
            "power_cap_w": power_cap_w,
            "clock_mhz": clock_mhz,
            "temp_c": temp_c
        }
        self._samples.append(sample)
        
        # Check for power throttling
        if power_cap_w > 0:
            power_pct = power_w / power_cap_w * 100
            
            if power_pct >= 99.0:  # At power limit
                # Check if clocks dropped
                if self._samples and len(self._samples) >= 2:
                    prev_clock = self._samples[-2].get("clock_mhz", 0)
                    if prev_clock > 0 and clock_mhz < prev_clock:
                        event = ThrottleEvent(
                            timestamp_ns=timestamp,
                            throttle_type=ThrottleType.POWER_GPU,
                            severity=ViolationSeverity.VIOLATION,
                            trigger_value=power_w,
                            threshold=power_cap_w,
                            clock_before_mhz=prev_clock,
                            clock_after_mhz=clock_mhz,
                            temp_c=temp_c,
                            power_w=power_w,
                            description=f"Power throttle at {power_w:.1f}W (cap: {power_cap_w:.1f}W)"
                        )
                        self._throttle_events.append(event)
                        return event
        
        return None
    
    def get_throttle_events(self) -> List[ThrottleEvent]:
        """Get all detected throttle events."""
        return self._throttle_events.copy()


def create_guardrail_config(
    gpu_temp_limit_c: float = 85.0,
    power_cap_headroom_pct: float = 10.0,
    clock_variance_pct: float = 5.0
) -> GuardrailConfig:
    """Create a guardrail configuration."""
    return GuardrailConfig(
        gpu_temp_violation_c=gpu_temp_limit_c,
        gpu_temp_warning_c=gpu_temp_limit_c - 10,
        gpu_temp_critical_c=gpu_temp_limit_c + 10,
        gpu_power_warning_pct=100 - power_cap_headroom_pct,
        clock_variance_violation_pct=clock_variance_pct,
        clock_variance_warning_pct=clock_variance_pct / 2
    )
