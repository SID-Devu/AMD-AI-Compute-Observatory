"""
AACO-SIGMA Deterministic Measurement Capsule V2 (DMC++)

A controlled experiment environment, not just "a run."
Provides cgroup v2 isolation, topology pinning, clock enforcement,
and comprehensive noise detection.
"""

import os
import json
import time
import uuid
import hashlib
import subprocess
import threading
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable, Union
from datetime import datetime
from contextlib import contextmanager


class IsolationLevelV2(Enum):
    """Isolation levels for measurement capsules."""
    NONE = auto()           # No isolation, observational only
    SOFT = auto()           # CPU affinity, nice priority
    STANDARD = auto()       # cgroup v2 basic isolation
    STRICT = auto()         # Full cgroup v2 + topology pinning
    PARANOID = auto()       # All of above + clock enforcement + thermal guards


@dataclass
class CapsuleHealthScore:
    """
    Health score for a measurement capsule session.
    Score 0-100 indicates measurement quality.
    """
    overall_score: float  # 0-100
    
    # Component scores
    isolation_score: float  # How well isolation held
    stability_score: float  # Clock/thermal stability
    noise_score: float      # Absence of interference
    repeatability_score: float  # Variance in measurements
    
    # Penalties
    penalties: List[Dict[str, Any]] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Raw metrics
    clock_variance_pct: float = 0.0
    temperature_variance_c: float = 0.0
    noise_events_count: int = 0
    cgroup_violations: int = 0
    
    def is_valid(self, threshold: float = 70.0) -> bool:
        """Check if measurement quality is acceptable."""
        return self.overall_score >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CapsulePolicyV2:
    """
    Policy configuration for measurement capsules.
    Defines isolation, resource limits, and enforcement rules.
    """
    # Isolation level
    isolation_level: IsolationLevelV2 = IsolationLevelV2.STANDARD
    
    # CPU isolation
    cpuset_cores: Optional[List[int]] = None  # Specific cores to use
    cpu_exclusive: bool = False                # Exclusive access to cores
    numa_node: Optional[int] = None            # Preferred NUMA node
    hyperthreading_policy: str = "disable"     # disable, enable, smt_first
    
    # CPU resource limits (cgroup v2)
    cpu_quota_us: Optional[int] = None         # CPU time quota
    cpu_period_us: int = 100000                # CPU period (100ms default)
    cpu_weight: int = 100                      # CPU weight (1-10000)
    
    # Memory limits
    memory_high_bytes: Optional[int] = None    # Memory high watermark
    memory_max_bytes: Optional[int] = None     # Hard memory limit
    swap_max_bytes: int = 0                    # Swap limit (0 = no swap)
    
    # IO limits
    io_max_rbps: Optional[int] = None          # Read bytes/sec limit
    io_max_wbps: Optional[int] = None          # Write bytes/sec limit
    io_max_riops: Optional[int] = None         # Read IOPS limit
    io_max_wiops: Optional[int] = None         # Write IOPS limit
    
    # Clock policy
    enforce_cpu_governor: bool = True          # Enforce CPU governor
    target_cpu_governor: str = "performance"   # Target governor
    enforce_gpu_clocks: bool = False           # Enforce GPU clocks (needs root)
    target_gpu_clock_mhz: Optional[int] = None # Target GPU clock
    
    # Thermal guardrails
    max_gpu_temp_c: float = 85.0               # Max GPU temperature
    max_cpu_temp_c: float = 90.0               # Max CPU temperature
    throttle_threshold_c: float = 80.0         # Warning threshold
    
    # Noise detection
    detect_irq_storms: bool = True
    detect_memory_pressure: bool = True
    detect_io_pressure: bool = True
    detect_scheduler_interference: bool = True
    
    # Thresholds for noise detection
    irq_storm_threshold: int = 10000           # IRQs/sec
    memory_pressure_threshold: float = 0.5     # PSI threshold
    context_switch_threshold: int = 50000      # Switches/sec
    
    # Measurement quality
    min_health_score: float = 70.0             # Minimum acceptable score
    warmup_iterations: int = 5                 # Warmup before measurement
    cooldown_seconds: float = 2.0              # Cooldown between runs
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['isolation_level'] = self.isolation_level.name
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'CapsulePolicyV2':
        d = d.copy()
        if 'isolation_level' in d:
            d['isolation_level'] = IsolationLevelV2[d['isolation_level']]
        return cls(**d)


@dataclass
class CapsuleManifestV2:
    """
    The capsule manifest - a contract describing the measurement environment.
    This is the primary artifact for reproducibility.
    """
    # Identity
    capsule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Policy applied
    policy: Dict[str, Any] = field(default_factory=dict)
    
    # System state at measurement time
    hostname: str = ""
    kernel_version: str = ""
    rocm_version: str = ""
    
    # CPU state
    cpu_model: str = ""
    cpu_cores_total: int = 0
    cpu_cores_used: List[int] = field(default_factory=list)
    cpu_governor_actual: str = ""
    numa_topology: Dict[str, Any] = field(default_factory=dict)
    
    # GPU state
    gpu_model: str = ""
    gpu_vbios: str = ""
    gpu_clock_mhz: int = 0
    gpu_mem_clock_mhz: int = 0
    gpu_power_cap_w: float = 0.0
    gpu_temp_c: float = 0.0
    gpu_memory_total_mb: int = 0
    gpu_memory_used_mb: int = 0
    
    # cgroup state
    cgroup_path: str = ""
    cgroup_controllers: List[str] = field(default_factory=list)
    
    # Environment
    environment_hash: str = ""  # Hash of relevant env vars
    ld_preload: str = ""
    rocr_visible_devices: str = ""
    
    # Workload identity
    workload_signature: str = ""  # Hash of model + inputs + EP
    model_hash: str = ""
    input_signature: str = ""
    execution_provider: str = ""
    
    # Timing
    session_start_ns: int = 0
    session_end_ns: int = 0
    duration_ns: int = 0
    
    def compute_environment_hash(self) -> str:
        """Compute hash of reproducibility-relevant environment."""
        env_keys = [
            'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES',
            'OMP_NUM_THREADS', 'ORT_NUM_THREADS',
            'MIOPEN_DEBUG_DISABLE_FIND_DB',
            'HSA_ENABLE_SDMA', 'GPU_MAX_HW_QUEUES',
        ]
        env_str = "|".join(f"{k}={os.environ.get(k, '')}" for k in sorted(env_keys))
        self.environment_hash = hashlib.sha256(env_str.encode()).hexdigest()[:16]
        return self.environment_hash
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'CapsuleManifestV2':
        """Load manifest from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))


class CgroupV2Controller:
    """
    cgroup v2 controller for process isolation.
    Manages CPU, memory, and IO resource limits.
    """
    
    CGROUP_ROOT = Path("/sys/fs/cgroup")
    
    def __init__(self, name: str, parent: str = "aaco"):
        self.name = name
        self.parent = parent
        self.cgroup_path = self.CGROUP_ROOT / parent / name
        self.active = False
        self._original_cgroup: Optional[str] = None
    
    def create(self) -> bool:
        """Create cgroup hierarchy."""
        try:
            # Create parent if needed
            parent_path = self.CGROUP_ROOT / self.parent
            if not parent_path.exists():
                parent_path.mkdir(parents=True, exist_ok=True)
                # Enable controllers in parent
                self._enable_controllers(parent_path.parent)
            
            # Create our cgroup
            if not self.cgroup_path.exists():
                self.cgroup_path.mkdir(parents=True, exist_ok=True)
            
            # Enable controllers
            self._enable_controllers(parent_path)
            
            return True
        except PermissionError:
            return False
        except Exception:
            return False
    
    def _enable_controllers(self, path: Path) -> None:
        """Enable cgroup v2 controllers."""
        subtree_control = path / "cgroup.subtree_control"
        if subtree_control.exists():
            try:
                with open(subtree_control, 'w') as f:
                    f.write("+cpu +memory +io +cpuset")
            except (PermissionError, OSError):
                pass
    
    def get_available_controllers(self) -> List[str]:
        """Get list of available controllers."""
        controllers_file = self.cgroup_path / "cgroup.controllers"
        if controllers_file.exists():
            return controllers_file.read_text().strip().split()
        return []
    
    def set_cpuset(self, cores: List[int]) -> bool:
        """Set CPU affinity via cpuset."""
        try:
            cpuset_file = self.cgroup_path / "cpuset.cpus"
            if cpuset_file.exists():
                cores_str = ",".join(str(c) for c in cores)
                cpuset_file.write_text(cores_str)
                return True
        except (PermissionError, OSError):
            pass
        return False
    
    def set_cpu_max(self, quota_us: int, period_us: int = 100000) -> bool:
        """Set CPU bandwidth limit."""
        try:
            cpu_max_file = self.cgroup_path / "cpu.max"
            if cpu_max_file.exists():
                cpu_max_file.write_text(f"{quota_us} {period_us}")
                return True
        except (PermissionError, OSError):
            pass
        return False
    
    def set_cpu_weight(self, weight: int) -> bool:
        """Set CPU weight (priority)."""
        try:
            cpu_weight_file = self.cgroup_path / "cpu.weight"
            if cpu_weight_file.exists():
                cpu_weight_file.write_text(str(weight))
                return True
        except (PermissionError, OSError):
            pass
        return False
    
    def set_memory_high(self, bytes_limit: int) -> bool:
        """Set memory high watermark."""
        try:
            mem_high_file = self.cgroup_path / "memory.high"
            if mem_high_file.exists():
                mem_high_file.write_text(str(bytes_limit))
                return True
        except (PermissionError, OSError):
            pass
        return False
    
    def set_memory_max(self, bytes_limit: int) -> bool:
        """Set hard memory limit."""
        try:
            mem_max_file = self.cgroup_path / "memory.max"
            if mem_max_file.exists():
                mem_max_file.write_text(str(bytes_limit))
                return True
        except (PermissionError, OSError):
            pass
        return False
    
    def set_swap_max(self, bytes_limit: int) -> bool:
        """Set swap limit."""
        try:
            swap_max_file = self.cgroup_path / "memory.swap.max"
            if swap_max_file.exists():
                swap_max_file.write_text(str(bytes_limit))
                return True
        except (PermissionError, OSError):
            pass
        return False
    
    def set_io_max(self, device: str, rbps: Optional[int] = None,
                   wbps: Optional[int] = None, riops: Optional[int] = None,
                   wiops: Optional[int] = None) -> bool:
        """Set IO bandwidth limits."""
        try:
            io_max_file = self.cgroup_path / "io.max"
            if io_max_file.exists():
                parts = [device]
                if rbps is not None:
                    parts.append(f"rbps={rbps}")
                if wbps is not None:
                    parts.append(f"wbps={wbps}")
                if riops is not None:
                    parts.append(f"riops={riops}")
                if wiops is not None:
                    parts.append(f"wiops={wiops}")
                io_max_file.write_text(" ".join(parts))
                return True
        except (PermissionError, OSError):
            pass
        return False
    
    def add_process(self, pid: int) -> bool:
        """Add process to this cgroup."""
        try:
            # Save original cgroup
            proc_cgroup = Path(f"/proc/{pid}/cgroup")
            if proc_cgroup.exists():
                self._original_cgroup = proc_cgroup.read_text().strip()
            
            # Move to our cgroup
            procs_file = self.cgroup_path / "cgroup.procs"
            if procs_file.exists():
                procs_file.write_text(str(pid))
                self.active = True
                return True
        except (PermissionError, OSError):
            pass
        return False
    
    def remove_process(self, pid: int) -> bool:
        """Remove process from this cgroup (move to root)."""
        try:
            root_procs = self.CGROUP_ROOT / "cgroup.procs"
            if root_procs.exists():
                root_procs.write_text(str(pid))
                self.active = False
                return True
        except (PermissionError, OSError):
            pass
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cgroup statistics."""
        stats = {}
        
        # CPU stats
        cpu_stat = self.cgroup_path / "cpu.stat"
        if cpu_stat.exists():
            for line in cpu_stat.read_text().strip().split('\n'):
                if ' ' in line:
                    key, val = line.split(' ', 1)
                    stats[f"cpu_{key}"] = int(val)
        
        # Memory stats
        mem_current = self.cgroup_path / "memory.current"
        if mem_current.exists():
            stats["memory_current_bytes"] = int(mem_current.read_text().strip())
        
        # IO stats
        io_stat = self.cgroup_path / "io.stat"
        if io_stat.exists():
            stats["io_stat"] = io_stat.read_text().strip()
        
        return stats
    
    def cleanup(self) -> None:
        """Clean up cgroup."""
        try:
            if self.cgroup_path.exists():
                # Move any remaining processes to parent
                procs_file = self.cgroup_path / "cgroup.procs"
                if procs_file.exists():
                    parent_procs = self.cgroup_path.parent / "cgroup.procs"
                    pids = procs_file.read_text().strip().split('\n')
                    for pid in pids:
                        if pid:
                            try:
                                parent_procs.write_text(pid)
                            except:
                                pass
                
                # Remove cgroup directory
                self.cgroup_path.rmdir()
        except (PermissionError, OSError):
            pass


class MeasurementCapsuleV2:
    """
    AACO-SIGMA Deterministic Measurement Capsule V2.
    
    Provides a controlled, repeatable measurement environment with:
    - cgroup v2 resource isolation
    - CPU topology pinning
    - Clock policy enforcement
    - Thermal guardrails
    - Comprehensive noise detection
    """
    
    def __init__(self, policy: Optional[CapsulePolicyV2] = None):
        self.policy = policy or CapsulePolicyV2()
        self.manifest = CapsuleManifestV2()
        self.cgroup: Optional[CgroupV2Controller] = None
        
        # State tracking
        self._active = False
        self._start_time_ns: int = 0
        self._noise_events: List[Dict[str, Any]] = []
        self._thermal_events: List[Dict[str, Any]] = []
        self._clock_samples: List[Dict[str, Any]] = []
        
        # Background monitors
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Original system state (for restoration)
        self._original_cpu_governor: Dict[int, str] = {}
        self._original_affinity: Optional[List[int]] = None
    
    def __enter__(self) -> 'MeasurementCapsuleV2':
        self.enter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.exit()
    
    def enter(self) -> bool:
        """Enter the measurement capsule."""
        if self._active:
            return True
        
        self._start_time_ns = time.time_ns()
        self.manifest.session_start_ns = self._start_time_ns
        
        # Capture system state
        self._capture_system_state()
        
        # Apply isolation based on level
        success = True
        level = self.policy.isolation_level
        
        if level.value >= IsolationLevelV2.SOFT.value:
            success &= self._apply_soft_isolation()
        
        if level.value >= IsolationLevelV2.STANDARD.value:
            success &= self._apply_cgroup_isolation()
        
        if level.value >= IsolationLevelV2.STRICT.value:
            success &= self._apply_topology_pinning()
        
        if level.value >= IsolationLevelV2.PARANOID.value:
            success &= self._apply_clock_enforcement()
            self._start_thermal_monitoring()
        
        # Start noise monitoring
        self._start_noise_monitoring()
        
        self._active = True
        return success
    
    def exit(self) -> CapsuleHealthScore:
        """Exit the measurement capsule and compute health score."""
        if not self._active:
            return self._compute_health_score()
        
        # Stop monitoring
        self._stop_monitoring.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        
        # Record end time
        self.manifest.session_end_ns = time.time_ns()
        self.manifest.duration_ns = self.manifest.session_end_ns - self.manifest.session_start_ns
        
        # Restore system state
        self._restore_system_state()
        
        # Clean up cgroup
        if self.cgroup:
            self.cgroup.cleanup()
        
        self._active = False
        
        return self._compute_health_score()
    
    def _capture_system_state(self) -> None:
        """Capture current system state for manifest."""
        import platform
        
        self.manifest.hostname = platform.node()
        self.manifest.kernel_version = platform.release()
        
        # CPU info
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
                for line in cpuinfo.split('\n'):
                    if line.startswith("model name"):
                        self.manifest.cpu_model = line.split(':')[1].strip()
                        break
            
            self.manifest.cpu_cores_total = os.cpu_count() or 0
        except:
            pass
        
        # GPU info via rocm-smi
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname", "--showvbios", "--showclocks",
                 "--showtemp", "--showmeminfo", "vram", "--showpower"],
                capture_output=True, text=True, timeout=10
            )
            self._parse_rocm_smi_output(result.stdout)
        except:
            pass
        
        # ROCm version
        try:
            result = subprocess.run(
                ["rocm-smi", "--version"],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split('\n'):
                if 'version' in line.lower():
                    self.manifest.rocm_version = line.strip()
                    break
        except:
            pass
        
        # Compute environment hash
        self.manifest.compute_environment_hash()
        self.manifest.ld_preload = os.environ.get('LD_PRELOAD', '')
        self.manifest.rocr_visible_devices = os.environ.get('ROCR_VISIBLE_DEVICES', '')
    
    def _parse_rocm_smi_output(self, output: str) -> None:
        """Parse rocm-smi output for GPU state."""
        lines = output.split('\n')
        for line in lines:
            line_lower = line.lower()
            if 'card series' in line_lower or 'gpu' in line_lower:
                if ':' in line:
                    self.manifest.gpu_model = line.split(':')[-1].strip()
            elif 'vbios' in line_lower:
                if ':' in line:
                    self.manifest.gpu_vbios = line.split(':')[-1].strip()
            elif 'sclk' in line_lower or 'gpu clock' in line_lower:
                # Extract GPU clock
                import re
                match = re.search(r'(\d+)\s*mhz', line_lower)
                if match:
                    self.manifest.gpu_clock_mhz = int(match.group(1))
            elif 'mclk' in line_lower or 'mem clock' in line_lower:
                import re
                match = re.search(r'(\d+)\s*mhz', line_lower)
                if match:
                    self.manifest.gpu_mem_clock_mhz = int(match.group(1))
            elif 'temperature' in line_lower or 'temp' in line_lower:
                import re
                match = re.search(r'(\d+\.?\d*)\s*c', line_lower)
                if match:
                    self.manifest.gpu_temp_c = float(match.group(1))
            elif 'power cap' in line_lower:
                import re
                match = re.search(r'(\d+\.?\d*)\s*w', line_lower)
                if match:
                    self.manifest.gpu_power_cap_w = float(match.group(1))
    
    def _apply_soft_isolation(self) -> bool:
        """Apply soft isolation (affinity, nice)."""
        try:
            pid = os.getpid()
            
            # Save original affinity
            self._original_affinity = os.sched_getaffinity(pid)
            
            # Set CPU affinity if specified
            if self.policy.cpuset_cores:
                os.sched_setaffinity(pid, set(self.policy.cpuset_cores))
                self.manifest.cpu_cores_used = self.policy.cpuset_cores
            
            # Set nice priority
            os.nice(-5)  # Higher priority
            
            return True
        except (PermissionError, OSError):
            return False
    
    def _apply_cgroup_isolation(self) -> bool:
        """Apply cgroup v2 isolation."""
        capsule_name = f"capsule_{self.manifest.capsule_id[:8]}"
        self.cgroup = CgroupV2Controller(capsule_name)
        
        if not self.cgroup.create():
            return False
        
        self.manifest.cgroup_path = str(self.cgroup.cgroup_path)
        self.manifest.cgroup_controllers = self.cgroup.get_available_controllers()
        
        # Apply resource limits
        policy = self.policy
        
        if policy.cpuset_cores:
            self.cgroup.set_cpuset(policy.cpuset_cores)
        
        if policy.cpu_quota_us:
            self.cgroup.set_cpu_max(policy.cpu_quota_us, policy.cpu_period_us)
        
        self.cgroup.set_cpu_weight(policy.cpu_weight)
        
        if policy.memory_high_bytes:
            self.cgroup.set_memory_high(policy.memory_high_bytes)
        
        if policy.memory_max_bytes:
            self.cgroup.set_memory_max(policy.memory_max_bytes)
        
        self.cgroup.set_swap_max(policy.swap_max_bytes)
        
        # Add current process to cgroup
        return self.cgroup.add_process(os.getpid())
    
    def _apply_topology_pinning(self) -> bool:
        """Apply strict topology pinning."""
        # This would involve more advanced NUMA-aware pinning
        # For now, rely on cpuset
        return True
    
    def _apply_clock_enforcement(self) -> bool:
        """Apply clock policy enforcement."""
        if not self.policy.enforce_cpu_governor:
            return True
        
        try:
            # Get current governors and set to performance
            cpu_count = os.cpu_count() or 1
            for cpu in range(cpu_count):
                gov_path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
                if gov_path.exists():
                    self._original_cpu_governor[cpu] = gov_path.read_text().strip()
                    try:
                        gov_path.write_text(self.policy.target_cpu_governor)
                    except PermissionError:
                        pass
            
            # Record actual governor
            if 0 in self._original_cpu_governor:
                self.manifest.cpu_governor_actual = self._original_cpu_governor[0]
            
            return True
        except Exception:
            return False
    
    def _start_noise_monitoring(self) -> None:
        """Start background noise monitoring thread."""
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(target=self._noise_monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def _start_thermal_monitoring(self) -> None:
        """Thermal monitoring is part of the main monitor loop."""
        pass
    
    def _noise_monitor_loop(self) -> None:
        """Background loop for noise detection."""
        interval = 0.1  # 100ms
        
        prev_ctx_switches = self._get_context_switches()
        prev_irq_count = self._get_irq_count()
        
        while not self._stop_monitoring.is_set():
            try:
                # Check context switches
                ctx_switches = self._get_context_switches()
                if prev_ctx_switches and ctx_switches:
                    delta = ctx_switches - prev_ctx_switches
                    rate = delta / interval
                    if rate > self.policy.context_switch_threshold:
                        self._noise_events.append({
                            "type": "context_switch_storm",
                            "timestamp_ns": time.time_ns(),
                            "rate": rate,
                            "threshold": self.policy.context_switch_threshold
                        })
                prev_ctx_switches = ctx_switches
                
                # Check IRQ storms
                irq_count = self._get_irq_count()
                if prev_irq_count and irq_count:
                    delta = irq_count - prev_irq_count
                    rate = delta / interval
                    if rate > self.policy.irq_storm_threshold:
                        self._noise_events.append({
                            "type": "irq_storm",
                            "timestamp_ns": time.time_ns(),
                            "rate": rate,
                            "threshold": self.policy.irq_storm_threshold
                        })
                prev_irq_count = irq_count
                
                # Check memory pressure
                mem_pressure = self._get_memory_pressure()
                if mem_pressure > self.policy.memory_pressure_threshold:
                    self._noise_events.append({
                        "type": "memory_pressure",
                        "timestamp_ns": time.time_ns(),
                        "pressure": mem_pressure,
                        "threshold": self.policy.memory_pressure_threshold
                    })
                
                # Check thermal (if paranoid)
                if self.policy.isolation_level == IsolationLevelV2.PARANOID:
                    self._check_thermal_state()
                
            except Exception:
                pass
            
            self._stop_monitoring.wait(interval)
    
    def _get_context_switches(self) -> Optional[int]:
        """Get total context switches from /proc/stat."""
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if line.startswith("ctxt "):
                        return int(line.split()[1])
        except:
            pass
        return None
    
    def _get_irq_count(self) -> Optional[int]:
        """Get total IRQ count from /proc/stat."""
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if line.startswith("intr "):
                        return int(line.split()[1])
        except:
            pass
        return None
    
    def _get_memory_pressure(self) -> float:
        """Get memory pressure from PSI."""
        try:
            psi_path = Path("/proc/pressure/memory")
            if psi_path.exists():
                content = psi_path.read_text()
                for line in content.split('\n'):
                    if line.startswith('some'):
                        # Parse: some avg10=0.00 avg60=0.00 avg300=0.00 total=0
                        import re
                        match = re.search(r'avg10=(\d+\.?\d*)', line)
                        if match:
                            return float(match.group(1)) / 100.0
        except:
            pass
        return 0.0
    
    def _check_thermal_state(self) -> None:
        """Check GPU/CPU thermal state."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showtemp"],
                capture_output=True, text=True, timeout=5
            )
            import re
            match = re.search(r'(\d+\.?\d*)\s*c', result.stdout.lower())
            if match:
                temp = float(match.group(1))
                self._clock_samples.append({
                    "timestamp_ns": time.time_ns(),
                    "gpu_temp_c": temp
                })
                
                if temp > self.policy.max_gpu_temp_c:
                    self._thermal_events.append({
                        "type": "gpu_over_temp",
                        "timestamp_ns": time.time_ns(),
                        "temp_c": temp,
                        "threshold": self.policy.max_gpu_temp_c
                    })
                elif temp > self.policy.throttle_threshold_c:
                    self._thermal_events.append({
                        "type": "gpu_throttle_warning",
                        "timestamp_ns": time.time_ns(),
                        "temp_c": temp,
                        "threshold": self.policy.throttle_threshold_c
                    })
        except:
            pass
    
    def _restore_system_state(self) -> None:
        """Restore original system state."""
        # Restore CPU governors
        for cpu, gov in self._original_cpu_governor.items():
            try:
                gov_path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
                if gov_path.exists():
                    gov_path.write_text(gov)
            except:
                pass
        
        # Restore CPU affinity
        if self._original_affinity:
            try:
                os.sched_setaffinity(os.getpid(), self._original_affinity)
            except:
                pass
    
    def _compute_health_score(self) -> CapsuleHealthScore:
        """Compute overall health score for the measurement session."""
        penalties = []
        recommendations = []
        
        # Base scores (start at 100, deduct for issues)
        isolation_score = 100.0
        stability_score = 100.0
        noise_score = 100.0
        repeatability_score = 100.0  # Requires external input
        
        # Penalize for noise events
        noise_count = len(self._noise_events)
        if noise_count > 0:
            noise_penalty = min(30, noise_count * 5)
            noise_score -= noise_penalty
            penalties.append({
                "type": "noise_events",
                "count": noise_count,
                "penalty": noise_penalty
            })
            recommendations.append("Consider isolating cores or reducing system load")
        
        # Penalize for thermal events
        thermal_count = len(self._thermal_events)
        if thermal_count > 0:
            thermal_penalty = min(25, thermal_count * 10)
            stability_score -= thermal_penalty
            penalties.append({
                "type": "thermal_events",
                "count": thermal_count,
                "penalty": thermal_penalty
            })
            recommendations.append("Allow GPU to cool or reduce power cap")
        
        # Check cgroup effectiveness
        if self.cgroup and not self.cgroup.active:
            isolation_score -= 20
            penalties.append({
                "type": "cgroup_inactive",
                "penalty": 20
            })
            recommendations.append("Run with elevated privileges for cgroup isolation")
        
        # Check clock variance
        if self._clock_samples:
            temps = [s.get('gpu_temp_c', 0) for s in self._clock_samples if s.get('gpu_temp_c')]
            if temps:
                temp_variance = max(temps) - min(temps)
                if temp_variance > 10:
                    stability_score -= min(20, temp_variance)
                    penalties.append({
                        "type": "temperature_variance",
                        "variance_c": temp_variance,
                        "penalty": min(20, temp_variance)
                    })
        
        # Compute overall score (weighted average)
        overall = (
            isolation_score * 0.25 +
            stability_score * 0.30 +
            noise_score * 0.30 +
            repeatability_score * 0.15
        )
        
        return CapsuleHealthScore(
            overall_score=max(0, overall),
            isolation_score=max(0, isolation_score),
            stability_score=max(0, stability_score),
            noise_score=max(0, noise_score),
            repeatability_score=max(0, repeatability_score),
            penalties=penalties,
            recommendations=recommendations,
            noise_events_count=noise_count,
            temperature_variance_c=max(temps) - min(temps) if temps else 0.0
        )
    
    def get_noise_report(self) -> Dict[str, Any]:
        """Get comprehensive noise report."""
        return {
            "capsule_id": self.manifest.capsule_id,
            "duration_ns": self.manifest.duration_ns,
            "noise_events": self._noise_events,
            "thermal_events": self._thermal_events,
            "total_noise_events": len(self._noise_events),
            "total_thermal_events": len(self._thermal_events),
            "event_types": list(set(e["type"] for e in self._noise_events)),
        }
    
    def set_workload_signature(self, model_hash: str, input_signature: str,
                               execution_provider: str) -> None:
        """Set workload signature for reproducibility tracking."""
        self.manifest.model_hash = model_hash
        self.manifest.input_signature = input_signature
        self.manifest.execution_provider = execution_provider
        
        sig_str = f"{model_hash}|{input_signature}|{execution_provider}"
        self.manifest.workload_signature = hashlib.sha256(sig_str.encode()).hexdigest()[:16]
    
    def save_artifacts(self, output_dir: Path) -> Dict[str, Path]:
        """Save all capsule artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        artifacts = {}
        
        # Save manifest
        manifest_path = output_dir / "capsule_manifest.json"
        self.manifest.policy = self.policy.to_dict()
        self.manifest.save(manifest_path)
        artifacts["manifest"] = manifest_path
        
        # Save noise report
        noise_path = output_dir / "noise_report.json"
        with open(noise_path, 'w') as f:
            json.dump(self.get_noise_report(), f, indent=2)
        artifacts["noise_report"] = noise_path
        
        # Save health score
        health_score = self._compute_health_score()
        health_path = output_dir / "capsule_health.json"
        with open(health_path, 'w') as f:
            json.dump(health_score.to_dict(), f, indent=2)
        artifacts["health_score"] = health_path
        
        return artifacts


@contextmanager
def run_in_capsule(policy: Optional[CapsulePolicyV2] = None):
    """
    Context manager for running code inside a measurement capsule.
    
    Usage:
        with run_in_capsule(policy) as capsule:
            # Your benchmark code here
            results = run_benchmark()
        
        # Access health score after exit
        health = capsule.exit()  # Called automatically
    """
    capsule = MeasurementCapsuleV2(policy)
    try:
        capsule.enter()
        yield capsule
    finally:
        capsule.exit()
