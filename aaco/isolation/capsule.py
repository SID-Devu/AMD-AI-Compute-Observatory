"""
Deterministic Measurement Capsule (DMC)
Configures the machine into a controlled "measurement contract" state.
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Isolation Levels
# ============================================================================


class IsolationLevel(str, Enum):
    """Isolation strictness levels."""

    NONE = "none"  # No isolation, just record state
    BASIC = "basic"  # CPU affinity + governor recording
    STANDARD = "standard"  # cgroups + pinning + governor control
    STRICT = "strict"  # Full isolation: cpuset + memory limits + IO throttle
    PARANOID = "paranoid"  # Maximum: isolcpus-style + nohz hints


class GovernorPolicy(str, Enum):
    """CPU frequency governor policies."""

    PERFORMANCE = "performance"
    POWERSAVE = "powersave"
    ONDEMAND = "ondemand"
    CONSERVATIVE = "conservative"
    SCHEDUTIL = "schedutil"
    RECORD_ONLY = "record_only"  # Don't change, just record


class GPUClockPolicy(str, Enum):
    """GPU clock management policy."""

    RECORD_ONLY = "record_only"
    HIGH = "high"  # rocm-smi --setperflevel high
    LOW = "low"
    AUTO = "auto"
    MANUAL = "manual"  # Lock to specific values


# ============================================================================
# Capsule Policy
# ============================================================================


@dataclass
class CapsulePolicy:
    """
    Policy configuration for measurement capsule.

    Defines what isolation measures to apply.
    """

    # Overall level
    isolation_level: IsolationLevel = IsolationLevel.STANDARD

    # CPU settings
    cpu_cores: Optional[List[int]] = None  # None = auto-detect available
    cpu_governor: GovernorPolicy = GovernorPolicy.PERFORMANCE
    thread_affinity: bool = True
    numa_aware: bool = True

    # Memory settings
    memory_limit_mb: Optional[int] = None  # cgroup memory.high
    pin_memory: bool = True  # mlock hints

    # GPU settings
    gpu_clock_policy: GPUClockPolicy = GPUClockPolicy.HIGH
    gpu_power_limit_w: Optional[int] = None

    # cgroup settings
    cgroup_name: str = "aaco_capsule"
    io_throttle: bool = False
    io_max_mbps: Optional[int] = None

    # Process settings
    nice_value: int = -10  # Higher priority (requires privileges)
    use_realtime: bool = False  # SCHED_FIFO (dangerous, optional)

    # Noise control
    disable_swap: bool = False
    drop_caches: bool = True
    sync_before_run: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["isolation_level"] = self.isolation_level.value
        d["cpu_governor"] = self.cpu_governor.value
        d["gpu_clock_policy"] = self.gpu_clock_policy.value
        return d


# ============================================================================
# Capsule Manifest
# ============================================================================


@dataclass
class CapsuleManifest:
    """
    Record of the measurement environment state.

    This is the "measurement contract" - what was controlled and what was observed.
    """

    # Identification
    capsule_id: str = ""
    timestamp_utc: float = 0.0

    # Applied policy
    policy: Optional[CapsulePolicy] = None

    # CPU state (recorded)
    cpu_cores_used: List[int] = field(default_factory=list)
    cpu_governors_before: Dict[int, str] = field(default_factory=dict)
    cpu_governors_after: Dict[int, str] = field(default_factory=dict)
    cpu_frequencies_mhz: Dict[int, float] = field(default_factory=dict)
    numa_topology: Dict[str, Any] = field(default_factory=dict)

    # GPU state (recorded)
    gpu_clocks_before: Dict[str, int] = field(default_factory=dict)
    gpu_clocks_after: Dict[str, int] = field(default_factory=dict)
    gpu_power_limit_w: Optional[int] = None
    gpu_perf_level: str = ""

    # cgroup state
    cgroup_path: str = ""
    cgroup_cpu_quota: Optional[int] = None
    cgroup_memory_high: Optional[int] = None

    # System state
    kernel_version: str = ""
    hostname: str = ""
    total_memory_mb: int = 0
    swap_enabled: bool = True
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Isolation verification
    isolation_verified: bool = False
    isolation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.policy:
            d["policy"] = self.policy.to_dict()
        return d

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# Measurement Capsule
# ============================================================================


class MeasurementCapsule:
    """
    Deterministic Measurement Capsule (DMC).

    Creates a controlled environment for reproducible performance measurements.

    Features:
    - cgroups v2 isolation (cpuset, memory, io)
    - CPU governor control
    - GPU clock policy management
    - Thread affinity control
    - Noise detection and reporting

    Usage:
        policy = CapsulePolicy(isolation_level=IsolationLevel.STANDARD)
        capsule = MeasurementCapsule(policy)

        with capsule:
            # Run benchmark in isolated environment
            results = run_benchmark()

        # Capsule automatically cleaned up
        manifest = capsule.get_manifest()
    """

    def __init__(self, policy: Optional[CapsulePolicy] = None):
        self.policy = policy or CapsulePolicy()
        self._manifest: Optional[CapsuleManifest] = None
        self._original_state: Dict[str, Any] = {}
        self._cgroup_created = False
        self._entered = False

    def __enter__(self) -> "MeasurementCapsule":
        self.enter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.exit()

    def enter(self) -> CapsuleManifest:
        """Enter the measurement capsule, applying isolation."""
        if self._entered:
            raise RuntimeError("Capsule already entered")

        self._entered = True
        self._manifest = CapsuleManifest(
            capsule_id=f"capsule_{int(time.time())}_{os.getpid()}",
            timestamp_utc=time.time(),
            policy=self.policy,
        )

        logger.info(f"Entering measurement capsule: {self._manifest.capsule_id}")
        logger.info(f"Isolation level: {self.policy.isolation_level.value}")

        # Record initial state
        self._record_system_state()

        # Apply isolation based on level
        if self.policy.isolation_level != IsolationLevel.NONE:
            self._apply_isolation()

        # Verify isolation
        self._verify_isolation()

        return self._manifest

    def exit(self) -> None:
        """Exit the capsule, restoring original state."""
        if not self._entered:
            return

        logger.info(f"Exiting measurement capsule: {self._manifest.capsule_id}")

        # Restore original state
        if self.policy.isolation_level != IsolationLevel.NONE:
            self._restore_state()

        self._entered = False

    def get_manifest(self) -> Optional[CapsuleManifest]:
        """Get the capsule manifest."""
        return self._manifest

    # ==========================================================================
    # State Recording
    # ==========================================================================

    def _record_system_state(self) -> None:
        """Record initial system state."""
        manifest = self._manifest

        # Kernel version
        try:
            manifest.kernel_version = self._run_cmd("uname -r").strip()
        except:
            manifest.kernel_version = "unknown"

        # Hostname
        manifest.hostname = os.uname().nodename if hasattr(os, "uname") else "unknown"

        # Memory
        manifest.total_memory_mb = self._get_total_memory_mb()
        manifest.swap_enabled = self._is_swap_enabled()

        # Load average
        try:
            with open("/proc/loadavg") as f:
                parts = f.read().split()
                manifest.load_average = (
                    float(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                )
        except:
            pass

        # CPU governors
        manifest.cpu_governors_before = self._get_cpu_governors()
        manifest.cpu_frequencies_mhz = self._get_cpu_frequencies()

        # Detect available CPU cores
        if self.policy.cpu_cores:
            manifest.cpu_cores_used = self.policy.cpu_cores
        else:
            manifest.cpu_cores_used = self._detect_available_cores()

        # NUMA topology
        manifest.numa_topology = self._get_numa_topology()

        # GPU state
        manifest.gpu_clocks_before = self._get_gpu_clocks()
        manifest.gpu_perf_level = self._get_gpu_perf_level()

        # Store for restoration
        self._original_state = {
            "governors": manifest.cpu_governors_before.copy(),
            "gpu_perf_level": manifest.gpu_perf_level,
        }

    def _get_total_memory_mb(self) -> int:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) // 1024
        except:
            pass
        return 0

    def _is_swap_enabled(self) -> bool:
        try:
            with open("/proc/swaps") as f:
                lines = f.readlines()
                return len(lines) > 1
        except:
            return True

    def _get_cpu_governors(self) -> Dict[int, str]:
        governors = {}
        try:
            for cpu_dir in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
                cpu_id = int(cpu_dir.name[3:])
                gov_path = cpu_dir / "cpufreq" / "scaling_governor"
                if gov_path.exists():
                    governors[cpu_id] = gov_path.read_text().strip()
        except:
            pass
        return governors

    def _get_cpu_frequencies(self) -> Dict[int, float]:
        freqs = {}
        try:
            for cpu_dir in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
                cpu_id = int(cpu_dir.name[3:])
                freq_path = cpu_dir / "cpufreq" / "scaling_cur_freq"
                if freq_path.exists():
                    freqs[cpu_id] = int(freq_path.read_text().strip()) / 1000  # MHz
        except:
            pass
        return freqs

    def _detect_available_cores(self) -> List[int]:
        cores = []
        try:
            for cpu_dir in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
                cpu_id = int(cpu_dir.name[3:])
                online_path = cpu_dir / "online"
                if not online_path.exists() or online_path.read_text().strip() == "1":
                    cores.append(cpu_id)
        except:
            cores = list(range(os.cpu_count() or 1))
        return sorted(cores)

    def _get_numa_topology(self) -> Dict[str, Any]:
        topology = {"nodes": {}}
        try:
            numa_dir = Path("/sys/devices/system/node")
            for node_dir in numa_dir.glob("node[0-9]*"):
                node_id = int(node_dir.name[4:])
                cpulist_path = node_dir / "cpulist"
                if cpulist_path.exists():
                    topology["nodes"][node_id] = {"cpulist": cpulist_path.read_text().strip()}
        except:
            pass
        return topology

    def _get_gpu_clocks(self) -> Dict[str, int]:
        clocks = {}
        try:
            result = self._run_cmd("rocm-smi --showclocks --json")
            if result:
                data = json.loads(result)
                for card, info in data.items():
                    if "clocks" in info:
                        clocks[card] = info["clocks"]
        except:
            pass
        return clocks

    def _get_gpu_perf_level(self) -> str:
        try:
            result = self._run_cmd("rocm-smi --showperflevel")
            if result:
                for line in result.split("\n"):
                    if "Performance Level" in line:
                        return line.split(":")[-1].strip()
        except:
            pass
        return "unknown"

    # ==========================================================================
    # Isolation Application
    # ==========================================================================

    def _apply_isolation(self) -> None:
        """Apply isolation measures based on policy."""
        level = self.policy.isolation_level

        # Sync and drop caches (reduce memory pressure noise)
        if self.policy.sync_before_run:
            self._sync_and_drop_caches()

        # CPU governor
        if self.policy.cpu_governor != GovernorPolicy.RECORD_ONLY:
            self._set_cpu_governors()

        # GPU clocks
        if self.policy.gpu_clock_policy != GPUClockPolicy.RECORD_ONLY:
            self._set_gpu_clocks()

        # cgroups (for STANDARD and above)
        if level in (
            IsolationLevel.STANDARD,
            IsolationLevel.STRICT,
            IsolationLevel.PARANOID,
        ):
            self._setup_cgroup()

        # Record final state
        self._manifest.cpu_governors_after = self._get_cpu_governors()
        self._manifest.gpu_clocks_after = self._get_gpu_clocks()

    def _sync_and_drop_caches(self) -> None:
        """Sync filesystems and drop caches."""
        try:
            os.sync()
            if self.policy.drop_caches:
                # Requires root
                try:
                    Path("/proc/sys/vm/drop_caches").write_text("3")
                except PermissionError:
                    self._manifest.isolation_warnings.append("Cannot drop caches (not root)")
        except Exception as e:
            logger.warning(f"Failed to sync/drop caches: {e}")

    def _set_cpu_governors(self) -> None:
        """Set CPU governor on all cores."""
        governor = self.policy.cpu_governor.value
        cores = self._manifest.cpu_cores_used

        for cpu_id in cores:
            gov_path = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_governor")
            if gov_path.exists():
                try:
                    gov_path.write_text(governor)
                    logger.debug(f"Set CPU {cpu_id} governor to {governor}")
                except PermissionError:
                    self._manifest.isolation_warnings.append(
                        f"Cannot set governor on CPU {cpu_id} (not root)"
                    )

    def _set_gpu_clocks(self) -> None:
        """Set GPU performance level."""
        policy = self.policy.gpu_clock_policy

        if policy == GPUClockPolicy.HIGH:
            self._run_cmd("rocm-smi --setperflevel high")
        elif policy == GPUClockPolicy.LOW:
            self._run_cmd("rocm-smi --setperflevel low")
        elif policy == GPUClockPolicy.AUTO:
            self._run_cmd("rocm-smi --setperflevel auto")

    def _setup_cgroup(self) -> None:
        """Setup cgroup v2 for isolation."""
        cgroup_name = self.policy.cgroup_name
        cgroup_path = Path(f"/sys/fs/cgroup/{cgroup_name}")

        try:
            # Create cgroup
            if not cgroup_path.exists():
                cgroup_path.mkdir(parents=True)
                self._cgroup_created = True

            self._manifest.cgroup_path = str(cgroup_path)

            # Set cpuset
            if self._manifest.cpu_cores_used:
                cpuset_path = cgroup_path / "cpuset.cpus"
                if cpuset_path.exists():
                    cpulist = ",".join(str(c) for c in self._manifest.cpu_cores_used)
                    cpuset_path.write_text(cpulist)

            # Set memory limit
            if self.policy.memory_limit_mb:
                mem_high_path = cgroup_path / "memory.high"
                if mem_high_path.exists():
                    mem_bytes = self.policy.memory_limit_mb * 1024 * 1024
                    mem_high_path.write_text(str(mem_bytes))
                    self._manifest.cgroup_memory_high = mem_bytes

            # Set IO throttle
            if self.policy.io_throttle and self.policy.io_max_mbps:
                # Would need device major:minor numbers
                pass

            # Move current process into cgroup
            procs_path = cgroup_path / "cgroup.procs"
            if procs_path.exists():
                procs_path.write_text(str(os.getpid()))

            logger.info(f"Created cgroup: {cgroup_path}")

        except PermissionError:
            self._manifest.isolation_warnings.append("Cannot create cgroup (not root)")
        except Exception as e:
            self._manifest.isolation_warnings.append(f"cgroup error: {e}")

    # ==========================================================================
    # Verification
    # ==========================================================================

    def _verify_isolation(self) -> None:
        """Verify isolation was applied correctly."""
        manifest = self._manifest
        warnings = []

        # Check governors
        if self.policy.cpu_governor != GovernorPolicy.RECORD_ONLY:
            expected = self.policy.cpu_governor.value
            for cpu_id, gov in manifest.cpu_governors_after.items():
                if gov != expected:
                    warnings.append(f"CPU {cpu_id} governor is {gov}, expected {expected}")

        # Check GPU perf level
        if self.policy.gpu_clock_policy == GPUClockPolicy.HIGH:
            new_level = self._get_gpu_perf_level()
            if "high" not in new_level.lower():
                warnings.append(f"GPU perf level is {new_level}, expected high")

        # Check load average (warn if high)
        if manifest.load_average[0] > 2.0:
            warnings.append(f"High system load: {manifest.load_average[0]:.1f}")

        # Check swap (warn if enabled in strict mode)
        if self.policy.isolation_level == IsolationLevel.STRICT and manifest.swap_enabled:
            warnings.append("Swap is enabled (may cause latency spikes)")

        manifest.isolation_warnings.extend(warnings)
        manifest.isolation_verified = len(warnings) == 0

        if warnings:
            for w in warnings:
                logger.warning(f"Isolation warning: {w}")
        else:
            logger.info("Isolation verified successfully")

    # ==========================================================================
    # Restoration
    # ==========================================================================

    def _restore_state(self) -> None:
        """Restore original system state."""
        # Restore CPU governors
        original_governors = self._original_state.get("governors", {})
        for cpu_id, gov in original_governors.items():
            gov_path = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/cpufreq/scaling_governor")
            if gov_path.exists():
                try:
                    gov_path.write_text(gov)
                except:
                    pass

        # Restore GPU perf level
        original_level = self._original_state.get("gpu_perf_level", "auto")
        if "auto" in original_level.lower():
            self._run_cmd("rocm-smi --setperflevel auto")

        # Remove cgroup (if we created it)
        if self._cgroup_created and self._manifest.cgroup_path:
            try:
                cgroup_path = Path(self._manifest.cgroup_path)
                # Move process out first
                root_procs = Path("/sys/fs/cgroup/cgroup.procs")
                if root_procs.exists():
                    root_procs.write_text(str(os.getpid()))
                # Remove cgroup
                if cgroup_path.exists():
                    cgroup_path.rmdir()
            except:
                pass

    # ==========================================================================
    # Utilities
    # ==========================================================================

    def _run_cmd(self, cmd: str) -> str:
        """Run a shell command and return output."""
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            return result.stdout
        except:
            return ""


# ============================================================================
# Quick Capsule Factory
# ============================================================================


def create_standard_capsule() -> MeasurementCapsule:
    """Create a standard measurement capsule with reasonable defaults."""
    policy = CapsulePolicy(
        isolation_level=IsolationLevel.STANDARD,
        cpu_governor=GovernorPolicy.PERFORMANCE,
        gpu_clock_policy=GPUClockPolicy.HIGH,
        drop_caches=True,
        sync_before_run=True,
    )
    return MeasurementCapsule(policy)


def create_minimal_capsule() -> MeasurementCapsule:
    """Create a minimal capsule that only records state."""
    policy = CapsulePolicy(
        isolation_level=IsolationLevel.NONE,
        cpu_governor=GovernorPolicy.RECORD_ONLY,
        gpu_clock_policy=GPUClockPolicy.RECORD_ONLY,
    )
    return MeasurementCapsule(policy)
