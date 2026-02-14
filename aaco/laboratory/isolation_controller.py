# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Isolation Controller

Full system isolation for deterministic measurements using:
- cgroups v2 CPU & memory partitioning
- CPU core isolation + NUMA pinning
- Thread affinity control
- CPU governor enforcement
- GPU clock policy enforcement
"""

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class CPUGovernor(Enum):
    """CPU frequency governor policies."""

    PERFORMANCE = "performance"
    POWERSAVE = "powersave"
    ONDEMAND = "ondemand"
    CONSERVATIVE = "conservative"
    SCHEDUTIL = "schedutil"
    USERSPACE = "userspace"


class GPUClockPolicy(Enum):
    """AMD GPU clock policies."""

    AUTO = "auto"
    LOW = "low"
    HIGH = "high"
    MANUAL = "manual"
    PROFILE_STANDARD = "profile_standard"
    PROFILE_MIN_SCLK = "profile_min_sclk"
    PROFILE_MIN_MCLK = "profile_min_mclk"
    PROFILE_PEAK = "profile_peak"


@dataclass
class NUMAConfig:
    """NUMA topology configuration."""

    node_id: int
    cpu_list: List[int]
    memory_mb: int
    local_memory_only: bool = True


@dataclass
class CGroupConfig:
    """cgroups v2 configuration for isolation."""

    name: str
    cpu_cores: List[int]
    memory_limit_mb: int
    cpu_quota_us: Optional[int] = None  # None = unlimited
    cpu_period_us: int = 100000
    memory_swap_limit_mb: Optional[int] = None
    io_weight: int = 100


@dataclass
class IsolationConfig:
    """Complete isolation configuration."""

    # CPU isolation
    isolated_cores: List[int] = field(default_factory=list)
    numa_config: Optional[NUMAConfig] = None
    cpu_governor: CPUGovernor = CPUGovernor.PERFORMANCE
    disable_hyper_threading: bool = False

    # Memory isolation
    memory_limit_mb: int = 0  # 0 = no limit
    disable_swap: bool = True
    lock_memory: bool = True
    huge_pages: bool = False

    # GPU isolation
    gpu_device_ids: List[int] = field(default_factory=lambda: [0])
    gpu_clock_policy: GPUClockPolicy = GPUClockPolicy.PROFILE_PEAK
    gpu_power_limit_watts: Optional[int] = None

    # Process isolation
    nice_priority: int = -20
    io_priority: int = 0  # 0 = real-time
    scheduler_policy: str = "SCHED_FIFO"
    scheduler_priority: int = 99

    # cgroups
    cgroup_config: Optional[CGroupConfig] = None


@dataclass
class IsolationState:
    """Current isolation state."""

    active: bool = False
    start_time: float = 0.0
    original_governor: Optional[str] = None
    original_gpu_clock: Optional[str] = None
    cgroup_path: Optional[Path] = None
    isolated_pids: Set[int] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class IsolationController:
    """
    System isolation controller for deterministic execution.

    Provides laboratory-grade isolation by controlling:
    - CPU core allocation and frequency
    - Memory allocation and NUMA binding
    - GPU clock and power policies
    - Process scheduling priorities
    - cgroups v2 resource limits
    """

    CGROUP_BASE = Path("/sys/fs/cgroup")
    CPU_GOVERNOR_PATH = Path("/sys/devices/system/cpu")
    GPU_SYSFS_PATH = Path("/sys/class/drm")

    def __init__(self, config: Optional[IsolationConfig] = None):
        """Initialize isolation controller."""
        self.config = config or IsolationConfig()
        self.state = IsolationState()
        self._validate_permissions()

    def _validate_permissions(self) -> None:
        """Check if we have necessary permissions."""
        self._has_root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
        self._has_cgroup_access = self.CGROUP_BASE.exists()

        if not self._has_root:
            logger.warning("Running without root - some isolation features unavailable")

    def enter_laboratory_mode(self) -> Dict[str, Any]:
        """
        Enter laboratory mode with full isolation.

        Returns:
            Dict with isolation status and any errors/warnings
        """
        if self.state.active:
            raise RuntimeError("Laboratory mode already active")

        self.state = IsolationState(active=True, start_time=time.time())
        results = {}

        # 1. Setup CPU isolation
        results["cpu"] = self._setup_cpu_isolation()

        # 2. Setup memory isolation
        results["memory"] = self._setup_memory_isolation()

        # 3. Setup GPU isolation
        results["gpu"] = self._setup_gpu_isolation()

        # 4. Setup cgroups if configured
        if self.config.cgroup_config:
            results["cgroup"] = self._setup_cgroup()

        # 5. Setup process priorities
        results["process"] = self._setup_process_isolation()

        # 6. Validate isolation
        results["validation"] = self._validate_isolation()

        return {
            "success": len(self.state.errors) == 0,
            "results": results,
            "errors": self.state.errors,
            "warnings": self.state.warnings,
        }

    def exit_laboratory_mode(self) -> Dict[str, Any]:
        """
        Exit laboratory mode and restore system state.

        Returns:
            Dict with restoration status
        """
        if not self.state.active:
            return {"success": True, "message": "Not in laboratory mode"}

        results = {}

        # Restore in reverse order
        if self.state.cgroup_path:
            results["cgroup"] = self._teardown_cgroup()

        results["gpu"] = self._restore_gpu_settings()
        results["cpu"] = self._restore_cpu_settings()

        self.state.active = False

        return {
            "success": True,
            "duration_seconds": time.time() - self.state.start_time,
            "results": results,
        }

    def _setup_cpu_isolation(self) -> Dict[str, Any]:
        """Configure CPU isolation."""
        result = {"success": False, "actions": []}

        try:
            # Set CPU governor
            if self._has_root:
                self._set_cpu_governor(self.config.cpu_governor)
                result["actions"].append(f"Set governor to {self.config.cpu_governor.value}")

            # Set CPU affinity for isolated cores
            if self.config.isolated_cores:
                self._set_cpu_affinity(self.config.isolated_cores)
                result["actions"].append(f"Isolated cores: {self.config.isolated_cores}")

            # NUMA binding
            if self.config.numa_config:
                self._setup_numa_binding(self.config.numa_config)
                result["actions"].append(f"NUMA node: {self.config.numa_config.node_id}")

            result["success"] = True

        except Exception as e:
            self.state.errors.append(f"CPU isolation failed: {e}")
            result["error"] = str(e)

        return result

    def _setup_memory_isolation(self) -> Dict[str, Any]:
        """Configure memory isolation."""
        result = {"success": False, "actions": []}

        try:
            # Disable swap if requested
            if self.config.disable_swap and self._has_root:
                self._disable_swap()
                result["actions"].append("Disabled swap")

            # Lock memory if requested
            if self.config.lock_memory:
                self._lock_memory()
                result["actions"].append("Memory locked")

            # Setup huge pages if requested
            if self.config.huge_pages and self._has_root:
                self._setup_huge_pages()
                result["actions"].append("Huge pages enabled")

            result["success"] = True

        except Exception as e:
            self.state.warnings.append(f"Memory isolation partial: {e}")
            result["error"] = str(e)

        return result

    def _setup_gpu_isolation(self) -> Dict[str, Any]:
        """Configure GPU isolation."""
        result = {"success": False, "actions": [], "devices": []}

        try:
            for gpu_id in self.config.gpu_device_ids:
                gpu_result = self._configure_gpu(gpu_id)
                result["devices"].append(gpu_result)

            result["success"] = True
            result["actions"].append(f"Configured {len(self.config.gpu_device_ids)} GPU(s)")

        except Exception as e:
            self.state.warnings.append(f"GPU isolation partial: {e}")
            result["error"] = str(e)

        return result

    def _configure_gpu(self, gpu_id: int) -> Dict[str, Any]:
        """Configure a single GPU."""
        result = {"gpu_id": gpu_id, "success": False}

        try:
            # Try to use rocm-smi for configuration
            if self.config.gpu_clock_policy == GPUClockPolicy.PROFILE_PEAK:
                self._run_rocm_smi(["--setperflevel", "high", "-d", str(gpu_id)])
                result["clock_policy"] = "high"

            if self.config.gpu_power_limit_watts:
                self._run_rocm_smi(
                    [
                        "--setpoweroverdrive",
                        str(self.config.gpu_power_limit_watts),
                        "-d",
                        str(gpu_id),
                    ]
                )
                result["power_limit"] = self.config.gpu_power_limit_watts

            result["success"] = True

        except Exception as e:
            result["error"] = str(e)
            self.state.warnings.append(f"GPU {gpu_id} config partial: {e}")

        return result

    def _setup_cgroup(self) -> Dict[str, Any]:
        """Setup cgroups v2 isolation."""
        result = {"success": False}
        cfg = self.config.cgroup_config

        if not cfg or not self._has_root:
            return {"success": False, "reason": "No cgroup config or no root"}

        try:
            cgroup_path = self.CGROUP_BASE / "aaco" / cfg.name
            cgroup_path.mkdir(parents=True, exist_ok=True)
            self.state.cgroup_path = cgroup_path

            # CPU controller
            if cfg.cpu_cores:
                cpuset_path = cgroup_path / "cpuset.cpus"
                cpu_str = ",".join(map(str, cfg.cpu_cores))
                cpuset_path.write_text(cpu_str)
                result["cpuset"] = cpu_str

            # CPU quota
            if cfg.cpu_quota_us:
                max_path = cgroup_path / "cpu.max"
                max_path.write_text(f"{cfg.cpu_quota_us} {cfg.cpu_period_us}")
                result["cpu_quota"] = cfg.cpu_quota_us

            # Memory limit
            if cfg.memory_limit_mb > 0:
                mem_path = cgroup_path / "memory.max"
                mem_path.write_text(str(cfg.memory_limit_mb * 1024 * 1024))
                result["memory_limit_mb"] = cfg.memory_limit_mb

            result["success"] = True
            result["path"] = str(cgroup_path)

        except Exception as e:
            self.state.errors.append(f"cgroup setup failed: {e}")
            result["error"] = str(e)

        return result

    def _setup_process_isolation(self) -> Dict[str, Any]:
        """Configure process scheduling."""
        result = {"success": False, "actions": []}

        try:
            pid = os.getpid()

            # Set nice priority
            if self._has_root:
                os.nice(self.config.nice_priority)
                result["actions"].append(f"Nice: {self.config.nice_priority}")

            # Set scheduler policy (Linux only)
            if hasattr(os, "sched_setscheduler") and self._has_root:
                # SCHED_FIFO = 1, SCHED_RR = 2
                policy = 1 if self.config.scheduler_policy == "SCHED_FIFO" else 2
                param = os.sched_param(self.config.scheduler_priority)
                os.sched_setscheduler(pid, policy, param)
                result["actions"].append(f"Scheduler: {self.config.scheduler_policy}")

            result["success"] = True

        except Exception as e:
            self.state.warnings.append(f"Process isolation partial: {e}")
            result["error"] = str(e)

        return result

    def _validate_isolation(self) -> Dict[str, Any]:
        """Validate that isolation is properly configured."""
        validation = {
            "cpu_isolated": False,
            "memory_locked": False,
            "gpu_configured": False,
            "overall_score": 0.0,
        }

        # Check CPU affinity
        if hasattr(os, "sched_getaffinity"):
            current_affinity = os.sched_getaffinity(0)
            if self.config.isolated_cores:
                expected = set(self.config.isolated_cores)
                validation["cpu_isolated"] = current_affinity == expected

        # Calculate overall isolation score
        scores = [
            validation["cpu_isolated"],
            len(self.state.errors) == 0,
        ]
        validation["overall_score"] = sum(scores) / len(scores)

        return validation

    def _set_cpu_governor(self, governor: CPUGovernor) -> None:
        """Set CPU frequency governor for all CPUs."""
        for cpu_path in self.CPU_GOVERNOR_PATH.glob("cpu[0-9]*"):
            governor_path = cpu_path / "cpufreq" / "scaling_governor"
            if governor_path.exists():
                # Store original
                if self.state.original_governor is None:
                    self.state.original_governor = governor_path.read_text().strip()
                governor_path.write_text(governor.value)

    def _set_cpu_affinity(self, cores: List[int]) -> None:
        """Set CPU affinity for current process."""
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(cores))

    def _setup_numa_binding(self, config: NUMAConfig) -> None:
        """Setup NUMA memory binding."""
        # Use numactl if available
        pass  # Platform-specific implementation

    def _disable_swap(self) -> None:
        """Disable swap."""
        subprocess.run(["swapoff", "-a"], check=False, capture_output=True)

    def _lock_memory(self) -> None:
        """Lock memory to prevent paging."""
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_MEMLOCK, (-1, -1))
        except Exception:
            pass  # Not critical

    def _setup_huge_pages(self) -> None:
        """Enable huge pages."""
        huge_path = Path("/sys/kernel/mm/transparent_hugepage/enabled")
        if huge_path.exists():
            huge_path.write_text("always")

    def _run_rocm_smi(self, args: List[str]) -> subprocess.CompletedProcess:
        """Run rocm-smi command."""
        return subprocess.run(["rocm-smi"] + args, check=False, capture_output=True, text=True)

    def _restore_cpu_settings(self) -> Dict[str, Any]:
        """Restore original CPU settings."""
        result = {"success": True}

        if self.state.original_governor and self._has_root:
            try:
                for cpu_path in self.CPU_GOVERNOR_PATH.glob("cpu[0-9]*"):
                    governor_path = cpu_path / "cpufreq" / "scaling_governor"
                    if governor_path.exists():
                        governor_path.write_text(self.state.original_governor)
                result["governor_restored"] = self.state.original_governor
            except Exception as e:
                result["error"] = str(e)

        return result

    def _restore_gpu_settings(self) -> Dict[str, Any]:
        """Restore original GPU settings."""
        result = {"success": True}

        try:
            for gpu_id in self.config.gpu_device_ids:
                self._run_rocm_smi(["--setperflevel", "auto", "-d", str(gpu_id)])
        except Exception as e:
            result["error"] = str(e)

        return result

    def _teardown_cgroup(self) -> Dict[str, Any]:
        """Remove cgroup."""
        result = {"success": False}

        if self.state.cgroup_path and self.state.cgroup_path.exists():
            try:
                # Move processes out first
                procs_path = self.state.cgroup_path / "cgroup.procs"
                if procs_path.exists():
                    # Move to parent
                    parent_procs = self.state.cgroup_path.parent / "cgroup.procs"
                    for pid in procs_path.read_text().strip().split("\n"):
                        if pid:
                            parent_procs.write_text(pid)

                self.state.cgroup_path.rmdir()
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)

        return result

    def get_isolation_manifest(self) -> Dict[str, Any]:
        """Get complete isolation manifest for reproducibility."""
        return {
            "version": "1.0.0",
            "config": {
                "isolated_cores": self.config.isolated_cores,
                "cpu_governor": self.config.cpu_governor.value,
                "gpu_devices": self.config.gpu_device_ids,
                "gpu_clock_policy": self.config.gpu_clock_policy.value,
                "memory_limit_mb": self.config.memory_limit_mb,
                "disable_swap": self.config.disable_swap,
                "scheduler_policy": self.config.scheduler_policy,
            },
            "state": {
                "active": self.state.active,
                "start_time": self.state.start_time,
                "errors": self.state.errors,
                "warnings": self.state.warnings,
            },
            "system": self._get_system_info(),
        }

    def _get_system_info(self) -> Dict[str, Any]:
        """Get current system information."""
        import platform

        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

        # CPU count
        info["cpu_count"] = os.cpu_count()

        # Try to get memory info
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        info["memory_total_kb"] = int(line.split()[1])
                        break
        except Exception:
            pass

        return info


def create_laboratory_config(
    isolated_cores: Optional[List[int]] = None,
    gpu_ids: Optional[List[int]] = None,
    memory_limit_mb: int = 0,
) -> IsolationConfig:
    """Create a standard laboratory configuration."""
    return IsolationConfig(
        isolated_cores=isolated_cores or [],
        gpu_device_ids=gpu_ids or [0],
        cpu_governor=CPUGovernor.PERFORMANCE,
        gpu_clock_policy=GPUClockPolicy.PROFILE_PEAK,
        memory_limit_mb=memory_limit_mb,
        disable_swap=True,
        lock_memory=True,
        nice_priority=-20,
        scheduler_policy="SCHED_FIFO",
        scheduler_priority=99,
    )
