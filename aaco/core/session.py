"""
AACO Session Management
Handles session lifecycle, folder creation, and artifact management.
"""

import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from aaco.core.schema import (
    SessionMetadata,
    HostInfo,
    GPUInfo,
    WorkloadConfig,
    BackendConfig,
)
from aaco.core.utils import get_monotonic_ns, run_command, safe_json_dump


class Session:
    """
    Represents a single AACO profiling session.
    Manages the session folder, metadata, and artifact collection.
    """

    def __init__(
        self,
        base_dir: Path,
        model_name: str,
        backend: str,
        batch_size: int = 1,
        session_id: Optional[str] = None,
    ):
        self.session_id = session_id or self._generate_session_id()
        self.model_name = model_name
        self.backend = backend
        self.batch_size = batch_size

        # Create session directory
        date_str = datetime.now().strftime("%Y%m%d")
        self.session_dir = base_dir / date_str / self.session_id
        self._create_directory_structure()

        # Timing reference
        self.t0_monotonic_ns = get_monotonic_ns()
        self.t0_utc = datetime.now(timezone.utc)

        # Metadata
        self.metadata: Optional[SessionMetadata] = None
        self._initialized = False

    def _generate_session_id(self) -> str:
        """Generate unique session ID with timestamp and random suffix."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:6]
        return f"{timestamp}_{suffix}"

    def _create_directory_structure(self) -> None:
        """Create the session folder hierarchy."""
        subdirs = [
            "model",
            "runtime",
            "telemetry",
            "profiler/rocprof_raw",
            "attribution",
            "metrics",
            "regress",
            "report/plots",
        ]

        self.session_dir.mkdir(parents=True, exist_ok=True)
        for subdir in subdirs:
            (self.session_dir / subdir).mkdir(parents=True, exist_ok=True)

    def initialize(
        self,
        model_path: Optional[str] = None,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        dtype: str = "float32",
        warmup: int = 10,
        iterations: int = 100,
        ep_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize session with full metadata.
        Must be called before any data collection.
        """
        host_info = self._collect_host_info()
        gpu_info = self._collect_gpu_info()

        workload = WorkloadConfig(
            framework="onnxruntime",
            model_name=self.model_name,
            model_path=model_path or f"models/{self.model_name}.onnx",
            input_shapes=input_shapes or {},
            dtype=dtype,
            batch_size=self.batch_size,
            warmup_iterations=warmup,
            measure_iterations=iterations,
        )

        backend_config = BackendConfig(
            name=self.backend,
            provider=self._get_ep_name(self.backend),
            device_id=0,
            config=ep_config or {},
        )

        self.metadata = SessionMetadata(
            session_id=self.session_id,
            created_utc=self.t0_utc.isoformat(),
            t0_monotonic_ns=self.t0_monotonic_ns,
            host=host_info,
            gpu=gpu_info,
            workload=workload,
            backend=backend_config,
        )

        # Save session.json
        self._save_session_metadata()

        # Save environment lockbox
        self._save_environment()

        self._initialized = True

    def _collect_host_info(self) -> HostInfo:
        """Collect host system information."""
        uname = platform.uname()

        # Get kernel version on Linux
        kernel_version = uname.release

        # Try to get CPU model
        cpu_model = platform.processor() or "Unknown"
        if os.path.exists("/proc/cpuinfo"):
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_model = line.split(":")[1].strip()
                            break
            except Exception:
                pass

        # Get RAM
        try:
            import psutil

            ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        except ImportError:
            ram_gb = 0.0

        return HostInfo(
            hostname=uname.node,
            os=f"{uname.system} {uname.release}",
            kernel=kernel_version,
            cpu_model=cpu_model,
            ram_gb=ram_gb,
            architecture=uname.machine,
        )

    def _collect_gpu_info(self) -> GPUInfo:
        """Collect GPU information using rocm-smi."""
        gpu_info = GPUInfo(
            vendor="AMD",
            name="Unknown",
            driver="Unknown",
            rocm_version="Unknown",
            vram_gb=0.0,
        )

        # Try rocm-smi for GPU name
        result = run_command(["rocm-smi", "--showproductname"])
        if result and "GPU" in result:
            for line in result.split("\n"):
                if "Card series" in line or "GPU" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        gpu_info.name = parts[-1].strip()
                        break

        # Try to get ROCm version
        result = run_command(["rocm-smi", "--version"])
        if result:
            for line in result.split("\n"):
                if "ROCm" in line or "version" in line.lower():
                    gpu_info.rocm_version = line.strip()
                    break

        # Try to get VRAM
        result = run_command(["rocm-smi", "--showmeminfo", "vram"])
        if result:
            for line in result.split("\n"):
                if "Total" in line:
                    try:
                        # Parse total VRAM in bytes and convert to GB
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p.isdigit():
                                vram_bytes = int(p)
                                gpu_info.vram_gb = round(vram_bytes / (1024**3), 1)
                                break
                    except Exception:
                        pass

        # Get driver version
        if os.path.exists("/sys/module/amdgpu/version"):
            try:
                with open("/sys/module/amdgpu/version", "r") as f:
                    gpu_info.driver = f.read().strip()
            except Exception:
                pass

        return gpu_info

    def _get_ep_name(self, backend: str) -> str:
        """Map backend name to ONNX Runtime ExecutionProvider name."""
        mapping = {
            "migraphx": "MIGraphXExecutionProvider",
            "rocm": "ROCMExecutionProvider",
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "tensorrt": "TensorrtExecutionProvider",
            "openvino": "OpenVINOExecutionProvider",
        }
        return mapping.get(backend.lower(), f"{backend}ExecutionProvider")

    def _save_session_metadata(self) -> None:
        """Save session.json with full metadata."""
        if not self.metadata:
            return

        session_path = self.session_dir / "session.json"
        safe_json_dump(self.metadata.to_dict(), session_path)

    def _save_environment(self) -> None:
        """Save env.json reproducibility lockbox."""
        env_data = {
            "git": self._get_git_info(),
            "packages": self._get_package_versions(),
            "system": self._get_system_config(),
            "cmdline": " ".join(os.sys.argv) if hasattr(os, "sys") else "",
            "env_vars": self._get_relevant_env_vars(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        env_path = self.session_dir / "env.json"
        safe_json_dump(env_data, env_path)

    def _get_git_info(self) -> Dict[str, Any]:
        """Get git repository information."""
        git_info = {"commit": "unknown", "branch": "unknown", "dirty": False}

        commit = run_command(["git", "rev-parse", "HEAD"])
        if commit:
            git_info["commit"] = commit.strip()[:12]

        branch = run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        if branch:
            git_info["branch"] = branch.strip()

        status = run_command(["git", "status", "--porcelain"])
        git_info["dirty"] = bool(status and status.strip())

        return git_info

    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        packages = {}

        try:
            import onnxruntime

            packages["onnxruntime"] = onnxruntime.__version__
        except ImportError:
            pass

        try:
            import numpy

            packages["numpy"] = numpy.__version__
        except ImportError:
            pass

        try:
            import pandas

            packages["pandas"] = pandas.__version__
        except ImportError:
            pass

        # ROCm version from rocm-smi
        result = run_command(["rocm-smi", "--version"])
        if result:
            packages["rocm"] = result.strip().split("\n")[0]

        return packages

    def _get_system_config(self) -> Dict[str, Any]:
        """Get system configuration relevant to performance."""
        config = {
            "cpu_governor": "unknown",
            "numa_enabled": False,
            "hugepages": "unknown",
        }

        # CPU governor
        governor_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        if os.path.exists(governor_path):
            try:
                with open(governor_path, "r") as f:
                    config["cpu_governor"] = f.read().strip()
            except Exception:
                pass

        # NUMA
        if os.path.exists("/sys/devices/system/node/node1"):
            config["numa_enabled"] = True

        # Hugepages
        hp_path = "/proc/sys/vm/nr_hugepages"
        if os.path.exists(hp_path):
            try:
                with open(hp_path, "r") as f:
                    config["hugepages"] = f.read().strip()
            except Exception:
                pass

        return config

    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """Get environment variables relevant to ROCm/GPU performance."""
        relevant_vars = [
            "HSA_OVERRIDE_GFX_VERSION",
            "MIGRAPHX_TRACE",
            "MIGRAPHX_DISABLE_FAST_GELU",
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "ORT_TENSORRT_FP16_ENABLE",
            "CUDA_VISIBLE_DEVICES",
        ]

        return {var: os.environ.get(var, "") for var in relevant_vars if os.environ.get(var)}

    def get_relative_time_ns(self) -> int:
        """Get current time relative to session start in nanoseconds."""
        return get_monotonic_ns() - self.t0_monotonic_ns

    def get_artifact_path(self, category: str, filename: str) -> Path:
        """Get the full path for an artifact file."""
        return self.session_dir / category / filename

    def save_artifact(self, category: str, filename: str, data: Any) -> Path:
        """Save an artifact (JSON, parquet, or raw)."""
        path = self.get_artifact_path(category, filename)

        if filename.endswith(".json"):
            safe_json_dump(data, path)
        elif filename.endswith(".parquet"):
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                data.to_parquet(path, index=False)
            else:
                pd.DataFrame(data).to_parquet(path, index=False)
        else:
            with open(path, "w") as f:
                f.write(str(data))

        return path

    def load_artifact(self, category: str, filename: str) -> Any:
        """Load an artifact from the session folder."""
        path = self.get_artifact_path(category, filename)

        if not path.exists():
            return None

        if filename.endswith(".json"):
            with open(path, "r") as f:
                return json.load(f)
        elif filename.endswith(".parquet"):
            import pandas as pd

            return pd.read_parquet(path)
        else:
            with open(path, "r") as f:
                return f.read()

    def finalize(self) -> None:
        """Finalize session - compute final timestamps and close."""
        if self.metadata:
            self.metadata.duration_s = (get_monotonic_ns() - self.t0_monotonic_ns) / 1e9
            self._save_session_metadata()


class SessionManager:
    """
    Manages multiple AACO sessions - creation, loading, and baseline comparison.
    """

    def __init__(self, base_dir: str = "sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        model_name: str,
        backend: str,
        batch_size: int = 1,
        **kwargs,
    ) -> Session:
        """Create a new profiling session."""
        session = Session(
            base_dir=self.base_dir,
            model_name=model_name,
            backend=backend,
            batch_size=batch_size,
        )
        return session

    def load_session(self, session_path: str) -> Optional[Session]:
        """Load an existing session from disk."""
        path = Path(session_path)

        if not path.exists():
            return None

        session_json = path / "session.json"
        if not session_json.exists():
            return None

        with open(session_json, "r") as f:
            metadata = json.load(f)

        # Reconstruct session
        session = Session.__new__(Session)
        session.session_id = metadata["session_id"]
        session.session_dir = path
        session.model_name = metadata["workload"]["model_name"]
        session.backend = metadata["backend"]["name"]
        session.batch_size = metadata["workload"]["batch_size"]
        session.t0_monotonic_ns = metadata.get("t0_monotonic_ns", 0)
        session._initialized = True

        return session

    def get_latest_session(self) -> Optional[Path]:
        """Get the path to the most recent session."""
        all_sessions = []

        for date_dir in self.base_dir.iterdir():
            if date_dir.is_dir():
                for session_dir in date_dir.iterdir():
                    if (session_dir / "session.json").exists():
                        all_sessions.append(session_dir)

        if not all_sessions:
            return None

        # Sort by modification time
        all_sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return all_sessions[0]

    def list_sessions(
        self,
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List sessions with optional filtering."""
        sessions = []

        for date_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if not date_dir.is_dir():
                continue

            for session_dir in sorted(date_dir.iterdir(), reverse=True):
                session_json = session_dir / "session.json"
                if not session_json.exists():
                    continue

                try:
                    with open(session_json, "r") as f:
                        metadata = json.load(f)

                    # Apply filters
                    if model_name and metadata["workload"]["model_name"] != model_name:
                        continue
                    if backend and metadata["backend"]["name"] != backend:
                        continue

                    sessions.append(
                        {
                            "session_id": metadata["session_id"],
                            "path": str(session_dir),
                            "model_name": metadata["workload"]["model_name"],
                            "backend": metadata["backend"]["name"],
                            "batch_size": metadata["workload"]["batch_size"],
                            "created": metadata["created_utc"],
                        }
                    )

                    if len(sessions) >= limit:
                        return sessions

                except Exception:
                    continue

        return sessions

    def create_symlink_latest(self, session: Session) -> None:
        """Create a 'latest' symlink pointing to the most recent session."""
        latest_link = self.base_dir / "latest"

        # Remove existing symlink
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()

        # Create new symlink
        try:
            latest_link.symlink_to(session.session_dir)
        except OSError:
            # On Windows, may need admin rights for symlinks
            pass
