"""
AACO-SIGMA Forensic Collector

Collects forensic data from various sources.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
from enum import Enum, auto
import time
import os
import platform
import subprocess
import json
import uuid

from .bundle import (
    ForensicBundle,
    BundleSection,
    EnvironmentInfo,
    TraceData,
    CounterData,
    MetricsData,
)


class CollectorMode(Enum):
    """Collection mode."""

    MINIMAL = auto()  # Metrics only
    STANDARD = auto()  # Metrics + traces + counters
    COMPREHENSIVE = auto()  # All sections
    CUSTOM = auto()  # User-defined sections


@dataclass
class CollectorConfig:
    """Configuration for forensic collection."""

    # Mode
    mode: CollectorMode = CollectorMode.STANDARD

    # Sections to collect (for CUSTOM mode)
    sections: List[BundleSection] = field(default_factory=list)

    # Collection parameters
    collect_env: bool = True
    collect_traces: bool = True
    collect_counters: bool = True
    collect_ir: bool = False
    collect_logs: bool = True

    # Trace settings
    trace_duration_s: float = 10.0
    rocprof_counters: List[str] = field(default_factory=lambda: ["GRBM_COUNT", "GRBM_GUI_ACTIVE"])

    # Size limits
    max_trace_events: int = 100000
    max_log_lines: int = 10000

    # Tagging
    auto_tag: bool = True
    custom_tags: List[str] = field(default_factory=list)


class ForensicCollector:
    """
    Collects forensic data from various sources.

    Sources:
    - rocprof traces
    - HIP runtime
    - Environment queries
    - Log files
    """

    def __init__(self, config: Optional[CollectorConfig] = None):
        self.config = config or CollectorConfig()
        self.collectors: Dict[BundleSection, Callable] = {}
        self._setup_collectors()

    def _setup_collectors(self) -> None:
        """Setup section collectors."""
        self.collectors = {
            BundleSection.ENVIRONMENT: self._collect_environment,
            BundleSection.TRACES: self._collect_traces,
            BundleSection.COUNTERS: self._collect_counters,
            BundleSection.METRICS: self._collect_metrics,
            BundleSection.IR: self._collect_ir,
            BundleSection.LOGS: self._collect_logs,
        }

    def collect(
        self,
        workload_id: str = "",
        name: str = "",
        description: str = "",
    ) -> ForensicBundle:
        """
        Collect forensic bundle.

        Args:
            workload_id: ID of the workload
            name: Bundle name
            description: Bundle description

        Returns:
            Complete forensic bundle
        """
        start_time = time.time()

        # Create bundle
        bundle = ForensicBundle()
        bundle.metadata.bundle_id = str(uuid.uuid4())[:8]
        bundle.metadata.name = name or f"bundle_{bundle.metadata.bundle_id}"
        bundle.metadata.description = description
        bundle.metadata.workload_id = workload_id
        bundle.metadata.created_at = start_time

        # Determine sections to collect
        sections = self._get_sections_for_mode()

        # Collect each section
        for section in sections:
            if section in self.collectors:
                try:
                    self.collectors[section](bundle)
                    bundle.add_log(f"Collected section: {section.value}")
                except Exception as e:
                    bundle.add_log(f"Failed to collect {section.value}: {e}")

        # Finalize
        bundle.metadata.capture_duration_s = time.time() - start_time

        # Auto-tag
        if self.config.auto_tag:
            self._auto_tag(bundle)

        # Add custom tags
        bundle.metadata.tags.extend(self.config.custom_tags)

        # Compute checksum
        bundle.compute_checksum()

        return bundle

    def _get_sections_for_mode(self) -> List[BundleSection]:
        """Get sections based on mode."""
        if self.config.mode == CollectorMode.MINIMAL:
            return [BundleSection.ENVIRONMENT, BundleSection.METRICS]

        elif self.config.mode == CollectorMode.STANDARD:
            return [
                BundleSection.ENVIRONMENT,
                BundleSection.TRACES,
                BundleSection.COUNTERS,
                BundleSection.METRICS,
            ]

        elif self.config.mode == CollectorMode.COMPREHENSIVE:
            return list(BundleSection)

        else:  # CUSTOM
            return self.config.sections

    def _collect_environment(self, bundle: ForensicBundle) -> None:
        """Collect environment information."""
        env = bundle.environment

        # System info
        env.hostname = platform.node()
        env.os_name = platform.system()
        env.os_version = platform.release()

        # CPU
        env.cpu_model = platform.processor() or "Unknown"
        env.cpu_cores = os.cpu_count() or 0

        # Memory (simple approach)
        try:
            if platform.system() == "Windows":
                # Use wmic on Windows
                result = subprocess.run(
                    ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    mem_bytes = int(lines[1].strip())
                    env.memory_gb = mem_bytes / (1024**3)
            else:
                # On Linux
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            kb = int(line.split()[1])
                            env.memory_gb = kb / (1024**2)
                            break
        except Exception:
            env.memory_gb = 0.0

        # GPU info via rocm-smi
        self._collect_gpu_info(env)

        # Software versions
        self._collect_software_versions(env)

        bundle.metadata.sections.append(BundleSection.ENVIRONMENT.value)

    def _collect_gpu_info(self, env: EnvironmentInfo) -> None:
        """Collect GPU information via rocm-smi."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                gpu_models = []
                for line in lines:
                    if "GPU" in line and ":" in line:
                        model = line.split(":")[-1].strip()
                        gpu_models.append(model)

                env.gpu_count = len(gpu_models)
                env.gpu_models = gpu_models

            # Get driver version
            result = subprocess.run(
                ["rocm-smi", "--showdriverversion"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "Driver" in line and ":" in line:
                        env.gpu_driver_version = line.split(":")[-1].strip()
                        break
        except FileNotFoundError:
            pass  # rocm-smi not available
        except Exception:
            pass

    def _collect_software_versions(self, env: EnvironmentInfo) -> None:
        """Collect software versions."""
        # ROCm version
        try:
            if os.path.exists("/opt/rocm/.info/version"):
                with open("/opt/rocm/.info/version") as f:
                    env.rocm_version = f.read().strip()
            else:
                result = subprocess.run(
                    ["apt", "list", "--installed", "rocm-libs"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if "rocm" in result.stdout:
                    env.rocm_version = result.stdout.split()[1]
        except Exception:
            pass

        # Python version
        env.python_version = platform.python_version()

        # PyTorch version
        try:
            import torch

            env.pytorch_version = torch.__version__
        except ImportError:
            pass

    def _collect_traces(self, bundle: ForensicBundle) -> None:
        """Collect trace data."""
        # This would interface with actual profiler
        # Placeholder implementation

        trace = TraceData()
        trace.trace_id = str(uuid.uuid4())[:8]
        trace.trace_type = "rocprof"
        trace.event_count = 0
        trace.events = []

        bundle.add_trace(trace)

    def _collect_counters(self, bundle: ForensicBundle) -> None:
        """Collect counter data."""
        # Placeholder - would run rocprof with counters
        bundle.counters = CounterData()
        bundle.metadata.sections.append(BundleSection.COUNTERS.value)

    def _collect_metrics(self, bundle: ForensicBundle) -> None:
        """Collect performance metrics."""
        # Placeholder - would collect actual metrics
        bundle.metrics = MetricsData()
        bundle.metadata.sections.append(BundleSection.METRICS.value)

    def _collect_ir(self, bundle: ForensicBundle) -> None:
        """Collect IR/graph data."""
        # Placeholder - would collect from PyTorch/runtime
        bundle.metadata.sections.append(BundleSection.IR.value)

    def _collect_logs(self, bundle: ForensicBundle) -> None:
        """Collect relevant logs."""
        bundle.add_log("Forensic collection started")
        bundle.add_log(f"Collection mode: {self.config.mode.name}")

    def _auto_tag(self, bundle: ForensicBundle) -> None:
        """Automatically tag bundle based on content."""
        tags = []

        # Tag by GPU
        if bundle.environment.gpu_models:
            for model in bundle.environment.gpu_models:
                if "MI300" in model:
                    tags.append("mi300")
                elif "MI250" in model:
                    tags.append("mi250")
                elif "RX" in model:
                    tags.append("rdna")

        # Tag by software
        if bundle.environment.rocm_version:
            major = bundle.environment.rocm_version.split(".")[0]
            tags.append(f"rocm{major}")

        if bundle.environment.pytorch_version:
            tags.append("pytorch")

        # Tag by sections
        if bundle.traces:
            tags.append("traced")

        bundle.metadata.tags.extend(tags)

    def collect_from_rocprof(
        self,
        csv_file: str,
        bundle: Optional[ForensicBundle] = None,
    ) -> ForensicBundle:
        """
        Collect trace data from rocprof CSV output.

        Args:
            csv_file: Path to rocprof CSV file
            bundle: Existing bundle to add to (or create new)

        Returns:
            Bundle with trace data
        """
        if bundle is None:
            bundle = ForensicBundle()
            bundle.metadata.bundle_id = str(uuid.uuid4())[:8]

        trace = TraceData()
        trace.trace_id = str(uuid.uuid4())[:8]
        trace.trace_type = "rocprof"
        trace.source_file = csv_file

        try:
            with open(csv_file) as f:
                import csv

                reader = csv.DictReader(f)

                events = []
                for i, row in enumerate(reader):
                    if i >= self.config.max_trace_events:
                        break
                    events.append(dict(row))

                trace.events = events
                trace.event_count = len(events)
        except Exception as e:
            bundle.add_log(f"Error reading rocprof CSV: {e}")

        bundle.add_trace(trace)
        return bundle

    def collect_from_json(
        self,
        json_file: str,
        bundle: Optional[ForensicBundle] = None,
    ) -> ForensicBundle:
        """
        Collect trace data from JSON trace file.

        Args:
            json_file: Path to JSON trace file
            bundle: Existing bundle to add to

        Returns:
            Bundle with trace data
        """
        if bundle is None:
            bundle = ForensicBundle()
            bundle.metadata.bundle_id = str(uuid.uuid4())[:8]

        trace = TraceData()
        trace.trace_id = str(uuid.uuid4())[:8]
        trace.trace_type = "chrome"
        trace.source_file = json_file

        try:
            with open(json_file) as f:
                data = json.load(f)

            # Chrome trace format
            if isinstance(data, dict) and "traceEvents" in data:
                events = data["traceEvents"]
            elif isinstance(data, list):
                events = data
            else:
                events = []

            trace.events = events[: self.config.max_trace_events]
            trace.event_count = len(trace.events)
        except Exception as e:
            bundle.add_log(f"Error reading JSON trace: {e}")

        bundle.add_trace(trace)
        return bundle
