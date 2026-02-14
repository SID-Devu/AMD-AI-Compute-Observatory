"""
AACO-SIGMA Forensic Bundle

Container for complete forensic capture of performance data.
Enables reproducible analysis and debugging.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import time
import hashlib


BUNDLE_FORMAT_VERSION = "1.0.0"


class BundleVersion:
    """Bundle format version handling."""

    CURRENT = BUNDLE_FORMAT_VERSION
    SUPPORTED = ["1.0.0"]

    @classmethod
    def is_supported(cls, version: str) -> bool:
        return version in cls.SUPPORTED


class BundleSection(Enum):
    """Sections of the forensic bundle."""

    METADATA = "metadata"
    ENVIRONMENT = "environment"
    CONFIGURATION = "configuration"
    TRACES = "traces"
    COUNTERS = "counters"
    METRICS = "metrics"
    BASELINES = "baselines"
    GRAPHS = "graphs"
    IR = "ir"
    LOGS = "logs"
    ARTIFACTS = "artifacts"


@dataclass
class EnvironmentInfo:
    """Captured environment information."""

    # System
    hostname: str = ""
    os_name: str = ""
    os_version: str = ""
    kernel_version: str = ""

    # Hardware
    cpu_model: str = ""
    cpu_cores: int = 0
    memory_gb: float = 0.0

    # GPU
    gpu_count: int = 0
    gpu_models: List[str] = field(default_factory=list)
    gpu_driver_version: str = ""

    # Software
    rocm_version: str = ""
    hip_version: str = ""
    python_version: str = ""
    pytorch_version: str = ""

    # Custom
    custom_env: Dict[str, str] = field(default_factory=dict)


@dataclass
class BundleMetadata:
    """Metadata for the forensic bundle."""

    # Identity
    bundle_id: str = ""
    name: str = ""
    description: str = ""

    # Version
    format_version: str = BUNDLE_FORMAT_VERSION

    # Source
    workload_id: str = ""
    model_name: str = ""

    # Timing
    created_at: float = field(default_factory=time.time)
    capture_duration_s: float = 0.0

    # Content
    sections: List[str] = field(default_factory=list)

    # Tags and labels
    tags: List[str] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    # Creator
    created_by: str = ""

    # Integrity
    checksum: str = ""


@dataclass
class TraceData:
    """Captured trace data."""

    trace_id: str = ""
    trace_type: str = ""  # "rocprof", "chrome", "custom"

    # Data
    events: List[Dict[str, Any]] = field(default_factory=list)

    # Source file (if from file)
    source_file: str = ""

    # Size
    event_count: int = 0
    raw_size_bytes: int = 0


@dataclass
class CounterData:
    """Captured counter data."""

    # Kernel -> counter -> values
    kernel_counters: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    # Aggregate
    aggregate_counters: Dict[str, float] = field(default_factory=dict)

    # Counter metadata
    counter_descriptions: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricsData:
    """Captured performance metrics."""

    # Per-kernel metrics
    kernel_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Aggregate metrics
    latency_ms: float = 0.0
    throughput: float = 0.0
    memory_peak_mb: float = 0.0

    # Timing breakdown
    compute_time_ms: float = 0.0
    memory_time_ms: float = 0.0
    overhead_time_ms: float = 0.0


@dataclass
class ForensicBundle:
    """
    Complete forensic bundle containing all captured data.

    A forensic bundle contains everything needed to:
    - Reproduce the performance measurement
    - Analyze root causes
    - Compare against baselines
    - Debug issues
    """

    # Metadata
    metadata: BundleMetadata = field(default_factory=BundleMetadata)

    # Environment
    environment: EnvironmentInfo = field(default_factory=EnvironmentInfo)

    # Configuration (how the workload was run)
    configuration: Dict[str, Any] = field(default_factory=dict)

    # Data sections
    traces: List[TraceData] = field(default_factory=list)
    counters: CounterData = field(default_factory=CounterData)
    metrics: MetricsData = field(default_factory=MetricsData)

    # Baseline comparison
    baseline_metrics: Optional[MetricsData] = None

    # Graph/IR data
    graph_json: Optional[str] = None
    ir_data: Dict[str, str] = field(default_factory=dict)  # filename -> content

    # Logs
    logs: List[str] = field(default_factory=list)

    # Additional artifacts
    artifacts: Dict[str, bytes] = field(default_factory=dict)  # name -> data

    def add_trace(self, trace: TraceData) -> None:
        """Add trace data to bundle."""
        self.traces.append(trace)
        if BundleSection.TRACES.value not in self.metadata.sections:
            self.metadata.sections.append(BundleSection.TRACES.value)

    def add_counters(self, kernel: str, counters: Dict[str, float]) -> None:
        """Add counter data for a kernel."""
        if kernel not in self.counters.kernel_counters:
            self.counters.kernel_counters[kernel] = {}

        for name, value in counters.items():
            if name not in self.counters.kernel_counters[kernel]:
                self.counters.kernel_counters[kernel][name] = []
            self.counters.kernel_counters[kernel][name].append(value)

        if BundleSection.COUNTERS.value not in self.metadata.sections:
            self.metadata.sections.append(BundleSection.COUNTERS.value)

    def add_log(self, message: str) -> None:
        """Add log message."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")

        if BundleSection.LOGS.value not in self.metadata.sections:
            self.metadata.sections.append(BundleSection.LOGS.value)

    def add_artifact(self, name: str, data: bytes) -> None:
        """Add binary artifact."""
        self.artifacts[name] = data

        if BundleSection.ARTIFACTS.value not in self.metadata.sections:
            self.metadata.sections.append(BundleSection.ARTIFACTS.value)

    def set_ir(self, name: str, content: str) -> None:
        """Add IR data."""
        self.ir_data[name] = content

        if BundleSection.IR.value not in self.metadata.sections:
            self.metadata.sections.append(BundleSection.IR.value)

    def compute_checksum(self) -> str:
        """Compute checksum of bundle contents."""
        hasher = hashlib.sha256()

        # Hash metadata
        hasher.update(self.metadata.bundle_id.encode())
        hasher.update(str(self.metadata.created_at).encode())

        # Hash traces
        for trace in self.traces:
            hasher.update(trace.trace_id.encode())
            hasher.update(str(trace.event_count).encode())

        # Hash metrics
        hasher.update(str(self.metrics.latency_ms).encode())

        self.metadata.checksum = hasher.hexdigest()[:16]
        return self.metadata.checksum

    def validate(self) -> tuple:
        """
        Validate bundle integrity.

        Returns:
            (is_valid, error_messages)
        """
        errors = []

        # Check version
        if not BundleVersion.is_supported(self.metadata.format_version):
            errors.append(f"Unsupported version: {self.metadata.format_version}")

        # Check required fields
        if not self.metadata.bundle_id:
            errors.append("Missing bundle_id")

        # Check sections match content
        if BundleSection.TRACES.value in self.metadata.sections and not self.traces:
            errors.append("Traces section declared but empty")

        return (len(errors) == 0, errors)

    def to_dict(self) -> Dict[str, Any]:
        """Convert bundle to dictionary for serialization."""
        return {
            "metadata": {
                "bundle_id": self.metadata.bundle_id,
                "name": self.metadata.name,
                "description": self.metadata.description,
                "format_version": self.metadata.format_version,
                "workload_id": self.metadata.workload_id,
                "created_at": self.metadata.created_at,
                "sections": self.metadata.sections,
                "tags": self.metadata.tags,
                "checksum": self.metadata.checksum,
            },
            "environment": {
                "hostname": self.environment.hostname,
                "os_name": self.environment.os_name,
                "gpu_count": self.environment.gpu_count,
                "gpu_models": self.environment.gpu_models,
                "rocm_version": self.environment.rocm_version,
            },
            "configuration": self.configuration,
            "metrics": {
                "latency_ms": self.metrics.latency_ms,
                "throughput": self.metrics.throughput,
                "memory_peak_mb": self.metrics.memory_peak_mb,
            },
            "traces_count": len(self.traces),
            "logs_count": len(self.logs),
            "artifacts_count": len(self.artifacts),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ForensicBundle":
        """Create bundle from dictionary."""
        bundle = cls()

        if "metadata" in data:
            m = data["metadata"]
            bundle.metadata.bundle_id = m.get("bundle_id", "")
            bundle.metadata.name = m.get("name", "")
            bundle.metadata.format_version = m.get("format_version", BUNDLE_FORMAT_VERSION)
            bundle.metadata.sections = m.get("sections", [])
            bundle.metadata.tags = m.get("tags", [])

        if "environment" in data:
            e = data["environment"]
            bundle.environment.hostname = e.get("hostname", "")
            bundle.environment.rocm_version = e.get("rocm_version", "")

        if "configuration" in data:
            bundle.configuration = data["configuration"]

        if "metrics" in data:
            m = data["metrics"]
            bundle.metrics.latency_ms = m.get("latency_ms", 0.0)
            bundle.metrics.throughput = m.get("throughput", 0.0)

        return bundle
