"""
AACO Data Schema Definitions
Dataclasses for all AACO artifacts - session metadata, metrics, classifications, verdicts.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from enum import Enum


class BottleneckClass(Enum):
    """Performance bottleneck classification categories."""

    UNKNOWN = "unknown"
    LAUNCH_BOUND = "launch-bound"
    CPU_BOUND = "cpu-bound"
    MEMORY_BOUND = "memory-bound"
    COMPUTE_BOUND = "compute-bound"
    THROTTLING = "throttling"
    IO_BOUND = "io-bound"


class RegressionSeverity(Enum):
    """Regression severity levels."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Confidence(Enum):
    """Confidence levels for classifications."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class HostInfo:
    """Host system information."""

    hostname: str
    os: str
    kernel: str
    cpu_model: str
    ram_gb: float
    architecture: str = "x86_64"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GPUInfo:
    """GPU hardware information."""

    vendor: str
    name: str
    driver: str
    rocm_version: str
    vram_gb: float
    device_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkloadConfig:
    """Workload configuration for inference."""

    framework: str
    model_name: str
    model_path: str
    input_shapes: Dict[str, List[int]]
    dtype: str
    batch_size: int
    warmup_iterations: int
    measure_iterations: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BackendConfig:
    """Backend/Execution Provider configuration."""

    name: str
    provider: str
    device_id: int
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SessionMetadata:
    """Complete session metadata - the spine of every session bundle."""

    session_id: str
    created_utc: str
    t0_monotonic_ns: int
    host: HostInfo
    gpu: GPUInfo
    workload: WorkloadConfig
    backend: BackendConfig
    duration_s: float = 0.0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_utc": self.created_utc,
            "t0_monotonic_ns": self.t0_monotonic_ns,
            "host": self.host.to_dict(),
            "gpu": self.gpu.to_dict(),
            "workload": self.workload.to_dict(),
            "backend": self.backend.to_dict(),
            "duration_s": self.duration_s,
            "notes": self.notes,
        }


@dataclass
class InferenceIteration:
    """Single inference iteration measurement."""

    iter_idx: int
    t_start_ns: int
    t_end_ns: int
    latency_ms: float
    phase: str = "measure"  # "warmup" or "measure"
    token_idx: Optional[int] = None
    decode_phase: Optional[str] = None  # "prefill" or "decode"


@dataclass
class PhaseMetrics:
    """Metrics for a specific execution phase (warmup/measurement)."""

    name: str
    iterations: int
    total_time_ms: float
    mean_ms: float
    std_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    iqr_ms: float
    cov_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceResult:
    """Aggregated inference results with percentiles."""

    iterations: int
    warmup_iterations: int
    latencies_ms: List[float]
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput_samples_per_sec: float
    coefficient_of_variation: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_latencies(cls, latencies: List[float], warmup: int = 0) -> "InferenceResult":
        """Create InferenceResult from raw latency list."""
        import numpy as np

        arr = np.array(latencies)
        return cls(
            iterations=len(latencies),
            warmup_iterations=warmup,
            latencies_ms=latencies,
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p99_ms=float(np.percentile(arr, 99)),
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            throughput_samples_per_sec=1000.0 / float(np.mean(arr)) if np.mean(arr) > 0 else 0,
            coefficient_of_variation=float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else 0,
        )


@dataclass
class SystemEvent:
    """Single system telemetry sample."""

    t_ns: int
    cpu_pct: float
    rss_mb: float
    ctx_switches_delta: int
    majfault_delta: int
    runq_len: float
    load1: float
    pid: Optional[int] = None


@dataclass
class GPUEvent:
    """Single GPU telemetry sample."""

    t_ns: int
    gfx_clock_mhz: float
    mem_clock_mhz: float
    power_w: float
    temp_c: float
    vram_used_mb: float
    gpu_util_pct: float = 0.0


@dataclass
class KernelExecution:
    """Single GPU kernel execution record from rocprof."""

    t_start_ns: int
    t_end_ns: int
    dur_ns: int
    kernel_name: str
    queue_id: int = 0
    stream_id: int = 0
    grid_size: Optional[List[int]] = None
    workgroup_size: Optional[List[int]] = None


@dataclass
class KernelSummary:
    """Summary statistics for a GPU kernel."""

    kernel_name: str
    calls: int
    total_time_ms: float
    avg_time_us: float
    min_time_us: float
    max_time_us: float
    std_time_us: float
    pct_total: float


@dataclass
class KernelMetrics:
    """Derived kernel-level metrics for bottleneck analysis."""

    total_kernel_count: int
    unique_kernel_count: int
    total_kernel_time_ms: float
    avg_kernel_duration_us: float
    microkernel_count: int
    microkernel_pct: float  # % of kernels under threshold (e.g., 10us)
    microkernel_threshold_us: float
    launch_rate_per_sec: float
    launch_tax_score: float
    kernel_amplification_ratio: float  # kernels / ONNX nodes
    gpu_active_ratio: float  # kernel time / wall time
    top_kernels: List[KernelSummary] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["top_kernels"] = [asdict(k) for k in self.top_kernels]
        return result


@dataclass
class GraphNode:
    """ONNX graph node metadata."""

    node_id: int
    node_name: str
    op_type: str
    domain: str
    inputs: List[str]
    outputs: List[str]
    input_shapes: Dict[str, List[int]]
    output_shapes: Dict[str, List[int]]
    attributes: Dict[str, Any] = field(default_factory=dict)
    estimated_flops: Optional[float] = None
    estimated_bytes: Optional[float] = None


@dataclass
class KernelAttribution:
    """Mapping from ONNX node to GPU kernel group."""

    node_id: int
    op_type: str
    partition_id: Optional[int]
    kernel_group_id: Optional[int]
    kernel_names: List[str]
    attribution_method: str  # "exact", "name_heuristic", "time_correlation", "unknown"
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSignal:
    """Single evidence signal for bottleneck classification."""

    signal_name: str
    value: float
    weight: float
    threshold: Optional[float] = None
    direction: str = "higher_is_worse"  # or "lower_is_worse"


@dataclass
class BottleneckClassification:
    """Bottleneck classification result with evidence."""

    bottleneck_class: BottleneckClass
    confidence: Confidence
    score: float
    top_evidence: List[EvidenceSignal]
    explanation: str
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bottleneck_class": self.bottleneck_class.value,
            "confidence": self.confidence.value,
            "score": self.score,
            "top_evidence": [
                {
                    "signal": e.signal_name,
                    "value": e.value,
                    "weight": e.weight,
                }
                for e in self.top_evidence
            ],
            "explanation": self.explanation,
            "recommendations": self.recommendations,
        }


@dataclass
class MetricDelta:
    """Delta between two metrics for regression detection."""

    metric_name: str
    baseline_value: float
    current_value: float
    delta_absolute: float
    delta_pct: float
    significant: bool
    direction: str  # "better", "worse", "neutral"


@dataclass
class RegressionVerdict:
    """Final regression verdict with root cause analysis."""

    regression: bool
    severity: RegressionSeverity
    confidence: Confidence
    latency_delta_pct: float
    suspected_cause: BottleneckClass
    key_deltas: List[MetricDelta]
    evidence: Dict[str, Any]
    recommendation: str
    baseline_session_id: str
    comparison_session_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regression": self.regression,
            "severity": self.severity.value,
            "confidence": self.confidence.value,
            "latency_delta_pct": self.latency_delta_pct,
            "suspected_cause": self.suspected_cause.value,
            "key_deltas": [
                {
                    "metric": d.metric_name,
                    "baseline": d.baseline_value,
                    "current": d.current_value,
                    "delta_pct": d.delta_pct,
                    "significant": d.significant,
                    "direction": d.direction,
                }
                for d in self.key_deltas
            ],
            "evidence": self.evidence,
            "recommendation": self.recommendation,
            "baseline_session_id": self.baseline_session_id,
            "comparison_session_id": self.comparison_session_id,
        }


@dataclass
class DerivedMetrics:
    """All derived performance metrics for a session."""

    # Per-phase metrics
    warmup_phase: "PhaseMetrics"
    measurement_phase: "PhaseMetrics"

    # Aggregated metrics by category (Dict[str, float])
    throughput: Dict[str, Any]
    efficiency: Dict[str, Any]
    latency: Dict[str, Any]
    system: Dict[str, Any]
    gpu: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warmup_phase": self.warmup_phase.to_dict(),
            "measurement_phase": self.measurement_phase.to_dict(),
            "throughput": self.throughput,
            "efficiency": self.efficiency,
            "latency": self.latency,
            "system": self.system,
            "gpu": self.gpu,
        }
