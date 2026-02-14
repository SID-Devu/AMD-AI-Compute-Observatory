"""
Feature Store System
Session and iteration-level feature extraction and storage.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


# ============================================================================
# Session-Level Features
# ============================================================================


@dataclass
class SessionFeatures:
    """
    Session-level performance features.

    Computed once per profiling session to enable cross-session comparison
    and regression detection.
    """

    # Identification
    session_id: str
    model_name: str = ""
    backend: str = ""  # onnxruntime, pytorch, llamacpp, etc.
    device: str = ""  # GPU model
    timestamp_utc: float = 0.0

    # Latency Statistics (measurement phase only)
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0

    # Throughput
    throughput_its: float = 0.0  # Iterations per second
    throughput_tokens: float = 0.0  # Tokens per second (if applicable)

    # Stability Metrics
    cv_pct: float = 0.0  # Coefficient of variation
    spike_count: int = 0  # Number of latency spikes
    spike_ratio: float = 0.0  # Fraction of iterations that are spikes
    noise_score: float = 0.0  # Overall noise/instability score [0-1]

    # GPU Efficiency
    kar: float = 0.0  # Kernel Active Ratio
    launch_tax_score: float = 0.0  # Launch overhead score [0-1]
    gpu_util_mean_pct: float = 0.0
    gpu_util_std_pct: float = 0.0

    # Power/Thermal
    power_mean_w: float = 0.0
    power_max_w: float = 0.0
    temp_mean_c: float = 0.0
    temp_max_c: float = 0.0
    throttle_events: int = 0

    # Clock Stability
    clock_mean_mhz: float = 0.0
    clock_std_mhz: float = 0.0
    clock_stability_score: float = 0.0  # [0-1], higher = more stable

    # Memory
    vram_used_mb: float = 0.0
    vram_peak_mb: float = 0.0

    # CPU/System
    ctx_switches_per_iter: float = 0.0
    page_faults_per_iter: float = 0.0
    cpu_time_pct: float = 0.0

    # Composite Scores
    chi_score: float = 0.0  # Compute Health Index [0-100]

    # Run Configuration
    batch_size: int = 1
    warmup_iters: int = 0
    measure_iters: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionFeatures":
        return cls(**d)


# ============================================================================
# Iteration-Level Features
# ============================================================================


@dataclass
class IterationFeatures:
    """Per-iteration performance features for fine-grained analysis."""

    # Identification
    session_id: str
    iteration: int
    is_warmup: bool = False

    # Timing
    latency_ms: float = 0.0
    start_ns: int = 0
    end_ns: int = 0

    # GPU Metrics (sampled during this iteration)
    power_w: float = 0.0
    gfx_clock_mhz: float = 0.0
    mem_clock_mhz: float = 0.0
    temp_c: float = 0.0
    gpu_util_pct: float = 0.0

    # CPU/System (during this iteration)
    ctx_switches: int = 0
    page_faults: int = 0

    # Derived
    is_spike: bool = False
    spike_magnitude: float = 0.0  # How many std devs above mean

    # Kernel data
    kernel_count: int = 0
    total_kernel_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Arrow/Parquet Schemas
# ============================================================================

SESSION_FEATURES_SCHEMA = pa.schema(
    [
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("model_name", pa.string()),
        pa.field("backend", pa.string()),
        pa.field("device", pa.string()),
        pa.field("timestamp_utc", pa.float64()),
        pa.field("latency_mean_ms", pa.float64()),
        pa.field("latency_std_ms", pa.float64()),
        pa.field("latency_p50_ms", pa.float64()),
        pa.field("latency_p90_ms", pa.float64()),
        pa.field("latency_p95_ms", pa.float64()),
        pa.field("latency_p99_ms", pa.float64()),
        pa.field("latency_min_ms", pa.float64()),
        pa.field("latency_max_ms", pa.float64()),
        pa.field("throughput_its", pa.float64()),
        pa.field("throughput_tokens", pa.float64()),
        pa.field("cv_pct", pa.float64()),
        pa.field("spike_count", pa.int32()),
        pa.field("spike_ratio", pa.float64()),
        pa.field("noise_score", pa.float64()),
        pa.field("kar", pa.float64()),
        pa.field("launch_tax_score", pa.float64()),
        pa.field("gpu_util_mean_pct", pa.float64()),
        pa.field("gpu_util_std_pct", pa.float64()),
        pa.field("power_mean_w", pa.float64()),
        pa.field("power_max_w", pa.float64()),
        pa.field("temp_mean_c", pa.float64()),
        pa.field("temp_max_c", pa.float64()),
        pa.field("throttle_events", pa.int32()),
        pa.field("clock_mean_mhz", pa.float64()),
        pa.field("clock_std_mhz", pa.float64()),
        pa.field("clock_stability_score", pa.float64()),
        pa.field("vram_used_mb", pa.float64()),
        pa.field("vram_peak_mb", pa.float64()),
        pa.field("ctx_switches_per_iter", pa.float64()),
        pa.field("page_faults_per_iter", pa.float64()),
        pa.field("cpu_time_pct", pa.float64()),
        pa.field("chi_score", pa.float64()),
        pa.field("batch_size", pa.int32()),
        pa.field("warmup_iters", pa.int32()),
        pa.field("measure_iters", pa.int32()),
        pa.field("input_tokens", pa.int32()),
        pa.field("output_tokens", pa.int32()),
    ]
)

ITERATION_FEATURES_SCHEMA = pa.schema(
    [
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("iteration", pa.int32(), nullable=False),
        pa.field("is_warmup", pa.bool_()),
        pa.field("latency_ms", pa.float64()),
        pa.field("start_ns", pa.int64()),
        pa.field("end_ns", pa.int64()),
        pa.field("power_w", pa.float64()),
        pa.field("gfx_clock_mhz", pa.float64()),
        pa.field("mem_clock_mhz", pa.float64()),
        pa.field("temp_c", pa.float64()),
        pa.field("gpu_util_pct", pa.float64()),
        pa.field("ctx_switches", pa.int32()),
        pa.field("page_faults", pa.int32()),
        pa.field("is_spike", pa.bool_()),
        pa.field("spike_magnitude", pa.float64()),
        pa.field("kernel_count", pa.int32()),
        pa.field("total_kernel_time_ms", pa.float64()),
    ]
)


# ============================================================================
# Feature Extractor
# ============================================================================


class FeatureExtractor:
    """
    Extracts session and iteration features from raw AACO data.
    """

    def __init__(self, spike_threshold_std: float = 2.0):
        self.spike_threshold_std = spike_threshold_std

    def extract_session_features(
        self,
        session_id: str,
        latencies_ms: List[float],
        warmup_count: int = 10,
        gpu_samples: Optional[List[Dict]] = None,
        kernel_data: Optional[List[Dict]] = None,
        sched_data: Optional[Dict] = None,
        config: Optional[Dict] = None,
    ) -> SessionFeatures:
        """
        Extract session-level features from raw data.

        Args:
            session_id: Session identifier
            latencies_ms: All latency measurements (warmup + measure)
            warmup_count: Number of warmup iterations
            gpu_samples: GPU telemetry samples
            kernel_data: Kernel profile data
            sched_data: Scheduling metrics
            config: Run configuration

        Returns:
            SessionFeatures with all extracted metrics
        """
        features = SessionFeatures(
            session_id=session_id,
            timestamp_utc=time.time(),
            warmup_iters=warmup_count,
        )

        # Config
        if config:
            features.model_name = config.get("model_name", "")
            features.backend = config.get("backend", "")
            features.device = config.get("device", "")
            features.batch_size = config.get("batch_size", 1)
            features.input_tokens = config.get("input_tokens", 0)
            features.output_tokens = config.get("output_tokens", 0)

        # Measurement phase latencies only
        measure_latencies = (
            latencies_ms[warmup_count:] if len(latencies_ms) > warmup_count else latencies_ms
        )
        features.measure_iters = len(measure_latencies)

        if not measure_latencies:
            return features

        arr = np.array(measure_latencies)

        # Latency statistics
        features.latency_mean_ms = float(np.mean(arr))
        features.latency_std_ms = float(np.std(arr))
        features.latency_min_ms = float(np.min(arr))
        features.latency_max_ms = float(np.max(arr))
        features.latency_p50_ms = float(np.percentile(arr, 50))
        features.latency_p90_ms = float(np.percentile(arr, 90))
        features.latency_p95_ms = float(np.percentile(arr, 95))
        features.latency_p99_ms = float(np.percentile(arr, 99))

        # Throughput
        if features.latency_mean_ms > 0:
            features.throughput_its = 1000.0 / features.latency_mean_ms

        # Stability metrics
        if features.latency_mean_ms > 0:
            features.cv_pct = (features.latency_std_ms / features.latency_mean_ms) * 100

        # Spike detection
        if features.latency_std_ms > 0:
            spike_threshold = (
                features.latency_mean_ms + self.spike_threshold_std * features.latency_std_ms
            )
            spikes = arr[arr > spike_threshold]
            features.spike_count = len(spikes)
            features.spike_ratio = len(spikes) / len(arr)

        # Noise score: combination of CV and spike ratio
        features.noise_score = min(1.0, (features.cv_pct / 20.0) * 0.7 + features.spike_ratio * 0.3)

        # GPU metrics
        if gpu_samples:
            self._extract_gpu_features(features, gpu_samples)

        # Kernel metrics
        if kernel_data:
            self._extract_kernel_features(features, kernel_data, measure_latencies)

        # Scheduling metrics
        if sched_data:
            self._extract_sched_features(features, sched_data, len(measure_latencies))

        # Compute CHI score
        features.chi_score = self._compute_chi(features)

        return features

    def _extract_gpu_features(self, features: SessionFeatures, samples: List[Dict]) -> None:
        """Extract GPU telemetry features."""
        if not samples:
            return

        powers = [s.get("power_w", 0) for s in samples if "power_w" in s]
        clocks = [s.get("gfx_clock_mhz", 0) for s in samples if "gfx_clock_mhz" in s]
        temps = [s.get("temp_c", 0) for s in samples if "temp_c" in s]
        utils = [s.get("gpu_util_pct", 0) for s in samples if "gpu_util_pct" in s]
        vrams = [s.get("vram_used_mb", 0) for s in samples if "vram_used_mb" in s]

        if powers:
            features.power_mean_w = float(np.mean(powers))
            features.power_max_w = float(np.max(powers))

        if clocks:
            features.clock_mean_mhz = float(np.mean(clocks))
            features.clock_std_mhz = float(np.std(clocks))
            # Clock stability: lower std = more stable
            if features.clock_mean_mhz > 0:
                features.clock_stability_score = 1.0 - min(
                    1.0, features.clock_std_mhz / features.clock_mean_mhz
                )

        if temps:
            features.temp_mean_c = float(np.mean(temps))
            features.temp_max_c = float(np.max(temps))

        if utils:
            features.gpu_util_mean_pct = float(np.mean(utils))
            features.gpu_util_std_pct = float(np.std(utils))

        if vrams:
            features.vram_used_mb = float(np.mean(vrams))
            features.vram_peak_mb = float(np.max(vrams))

    def _extract_kernel_features(
        self, features: SessionFeatures, kernels: List[Dict], latencies: List[float]
    ) -> None:
        """Extract kernel profile features."""
        if not kernels or not latencies:
            return

        total_kernel_time_ms = sum(k.get("total_time_ms", 0) for k in kernels)
        total_latency_ms = sum(latencies)

        # Kernel Active Ratio
        if total_latency_ms > 0:
            features.kar = total_kernel_time_ms / total_latency_ms

        # Launch tax score: low KAR = high launch tax
        features.launch_tax_score = max(0.0, 1.0 - features.kar)

    def _extract_sched_features(
        self, features: SessionFeatures, sched_data: Dict, iter_count: int
    ) -> None:
        """Extract CPU scheduling features."""
        if iter_count == 0:
            return

        total_ctx_switches = sched_data.get("context_switches", 0)
        total_faults = sched_data.get("page_faults", 0)

        features.ctx_switches_per_iter = total_ctx_switches / iter_count
        features.page_faults_per_iter = total_faults / iter_count

    def _compute_chi(self, features: SessionFeatures) -> float:
        """
        Compute Compute Health Index (CHI).

        A weighted composite score [0-100] indicating overall compute health.
        Higher is better.
        """
        # Component scores (each 0-1, higher = better)

        # Stability score: based on CV (lower CV = better)
        stability = 1.0 - min(1.0, features.cv_pct / 30.0)

        # Efficiency score: based on KAR (higher KAR = better)
        efficiency = features.kar if features.kar > 0 else 0.5

        # Launch tax score: (lower tax = better, so invert)
        launch = 1.0 - features.launch_tax_score

        # Clock stability score (already 0-1)
        clock_stability = features.clock_stability_score

        # Spike penalty
        spike_penalty = 1.0 - features.spike_ratio

        # Thermal score (temps below 80C = good)
        thermal = (
            1.0 - min(1.0, max(0, features.temp_max_c - 60) / 40.0)
            if features.temp_max_c > 0
            else 1.0
        )

        # Weighted combination
        chi = (
            stability * 0.25
            + efficiency * 0.25
            + launch * 0.15
            + clock_stability * 0.15
            + spike_penalty * 0.10
            + thermal * 0.10
        ) * 100

        return round(chi, 1)

    def extract_iteration_features(
        self,
        session_id: str,
        latencies_ms: List[float],
        warmup_count: int = 10,
        gpu_samples: Optional[List[Dict]] = None,
    ) -> List[IterationFeatures]:
        """
        Extract per-iteration features.

        Args:
            session_id: Session identifier
            latencies_ms: All latency measurements
            warmup_count: Number of warmup iterations
            gpu_samples: GPU telemetry samples (time-aligned)

        Returns:
            List of IterationFeatures
        """
        iterations = []

        # Compute spike threshold from measurement phase
        measure_latencies = (
            latencies_ms[warmup_count:] if len(latencies_ms) > warmup_count else latencies_ms
        )
        if measure_latencies:
            mean = np.mean(measure_latencies)
            std = np.std(measure_latencies)
            spike_threshold = mean + self.spike_threshold_std * std
        else:
            mean, std, spike_threshold = 0, 0, float("inf")

        # Sample index for GPU data alignment
        samples_per_iter = (
            len(gpu_samples) / len(latencies_ms) if gpu_samples and latencies_ms else 0
        )

        for i, latency_ms in enumerate(latencies_ms):
            is_warmup = i < warmup_count
            is_spike = not is_warmup and latency_ms > spike_threshold
            spike_mag = (latency_ms - mean) / std if std > 0 and is_spike else 0.0

            iter_features = IterationFeatures(
                session_id=session_id,
                iteration=i,
                is_warmup=is_warmup,
                latency_ms=latency_ms,
                is_spike=is_spike,
                spike_magnitude=spike_mag,
            )

            # Align GPU sample data
            if gpu_samples and samples_per_iter > 0:
                sample_idx = int(i * samples_per_iter)
                if 0 <= sample_idx < len(gpu_samples):
                    sample = gpu_samples[sample_idx]
                    iter_features.power_w = sample.get("power_w", 0)
                    iter_features.gfx_clock_mhz = sample.get("gfx_clock_mhz", 0)
                    iter_features.mem_clock_mhz = sample.get("mem_clock_mhz", 0)
                    iter_features.temp_c = sample.get("temp_c", 0)
                    iter_features.gpu_util_pct = sample.get("gpu_util_pct", 0)

            iterations.append(iter_features)

        return iterations


# ============================================================================
# Feature Store
# ============================================================================


class FeatureStore:
    """
    Persistent feature store for session and iteration features.

    Supports:
    - Append new sessions/iterations
    - Query by session ID, time range, model
    - Load/save to Parquet
    - Baseline comparison
    """

    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = Path(store_path) if store_path else None
        self._session_features: List[SessionFeatures] = []
        self._iteration_features: List[IterationFeatures] = []

        # Load existing data if store exists
        if self.store_path and self.store_path.exists():
            self._load()

    def add_session(self, features: SessionFeatures) -> None:
        """Add session features."""
        self._session_features.append(features)

    def add_iterations(self, iterations: List[IterationFeatures]) -> None:
        """Add iteration features."""
        self._iteration_features.extend(iterations)

    def get_session(self, session_id: str) -> Optional[SessionFeatures]:
        """Get session by ID."""
        for sf in self._session_features:
            if sf.session_id == session_id:
                return sf
        return None

    def get_sessions_for_model(self, model_name: str) -> List[SessionFeatures]:
        """Get all sessions for a model."""
        return [sf for sf in self._session_features if sf.model_name == model_name]

    def get_baseline(self, model_name: str, n_sessions: int = 5) -> Optional[SessionFeatures]:
        """
        Get baseline features for a model (average of last N sessions).

        Returns None if insufficient history.
        """
        sessions = self.get_sessions_for_model(model_name)
        if len(sessions) < n_sessions:
            return None

        # Get most recent N sessions
        recent = sorted(sessions, key=lambda s: s.timestamp_utc, reverse=True)[:n_sessions]

        # Compute average baseline
        baseline = SessionFeatures(
            session_id=f"baseline_{model_name}",
            model_name=model_name,
            timestamp_utc=time.time(),
        )

        # Average numeric fields
        for field_name in [
            "latency_mean_ms",
            "latency_std_ms",
            "latency_p50_ms",
            "latency_p90_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "throughput_its",
            "cv_pct",
            "spike_ratio",
            "kar",
            "launch_tax_score",
            "power_mean_w",
            "temp_mean_c",
            "clock_mean_mhz",
            "chi_score",
        ]:
            values = [getattr(s, field_name, 0) for s in recent]
            setattr(baseline, field_name, float(np.mean(values)) if values else 0)

        return baseline

    def get_iterations(self, session_id: str) -> List[IterationFeatures]:
        """Get iterations for a session."""
        return [it for it in self._iteration_features if it.session_id == session_id]

    def save(self) -> None:
        """Save feature store to Parquet files."""
        if not self.store_path:
            raise ValueError("No store path configured")

        self.store_path.mkdir(parents=True, exist_ok=True)

        # Save session features
        session_path = self.store_path / "feature_store_session.parquet"
        self._save_sessions_parquet(session_path)

        # Save iteration features
        iter_path = self.store_path / "feature_store_iter.parquet"
        self._save_iterations_parquet(iter_path)

        logger.info(
            f"Feature store saved: {len(self._session_features)} sessions, "
            f"{len(self._iteration_features)} iterations"
        )

    def _save_sessions_parquet(self, path: Path) -> None:
        """Save session features to Parquet."""
        if not self._session_features:
            return

        data = {field: [] for field in asdict(self._session_features[0]).keys()}

        for sf in self._session_features:
            d = asdict(sf)
            for field, value in d.items():
                data[field].append(value)

        table = pa.table(data, schema=SESSION_FEATURES_SCHEMA)
        pq.write_table(table, str(path), compression="zstd")

    def _save_iterations_parquet(self, path: Path) -> None:
        """Save iteration features to Parquet."""
        if not self._iteration_features:
            return

        data = {field: [] for field in asdict(self._iteration_features[0]).keys()}

        for it in self._iteration_features:
            d = asdict(it)
            for field, value in d.items():
                data[field].append(value)

        table = pa.table(data, schema=ITERATION_FEATURES_SCHEMA)
        pq.write_table(table, str(path), compression="zstd")

    def _load(self) -> None:
        """Load feature store from Parquet files."""
        session_path = self.store_path / "feature_store_session.parquet"
        if session_path.exists():
            table = pq.read_table(str(session_path))
            self._session_features = self._load_sessions_from_table(table)

        iter_path = self.store_path / "feature_store_iter.parquet"
        if iter_path.exists():
            table = pq.read_table(str(iter_path))
            self._iteration_features = self._load_iterations_from_table(table)

    def _load_sessions_from_table(self, table: pa.Table) -> List[SessionFeatures]:
        """Load session features from Arrow table."""
        sessions = []

        for i in range(table.num_rows):
            row_dict = {}
            for col_name in table.column_names:
                value = table.column(col_name)[i].as_py()
                row_dict[col_name] = value if value is not None else 0

            # Handle missing fields gracefully
            for field in asdict(SessionFeatures(session_id="")).keys():
                if field not in row_dict:
                    row_dict[field] = (
                        "" if field in ("session_id", "model_name", "backend", "device") else 0
                    )

            sessions.append(SessionFeatures(**row_dict))

        return sessions

    def _load_iterations_from_table(self, table: pa.Table) -> List[IterationFeatures]:
        """Load iteration features from Arrow table."""
        iterations = []

        for i in range(table.num_rows):
            row_dict = {}
            for col_name in table.column_names:
                value = table.column(col_name)[i].as_py()
                row_dict[col_name] = value if value is not None else 0

            iterations.append(IterationFeatures(**row_dict))

        return iterations

    def export_json(self, path: Path) -> None:
        """Export feature store to JSON for inspection."""
        data = {
            "sessions": [sf.to_dict() for sf in self._session_features],
            "iterations_count": len(self._iteration_features),
            "session_count": len(self._session_features),
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# ============================================================================
# Convenience Functions
# ============================================================================


def extract_and_store_features(
    session_path: Path,
    store: Optional[FeatureStore] = None,
) -> Tuple[SessionFeatures, List[IterationFeatures]]:
    """
    Extract features from a session and optionally store them.

    Args:
        session_path: Path to session directory
        store: Optional feature store to save to

    Returns:
        Tuple of (SessionFeatures, List[IterationFeatures])
    """
    session_path = Path(session_path)
    session_id = session_path.name

    # Load session data
    config = {}
    latencies_ms = []
    gpu_samples = []
    kernel_data = []

    config_file = session_path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    results_file = session_path / "inference_results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
            latencies_ms = [r.get("latency_ms", 0) for r in results]

    gpu_file = session_path / "gpu_samples.json"
    if gpu_file.exists():
        with open(gpu_file) as f:
            gpu_samples = json.load(f)

    kernel_file = session_path / "kernel_summary.json"
    if kernel_file.exists():
        with open(kernel_file) as f:
            kernel_data = json.load(f)

    # Extract features
    extractor = FeatureExtractor()

    warmup_count = config.get("warmup_iters", 10)

    session_features = extractor.extract_session_features(
        session_id=session_id,
        latencies_ms=latencies_ms,
        warmup_count=warmup_count,
        gpu_samples=gpu_samples,
        kernel_data=kernel_data,
        config=config,
    )

    iteration_features = extractor.extract_iteration_features(
        session_id=session_id,
        latencies_ms=latencies_ms,
        warmup_count=warmup_count,
        gpu_samples=gpu_samples,
    )

    # Store if provided
    if store:
        store.add_session(session_features)
        store.add_iterations(iteration_features)

    return session_features, iteration_features
