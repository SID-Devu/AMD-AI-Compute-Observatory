"""
AACO-SIGMA Parquet Feature Store

Secondary trace format for ML-ready feature extraction.
Provides efficient columnar storage for analytics.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict


# Try to import pyarrow for Parquet support
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


@dataclass
class SessionFeatureTable:
    """Session-level features for ML/analytics."""
    session_id: str
    
    # Timing
    start_timestamp_ns: int = 0
    end_timestamp_ns: int = 0
    duration_ns: int = 0
    
    # Workload identity
    model_hash: str = ""
    input_signature: str = ""
    execution_provider: str = ""
    workload_signature: str = ""
    
    # Hardware
    gpu_model: str = ""
    gpu_clock_mhz: int = 0
    rocm_version: str = ""
    
    # Summary metrics
    iteration_count: int = 0
    latency_p50_us: float = 0.0
    latency_p90_us: float = 0.0
    latency_p99_us: float = 0.0
    latency_mean_us: float = 0.0
    latency_std_us: float = 0.0
    
    throughput_samples_per_sec: float = 0.0
    
    # Kernel metrics
    total_kernels: int = 0
    unique_kernels: int = 0
    gpu_time_ns: int = 0
    gpu_util_mean: float = 0.0
    
    # Derived metrics
    kernel_amplification_ratio: float = 0.0
    microkernel_pct: float = 0.0
    launch_tax_pct: float = 0.0
    memory_bound_pct: float = 0.0
    compute_bound_pct: float = 0.0
    
    # Quality metrics
    capsule_health_score: float = 100.0
    noise_score: float = 0.0
    stability_score: float = 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SessionFeatureTable':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class KernelFeatureTable:
    """Kernel-level features for ML/analytics."""
    session_id: str
    kernel_name: str
    kernel_family_id: str = ""
    
    # Timing
    total_time_ns: int = 0
    call_count: int = 0
    
    # Statistics
    mean_duration_ns: float = 0.0
    std_duration_ns: float = 0.0
    min_duration_ns: int = 0
    max_duration_ns: int = 0
    p50_duration_ns: float = 0.0
    p90_duration_ns: float = 0.0
    p99_duration_ns: float = 0.0
    
    # Launch config
    grid_x: int = 0
    grid_y: int = 0
    grid_z: int = 0
    block_x: int = 0
    block_y: int = 0
    block_z: int = 0
    
    # Classification
    is_microkernel: bool = False
    category: str = ""  # gemm, conv, attention, etc.
    
    # Counter-based metrics (if available)
    occupancy_pct: float = 0.0
    memory_throughput_pct: float = 0.0
    compute_throughput_pct: float = 0.0
    l2_hit_rate: float = 0.0
    
    # Derived
    pct_of_gpu_time: float = 0.0
    pct_of_calls: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IterationFeatureTable:
    """Iteration-level features for ML/analytics."""
    session_id: str
    iteration: int
    
    # Timing
    timestamp_ns: int = 0
    latency_ns: int = 0
    
    # Phase
    phase: str = ""  # warmup, measure, prefill, decode
    
    # GPU metrics
    kernel_count: int = 0
    gpu_time_ns: int = 0
    gpu_idle_ns: int = 0
    
    # Counter aggregates
    gpu_util: float = 0.0
    gpu_clock_mhz: int = 0
    gpu_temp_c: float = 0.0
    gpu_power_w: float = 0.0
    gpu_memory_used_mb: int = 0
    
    # Derived
    compute_efficiency: float = 0.0
    memory_efficiency: float = 0.0
    
    # Quality
    had_noise: bool = False
    had_throttle: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ParquetFeatureStore:
    """
    Parquet-based feature store for efficient analytics.
    
    Stores features in columnar format for:
    - Session-level aggregates
    - Kernel-level statistics
    - Iteration-level time series
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._session_features: List[SessionFeatureTable] = []
        self._kernel_features: List[KernelFeatureTable] = []
        self._iteration_features: List[IterationFeatureTable] = []
    
    def add_session(self, features: SessionFeatureTable) -> None:
        """Add session-level features."""
        self._session_features.append(features)
    
    def add_kernel(self, features: KernelFeatureTable) -> None:
        """Add kernel-level features."""
        self._kernel_features.append(features)
    
    def add_iteration(self, features: IterationFeatureTable) -> None:
        """Add iteration-level features."""
        self._iteration_features.append(features)
    
    def add_kernels_bulk(self, kernels: List[Dict[str, Any]], session_id: str) -> None:
        """Add multiple kernel features from raw data."""
        for kernel in kernels:
            features = KernelFeatureTable(
                session_id=session_id,
                kernel_name=kernel.get("name", kernel.get("kernel_name", "")),
                kernel_family_id=kernel.get("family_id", ""),
                total_time_ns=kernel.get("total_time_ns", 0),
                call_count=kernel.get("count", kernel.get("call_count", 1)),
                mean_duration_ns=kernel.get("mean_ns", kernel.get("mean_duration_ns", 0)),
                std_duration_ns=kernel.get("std_ns", kernel.get("std_duration_ns", 0)),
                min_duration_ns=kernel.get("min_ns", kernel.get("min_duration_ns", 0)),
                max_duration_ns=kernel.get("max_ns", kernel.get("max_duration_ns", 0)),
                p50_duration_ns=kernel.get("p50_ns", kernel.get("p50_duration_ns", 0)),
                p90_duration_ns=kernel.get("p90_ns", kernel.get("p90_duration_ns", 0)),
                p99_duration_ns=kernel.get("p99_ns", kernel.get("p99_duration_ns", 0)),
                is_microkernel=kernel.get("is_microkernel", False),
                category=kernel.get("category", ""),
                pct_of_gpu_time=kernel.get("pct_time", kernel.get("pct_of_gpu_time", 0)),
            )
            self._kernel_features.append(features)
    
    def add_iterations_bulk(self, iterations: List[Dict[str, Any]], session_id: str) -> None:
        """Add multiple iteration features from raw data."""
        for iter_data in iterations:
            features = IterationFeatureTable(
                session_id=session_id,
                iteration=iter_data.get("iteration", 0),
                timestamp_ns=iter_data.get("timestamp_ns", 0),
                latency_ns=iter_data.get("latency_ns", iter_data.get("duration_ns", 0)),
                phase=iter_data.get("phase", "measure"),
                kernel_count=iter_data.get("kernel_count", 0),
                gpu_time_ns=iter_data.get("gpu_time_ns", 0),
                gpu_util=iter_data.get("gpu_util", 0),
                gpu_temp_c=iter_data.get("gpu_temp_c", 0),
                gpu_power_w=iter_data.get("gpu_power_w", 0),
            )
            self._iteration_features.append(features)
    
    def save_parquet(self, prefix: str = "") -> Dict[str, Path]:
        """Save all features as Parquet files."""
        paths = {}
        
        if not HAS_PYARROW:
            # Fallback to JSON if PyArrow not available
            return self.save_json(prefix)
        
        # Save session features
        if self._session_features:
            path = self.output_dir / f"{prefix}session_features.parquet"
            table = pa.Table.from_pylist([f.to_dict() for f in self._session_features])
            pq.write_table(table, path)
            paths["session_features"] = path
        
        # Save kernel features
        if self._kernel_features:
            path = self.output_dir / f"{prefix}kernel_features.parquet"
            table = pa.Table.from_pylist([f.to_dict() for f in self._kernel_features])
            pq.write_table(table, path)
            paths["kernel_features"] = path
        
        # Save iteration features
        if self._iteration_features:
            path = self.output_dir / f"{prefix}iteration_features.parquet"
            table = pa.Table.from_pylist([f.to_dict() for f in self._iteration_features])
            pq.write_table(table, path)
            paths["iteration_features"] = path
        
        return paths
    
    def save_json(self, prefix: str = "") -> Dict[str, Path]:
        """Fallback: save as JSON files."""
        paths = {}
        
        if self._session_features:
            path = self.output_dir / f"{prefix}session_features.json"
            with open(path, 'w') as f:
                json.dump([f.to_dict() for f in self._session_features], f, indent=2)
            paths["session_features"] = path
        
        if self._kernel_features:
            path = self.output_dir / f"{prefix}kernel_features.json"
            with open(path, 'w') as f:
                json.dump([f.to_dict() for f in self._kernel_features], f, indent=2)
            paths["kernel_features"] = path
        
        if self._iteration_features:
            path = self.output_dir / f"{prefix}iteration_features.json"
            with open(path, 'w') as f:
                json.dump([f.to_dict() for f in self._iteration_features], f, indent=2)
            paths["iteration_features"] = path
        
        return paths
    
    @classmethod
    def load_session_features(cls, path: Path) -> List[SessionFeatureTable]:
        """Load session features from file."""
        if path.suffix == '.parquet' and HAS_PYARROW:
            table = pq.read_table(path)
            return [SessionFeatureTable.from_dict(row) for row in table.to_pylist()]
        else:
            with open(path) as f:
                data = json.load(f)
            return [SessionFeatureTable.from_dict(d) for d in data]
    
    def get_session_features(self) -> List[SessionFeatureTable]:
        """Get all session features."""
        return self._session_features.copy()
    
    def get_kernel_features(self) -> List[KernelFeatureTable]:
        """Get all kernel features."""
        return self._kernel_features.copy()
    
    def get_iteration_features(self) -> List[IterationFeatureTable]:
        """Get all iteration features."""
        return self._iteration_features.copy()


class FeatureStoreBuilder:
    """Builder for constructing feature store from session data."""
    
    def __init__(self, session_id: str, output_dir: Path):
        self.session_id = session_id
        self.store = ParquetFeatureStore(output_dir)
        self._session = SessionFeatureTable(session_id=session_id)
    
    def set_timing(self, start_ns: int, end_ns: int) -> 'FeatureStoreBuilder':
        """Set session timing."""
        self._session.start_timestamp_ns = start_ns
        self._session.end_timestamp_ns = end_ns
        self._session.duration_ns = end_ns - start_ns
        return self
    
    def set_workload(self, model_hash: str, input_signature: str,
                     execution_provider: str) -> 'FeatureStoreBuilder':
        """Set workload identity."""
        self._session.model_hash = model_hash
        self._session.input_signature = input_signature
        self._session.execution_provider = execution_provider
        return self
    
    def set_latency_stats(self, p50_us: float, p90_us: float, p99_us: float,
                          mean_us: float, std_us: float) -> 'FeatureStoreBuilder':
        """Set latency statistics."""
        self._session.latency_p50_us = p50_us
        self._session.latency_p90_us = p90_us
        self._session.latency_p99_us = p99_us
        self._session.latency_mean_us = mean_us
        self._session.latency_std_us = std_us
        return self
    
    def set_kernel_metrics(self, total: int, unique: int, gpu_time_ns: int,
                           kar: float, microkernel_pct: float) -> 'FeatureStoreBuilder':
        """Set kernel metrics."""
        self._session.total_kernels = total
        self._session.unique_kernels = unique
        self._session.gpu_time_ns = gpu_time_ns
        self._session.kernel_amplification_ratio = kar
        self._session.microkernel_pct = microkernel_pct
        return self
    
    def set_quality(self, capsule_health: float, noise_score: float,
                    stability: float) -> 'FeatureStoreBuilder':
        """Set measurement quality metrics."""
        self._session.capsule_health_score = capsule_health
        self._session.noise_score = noise_score
        self._session.stability_score = stability
        return self
    
    def add_kernels(self, kernels: List[Dict[str, Any]]) -> 'FeatureStoreBuilder':
        """Add kernel features."""
        self.store.add_kernels_bulk(kernels, self.session_id)
        return self
    
    def add_iterations(self, iterations: List[Dict[str, Any]]) -> 'FeatureStoreBuilder':
        """Add iteration features."""
        self.store.add_iterations_bulk(iterations, self.session_id)
        self._session.iteration_count = len(iterations)
        return self
    
    def build(self) -> ParquetFeatureStore:
        """Finalize and return the feature store."""
        self.store.add_session(self._session)
        return self.store
