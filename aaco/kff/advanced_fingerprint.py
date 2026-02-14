# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Kernel Fingerprint Family (Advanced)

Advanced kernel fingerprinting with:
- Duration distribution vectors
- Grid/workgroup signatures
- Counter signatures (memory vs compute)
- Launch rate statistics
- Stability fingerprints
- Cross-run clustering and family assignment
"""

import hashlib
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class KernelFamily(Enum):
    """GPU kernel family classifications."""

    GEMM = "gemm"
    CONV = "convolution"
    REDUCE = "reduction"
    ELEMENTWISE = "elementwise"
    ATTENTION = "attention"
    NORM = "normalization"
    POOLING = "pooling"
    SOFTMAX = "softmax"
    TRANSPOSE = "transpose"
    MEMORY = "memory_operation"
    UNKNOWN = "unknown"


@dataclass
class DurationDistribution:
    """Duration distribution vector for a kernel."""

    count: int = 0
    mean_ns: float = 0.0
    std_dev_ns: float = 0.0
    min_ns: float = 0.0
    max_ns: float = 0.0
    percentiles: Dict[int, float] = field(default_factory=dict)  # p50, p90, p95, p99
    histogram: List[int] = field(default_factory=list)  # 10-bin histogram

    def compute(self, durations: List[float]) -> "DurationDistribution":
        """Compute distribution from duration samples."""
        if not durations:
            return self

        self.count = len(durations)
        self.mean_ns = statistics.mean(durations)
        self.std_dev_ns = statistics.stdev(durations) if len(durations) > 1 else 0
        self.min_ns = min(durations)
        self.max_ns = max(durations)

        sorted_d = sorted(durations)
        n = len(sorted_d)
        self.percentiles = {
            50: sorted_d[int(n * 0.50)],
            90: sorted_d[int(n * 0.90)] if n > 10 else sorted_d[-1],
            95: sorted_d[int(n * 0.95)] if n > 20 else sorted_d[-1],
            99: sorted_d[int(n * 0.99)] if n > 100 else sorted_d[-1],
        }

        # Compute histogram (log-scale bins)
        if self.max_ns > self.min_ns:
            import math

            log_min = math.log10(max(1, self.min_ns))
            log_max = math.log10(self.max_ns)
            bin_width = (log_max - log_min) / 10

            self.histogram = [0] * 10
            for d in durations:
                bin_idx = min(9, int((math.log10(max(1, d)) - log_min) / bin_width))
                self.histogram[bin_idx] += 1

        return self

    def similarity(self, other: "DurationDistribution") -> float:
        """Compute similarity to another distribution (0-1)."""
        if self.count == 0 or other.count == 0:
            return 0.0

        # Compare means (log-scale)
        import math

        mean_sim = 1.0 / (
            1.0 + abs(math.log10(max(1, self.mean_ns)) - math.log10(max(1, other.mean_ns)))
        )

        # Compare histograms (cosine similarity)
        if self.histogram and other.histogram:
            dot = sum(a * b for a, b in zip(self.histogram, other.histogram))
            norm_a = sum(a * a for a in self.histogram) ** 0.5
            norm_b = sum(b * b for b in other.histogram) ** 0.5
            hist_sim = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0
        else:
            hist_sim = mean_sim

        return 0.6 * mean_sim + 0.4 * hist_sim


@dataclass
class GridSignature:
    """Grid/workgroup signature for a kernel."""

    grid_x: int = 0
    grid_y: int = 0
    grid_z: int = 0
    block_x: int = 0
    block_y: int = 0
    block_z: int = 0

    # Derived
    total_threads: int = 0
    total_workgroups: int = 0
    threads_per_workgroup: int = 0

    def compute(self) -> "GridSignature":
        """Compute derived values."""
        self.total_workgroups = self.grid_x * self.grid_y * self.grid_z
        self.threads_per_workgroup = self.block_x * self.block_y * self.block_z
        self.total_threads = self.total_workgroups * self.threads_per_workgroup
        return self

    def signature_hash(self) -> str:
        """Get signature hash."""
        data = (
            f"{self.grid_x}:{self.grid_y}:{self.grid_z}:"
            f"{self.block_x}:{self.block_y}:{self.block_z}"
        )
        return hashlib.md5(data.encode()).hexdigest()[:8]


@dataclass
class CounterSignature:
    """Hardware counter signature for a kernel."""

    # Memory counters
    fetch_size_bytes: int = 0
    write_size_bytes: int = 0
    l2_cache_hit_ratio: float = 0.0
    vram_read_bytes: int = 0
    vram_write_bytes: int = 0

    # Compute counters
    valu_cycles: int = 0
    salu_cycles: int = 0
    lds_cycles: int = 0

    # Occupancy
    achieved_occupancy: float = 0.0
    wavefronts_launched: int = 0

    # Derived ratios
    memory_intensity: float = 0.0  # bytes per cycle
    compute_intensity: float = 0.0  # FLOPs per byte
    is_memory_bound: bool = False
    is_compute_bound: bool = False

    def compute_intensity_metrics(self, duration_ns: float) -> "CounterSignature":
        """Compute derived intensity metrics."""
        total_bytes = self.fetch_size_bytes + self.write_size_bytes

        if duration_ns > 0:
            # Approximate cycles (assuming 1GHz)
            cycles = duration_ns
            self.memory_intensity = total_bytes / cycles if cycles > 0 else 0

        if total_bytes > 0:
            flops = self.valu_cycles + self.salu_cycles
            self.compute_intensity = flops / total_bytes

        # Classification heuristics
        self.is_memory_bound = self.memory_intensity > 10  # High memory traffic
        self.is_compute_bound = self.compute_intensity > 5  # High compute ratio

        return self


@dataclass
class KernelFingerprint:
    """Complete fingerprint for a GPU kernel."""

    # Identity
    kernel_name: str = ""
    kernel_hash: str = ""
    kernel_family_id: str = ""

    # Family classification
    family: KernelFamily = KernelFamily.UNKNOWN
    family_confidence: float = 0.0

    # Duration analysis
    duration_distribution: DurationDistribution = field(default_factory=DurationDistribution)

    # Grid/workgroup signature
    grid_signature: GridSignature = field(default_factory=GridSignature)
    grid_variance: float = 0.0  # Variance in grid sizes across invocations

    # Counter signature
    counter_signature: CounterSignature = field(default_factory=CounterSignature)

    # Launch statistics
    launch_count: int = 0
    launch_rate_hz: float = 0.0
    avg_gap_ns: float = 0.0
    gap_variance: float = 0.0

    # Stability fingerprint
    stability_score: float = 1.0
    cov_duration: float = 0.0  # Coefficient of variation

    # Family drift (compared to baseline)
    family_drift_score: float = 0.0
    family_regression_weight: float = 0.0

    def compute_hash(self) -> str:
        """Compute kernel fingerprint hash."""
        data = f"{self.kernel_name}:{self.grid_signature.signature_hash()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class KernelFamilyCluster:
    """A cluster of related kernels forming a family."""

    family_id: str = ""
    family_type: KernelFamily = KernelFamily.UNKNOWN

    # Member kernels
    member_hashes: Set[str] = field(default_factory=set)
    member_count: int = 0

    # Aggregate statistics
    total_time_ns: float = 0.0
    total_invocations: int = 0
    time_share_pct: float = 0.0

    # Representative fingerprint
    representative: Optional[KernelFingerprint] = None

    # Drift tracking
    baseline_mean_ns: float = 0.0
    current_mean_ns: float = 0.0
    drift_pct: float = 0.0
    regression_confidence: float = 0.0


class KernelFamilyClassifier:
    """
    Advanced kernel family classification using:
    - Kernel name patterns
    - Grid/block signatures
    - Counter signatures
    - Cross-run clustering
    """

    # Name patterns for family classification
    FAMILY_PATTERNS = {
        KernelFamily.GEMM: ["gemm", "matmul", "dot", "mm_", "_mm"],
        KernelFamily.CONV: ["conv", "winograd", "im2col", "col2im"],
        KernelFamily.REDUCE: ["reduce", "sum", "mean", "max_reduce", "min_reduce"],
        KernelFamily.ELEMENTWISE: [
            "elementwise",
            "ewise",
            "pointwise",
            "add_",
            "mul_",
            "relu",
        ],
        KernelFamily.ATTENTION: ["attention", "softmax_", "scaled_dot", "mha_"],
        KernelFamily.NORM: ["batch_norm", "layer_norm", "norm_", "bn_", "ln_"],
        KernelFamily.POOLING: ["pool", "maxpool", "avgpool"],
        KernelFamily.SOFTMAX: ["softmax"],
        KernelFamily.TRANSPOSE: ["transpose", "permute", "reshape"],
        KernelFamily.MEMORY: ["memcpy", "memset", "copy_", "fill_"],
    }

    def __init__(self):
        """Initialize classifier."""
        self._family_clusters: Dict[str, KernelFamilyCluster] = {}
        self._kernel_fingerprints: Dict[str, KernelFingerprint] = {}

    def classify_kernel(
        self,
        kernel_name: str,
        durations: List[float],
        grid: Optional[GridSignature] = None,
        counters: Optional[CounterSignature] = None,
    ) -> KernelFingerprint:
        """
        Classify a kernel and create fingerprint.

        Args:
            kernel_name: GPU kernel function name
            durations: List of execution durations in nanoseconds
            grid: Grid/workgroup configuration
            counters: Hardware counter values

        Returns:
            Complete kernel fingerprint
        """
        fp = KernelFingerprint(kernel_name=kernel_name)

        # Classify by name
        fp.family, fp.family_confidence = self._classify_by_name(kernel_name)

        # Compute duration distribution
        if durations:
            fp.duration_distribution.compute(durations)
            fp.launch_count = len(durations)

            # Stability metrics
            if fp.duration_distribution.mean_ns > 0:
                fp.cov_duration = (
                    fp.duration_distribution.std_dev_ns / fp.duration_distribution.mean_ns
                )
                fp.stability_score = max(0, 1.0 - fp.cov_duration)

        # Grid signature
        if grid:
            fp.grid_signature = grid.compute()

        # Counter signature
        if counters:
            fp.counter_signature = counters
            if fp.counter_signature.is_memory_bound:
                fp.family = KernelFamily.MEMORY
                fp.family_confidence = max(fp.family_confidence, 0.7)

        # Compute fingerprint hash
        fp.kernel_hash = fp.compute_hash()

        # Assign to family cluster
        fp.kernel_family_id = self._assign_to_family(fp)

        # Store fingerprint
        self._kernel_fingerprints[fp.kernel_hash] = fp

        return fp

    def _classify_by_name(self, kernel_name: str) -> Tuple[KernelFamily, float]:
        """Classify kernel family by name patterns."""
        name_lower = kernel_name.lower()

        for family, patterns in self.FAMILY_PATTERNS.items():
            for pattern in patterns:
                if pattern in name_lower:
                    # Confidence based on pattern specificity
                    confidence = 0.8 if len(pattern) > 4 else 0.6
                    return family, confidence

        return KernelFamily.UNKNOWN, 0.0

    def _assign_to_family(self, fp: KernelFingerprint) -> str:
        """Assign kernel to a family cluster."""
        # Find best matching cluster
        best_cluster = None
        best_similarity = 0.0

        for cluster_id, cluster in self._family_clusters.items():
            if cluster.family_type == fp.family and cluster.representative:
                sim = fp.duration_distribution.similarity(
                    cluster.representative.duration_distribution
                )
                if sim > best_similarity and sim > 0.7:
                    best_similarity = sim
                    best_cluster = cluster

        # Create new cluster or join existing
        if best_cluster:
            best_cluster.member_hashes.add(fp.kernel_hash)
            best_cluster.member_count += 1
            best_cluster.total_invocations += fp.launch_count
            best_cluster.total_time_ns += fp.duration_distribution.mean_ns * fp.launch_count
            return best_cluster.family_id
        else:
            # Create new cluster
            family_id = f"{fp.family.value}_{len(self._family_clusters)}"
            cluster = KernelFamilyCluster(
                family_id=family_id,
                family_type=fp.family,
                member_hashes={fp.kernel_hash},
                member_count=1,
                total_invocations=fp.launch_count,
                total_time_ns=fp.duration_distribution.mean_ns * fp.launch_count,
                representative=fp,
            )
            self._family_clusters[family_id] = cluster
            return family_id

    def compute_family_drift(
        self, baseline_fingerprints: Dict[str, KernelFingerprint]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute drift for each kernel family compared to baseline.

        Returns:
            Dict mapping family_id to drift analysis
        """
        drift_analysis = {}

        for family_id, cluster in self._family_clusters.items():
            drift = {
                "family_id": family_id,
                "family_type": cluster.family_type.value,
                "member_count": cluster.member_count,
                "has_drift": False,
                "drift_pct": 0.0,
                "regression_confidence": 0.0,
            }

            # Find matching baseline kernels
            baseline_times = []
            current_times = []

            for kernel_hash in cluster.member_hashes:
                current_fp = self._kernel_fingerprints.get(kernel_hash)
                if current_fp:
                    current_times.append(current_fp.duration_distribution.mean_ns)

                # Try to find in baseline
                for base_hash, base_fp in baseline_fingerprints.items():
                    if base_fp.kernel_name == current_fp.kernel_name:
                        baseline_times.append(base_fp.duration_distribution.mean_ns)
                        break

            # Compute drift
            if baseline_times and current_times:
                baseline_mean = statistics.mean(baseline_times)
                current_mean = statistics.mean(current_times)

                if baseline_mean > 0:
                    drift_pct = ((current_mean - baseline_mean) / baseline_mean) * 100
                    drift["drift_pct"] = drift_pct
                    drift["has_drift"] = abs(drift_pct) > 5

                    # Confidence based on sample size and consistency
                    n = min(len(baseline_times), len(current_times))
                    drift["regression_confidence"] = min(0.95, 0.5 + 0.05 * n)

                    cluster.baseline_mean_ns = baseline_mean
                    cluster.current_mean_ns = current_mean
                    cluster.drift_pct = drift_pct
                    cluster.regression_confidence = drift["regression_confidence"]

            drift_analysis[family_id] = drift

        return drift_analysis

    def get_family_summary(self) -> Dict[str, Any]:
        """Get summary of all kernel families."""
        total_time = sum(c.total_time_ns for c in self._family_clusters.values())

        families = []
        for cluster in self._family_clusters.values():
            families.append(
                {
                    "family_id": cluster.family_id,
                    "family_type": cluster.family_type.value,
                    "member_count": cluster.member_count,
                    "invocations": cluster.total_invocations,
                    "total_time_ns": cluster.total_time_ns,
                    "time_share_pct": (
                        cluster.total_time_ns / total_time * 100 if total_time > 0 else 0
                    ),
                    "drift_pct": cluster.drift_pct,
                }
            )

        # Sort by time share
        families.sort(key=lambda x: x["time_share_pct"], reverse=True)

        return {
            "total_families": len(self._family_clusters),
            "total_kernels": len(self._kernel_fingerprints),
            "total_time_ns": total_time,
            "families": families,
        }

    def export_fingerprints(self) -> Dict[str, Any]:
        """Export all fingerprints for storage."""
        return {
            "fingerprints": {
                h: {
                    "kernel_name": fp.kernel_name,
                    "kernel_hash": fp.kernel_hash,
                    "family": fp.family.value,
                    "family_id": fp.kernel_family_id,
                    "duration_mean_ns": fp.duration_distribution.mean_ns,
                    "duration_std_ns": fp.duration_distribution.std_dev_ns,
                    "launch_count": fp.launch_count,
                    "stability_score": fp.stability_score,
                }
                for h, fp in self._kernel_fingerprints.items()
            },
            "clusters": {
                c.family_id: {
                    "family_type": c.family_type.value,
                    "member_count": c.member_count,
                    "total_time_ns": c.total_time_ns,
                }
                for c in self._family_clusters.values()
            },
        }
