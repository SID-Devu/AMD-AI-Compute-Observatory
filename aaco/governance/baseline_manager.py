"""
AACO-SIGMA Baseline Manager

Manages performance baselines for regression comparison.
Versioned storage and comparison of baseline snapshots.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import time
import json
import hashlib
import statistics


@dataclass
class BaselineMetric:
    """A single metric in a baseline."""

    name: str
    mean: float
    stddev: float
    min_value: float
    max_value: float
    sample_count: int
    unit: str = ""


@dataclass
class BaselineVersion:
    """Version metadata for a baseline."""

    version: str
    created_at: float
    created_by: str = ""

    # Git info
    git_commit: str = ""
    git_branch: str = ""

    # Environment
    gpu_model: str = ""
    driver_version: str = ""
    rocm_version: str = ""

    # Tags
    tags: List[str] = field(default_factory=list)

    # Validity
    is_active: bool = True
    superseded_by: str = ""


@dataclass
class PerformanceBaseline:
    """A performance baseline snapshot."""

    # Identity
    name: str
    model_name: str
    version: BaselineVersion = field(
        default_factory=lambda: BaselineVersion(version="1.0.0", created_at=time.time())
    )

    # Metrics
    metrics: Dict[str, BaselineMetric] = field(default_factory=dict)

    # Kernel-level baselines
    kernel_metrics: Dict[str, Dict[str, BaselineMetric]] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Hash of contents
    content_hash: str = ""

    def add_metric(self, name: str, values: List[float], unit: str = "") -> None:
        """Add a metric from a list of values."""
        if not values:
            return

        self.metrics[name] = BaselineMetric(
            name=name,
            mean=statistics.mean(values),
            stddev=statistics.stdev(values) if len(values) > 1 else 0.0,
            min_value=min(values),
            max_value=max(values),
            sample_count=len(values),
            unit=unit,
        )

    def get_metric(self, name: str) -> Optional[BaselineMetric]:
        """Get a metric by name."""
        return self.metrics.get(name)

    def compute_hash(self) -> str:
        """Compute content hash for integrity."""
        content = {
            "name": self.name,
            "model_name": self.model_name,
            "metrics": {k: {"mean": v.mean, "stddev": v.stddev} for k, v in self.metrics.items()},
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]


@dataclass
class BaselineComparison:
    """Comparison between two baselines."""

    baseline_a: str  # Name/version
    baseline_b: str

    # Per-metric comparisons
    metric_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # metric_deltas[name] = {"delta_pct": X, "delta_abs": Y}

    # Summary
    improved_count: int = 0
    regressed_count: int = 0
    unchanged_count: int = 0

    # Significant changes
    significant_improvements: List[str] = field(default_factory=list)
    significant_regressions: List[str] = field(default_factory=list)


class BaselineManager:
    """
    Manages collection of performance baselines.

    Features:
    - Versioned baseline storage
    - Comparison between baselines
    - Baseline selection (latest, specific version)
    - Storage persistence
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        self.baselines: Dict[
            str, Dict[str, PerformanceBaseline]
        ] = {}  # name -> version -> baseline
        self.active_baselines: Dict[str, str] = {}  # name -> active version

        if storage_path:
            self._load_from_storage()

    def create_baseline(
        self,
        name: str,
        model_name: str,
        metrics: Dict[str, List[float]],
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PerformanceBaseline:
        """
        Create a new baseline from metric measurements.

        Args:
            name: Baseline name (e.g., "llama2-7b-inference")
            model_name: Model identifier
            metrics: Dict of metric name -> list of values
            version: Version string (auto-generated if not provided)
            metadata: Additional metadata
        """
        # Auto-generate version
        if version is None:
            existing = self.baselines.get(name, {})
            version = f"1.0.{len(existing)}"

        baseline = PerformanceBaseline(
            name=name,
            model_name=model_name,
            version=BaselineVersion(
                version=version,
                created_at=time.time(),
            ),
            metadata=metadata or {},
        )

        # Add metrics
        for metric_name, values in metrics.items():
            baseline.add_metric(metric_name, values)

        # Compute hash
        baseline.content_hash = baseline.compute_hash()

        # Store
        if name not in self.baselines:
            self.baselines[name] = {}
        self.baselines[name][version] = baseline

        # Set as active
        self.active_baselines[name] = version

        # Persist
        if self.storage_path:
            self._save_baseline(baseline)

        return baseline

    def get_baseline(
        self, name: str, version: Optional[str] = None
    ) -> Optional[PerformanceBaseline]:
        """
        Get a baseline by name and optional version.

        Args:
            name: Baseline name
            version: Specific version (default: active version)
        """
        if name not in self.baselines:
            return None

        if version is None:
            version = self.active_baselines.get(name)

        if version is None:
            return None

        return self.baselines[name].get(version)

    def get_active_baseline(self, name: str) -> Optional[PerformanceBaseline]:
        """Get the currently active baseline for a name."""
        return self.get_baseline(name)

    def set_active_version(self, name: str, version: str) -> bool:
        """Set the active version for a baseline."""
        if name not in self.baselines or version not in self.baselines[name]:
            return False

        # Mark old active as superseded
        old_version = self.active_baselines.get(name)
        if old_version and old_version in self.baselines[name]:
            self.baselines[name][old_version].version.is_active = False

        self.active_baselines[name] = version
        self.baselines[name][version].version.is_active = True
        return True

    def list_baselines(self, name: Optional[str] = None) -> List[Tuple[str, str]]:
        """List all baselines (name, version pairs)."""
        result = []

        if name:
            if name in self.baselines:
                for version in self.baselines[name]:
                    result.append((name, version))
        else:
            for baseline_name, versions in self.baselines.items():
                for version in versions:
                    result.append((baseline_name, version))

        return result

    def compare(
        self,
        baseline_a: PerformanceBaseline,
        baseline_b: PerformanceBaseline,
        significance_threshold_pct: float = 5.0,
    ) -> BaselineComparison:
        """
        Compare two baselines.

        Args:
            baseline_a: First baseline (typically older/reference)
            baseline_b: Second baseline (typically newer/current)
            significance_threshold_pct: Threshold for significant change
        """
        comparison = BaselineComparison(
            baseline_a=f"{baseline_a.name}@{baseline_a.version.version}",
            baseline_b=f"{baseline_b.name}@{baseline_b.version.version}",
        )

        # Compare all metrics in baseline_a
        all_metrics = set(baseline_a.metrics.keys()) | set(baseline_b.metrics.keys())

        for metric_name in all_metrics:
            metric_a = baseline_a.metrics.get(metric_name)
            metric_b = baseline_b.metrics.get(metric_name)

            if metric_a is None or metric_b is None:
                continue

            # Calculate delta
            delta_abs = metric_b.mean - metric_a.mean
            delta_pct = (delta_abs / metric_a.mean * 100) if metric_a.mean != 0 else 0

            comparison.metric_deltas[metric_name] = {
                "delta_abs": delta_abs,
                "delta_pct": delta_pct,
                "a_mean": metric_a.mean,
                "b_mean": metric_b.mean,
            }

            # Classify change
            if abs(delta_pct) < significance_threshold_pct:
                comparison.unchanged_count += 1
            elif delta_pct > 0:
                # For most metrics, increase is regression (latency, memory)
                comparison.regressed_count += 1
                if abs(delta_pct) >= significance_threshold_pct:
                    comparison.significant_regressions.append(metric_name)
            else:
                comparison.improved_count += 1
                if abs(delta_pct) >= significance_threshold_pct:
                    comparison.significant_improvements.append(metric_name)

        return comparison

    def _save_baseline(self, baseline: PerformanceBaseline) -> None:
        """Save baseline to storage."""
        if not self.storage_path:
            return

        self.storage_path.mkdir(parents=True, exist_ok=True)

        filepath = self.storage_path / f"{baseline.name}_{baseline.version.version}.json"

        data = {
            "name": baseline.name,
            "model_name": baseline.model_name,
            "version": {
                "version": baseline.version.version,
                "created_at": baseline.version.created_at,
                "is_active": baseline.version.is_active,
            },
            "metrics": {
                k: {
                    "mean": v.mean,
                    "stddev": v.stddev,
                    "min": v.min_value,
                    "max": v.max_value,
                    "count": v.sample_count,
                    "unit": v.unit,
                }
                for k, v in baseline.metrics.items()
            },
            "content_hash": baseline.content_hash,
            "metadata": baseline.metadata,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _load_from_storage(self) -> None:
        """Load baselines from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return

        for filepath in self.storage_path.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)

                baseline = PerformanceBaseline(
                    name=data["name"],
                    model_name=data["model_name"],
                    version=BaselineVersion(
                        version=data["version"]["version"],
                        created_at=data["version"]["created_at"],
                        is_active=data["version"].get("is_active", True),
                    ),
                    content_hash=data.get("content_hash", ""),
                    metadata=data.get("metadata", {}),
                )

                for k, v in data.get("metrics", {}).items():
                    baseline.metrics[k] = BaselineMetric(
                        name=k,
                        mean=v["mean"],
                        stddev=v["stddev"],
                        min_value=v["min"],
                        max_value=v["max"],
                        sample_count=v["count"],
                        unit=v.get("unit", ""),
                    )

                # Store
                if baseline.name not in self.baselines:
                    self.baselines[baseline.name] = {}
                self.baselines[baseline.name][baseline.version.version] = baseline

                if baseline.version.is_active:
                    self.active_baselines[baseline.name] = baseline.version.version

            except Exception:
                continue  # Skip invalid files

    def export_comparison_report(self, comparison: BaselineComparison) -> Dict[str, Any]:
        """Export comparison as a report."""
        return {
            "baseline_a": comparison.baseline_a,
            "baseline_b": comparison.baseline_b,
            "summary": {
                "improved": comparison.improved_count,
                "regressed": comparison.regressed_count,
                "unchanged": comparison.unchanged_count,
            },
            "significant_improvements": comparison.significant_improvements,
            "significant_regressions": comparison.significant_regressions,
            "metric_details": comparison.metric_deltas,
        }
