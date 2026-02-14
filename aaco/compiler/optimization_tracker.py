"""
AACO-SIGMA Optimization Tracker

Tracks compiler optimization passes and their effects on generated code.
Provides visibility into which optimizations were applied and their impact.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from enum import Enum, auto


class PassCategory(Enum):
    """Categories of compiler passes."""

    CANONICALIZATION = auto()  # Simplification and normalization
    FUSION = auto()  # Operation fusion
    TILING = auto()  # Loop tiling
    VECTORIZATION = auto()  # Vector operations
    MEMORY = auto()  # Memory optimizations
    LOWERING = auto()  # Dialect lowering
    CLEANUP = auto()  # Dead code, CSE, etc.
    GPU = auto()  # GPU-specific passes


@dataclass
class PassResult:
    """Result of running an optimization pass."""

    pass_name: str
    success: bool
    changed: bool = False  # Did the pass modify the IR?

    # Timing
    duration_us: int = 0

    # Statistics
    patterns_matched: int = 0
    transformations_applied: int = 0

    # Before/after metrics
    metrics_before: Dict[str, int] = field(default_factory=dict)
    metrics_after: Dict[str, int] = field(default_factory=dict)

    # Error info
    error_message: str = ""

    def get_delta(self, metric: str) -> int:
        """Get change in a metric."""
        before = self.metrics_before.get(metric, 0)
        after = self.metrics_after.get(metric, 0)
        return after - before


@dataclass
class OptimizationPass:
    """Description of an optimization pass."""

    name: str
    category: PassCategory
    description: str = ""

    # Ordering
    depends_on: List[str] = field(default_factory=list)

    # Configuration
    enabled: bool = True
    arguments: Dict[str, Any] = field(default_factory=dict)

    # Statistics across runs
    total_runs: int = 0
    total_changes: int = 0
    total_time_us: int = 0

    def average_time_us(self) -> float:
        """Get average execution time."""
        if self.total_runs == 0:
            return 0.0
        return self.total_time_us / self.total_runs


class PassPipeline:
    """A sequence of optimization passes."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.passes: List[OptimizationPass] = []
        self.results: List[PassResult] = []

    def add_pass(self, opt_pass: OptimizationPass) -> None:
        """Add a pass to the pipeline."""
        self.passes.append(opt_pass)

    def get_pass(self, name: str) -> Optional[OptimizationPass]:
        """Get pass by name."""
        for p in self.passes:
            if p.name == name:
                return p
        return None

    def get_enabled_passes(self) -> List[OptimizationPass]:
        """Get list of enabled passes."""
        return [p for p in self.passes if p.enabled]

    def get_results_by_category(self, category: PassCategory) -> List[PassResult]:
        """Get results for passes in a category."""
        category_passes = {p.name for p in self.passes if p.category == category}
        return [r for r in self.results if r.pass_name in category_passes]


class OptimizationTracker:
    """
    Tracks optimization passes during compilation.

    Provides:
    - Pass execution history
    - IR changes per pass
    - Performance impact analysis
    - Optimization recommendations
    """

    # Standard AMD GPU optimization passes
    STANDARD_PASSES = [
        OptimizationPass("canonicalize", PassCategory.CANONICALIZATION),
        OptimizationPass("cse", PassCategory.CLEANUP, "Common subexpression elimination"),
        OptimizationPass("dce", PassCategory.CLEANUP, "Dead code elimination"),
        OptimizationPass("loop-fusion", PassCategory.FUSION),
        OptimizationPass("linalg-fusion", PassCategory.FUSION),
        OptimizationPass("linalg-tiling", PassCategory.TILING),
        OptimizationPass("vectorize", PassCategory.VECTORIZATION),
        OptimizationPass("buffer-hoisting", PassCategory.MEMORY),
        OptimizationPass("promote-buffers-to-stack", PassCategory.MEMORY),
        OptimizationPass("gpu-map-parallel-loops", PassCategory.GPU),
        OptimizationPass("gpu-kernel-outlining", PassCategory.GPU),
        OptimizationPass("convert-linalg-to-gpu", PassCategory.LOWERING),
        OptimizationPass("convert-scf-to-gpu", PassCategory.LOWERING),
        OptimizationPass("convert-gpu-to-rocdl", PassCategory.LOWERING),
    ]

    def __init__(self):
        self.pipelines: Dict[str, PassPipeline] = {}
        self.active_pipeline: Optional[PassPipeline] = None
        self._hooks: Dict[str, List[Callable]] = {
            "before_pass": [],
            "after_pass": [],
            "pipeline_complete": [],
        }

    def create_pipeline(self, name: str, include_standard: bool = True) -> PassPipeline:
        """Create a new optimization pipeline."""
        pipeline = PassPipeline(name)

        if include_standard:
            for std_pass in self.STANDARD_PASSES:
                # Create copy to allow independent configuration
                pipeline.add_pass(
                    OptimizationPass(
                        name=std_pass.name,
                        category=std_pass.category,
                        description=std_pass.description,
                    )
                )

        self.pipelines[name] = pipeline
        return pipeline

    def start_pipeline(self, name: str) -> None:
        """Start tracking a pipeline execution."""
        if name not in self.pipelines:
            self.create_pipeline(name)
        self.active_pipeline = self.pipelines[name]
        self.active_pipeline.results = []

    def record_pass(self, result: PassResult) -> None:
        """Record a pass execution result."""
        if self.active_pipeline is None:
            return

        self.active_pipeline.results.append(result)

        # Update pass statistics
        opt_pass = self.active_pipeline.get_pass(result.pass_name)
        if opt_pass:
            opt_pass.total_runs += 1
            if result.changed:
                opt_pass.total_changes += 1
            opt_pass.total_time_us += result.duration_us

        # Call hooks
        for hook in self._hooks["after_pass"]:
            hook(result)

    def finish_pipeline(self) -> Dict[str, Any]:
        """Finish pipeline tracking and return summary."""
        if self.active_pipeline is None:
            return {}

        summary = self.get_pipeline_summary(self.active_pipeline)

        for hook in self._hooks["pipeline_complete"]:
            hook(summary)

        self.active_pipeline = None
        return summary

    def get_pipeline_summary(self, pipeline: PassPipeline) -> Dict[str, Any]:
        """Get summary statistics for a pipeline."""
        if not pipeline.results:
            return {}

        total_time = sum(r.duration_us for r in pipeline.results)
        passes_changed = sum(1 for r in pipeline.results if r.changed)

        # Aggregate metrics
        final_metrics: Dict[str, int] = {}
        if pipeline.results:
            final_metrics = pipeline.results[-1].metrics_after.copy()

        # Category breakdown
        category_times: Dict[str, int] = {}
        for category in PassCategory:
            cat_results = pipeline.get_results_by_category(category)
            category_times[category.name] = sum(r.duration_us for r in cat_results)

        return {
            "pipeline_name": pipeline.name,
            "total_passes": len(pipeline.results),
            "passes_changed_ir": passes_changed,
            "total_time_us": total_time,
            "category_breakdown_us": category_times,
            "final_metrics": final_metrics,
            "pass_results": [
                {
                    "name": r.pass_name,
                    "changed": r.changed,
                    "duration_us": r.duration_us,
                    "transformations": r.transformations_applied,
                }
                for r in pipeline.results
            ],
        }

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a callback hook."""
        if event in self._hooks:
            self._hooks[event].append(callback)

    def compare_pipelines(self, pipeline_a: str, pipeline_b: str) -> Dict[str, Any]:
        """Compare two pipeline executions."""
        pa = self.pipelines.get(pipeline_a)
        pb = self.pipelines.get(pipeline_b)

        if not pa or not pb:
            return {}

        return {
            "pipeline_a": pipeline_a,
            "pipeline_b": pipeline_b,
            "time_diff_us": sum(r.duration_us for r in pb.results)
            - sum(r.duration_us for r in pa.results),
            "passes_diff": len(pb.results) - len(pa.results),
        }


class PassPipelineAnalyzer:
    """
    Analyzes optimization pass pipelines to identify issues and opportunities.
    """

    def __init__(self, tracker: OptimizationTracker):
        self.tracker = tracker

    def find_ineffective_passes(self, pipeline: PassPipeline, min_runs: int = 3) -> List[str]:
        """Find passes that rarely change the IR."""
        ineffective = []

        for opt_pass in pipeline.passes:
            if opt_pass.total_runs >= min_runs:
                change_rate = opt_pass.total_changes / opt_pass.total_runs
                if change_rate < 0.1:  # Less than 10% effectiveness
                    ineffective.append(opt_pass.name)

        return ineffective

    def find_expensive_passes(
        self, pipeline: PassPipeline, threshold_pct: float = 20.0
    ) -> List[str]:
        """Find passes that take disproportionate time."""
        total_time = sum(r.duration_us for r in pipeline.results)
        if total_time == 0:
            return []

        expensive = []
        for result in pipeline.results:
            pct = (result.duration_us / total_time) * 100
            if pct > threshold_pct:
                expensive.append(result.pass_name)

        return expensive

    def suggest_optimizations(self, pipeline: PassPipeline) -> List[str]:
        """Suggest pipeline optimizations."""
        suggestions = []

        # Check for missed fusion opportunities
        fusion_results = pipeline.get_results_by_category(PassCategory.FUSION)
        if fusion_results:
            total_fusions = sum(r.transformations_applied for r in fusion_results)
            if total_fusions == 0:
                suggestions.append(
                    "No fusion transformations applied. Consider reviewing "
                    "fusion patterns for this workload."
                )

        # Check for vectorization
        vec_results = pipeline.get_results_by_category(PassCategory.VECTORIZATION)
        if vec_results:
            total_vecs = sum(r.transformations_applied for r in vec_results)
            if total_vecs == 0:
                suggestions.append(
                    "Vectorization pass made no changes. Loops may not be "
                    "vectorizable or already optimal."
                )

        # Check tiling
        tile_results = pipeline.get_results_by_category(PassCategory.TILING)
        if not any(r.changed for r in tile_results):
            suggestions.append(
                "No tiling transformations applied. Consider manual tiling "
                "annotations for better cache utilization."
            )

        return suggestions

    def analyze_for_gpu(self, pipeline: PassPipeline) -> Dict[str, Any]:
        """Analyze pipeline specifically for GPU compilation."""
        gpu_results = pipeline.get_results_by_category(PassCategory.GPU)

        kernel_outlined = any(
            r.pass_name == "gpu-kernel-outlining" and r.changed for r in gpu_results
        )

        parallel_mapped = any(
            r.pass_name == "gpu-map-parallel-loops" and r.changed for r in gpu_results
        )

        return {
            "gpu_passes_run": len(gpu_results),
            "kernel_outlined": kernel_outlined,
            "parallel_loops_mapped": parallel_mapped,
            "gpu_time_us": sum(r.duration_us for r in gpu_results),
            "recommendations": self._gpu_recommendations(kernel_outlined, parallel_mapped),
        }

    def _gpu_recommendations(self, kernel_outlined: bool, parallel_mapped: bool) -> List[str]:
        """Generate GPU-specific recommendations."""
        recs = []

        if not kernel_outlined:
            recs.append(
                "No GPU kernels were outlined. Ensure compute operations "
                "are marked for GPU execution."
            )

        if not parallel_mapped:
            recs.append(
                "Parallel loops not mapped to GPU. Check that loops have "
                "sufficient iteration count for GPU parallelization."
            )

        return recs
