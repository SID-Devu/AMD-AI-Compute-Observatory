"""
Auto-Experiment Engine.

Automated A/B testing framework for performance experiments:
- Hypothesis generation from root cause analysis
- Experiment design (parameter variations)
- Statistical significance testing
- Cause confirmation scoring
"""

import json
import logging
import random
import statistics
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================================
# Experiment Status & Types
# ============================================================================


class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentType(str, Enum):
    """Types of experiments."""

    AB_TEST = "ab_test"  # Compare two configurations
    PARAMETER_SWEEP = "parameter_sweep"  # Sweep a parameter range
    FACTORIAL = "factorial"  # Full factorial design
    HYPOTHESIS_TEST = "hypothesis_test"  # Test specific hypothesis


# ============================================================================
# Experiment Configuration
# ============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    name: str
    experiment_type: ExperimentType

    # Baseline configuration
    baseline_params: Dict[str, Any] = field(default_factory=dict)

    # Variation(s) to test
    variations: List[Dict[str, Any]] = field(default_factory=list)

    # Statistical parameters
    min_samples: int = 10
    max_samples: int = 100
    confidence_level: float = 0.95
    min_effect_size: float = 0.05  # Minimum detectable effect

    # Execution
    warmup_runs: int = 3
    timeout_per_run_s: float = 300

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["experiment_type"] = self.experiment_type.value
        return d


# ============================================================================
# Experiment Results
# ============================================================================


@dataclass
class VariationResult:
    """Results for a single variation."""

    variation_id: str
    params: Dict[str, Any]

    # Measurements
    measurements: List[float] = field(default_factory=list)

    # Statistics
    mean: float = 0.0
    std: float = 0.0
    median: float = 0.0
    p5: float = 0.0
    p95: float = 0.0

    # Comparison to baseline
    delta_vs_baseline: float = 0.0
    delta_pct: float = 0.0
    is_significant: bool = False
    p_value: float = 1.0
    effect_size: float = 0.0

    def compute_stats(self) -> None:
        """Compute statistics from measurements."""
        if not self.measurements:
            return

        self.mean = statistics.mean(self.measurements)
        self.std = statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0
        self.median = statistics.median(self.measurements)

        sorted_m = sorted(self.measurements)
        n = len(sorted_m)
        self.p5 = sorted_m[int(n * 0.05)]
        self.p95 = sorted_m[int(n * 0.95)] if n > 1 else sorted_m[0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Complete experiment results."""

    experiment_id: str
    config: ExperimentConfig
    status: ExperimentStatus

    # Results
    baseline_result: Optional[VariationResult] = None
    variation_results: List[VariationResult] = field(default_factory=list)

    # Best performing
    best_variation_id: Optional[str] = None
    best_improvement_pct: float = 0.0

    # Statistical summary
    significant_improvements: int = 0
    significant_regressions: int = 0

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    # Hypothesis confirmation
    hypothesis_confirmed: bool = False
    confirmation_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "experiment_id": self.experiment_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "baseline_result": self.baseline_result.to_dict() if self.baseline_result else None,
            "variation_results": [v.to_dict() for v in self.variation_results],
            "best_variation_id": self.best_variation_id,
            "best_improvement_pct": self.best_improvement_pct,
            "significant_improvements": self.significant_improvements,
            "significant_regressions": self.significant_regressions,
            "duration_s": self.end_time - self.start_time,
            "hypothesis_confirmed": self.hypothesis_confirmed,
            "confirmation_score": self.confirmation_score,
        }
        return d

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# Hypothesis
# ============================================================================


@dataclass
class Hypothesis:
    """
    A testable hypothesis about performance.
    """

    hypothesis_id: str
    description: str

    # What we expect
    expected_effect: str  # "improvement", "regression", "no_change"
    expected_magnitude_pct: float = 10.0  # Expected effect size

    # Parameters to vary
    parameter: str = ""
    baseline_value: Any = None
    test_value: Any = None

    # Root cause link
    root_cause_id: Optional[str] = None

    def to_experiment_config(self, name: str) -> ExperimentConfig:
        """Convert hypothesis to experiment configuration."""
        return ExperimentConfig(
            name=name,
            experiment_type=ExperimentType.HYPOTHESIS_TEST,
            baseline_params={self.parameter: self.baseline_value},
            variations=[{self.parameter: self.test_value}],
        )


# ============================================================================
# Experiment Runner Interface
# ============================================================================


class ExperimentRunner(ABC):
    """
    Abstract interface for running experiments.

    Implementations should execute the workload with given parameters
    and return a performance measurement.
    """

    @abstractmethod
    def run(self, params: Dict[str, Any]) -> float:
        """
        Run workload with given parameters.

        Args:
            params: Parameter dictionary

        Returns:
            Performance measurement (lower is better, e.g., latency)
        """
        pass

    def warmup(self, params: Dict[str, Any], n: int = 3) -> None:
        """Run warmup iterations."""
        for _ in range(n):
            self.run(params)


class DummyRunner(ExperimentRunner):
    """Dummy runner for testing."""

    def __init__(self, baseline_mean: float = 100.0, baseline_std: float = 5.0):
        self.baseline_mean = baseline_mean
        self.baseline_std = baseline_std

    def run(self, params: Dict[str, Any]) -> float:
        # Simulate effect of parameters
        mean = self.baseline_mean

        # Simulate some parameter having an effect
        if "optimization_level" in params:
            level = params["optimization_level"]
            mean = mean * (1 - level * 0.1)  # Each level gives ~10% improvement

        return random.gauss(mean, self.baseline_std)


# ============================================================================
# Auto-Experiment Engine
# ============================================================================


class AutoExperimentEngine:
    """
    Automated experiment execution engine.

    Features:
    - Sequential sampling with early stopping
    - Statistical significance testing (Welch's t-test)
    - Effect size calculation (Cohen's d)
    - Adaptive sample sizing

    Usage:
        runner = MyBenchmarkRunner()
        engine = AutoExperimentEngine(runner)

        config = ExperimentConfig(
            name="test_batch_size",
            experiment_type=ExperimentType.AB_TEST,
            baseline_params={"batch_size": 32},
            variations=[{"batch_size": 64}],
        )

        result = engine.run_experiment(config)
        print(f"Result: {result.best_improvement_pct:.1f}% improvement")
    """

    def __init__(self, runner: ExperimentRunner, output_dir: Optional[Path] = None):
        self.runner = runner
        self.output_dir = output_dir
        self._experiment_count = 0

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a complete experiment.

        Args:
            config: Experiment configuration

        Returns:
            Experiment results with statistical analysis
        """
        self._experiment_count += 1
        exp_id = f"exp_{int(time.time())}_{self._experiment_count}"

        result = ExperimentResult(
            experiment_id=exp_id,
            config=config,
            status=ExperimentStatus.RUNNING,
            start_time=time.time(),
        )

        logger.info(f"Starting experiment: {config.name} ({exp_id})")

        try:
            # Run baseline
            logger.info("Running baseline...")
            baseline_result = self._run_variation("baseline", config.baseline_params, config)
            result.baseline_result = baseline_result

            # Run variations
            for i, variation_params in enumerate(config.variations):
                var_id = f"variation_{i}"
                logger.info(f"Running {var_id}: {variation_params}")

                var_result = self._run_variation(var_id, variation_params, config)

                # Compare to baseline
                self._compare_to_baseline(baseline_result, var_result, config)

                result.variation_results.append(var_result)

                if var_result.is_significant:
                    if var_result.delta_pct < 0:
                        result.significant_improvements += 1
                    else:
                        result.significant_regressions += 1

            # Find best variation
            if result.variation_results:
                best = min(
                    result.variation_results,
                    key=lambda v: v.delta_pct if v.is_significant else 0,
                )
                if best.is_significant and best.delta_pct < 0:
                    result.best_variation_id = best.variation_id
                    result.best_improvement_pct = -best.delta_pct

            result.status = ExperimentStatus.COMPLETED

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            result.status = ExperimentStatus.FAILED

        result.end_time = time.time()

        # Save results
        if self.output_dir:
            result.save(self.output_dir / f"{exp_id}.json")

        logger.info(f"Experiment complete: {result.status.value}")
        if result.best_variation_id:
            logger.info(
                f"Best variation: {result.best_variation_id} "
                f"({result.best_improvement_pct:.1f}% improvement)"
            )

        return result

    def run_hypothesis_test(self, hypothesis: Hypothesis) -> ExperimentResult:
        """
        Run an experiment to test a hypothesis.

        Args:
            hypothesis: Hypothesis to test

        Returns:
            Experiment result with hypothesis confirmation
        """
        config = hypothesis.to_experiment_config(f"test_{hypothesis.hypothesis_id}")
        result = self.run_experiment(config)

        # Evaluate hypothesis
        if result.variation_results:
            var = result.variation_results[0]

            if hypothesis.expected_effect == "improvement":
                # Expect improvement (lower value)
                result.hypothesis_confirmed = var.is_significant and var.delta_pct < 0
            elif hypothesis.expected_effect == "regression":
                result.hypothesis_confirmed = var.is_significant and var.delta_pct > 0
            else:  # no_change
                result.hypothesis_confirmed = not var.is_significant

            # Confirmation score based on effect size match
            if hypothesis.expected_magnitude_pct > 0:
                actual_magnitude = abs(var.delta_pct)
                expected = hypothesis.expected_magnitude_pct
                ratio = min(actual_magnitude, expected) / max(actual_magnitude, expected)
                result.confirmation_score = ratio * (1.0 if result.hypothesis_confirmed else 0.5)
            else:
                result.confirmation_score = 1.0 if result.hypothesis_confirmed else 0.0

        return result

    def run_parameter_sweep(
        self, param_name: str, values: List[Any], baseline_params: Dict[str, Any]
    ) -> ExperimentResult:
        """
        Sweep a parameter across multiple values.

        Args:
            param_name: Parameter to sweep
            values: Values to test
            baseline_params: Other parameters (constant)

        Returns:
            Experiment results for all values
        """
        config = ExperimentConfig(
            name=f"sweep_{param_name}",
            experiment_type=ExperimentType.PARAMETER_SWEEP,
            baseline_params=baseline_params,
            variations=[{**baseline_params, param_name: v} for v in values],
        )

        return self.run_experiment(config)

    def _run_variation(
        self, variation_id: str, params: Dict[str, Any], config: ExperimentConfig
    ) -> VariationResult:
        """Run a single variation and collect measurements."""
        result = VariationResult(
            variation_id=variation_id,
            params=params,
        )

        # Warmup
        self.runner.warmup(params, config.warmup_runs)

        # Collect samples with adaptive sizing
        for i in range(config.max_samples):
            measurement = self.runner.run(params)
            result.measurements.append(measurement)

            # Check if we have enough samples
            if len(result.measurements) >= config.min_samples:
                result.compute_stats()

                # Early stopping if variance is low enough
                if result.std > 0:
                    ci_width = 1.96 * result.std / np.sqrt(len(result.measurements))
                    relative_ci = ci_width / result.mean if result.mean > 0 else float("inf")

                    if relative_ci < config.min_effect_size / 2:
                        logger.debug(f"Early stopping at {len(result.measurements)} samples")
                        break

        result.compute_stats()
        return result

    def _compare_to_baseline(
        self,
        baseline: VariationResult,
        variation: VariationResult,
        config: ExperimentConfig,
    ) -> None:
        """Compare variation to baseline and compute statistical significance."""
        if not baseline.measurements or not variation.measurements:
            return

        # Welch's t-test (two-sample, unequal variance)
        t_stat, p_value = stats.ttest_ind(
            baseline.measurements, variation.measurements, equal_var=False
        )

        variation.p_value = p_value
        variation.is_significant = p_value < (1 - config.confidence_level)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((baseline.std**2 + variation.std**2) / 2)
        if pooled_std > 0:
            variation.effect_size = (variation.mean - baseline.mean) / pooled_std

        # Delta calculations
        variation.delta_vs_baseline = variation.mean - baseline.mean
        if baseline.mean > 0:
            variation.delta_pct = (variation.delta_vs_baseline / baseline.mean) * 100


# ============================================================================
# Experiment Designer
# ============================================================================


class ExperimentDesigner:
    """
    Generates experiments based on root cause analysis.

    Takes root cause suspects and generates hypotheses/experiments
    to confirm or refute them.
    """

    def __init__(self):
        self._hypothesis_templates = self._build_templates()

    def design_experiments_for_root_cause(
        self, root_cause_id: str, current_params: Dict[str, Any]
    ) -> List[ExperimentConfig]:
        """
        Design experiments to investigate a root cause.

        Args:
            root_cause_id: ID of suspected root cause
            current_params: Current workload parameters

        Returns:
            List of experiment configurations
        """
        templates = self._hypothesis_templates.get(root_cause_id, [])
        experiments = []

        for template in templates:
            config = self._instantiate_template(template, current_params)
            if config:
                experiments.append(config)

        return experiments

    def _build_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build experiment templates for each root cause."""
        return {
            "memory_bandwidth": [
                {
                    "name": "test_prefetch",
                    "param": "enable_prefetch",
                    "variations": [True, False],
                },
                {
                    "name": "test_memory_order",
                    "param": "memory_order",
                    "variations": ["row_major", "col_major"],
                },
            ],
            "launch_overhead": [
                {
                    "name": "test_kernel_fusion",
                    "param": "fuse_kernels",
                    "variations": [True, False],
                },
                {
                    "name": "test_batch_size",
                    "param": "batch_size",
                    "variations": [1, 4, 16, 64],
                },
            ],
            "low_occupancy": [
                {
                    "name": "test_block_size",
                    "param": "block_size",
                    "variations": [64, 128, 256, 512],
                },
            ],
            "compute_bound": [
                {
                    "name": "test_precision",
                    "param": "precision",
                    "variations": ["fp32", "fp16", "bf16"],
                },
            ],
        }

    def _instantiate_template(
        self, template: Dict[str, Any], current_params: Dict[str, Any]
    ) -> Optional[ExperimentConfig]:
        """Instantiate a template with current parameters."""
        param = template["param"]
        variations = template["variations"]

        # Get current value for baseline
        baseline_value = current_params.get(param, variations[0])

        # Create variation configs
        var_configs = []
        for var_value in variations:
            if var_value != baseline_value:
                var_configs.append({**current_params, param: var_value})

        if not var_configs:
            return None

        return ExperimentConfig(
            name=template["name"],
            experiment_type=ExperimentType.AB_TEST,
            baseline_params={**current_params, param: baseline_value},
            variations=var_configs,
        )


# ============================================================================
# Convenience Functions
# ============================================================================


def quick_ab_test(
    runner: ExperimentRunner,
    baseline_params: Dict[str, Any],
    variation_params: Dict[str, Any],
    n_samples: int = 30,
) -> ExperimentResult:
    """
    Quick A/B test between two configurations.

    Args:
        runner: Experiment runner
        baseline_params: Baseline configuration
        variation_params: Variation to test
        n_samples: Number of samples per configuration

    Returns:
        Experiment result
    """
    config = ExperimentConfig(
        name="quick_ab_test",
        experiment_type=ExperimentType.AB_TEST,
        baseline_params=baseline_params,
        variations=[variation_params],
        min_samples=n_samples,
        max_samples=n_samples,
    )

    engine = AutoExperimentEngine(runner)
    return engine.run_experiment(config)
