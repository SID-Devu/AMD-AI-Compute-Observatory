"""
AACO-SIGMA Constraint Solver

Solves for optimal configurations under given constraints.
Finds the best-performing configuration within hardware limits.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum, auto

from .gpu_model import GPUModel
from .simulator import PerformanceSimulator, KernelProfile, SimulationConfig


class ConstraintType(Enum):
    """Types of constraints."""

    MEMORY_LIMIT = auto()  # Max memory usage
    LATENCY_TARGET = auto()  # Target latency
    THROUGHPUT_MIN = auto()  # Minimum throughput
    POWER_BUDGET = auto()  # Power limit
    OCCUPANCY_MIN = auto()  # Minimum occupancy
    REGISTER_MAX = auto()  # Max registers per thread


@dataclass
class Constraint:
    """A constraint on the optimization."""

    name: str
    constraint_type: ConstraintType

    # Bounds
    value: float  # Target/limit value
    is_hard: bool = True  # Hard constraint vs soft preference

    # Parameters
    parameter: str = ""  # Parameter this constrains
    unit: str = ""  # Unit of measure

    def is_satisfied(self, actual: float) -> bool:
        """Check if constraint is satisfied."""
        if self.constraint_type in [
            ConstraintType.MEMORY_LIMIT,
            ConstraintType.LATENCY_TARGET,
            ConstraintType.POWER_BUDGET,
            ConstraintType.REGISTER_MAX,
        ]:
            return actual <= self.value
        else:  # MIN constraints
            return actual >= self.value

    def violation(self, actual: float) -> float:
        """Calculate constraint violation (0 if satisfied)."""
        if self.is_satisfied(actual):
            return 0.0
        return abs(actual - self.value)


@dataclass
class ConfigurationSpace:
    """Space of possible configurations to search."""

    # Batch size options
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128])

    # Block size options
    block_sizes: List[Tuple[int, int, int]] = field(
        default_factory=lambda: [
            (64, 1, 1),
            (128, 1, 1),
            (256, 1, 1),
            (512, 1, 1),
            (16, 16, 1),
            (32, 8, 1),
            (32, 32, 1),
        ]
    )

    # Precision options
    precisions: List[str] = field(default_factory=lambda: ["fp32", "fp16", "bf16", "int8"])

    # Tile sizes
    tile_sizes: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])

    # Register limits
    register_limits: List[int] = field(default_factory=lambda: [32, 48, 64, 96, 128])


@dataclass
class Configuration:
    """A specific configuration."""

    config_id: str

    # Parameters
    batch_size: int = 32
    block_size: Tuple[int, int, int] = (256, 1, 1)
    precision: str = "fp32"
    tile_size: int = 64
    register_limit: int = 64

    # Derived
    estimated_time_us: float = 0.0
    estimated_memory_mb: float = 0.0
    estimated_occupancy: float = 0.0


@dataclass
class OptimalConfig:
    """Result of constraint solving."""

    # Best configuration
    config: Optional[Configuration] = None

    # All evaluated configs
    evaluated_configs: List[Configuration] = field(default_factory=list)

    # Constraint satisfaction
    constraints_satisfied: bool = False
    violated_constraints: List[str] = field(default_factory=list)

    # Pareto front (if multi-objective)
    pareto_front: List[Configuration] = field(default_factory=list)

    # Metrics
    best_time_us: float = float("inf")
    best_throughput: float = 0.0

    # Search stats
    configs_evaluated: int = 0
    search_time_ms: float = 0.0


class ConstraintSolver:
    """
    Constraint solver for optimal configuration.

    Finds configurations that satisfy constraints while
    optimizing performance objectives.
    """

    def __init__(self, gpu_model: GPUModel):
        self.gpu = gpu_model
        self.simulator = PerformanceSimulator(SimulationConfig(target_gpu=gpu_model.gfx_version))
        self._config_counter = 0

    def solve(
        self,
        kernel_profile: KernelProfile,
        constraints: List[Constraint],
        space: Optional[ConfigurationSpace] = None,
        objective: str = "latency",
    ) -> OptimalConfig:
        """
        Solve for optimal configuration.

        Args:
            kernel_profile: Base kernel profile
            constraints: List of constraints
            space: Configuration space to search
            objective: Optimization objective ("latency" or "throughput")

        Returns:
            Optimal configuration
        """
        space = space or ConfigurationSpace()
        result = OptimalConfig()

        import time

        start_time = time.time()

        # Generate configurations
        configs = self._enumerate_configs(space, kernel_profile)

        # Evaluate each
        best_config = None
        best_score = float("inf") if objective == "latency" else 0.0

        for config in configs:
            result.configs_evaluated += 1

            # Simulate
            self._evaluate_config(config, kernel_profile)
            result.evaluated_configs.append(config)

            # Check constraints
            satisfied = all(self._check_constraint(c, config) for c in constraints)

            if satisfied:
                score = (
                    config.estimated_time_us
                    if objective == "latency"
                    else -config.estimated_time_us
                )

                if (objective == "latency" and score < best_score) or (
                    objective == "throughput" and score < best_score
                ):
                    best_score = score
                    best_config = config

        if best_config:
            result.config = best_config
            result.constraints_satisfied = True
            result.best_time_us = best_config.estimated_time_us
            result.best_throughput = (
                1e6 / best_config.estimated_time_us if best_config.estimated_time_us > 0 else 0
            )
        else:
            # Find minimum violation config
            result.config = self._find_min_violation(result.evaluated_configs, constraints)
            result.constraints_satisfied = False
            result.violated_constraints = (
                [c.name for c in constraints if not self._check_constraint(c, result.config)]
                if result.config
                else []
            )

        # Build Pareto front
        result.pareto_front = self._build_pareto_front(result.evaluated_configs)

        result.search_time_ms = (time.time() - start_time) * 1000
        return result

    def _enumerate_configs(
        self, space: ConfigurationSpace, profile: KernelProfile
    ) -> List[Configuration]:
        """Enumerate configurations from space."""
        configs = []

        # For simplicity, enumerate key combinations
        for batch_size in space.batch_sizes:
            for block_size in space.block_sizes:
                for precision in space.precisions:
                    self._config_counter += 1
                    config = Configuration(
                        config_id=f"cfg_{self._config_counter:04d}",
                        batch_size=batch_size,
                        block_size=block_size,
                        precision=precision,
                    )
                    configs.append(config)

        return configs

    def _evaluate_config(self, config: Configuration, base_profile: KernelProfile) -> None:
        """Evaluate a configuration."""
        # Modify profile based on config
        modified = KernelProfile(
            name=base_profile.name,
            flops=base_profile.flops * (config.batch_size / 32),
            ops_type=config.precision,
            bytes_read=base_profile.bytes_read * (config.batch_size / 32),
            bytes_written=base_profile.bytes_written * (config.batch_size / 32),
            block_size=config.block_size,
            registers_per_thread=config.register_limit,
        )

        # Precision scaling
        if config.precision == "fp16":
            modified.flops = int(modified.flops * 0.5)  # Same ops, faster
            modified.bytes_read = int(modified.bytes_read * 0.5)
            modified.bytes_written = int(modified.bytes_written * 0.5)
        elif config.precision == "int8":
            modified.flops = int(modified.flops * 0.25)
            modified.bytes_read = int(modified.bytes_read * 0.25)
            modified.bytes_written = int(modified.bytes_written * 0.25)

        # Simulate
        result = self.simulator.simulate_kernel(modified)

        config.estimated_time_us = result.estimated_time_us
        config.estimated_occupancy = result.occupancy

        # Memory estimate (simplified)
        config.estimated_memory_mb = (
            (base_profile.bytes_read + base_profile.bytes_written)
            * config.batch_size
            / 32
            / 1024
            / 1024
        )

    def _check_constraint(self, constraint: Constraint, config: Configuration) -> bool:
        """Check if configuration satisfies constraint."""
        if constraint.constraint_type == ConstraintType.LATENCY_TARGET:
            return config.estimated_time_us <= constraint.value
        elif constraint.constraint_type == ConstraintType.MEMORY_LIMIT:
            return config.estimated_memory_mb <= constraint.value
        elif constraint.constraint_type == ConstraintType.OCCUPANCY_MIN:
            return config.estimated_occupancy >= constraint.value
        elif constraint.constraint_type == ConstraintType.REGISTER_MAX:
            return config.register_limit <= constraint.value
        return True

    def _find_min_violation(
        self, configs: List[Configuration], constraints: List[Constraint]
    ) -> Optional[Configuration]:
        """Find config with minimum total constraint violation."""
        if not configs:
            return None

        best_config = None
        best_violation = float("inf")

        for config in configs:
            total_violation = 0.0
            for c in constraints:
                if c.constraint_type == ConstraintType.LATENCY_TARGET:
                    total_violation += max(0, config.estimated_time_us - c.value)
                elif c.constraint_type == ConstraintType.MEMORY_LIMIT:
                    total_violation += max(0, config.estimated_memory_mb - c.value) * 100
                elif c.constraint_type == ConstraintType.OCCUPANCY_MIN:
                    total_violation += max(0, c.value - config.estimated_occupancy) * 100

            if total_violation < best_violation:
                best_violation = total_violation
                best_config = config

        return best_config

    def _build_pareto_front(self, configs: List[Configuration]) -> List[Configuration]:
        """Build Pareto front of non-dominated solutions."""
        if not configs:
            return []

        pareto = []

        for config in configs:
            dominated = False
            for other in configs:
                if other is config:
                    continue
                # Check if other dominates config (better on all objectives)
                if (
                    other.estimated_time_us <= config.estimated_time_us
                    and other.estimated_memory_mb <= config.estimated_memory_mb
                    and other.estimated_occupancy >= config.estimated_occupancy
                ):
                    # Check strict improvement on at least one
                    if (
                        other.estimated_time_us < config.estimated_time_us
                        or other.estimated_memory_mb < config.estimated_memory_mb
                        or other.estimated_occupancy > config.estimated_occupancy
                    ):
                        dominated = True
                        break

            if not dominated:
                pareto.append(config)

        return pareto

    def sensitivity_analysis(
        self, kernel_profile: KernelProfile, parameter: str, values: List[Any]
    ) -> Dict[Any, float]:
        """
        Analyze sensitivity to a parameter.

        Args:
            kernel_profile: Base kernel profile
            parameter: Parameter to vary
            values: Values to test

        Returns:
            Dict mapping parameter value to estimated time
        """
        results = {}

        for value in values:
            config = Configuration(
                config_id=f"sens_{parameter}_{value}",
            )

            # Set parameter
            if parameter == "batch_size":
                config.batch_size = value
            elif parameter == "block_size":
                config.block_size = value
            elif parameter == "precision":
                config.precision = value
            elif parameter == "register_limit":
                config.register_limit = value

            self._evaluate_config(config, kernel_profile)
            results[value] = config.estimated_time_us

        return results
