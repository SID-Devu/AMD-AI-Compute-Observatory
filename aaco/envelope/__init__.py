"""
AACO-SIGMA Hardware Envelope Digital Twin

Models hardware capabilities and performance boundaries.
Enables simulation and prediction without running on actual hardware.
"""

from .gpu_model import (
    GPUModel,
    GPUArchitecture,
    GPUSpecs,
    ComputeCapability,
)
from .envelope import (
    PerformanceEnvelope,
    EnvelopeBoundary,
    EnvelopePoint,
    EnvelopeRegion,
)
from .simulator import (
    PerformanceSimulator,
    SimulationResult,
    SimulationConfig,
)
from .constraint_solver import (
    ConstraintSolver,
    Constraint,
    OptimalConfig,
)

__all__ = [
    # GPU Model
    "GPUModel",
    "GPUArchitecture",
    "GPUSpecs",
    "ComputeCapability",
    # Envelope
    "PerformanceEnvelope",
    "EnvelopeBoundary",
    "EnvelopePoint",
    "EnvelopeRegion",
    # Simulator
    "PerformanceSimulator",
    "SimulationResult",
    "SimulationConfig",
    # Solver
    "ConstraintSolver",
    "Constraint",
    "OptimalConfig",
]
