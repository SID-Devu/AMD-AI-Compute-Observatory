"""
AACO-Λ Auto-Experiment Module.

Provides automated A/B testing and experiment framework.
"""

from .auto_tune import (
    AutoExperimentEngine,
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    ExperimentStatus,
    ExperimentType,
    ExperimentDesigner,
    Hypothesis,
    VariationResult,
    DummyRunner,
    quick_ab_test,
)

__all__ = [
    "AutoExperimentEngine",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "ExperimentStatus",
    "ExperimentType",
    "ExperimentDesigner",
    "Hypothesis",
    "VariationResult",
    "DummyRunner",
    "quick_ab_test",
]
