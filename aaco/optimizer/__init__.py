# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Auto-Optimization Engine

Closed-loop self-experimental optimization with:
- Hypothesis generation from bottleneck analysis
- A/B testing with statistical validation
- Automatic rollback on degradation
- Optimization history tracking
"""

from aaco.optimizer.auto_optimizer import (
    OptimizationType,
    ExperimentStatus,
    OptimizationHypothesis,
    ExperimentConfig,
    ExperimentResult,
    AutoOptimizationEngine,
    create_auto_optimizer,
)

__all__ = [
    "OptimizationType",
    "ExperimentStatus",
    "OptimizationHypothesis",
    "ExperimentConfig",
    "ExperimentResult",
    "AutoOptimizationEngine",
    "create_auto_optimizer",
]
