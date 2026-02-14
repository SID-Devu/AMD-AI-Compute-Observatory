"""
AACO-SIGMA Counter-Calibrated Attribution Module

Uses hardware performance counters to provide accurate,
deterministic performance attribution across the full stack.
"""

from .counter_model import (
    CounterBasedModel,
    PerformanceComponent,
    ResourceUtilization,
    BottleneckType,
)

from .attribution_engine import (
    AttributionEngine,
    AttributionResult,
    LayerAttribution,
    KernelAttribution,
)

from .calibration import (
    CounterCalibrator,
    CalibrationProfile,
    OverheadModel,
    CalibrationResult,
)

from .roofline import (
    RooflineModel,
    RooflinePoint,
    PerformanceBound,
    RooflineAnalyzer,
)

__all__ = [
    # Counter model
    "CounterBasedModel",
    "PerformanceComponent",
    "ResourceUtilization",
    "BottleneckType",
    # Attribution
    "AttributionEngine",
    "AttributionResult",
    "LayerAttribution",
    "KernelAttribution",
    # Calibration
    "CounterCalibrator",
    "CalibrationProfile",
    "OverheadModel",
    "CalibrationResult",
    # Roofline
    "RooflineModel",
    "RooflinePoint",
    "PerformanceBound",
    "RooflineAnalyzer",
]
