# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Hardware-Calibrated Digital Twin

Calibration-based hardware envelope with:
- Microbenchmark suite
- Calibrated ceiling measurements
- Hardware Envelope Utilization (HEU) scoring
"""

from aaco.twin.hardware_envelope import (
    BenchmarkType,
    CalibrationSample,
    HardwareEnvelope,
    CalibrationResult,
    MicrobenchmarkSuite,
    DigitalTwinCalibrator,
    create_digital_twin,
)

__all__ = [
    "BenchmarkType",
    "CalibrationSample",
    "HardwareEnvelope",
    "CalibrationResult",
    "MicrobenchmarkSuite",
    "DigitalTwinCalibrator",
    "create_digital_twin",
]
