"""
AACO-Λ Hardware Calibration Module.

Provides hardware envelope calibration for establishing peak capabilities.
"""

from .envelope import (
    HardwareEnvelope,
    HardwareEnvelopeCalibrator,
    BandwidthEnvelope,
    ComputeEnvelope,
    LaunchEnvelope,
    TransferEnvelope,
    GPU_SPECS,
    compare_envelopes,
    quick_calibrate,
    load_or_calibrate,
)

__all__ = [
    "HardwareEnvelope",
    "HardwareEnvelopeCalibrator",
    "BandwidthEnvelope",
    "ComputeEnvelope",
    "LaunchEnvelope",
    "TransferEnvelope",
    "GPU_SPECS",
    "compare_envelopes",
    "quick_calibrate",
    "load_or_calibrate",
]
