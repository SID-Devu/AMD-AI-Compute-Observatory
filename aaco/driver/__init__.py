"""
AACO-SIGMA Linux Kernel Driver Module

Provides kernel-level primitives for deterministic measurement:
- High-precision TSC timestamps
- Memory barriers
- CPU affinity control
- Preemption control
- IRQ/context switch statistics
"""

from .python_bindings import (
    # IOCTL definitions
    AAcoIoctl,
    AACO_GET_VERSION,
    AACO_GET_TSC_FREQ,
    AACO_READ_TSC,
    AACO_SET_CPU_AFFINITY,
    AACO_GET_CPU_AFFINITY,
    AACO_MEMORY_BARRIER,
    AACO_GET_IRQ_STATS,
    AACO_GET_CTX_SWITCHES,
    # Data structures
    DriverVersion,
    IRQStats,
    MeasurementStats,
    # TSC utilities
    TSCReader,
    # Main driver interface
    AAcoDriver,
    # Measurement session
    KernelMeasurementSession,
    # Module utilities
    is_module_loaded,
    get_module_info,
    install_module,
)

__all__ = [
    # IOCTL definitions
    "AAcoIoctl",
    "AACO_GET_VERSION",
    "AACO_GET_TSC_FREQ",
    "AACO_READ_TSC",
    "AACO_SET_CPU_AFFINITY",
    "AACO_GET_CPU_AFFINITY",
    "AACO_MEMORY_BARRIER",
    "AACO_GET_IRQ_STATS",
    "AACO_GET_CTX_SWITCHES",
    # Data structures
    "DriverVersion",
    "IRQStats",
    "MeasurementStats",
    # TSC utilities
    "TSCReader",
    # Main driver interface
    "AAcoDriver",
    # Measurement session
    "KernelMeasurementSession",
    # Module utilities
    "is_module_loaded",
    "get_module_info",
    "install_module",
]
