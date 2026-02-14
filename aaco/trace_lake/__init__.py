# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Unified Trace Lake

Perfetto-compatible trace integration with:
- CPU scheduler events
- GPU kernel execution
- Power/thermal markers
- Anomaly annotations
- Cross-layer correlation
"""

from aaco.trace_lake.unified_trace import (
    TraceCategory,
    EventPhase,
    TraceEvent,
    AnomalyMarker,
    TraceLakeConfig,
    UnifiedTraceLake,
    create_trace_lake,
)

__all__ = [
    "TraceCategory",
    "EventPhase",
    "TraceEvent",
    "AnomalyMarker",
    "TraceLakeConfig",
    "UnifiedTraceLake",
    "create_trace_lake",
]
