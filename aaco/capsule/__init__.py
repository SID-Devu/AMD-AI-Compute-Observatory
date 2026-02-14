"""
AACO-SIGMA Deterministic Measurement Capsules (DMC++)

Repeatability engineered, not hoped for.
- cgroup v2 isolation (cpuset, cpu.max, memory.high, io.max)
- CPU topology pinning with NUMA awareness
- Clock policy capture and enforcement
- Noise sentinels for interference detection
- Thermal/power guardrails
"""

from aaco.capsule.capsule_v2 import (
    MeasurementCapsuleV2,
    CapsuleManifestV2,
    CapsulePolicyV2,
    CapsuleHealthScore,
    IsolationLevelV2,
    run_in_capsule,
)

from aaco.capsule.topology import (
    CPUTopology,
    NUMANode,
    CoreSet,
    TopologyPolicy,
    pin_to_cores,
    isolate_cores,
)

from aaco.capsule.clock_policy import (
    ClockPolicy,
    CPUGovernorPolicy,
    GPUClockPolicy,
    ClockEnforcer,
    capture_clock_state,
    enforce_clock_policy,
)

from aaco.capsule.thermal_guard import (
    ThermalGuard,
    PowerGuard,
    ThrottleEvent,
    GuardrailViolation,
    StabilityReport,
)

from aaco.capsule.noise_sentinel_v2 import (
    NoiseSentinelV2,
    NoiseReportV2,
    NoiseEventV2,
    NoiseSourceV2,
    InterferenceClassifier,
)

__all__ = [
    # Core Capsule
    "MeasurementCapsuleV2",
    "CapsuleManifestV2",
    "CapsulePolicyV2",
    "CapsuleHealthScore",
    "IsolationLevelV2",
    "run_in_capsule",
    # Topology
    "CPUTopology",
    "NUMANode",
    "CoreSet",
    "TopologyPolicy",
    "pin_to_cores",
    "isolate_cores",
    # Clock Policy
    "ClockPolicy",
    "CPUGovernorPolicy",
    "GPUClockPolicy",
    "ClockEnforcer",
    "capture_clock_state",
    "enforce_clock_policy",
    # Thermal/Power
    "ThermalGuard",
    "PowerGuard",
    "ThrottleEvent",
    "GuardrailViolation",
    "StabilityReport",
    # Noise Sentinel
    "NoiseSentinelV2",
    "NoiseReportV2",
    "NoiseEventV2",
    "NoiseSourceV2",
    "InterferenceClassifier",
]
