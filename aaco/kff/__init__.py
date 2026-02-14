"""
AACO-Ω∞ ROCm Plane KFF++ (Kernel Fingerprint Framework Enhanced)

Advanced kernel identification, fingerprinting, and family classification with:
- Dual-mode rocprof integration (v1/v2)
- Duration distribution vectors
- Hardware counter signatures
- Family clustering and drift detection
"""

from .kernel_fingerprint import (
    KernelFingerprint,
    KernelSignature,
    FingerprintGenerator,
    FingerprintMatcher,
    FingerprintDatabase,
)

from .fusion_detector import (
    FusionPattern,
    FusionDetector,
    FusionCandidate,
    FusionAnalyzer,
)

from .perf_counters import (
    HardwareCounter,
    CounterSpec,
    CounterSession,
    CounterReader,
    RocprofCounterReader,
)

from .family_classifier import (
    KernelFamily,
    FamilyClassifier,
    ClassificationResult,
    FamilyHeuristics,
)

# AACO-Ω∞ Advanced Fingerprinting
from .advanced_fingerprint import (
    DurationDistribution,
    GridSignature,
    CounterSignature,
    KernelFamilyCluster,
    KernelFamilyClassifier,
)

from .rocprof_integration import (
    RocprofMode,
    RocprofVersion,
    KernelTrace,
    CounterConfig,
    RocprofResult,
    DualModeRocprof,
    create_rocprof_collector,
)

__all__ = [
    # Kernel fingerprinting
    "KernelFingerprint",
    "KernelSignature",
    "FingerprintGenerator",
    "FingerprintMatcher",
    "FingerprintDatabase",
    # Fusion detection
    "FusionPattern",
    "FusionDetector",
    "FusionCandidate",
    "FusionAnalyzer",
    # Performance counters
    "HardwareCounter",
    "CounterSpec",
    "CounterSession",
    "CounterReader",
    "RocprofCounterReader",
    # Family classification
    "KernelFamily",
    "FamilyClassifier",
    "ClassificationResult",
    "FamilyHeuristics",
    # Advanced fingerprinting (Ω∞)
    "DurationDistribution",
    "GridSignature",
    "CounterSignature",
    "KernelFamilyCluster",
    "KernelFamilyClassifier",
    # Dual-mode rocprof (Ω∞)
    "RocprofMode",
    "RocprofVersion",
    "KernelTrace",
    "CounterConfig",
    "RocprofResult",
    "DualModeRocprof",
    "create_rocprof_collector",
]
