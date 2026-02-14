"""
AACO-SIGMA ROCm Plane KFF++ (Kernel Fingerprint Framework Enhanced)

Advanced kernel identification, fingerprinting, and family classification.
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
]
