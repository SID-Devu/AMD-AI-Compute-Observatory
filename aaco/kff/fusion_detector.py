"""
AACO-SIGMA Kernel Fusion Detector

Detects and analyzes kernel fusion patterns in execution traces.
Identifies:
- Fused kernel sequences
- Missed fusion opportunities
- Fusion efficiency metrics
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto


class FusionType(Enum):
    """Types of kernel fusion."""

    HORIZONTAL = auto()  # Same kernel repeated
    VERTICAL = auto()  # Different kernels fused into one
    MIXED = auto()  # Combination
    GRAPH_LEVEL = auto()  # Compiler-level fusion
    LIBRARY = auto()  # Library-provided fusion (e.g., fused attention)


@dataclass
class FusionPattern:
    """
    A detected or expected fusion pattern.
    """

    pattern_id: str
    fusion_type: FusionType

    # Component kernels
    kernel_names: List[str] = field(default_factory=list)
    kernel_families: List[str] = field(default_factory=list)

    # Fused kernel (if detected)
    fused_kernel_name: str = ""

    # Expected savings
    expected_speedup: float = 1.0  # e.g., 1.5x
    memory_savings_pct: float = 0.0

    # Confidence
    confidence: float = 0.0

    # Source
    source: str = ""  # "detected", "heuristic", "known"

    def __hash__(self):
        return hash(self.pattern_id)


@dataclass
class FusionCandidate:
    """
    A potential fusion opportunity (unfused kernels that could be fused).
    """

    kernels: List[str]
    families: List[str]

    # Why this is a candidate
    reason: str = ""

    # Estimated benefit
    estimated_speedup: float = 1.0

    # Complexity
    implementation_difficulty: str = "unknown"  # easy, medium, hard

    # Actionability
    suggestion: str = ""


@dataclass
class FusionMatch:
    """Result of checking for fusion in a kernel trace."""

    pattern: FusionPattern
    matched_kernels: List[Dict[str, Any]]  # The actual kernel events
    fused_kernel: Optional[Dict[str, Any]] = None  # If fusion detected

    # Metrics
    unfused_time_ns: int = 0
    fused_time_ns: int = 0

    @property
    def speedup(self) -> float:
        if self.fused_time_ns > 0:
            return self.unfused_time_ns / self.fused_time_ns
        return 1.0


class FusionDetector:
    """
    Detects kernel fusion patterns in execution traces.

    Supports:
    - Known fusion pattern matching
    - Heuristic fusion detection
    - Fusion opportunity identification
    """

    # Known fusion patterns (kernel sequence -> fused kernel)
    KNOWN_FUSIONS = {
        # Attention patterns
        ("matmul", "softmax", "matmul"): "flash_attention",
        ("qk_matmul", "softmax", "sv_matmul"): "fused_attention",
        # Normalization + activation
        ("layernorm", "gelu"): "fused_ln_gelu",
        ("layernorm", "relu"): "fused_ln_relu",
        ("batchnorm", "relu"): "fused_bn_relu",
        # Bias + activation
        ("add", "gelu"): "fused_bias_gelu",
        ("add", "relu"): "fused_bias_relu",
        ("add", "dropout"): "fused_bias_dropout",
        # GEMM + bias + activation
        ("gemm", "add", "gelu"): "fused_gemm_bias_gelu",
        ("gemm", "add", "relu"): "fused_gemm_bias_relu",
        # Residual patterns
        ("add", "layernorm"): "fused_residual_ln",
    }

    # Kernel name patterns that indicate fusion
    FUSED_INDICATORS = [
        r"fused_",
        r"_fused",
        r"flash_",
        r"_flash",
        r"_combined",
        r"combined_",
        r"merged_",
    ]

    def __init__(self):
        self._patterns: Dict[str, FusionPattern] = {}
        self._register_known_patterns()

    def _register_known_patterns(self) -> None:
        """Register known fusion patterns."""
        for kernel_tuple, fused_name in self.KNOWN_FUSIONS.items():
            pattern = FusionPattern(
                pattern_id=f"known_{fused_name}",
                fusion_type=FusionType.VERTICAL,
                kernel_families=list(kernel_tuple),
                fused_kernel_name=fused_name,
                confidence=1.0,
                source="known",
            )
            self._patterns[pattern.pattern_id] = pattern

    def detect_fusions(self, kernel_trace: List[Dict[str, Any]]) -> List[FusionMatch]:
        """
        Detect fusions in a kernel trace.

        Args:
            kernel_trace: List of kernel events with 'name', 'duration_ns', etc.

        Returns:
            List of detected fusion matches.
        """
        matches = []

        # Check for known patterns
        for pattern in self._patterns.values():
            match = self._find_pattern_match(kernel_trace, pattern)
            if match:
                matches.append(match)

        # Detect heuristic fusions
        heuristic_matches = self._detect_heuristic_fusions(kernel_trace)
        matches.extend(heuristic_matches)

        return matches

    def _find_pattern_match(
        self, trace: List[Dict[str, Any]], pattern: FusionPattern
    ) -> Optional[FusionMatch]:
        """Find a specific pattern in the trace."""
        families = pattern.kernel_families
        if not families:
            return None

        # Look for consecutive kernels matching the pattern
        for i in range(len(trace) - len(families) + 1):
            matched = True
            matched_kernels = []

            for j, family in enumerate(families):
                kernel = trace[i + j]
                kernel_name = kernel.get("name", "").lower()

                if family.lower() not in kernel_name:
                    matched = False
                    break

                matched_kernels.append(kernel)

            if matched:
                unfused_time = sum(k.get("duration_ns", 0) for k in matched_kernels)

                return FusionMatch(
                    pattern=pattern,
                    matched_kernels=matched_kernels,
                    unfused_time_ns=unfused_time,
                    fused_time_ns=0,  # Not fused yet
                )

        return None

    def _detect_heuristic_fusions(self, trace: List[Dict[str, Any]]) -> List[FusionMatch]:
        """Detect fusions using heuristics."""
        matches = []

        # Detect by fused kernel names
        for kernel in trace:
            name = kernel.get("name", "").lower()

            for indicator in self.FUSED_INDICATORS:
                if re.search(indicator, name):
                    pattern = FusionPattern(
                        pattern_id=f"heuristic_{name[:20]}",
                        fusion_type=FusionType.LIBRARY,
                        fused_kernel_name=name,
                        confidence=0.8,
                        source="heuristic",
                    )

                    match = FusionMatch(
                        pattern=pattern,
                        matched_kernels=[],  # Unknown what was fused
                        fused_kernel=kernel,
                        fused_time_ns=kernel.get("duration_ns", 0),
                    )
                    matches.append(match)
                    break

        return matches

    def find_fusion_opportunities(
        self, kernel_trace: List[Dict[str, Any]]
    ) -> List[FusionCandidate]:
        """
        Find unfused kernel sequences that could benefit from fusion.
        """
        candidates = []

        # Look for consecutive small kernels
        window_size = 3
        for i in range(len(kernel_trace) - window_size + 1):
            window = kernel_trace[i : i + window_size]

            # Check if all kernels are small (<10µs)
            all_small = all(k.get("duration_ns", 0) < 10_000 for k in window)

            if all_small:
                families = [self._infer_family(k.get("name", "")) for k in window]
                sum(k.get("duration_ns", 0) for k in window)

                candidate = FusionCandidate(
                    kernels=[k.get("name", "") for k in window],
                    families=families,
                    reason="Consecutive micro-kernels",
                    estimated_speedup=1.3,  # Conservative estimate
                    implementation_difficulty="medium",
                    suggestion=f"Consider fusing {'+'.join(families)}",
                )
                candidates.append(candidate)

        # Look for known unfused patterns
        for pattern in self._patterns.values():
            match = self._find_pattern_match(kernel_trace, pattern)
            if match and not match.fused_kernel:
                candidate = FusionCandidate(
                    kernels=[k.get("name", "") for k in match.matched_kernels],
                    families=pattern.kernel_families,
                    reason=f"Known fusion pattern: {pattern.fused_kernel_name}",
                    estimated_speedup=pattern.expected_speedup,
                    implementation_difficulty="easy",
                    suggestion=f"Use fused {pattern.fused_kernel_name} kernel",
                )
                candidates.append(candidate)

        return candidates

    def _infer_family(self, name: str) -> str:
        """Infer kernel family from name."""
        name_lower = name.lower()

        families = [
            ("gemm", ["gemm", "matmul", "mm_"]),
            ("conv", ["conv"]),
            ("attention", ["attention", "attn"]),
            ("softmax", ["softmax"]),
            ("layernorm", ["layernorm", "layer_norm", "ln_"]),
            ("gelu", ["gelu"]),
            ("relu", ["relu"]),
            ("add", ["add", "bias"]),
            ("copy", ["copy", "memcpy"]),
        ]

        for family, patterns in families:
            for pattern in patterns:
                if pattern in name_lower:
                    return family

        return "unknown"


class FusionAnalyzer:
    """
    Analyzes fusion efficiency and provides recommendations.
    """

    def __init__(self, detector: Optional[FusionDetector] = None):
        self.detector = detector or FusionDetector()

    def analyze(self, kernel_trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive fusion analysis.
        """
        # Detect existing fusions
        detected_fusions = self.detector.detect_fusions(kernel_trace)

        # Find fusion opportunities
        opportunities = self.detector.find_fusion_opportunities(kernel_trace)

        # Calculate metrics
        total_kernels = len(kernel_trace)
        total_time = sum(k.get("duration_ns", 0) for k in kernel_trace)

        # Time in fused vs unfused kernels
        fused_time = sum(m.fused_time_ns for m in detected_fusions if m.fused_kernel)

        # Estimated savings from implementing opportunities
        potential_savings = 0
        for opp in opportunities:
            kernel_time = sum(
                k.get("duration_ns", 0) for k in kernel_trace if k.get("name") in opp.kernels
            )
            savings = kernel_time * (1 - 1 / opp.estimated_speedup)
            potential_savings += savings

        return {
            "total_kernels": total_kernels,
            "total_time_ns": total_time,
            "detected_fusions": len(detected_fusions),
            "fused_time_ns": fused_time,
            "fused_time_pct": fused_time / total_time * 100 if total_time > 0 else 0,
            "fusion_opportunities": len(opportunities),
            "potential_savings_ns": potential_savings,
            "potential_savings_pct": potential_savings / total_time * 100 if total_time > 0 else 0,
            "opportunities": [
                {
                    "kernels": opp.kernels,
                    "reason": opp.reason,
                    "suggestion": opp.suggestion,
                    "estimated_speedup": opp.estimated_speedup,
                }
                for opp in opportunities[:10]  # Top 10
            ],
            "detected": [
                {
                    "pattern": m.pattern.pattern_id,
                    "type": m.pattern.fusion_type.name,
                    "speedup": m.speedup,
                }
                for m in detected_fusions
            ],
        }
