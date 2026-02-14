"""
AACO-SIGMA Kernel Family Classifier

Classifies GPU kernels into semantic families for analysis.
Uses name patterns, launch config, and behavior to classify.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class KernelFamily(Enum):
    """Kernel family classification."""

    # Compute-bound families
    GEMM = "gemm"
    CONV = "conv"
    ATTENTION = "attention"

    # Memory-bound families
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    SOFTMAX = "softmax"
    NORMALIZATION = "normalization"

    # Data movement
    COPY = "copy"
    TRANSPOSE = "transpose"
    EMBEDDING = "embedding"

    # Utility
    INIT = "init"
    FILL = "fill"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of kernel classification."""

    kernel_name: str
    family: KernelFamily
    confidence: float = 1.0

    # Sub-classification
    sub_family: str = ""
    variant: str = ""

    # Heuristics used
    classification_method: str = ""  # pattern, config, behavior
    matched_patterns: List[str] = field(default_factory=list)

    # Characteristics
    is_compute_bound: bool = False
    is_memory_bound: bool = False
    is_microkernel: bool = False

    # Expected behavior
    expected_roofline_region: str = ""  # compute, memory, balanced


@dataclass
class FamilyHeuristics:
    """
    Heuristics for kernel family classification.
    """

    patterns: Dict[str, List[str]] = field(default_factory=dict)
    config_rules: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize default patterns
        if not self.patterns:
            self.patterns = self._default_patterns()
        if not self.config_rules:
            self.config_rules = self._default_config_rules()

    def _default_patterns(self) -> Dict[str, List[str]]:
        """Default name patterns for each family."""
        return {
            KernelFamily.GEMM.value: [
                r"\bgemm\b",
                r"\bmatmul\b",
                r"matrix_mult",
                r"_mm_",
                r"\bsgemm\b",
                r"\bdgemm\b",
                r"\bhgemm\b",
                r"rocblas.*gemm",
                r"ck::.*Gemm",
                r"cutlass.*gemm",
                r"cublas.*gemm",
            ],
            KernelFamily.CONV.value: [
                r"\bconv\b",
                r"convolution",
                r"winograd",
                r"im2col",
                r"miopen.*conv",
                r"ck::.*Conv",
                r"cudnn.*conv",
            ],
            KernelFamily.ATTENTION.value: [
                r"attention",
                r"self_attn",
                r"cross_attn",
                r"multi_head",
                r"flash.*attn",
                r"scaled_dot",
                r"qkv_",
                r"_qk_",
                r"_sv_",
            ],
            KernelFamily.SOFTMAX.value: [
                r"\bsoftmax\b",
                r"soft_max",
                r"log_softmax",
            ],
            KernelFamily.NORMALIZATION.value: [
                r"layernorm",
                r"batchnorm",
                r"groupnorm",
                r"rmsnorm",
                r"instancenorm",
                r"layer_norm",
                r"batch_norm",
                r"_ln_",
                r"_bn_",
            ],
            KernelFamily.ELEMENTWISE.value: [
                r"\brelu\b",
                r"\bgelu\b",
                r"\bsilu\b",
                r"\bsigmoid\b",
                r"\btanh\b",
                r"\bswish\b",
                r"activation",
                r"\badd\b",
                r"\bmul\b",
                r"\bdiv\b",
                r"\bsub\b",
                r"fused_.*add",
                r"bias_",
                r"_bias",
                r"element_?wise",
                r"point_?wise",
            ],
            KernelFamily.REDUCTION.value: [
                r"\breduce\b",
                r"\bsum\b",
                r"\bmean\b",
                r"\bmax\b",
                r"\bmin\b",
                r"argmax",
                r"argmin",
                r"all_reduce",
            ],
            KernelFamily.COPY.value: [
                r"\bcopy\b",
                r"memcpy",
                r"memset",
                r"_copy_",
                r"HtoD",
                r"DtoH",
                r"DtoD",
                r"async_copy",
            ],
            KernelFamily.TRANSPOSE.value: [
                r"\btranspose\b",
                r"\bpermute\b",
                r"_transpose",
            ],
            KernelFamily.EMBEDDING.value: [
                r"\bembed\b",
                r"embedding",
                r"lookup",
                r"gather",
                r"index_select",
                r"scatter",
            ],
            KernelFamily.INIT.value: [
                r"\binit\b",
                r"initialize",
                r"_init_",
            ],
            KernelFamily.FILL.value: [
                r"\bfill\b",
                r"\bzero\b",
                r"\bones\b",
                r"constant",
            ],
        }

    def _default_config_rules(self) -> Dict[str, Any]:
        """Default launch config rules for families."""
        return {
            KernelFamily.GEMM.value: {
                "min_grid_size": 64,
                "typical_block_size": [256, 128, 64],
                "compute_bound": True,
            },
            KernelFamily.CONV.value: {
                "min_grid_size": 64,
                "typical_block_size": [256, 128],
                "compute_bound": True,
            },
            KernelFamily.ELEMENTWISE.value: {
                "min_grid_size": 1,
                "typical_block_size": [256, 512, 1024],
                "memory_bound": True,
            },
            KernelFamily.REDUCTION.value: {
                "min_grid_size": 1,
                "typical_block_size": [256, 512],
                "memory_bound": True,
            },
        }


class FamilyClassifier:
    """
    Classifies GPU kernels into semantic families.

    Uses multiple methods:
    1. Name pattern matching
    2. Launch configuration analysis
    3. Timing/behavior analysis
    """

    def __init__(self, heuristics: Optional[FamilyHeuristics] = None):
        self.heuristics = heuristics or FamilyHeuristics()
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns."""
        for family, patterns in self.heuristics.patterns.items():
            self._compiled_patterns[family] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def classify(
        self,
        kernel_name: str,
        grid_size: int = 0,
        block_size: int = 0,
        duration_ns: int = 0,
    ) -> ClassificationResult:
        """
        Classify a kernel into a family.

        Args:
            kernel_name: Kernel name (possibly mangled)
            grid_size: Total grid size
            block_size: Total block/workgroup size
            duration_ns: Kernel duration in nanoseconds

        Returns:
            ClassificationResult with family and confidence.
        """
        # Try pattern-based classification
        result = self._classify_by_pattern(kernel_name)

        if result.family == KernelFamily.UNKNOWN:
            # Try config-based classification
            config_result = self._classify_by_config(grid_size, block_size)
            if config_result.family != KernelFamily.UNKNOWN:
                result = config_result
                result.kernel_name = kernel_name

        # Enhance with behavior analysis
        self._enhance_with_behavior(result, duration_ns, grid_size, block_size)

        return result

    def _classify_by_pattern(self, kernel_name: str) -> ClassificationResult:
        """Classify using name patterns."""
        matched: List[tuple] = []  # (family, pattern, confidence)

        for family, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(kernel_name):
                    # Higher confidence for more specific matches
                    confidence = 0.9 if len(pattern.pattern) > 10 else 0.7
                    matched.append((family, pattern.pattern, confidence))

        if not matched:
            return ClassificationResult(
                kernel_name=kernel_name,
                family=KernelFamily.UNKNOWN,
                confidence=0.5,
                classification_method="pattern",
            )

        # Take highest confidence match
        matched.sort(key=lambda x: x[2], reverse=True)
        best = matched[0]

        return ClassificationResult(
            kernel_name=kernel_name,
            family=KernelFamily(best[0]),
            confidence=best[2],
            classification_method="pattern",
            matched_patterns=[m[1] for m in matched[:3]],
        )

    def _classify_by_config(self, grid_size: int, block_size: int) -> ClassificationResult:
        """Classify using launch configuration."""
        # Large grid + 256 block = likely GEMM/CONV
        if grid_size >= 256 and block_size in [128, 256]:
            return ClassificationResult(
                kernel_name="",
                family=KernelFamily.GEMM,
                confidence=0.5,
                classification_method="config",
            )

        # Small grid = likely reduction
        if grid_size <= 16 and block_size >= 256:
            return ClassificationResult(
                kernel_name="",
                family=KernelFamily.REDUCTION,
                confidence=0.4,
                classification_method="config",
            )

        return ClassificationResult(
            kernel_name="",
            family=KernelFamily.UNKNOWN,
            confidence=0.3,
            classification_method="config",
        )

    def _enhance_with_behavior(
        self,
        result: ClassificationResult,
        duration_ns: int,
        grid_size: int,
        block_size: int,
    ) -> None:
        """Enhance classification with behavioral analysis."""
        family = result.family

        # Microkernel detection
        if duration_ns > 0 and duration_ns < 10_000:  # < 10µs
            result.is_microkernel = True

        # Compute vs memory bound
        compute_families = {
            KernelFamily.GEMM,
            KernelFamily.CONV,
            KernelFamily.ATTENTION,
        }
        memory_families = {
            KernelFamily.ELEMENTWISE,
            KernelFamily.COPY,
            KernelFamily.REDUCTION,
            KernelFamily.EMBEDDING,
        }

        if family in compute_families:
            result.is_compute_bound = True
            result.expected_roofline_region = "compute"
        elif family in memory_families:
            result.is_memory_bound = True
            result.expected_roofline_region = "memory"
        else:
            result.expected_roofline_region = "balanced"

        # Sub-family detection for GEMM
        if family == KernelFamily.GEMM:
            name_lower = result.kernel_name.lower()
            if "batch" in name_lower or "strided" in name_lower:
                result.sub_family = "batched_gemm"
            elif "trsm" in name_lower:
                result.sub_family = "triangular"
            elif "symm" in name_lower:
                result.sub_family = "symmetric"
            else:
                result.sub_family = "dense"

        # Sub-family for attention
        if family == KernelFamily.ATTENTION:
            name_lower = result.kernel_name.lower()
            if "flash" in name_lower:
                result.sub_family = "flash_attention"
                result.variant = "v2" if "v2" in name_lower else "v1"
            elif "self" in name_lower:
                result.sub_family = "self_attention"
            elif "cross" in name_lower:
                result.sub_family = "cross_attention"

    def classify_batch(self, kernels: List[Dict[str, Any]]) -> List[ClassificationResult]:
        """Classify a batch of kernels."""
        results = []
        for kernel in kernels:
            result = self.classify(
                kernel_name=kernel.get("name", ""),
                grid_size=kernel.get("grid_size", 0),
                block_size=kernel.get("block_size", 0),
                duration_ns=kernel.get("duration_ns", 0),
            )
            results.append(result)
        return results

    def summarize_families(self, results: List[ClassificationResult]) -> Dict[str, Any]:
        """Summarize classification results by family."""
        by_family: Dict[KernelFamily, List[ClassificationResult]] = {}

        for result in results:
            if result.family not in by_family:
                by_family[result.family] = []
            by_family[result.family].append(result)

        summary = {}
        for family, family_results in by_family.items():
            summary[family.value] = {
                "count": len(family_results),
                "microkernel_count": sum(1 for r in family_results if r.is_microkernel),
                "avg_confidence": sum(r.confidence for r in family_results) / len(family_results),
                "sub_families": list(set(r.sub_family for r in family_results if r.sub_family)),
            }

        return {
            "total_kernels": len(results),
            "family_distribution": summary,
            "compute_bound_count": sum(1 for r in results if r.is_compute_bound),
            "memory_bound_count": sum(1 for r in results if r.is_memory_bound),
            "microkernel_count": sum(1 for r in results if r.is_microkernel),
        }
