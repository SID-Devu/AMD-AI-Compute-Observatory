"""
Op-to-Kernel Mapper
Maps ONNX operators to GPU kernel executions for model-to-hardware attribution.
"""

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from aaco.core.schema import KernelSummary
from aaco.graph.extractor import GraphNode, GraphMetadata

logger = logging.getLogger(__name__)


@dataclass
class KernelGroup:
    """Group of related GPU kernels."""

    group_id: int
    group_name: str
    kernel_name_patterns: List[str]
    kernel_names: List[str]
    total_calls: int
    total_time_ms: float
    avg_time_us: float
    pct_total_gpu_time: float
    is_microkernel_group: bool = False


@dataclass
class AttributionResult:
    """Result of op-to-kernel attribution."""

    node_id: int
    node_name: str
    op_type: str
    kernel_group_id: Optional[int]
    kernel_names: List[str]
    attributed_time_ms: float
    attribution_method: str  # "exact", "name_heuristic", "time_correlation", "estimated"
    confidence: float  # 0-1
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MappingReport:
    """Complete mapping report for a session."""

    model_nodes: int
    total_kernels: int
    attributed_nodes: int
    unattributed_nodes: int
    attribution_coverage: float  # % of nodes with attribution
    kernel_amplification_ratio: float  # kernels / nodes
    op_type_mappings: Dict[str, List[str]]  # op_type -> kernel patterns
    attributions: List[AttributionResult]
    kernel_groups: List[KernelGroup]
    unmapped_kernels: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_nodes": self.model_nodes,
            "total_kernels": self.total_kernels,
            "attributed_nodes": self.attributed_nodes,
            "unattributed_nodes": self.unattributed_nodes,
            "attribution_coverage": self.attribution_coverage,
            "kernel_amplification_ratio": self.kernel_amplification_ratio,
            "op_type_mappings": self.op_type_mappings,
            "attributions": [a.to_dict() for a in self.attributions],
            "kernel_groups": [asdict(g) for g in self.kernel_groups],
            "unmapped_kernels": self.unmapped_kernels,
        }


class OpKernelMapper:
    """
    Maps ONNX operators to GPU kernels using multiple attribution methods.

    Attribution methods (in order of confidence):
    1. exact - Direct name matching from EP annotations
    2. name_heuristic - Pattern matching on kernel names
    3. time_correlation - Temporal alignment of kernel bursts
    4. estimated - Statistical estimation from op counts
    """

    # Kernel name pattern -> ONNX op type mappings
    # These patterns are ROCm/HIP/MIGraphX specific
    KERNEL_PATTERNS = {
        # GEMM/MatMul patterns
        "MatMul": [
            r"gemm",
            r"rocblas_gemm",
            r"hipblas_gemm",
            r"Cijk",
            r"matmul",
            r"sgemm",
            r"dgemm",
            r"hgemm",
            r"rocblas_internal",
        ],
        "Conv": [
            r"Conv",
            r"miopenConv",
            r"ImplicitGemm",
            r"winograd",
            r"ConvolutionBackward",
            r"col2im",
            r"im2col",
        ],
        "BatchNormalization": [
            r"batchnorm",
            r"BatchNorm",
            r"bn_",
            r"miopenBatchNorm",
        ],
        "LayerNormalization": [
            r"layernorm",
            r"LayerNorm",
            r"ln_",
            r"layer_norm",
        ],
        "Softmax": [
            r"softmax",
            r"Softmax",
            r"exp_",
            r"reduce_sum",
        ],
        "Attention": [
            r"attention",
            r"Attention",
            r"flash_attn",
            r"fused_attention",
            r"multihead",
            r"qkv",
            r"scaled_dot",
        ],
        "Relu": [r"relu", r"Relu", r"activation"],
        "Sigmoid": [r"sigmoid", r"Sigmoid"],
        "Tanh": [r"tanh", r"Tanh"],
        "Add": [r"add_kernel", r"elementwise_add", r"vadd"],
        "Mul": [r"mul_kernel", r"elementwise_mul", r"vmul"],
        "Concat": [r"concat", r"Concat", r"concatenate"],
        "Transpose": [r"transpose", r"Transpose", r"permute"],
        "Reshape": [r"reshape", r"view", r"contiguous"],
        "Gather": [r"gather", r"Gather", r"index_select", r"embedding"],
        "Reduce": [r"reduce", r"Reduce", r"sum_kernel", r"mean_kernel"],
        "Pool": [r"pool", r"Pool", r"maxpool", r"avgpool"],
    }

    # Microkernel threshold in microseconds
    MICROKERNEL_THRESHOLD_US = 10.0

    def __init__(self, graph_metadata: GraphMetadata, kernel_summaries: List[KernelSummary]):
        self.graph = graph_metadata
        self.kernels = kernel_summaries
        self.kernel_groups: List[KernelGroup] = []
        self.attributions: List[AttributionResult] = []
        self._op_to_kernels: Dict[str, List[str]] = {}

    def map(self) -> MappingReport:
        """
        Perform op-to-kernel mapping.

        Returns:
            MappingReport with all attributions and analysis.
        """
        logger.info(f"Mapping {len(self.graph.nodes)} ops to {len(self.kernels)} kernels")

        # Step 1: Group kernels by pattern
        self._group_kernels()

        # Step 2: Build op-type to kernel pattern mapping
        self._build_op_kernel_map()

        # Step 3: Attribute each node
        self._attribute_nodes()

        # Step 4: Identify unmapped kernels
        unmapped = self._find_unmapped_kernels()

        # Compute statistics
        attributed = sum(1 for a in self.attributions if a.kernel_group_id is not None)
        coverage = attributed / len(self.graph.nodes) if self.graph.nodes else 0
        kar = len(self.kernels) / len(self.graph.nodes) if self.graph.nodes else 0

        return MappingReport(
            model_nodes=len(self.graph.nodes),
            total_kernels=sum(k.calls for k in self.kernels),
            attributed_nodes=attributed,
            unattributed_nodes=len(self.graph.nodes) - attributed,
            attribution_coverage=coverage,
            kernel_amplification_ratio=kar,
            op_type_mappings=self._op_to_kernels,
            attributions=self.attributions,
            kernel_groups=self.kernel_groups,
            unmapped_kernels=unmapped,
        )

    def _group_kernels(self) -> None:
        """Group kernels by name patterns."""
        groups: Dict[str, List[KernelSummary]] = {}

        for kernel in self.kernels:
            # Find matching pattern group
            group_name = self._find_kernel_group(kernel.kernel_name)
            if group_name not in groups:
                groups[group_name] = []
            groups[group_name].append(kernel)

        # Create KernelGroup objects
        total_time = sum(k.total_time_ms for k in self.kernels)

        for group_id, (group_name, kernels) in enumerate(groups.items()):
            total_calls = sum(k.calls for k in kernels)
            total_time_ms = sum(k.total_time_ms for k in kernels)
            avg_time_us = (total_time_ms * 1000) / total_calls if total_calls > 0 else 0

            # Check if microkernel group
            is_micro = avg_time_us < self.MICROKERNEL_THRESHOLD_US

            self.kernel_groups.append(
                KernelGroup(
                    group_id=group_id,
                    group_name=group_name,
                    kernel_name_patterns=[],  # Will be filled from patterns
                    kernel_names=[k.kernel_name for k in kernels],
                    total_calls=total_calls,
                    total_time_ms=total_time_ms,
                    avg_time_us=avg_time_us,
                    pct_total_gpu_time=(total_time_ms / total_time * 100) if total_time > 0 else 0,
                    is_microkernel_group=is_micro,
                )
            )

    def _find_kernel_group(self, kernel_name: str) -> str:
        """Find the group name for a kernel based on pattern matching."""
        kernel_lower = kernel_name.lower()

        for op_type, patterns in self.KERNEL_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern.lower(), kernel_lower):
                    return op_type

        # Extract a reasonable group name from kernel name
        # Remove numeric suffixes and common prefixes
        clean_name = re.sub(r"_\d+$", "", kernel_name)
        clean_name = re.sub(r"^(hip|rocm|migraphx)_?", "", clean_name, flags=re.I)

        # Take first word/segment as group
        match = re.match(r"^(\w+)", clean_name)
        return match.group(1) if match else "unknown"

    def _build_op_kernel_map(self) -> None:
        """Build mapping from op types to kernel patterns."""
        for op_type in set(n.op_type for n in self.graph.nodes):
            matched_kernels = []

            # Check pattern matches
            patterns = self.KERNEL_PATTERNS.get(op_type, [])
            for kernel in self.kernels:
                for pattern in patterns:
                    if re.search(pattern, kernel.kernel_name, re.I):
                        if kernel.kernel_name not in matched_kernels:
                            matched_kernels.append(kernel.kernel_name)
                        break

            if matched_kernels:
                self._op_to_kernels[op_type] = matched_kernels

    def _attribute_nodes(self) -> None:
        """Attribute each graph node to kernel groups."""
        for node in self.graph.nodes:
            attribution = self._attribute_single_node(node)
            self.attributions.append(attribution)

    def _attribute_single_node(self, node: GraphNode) -> AttributionResult:
        """Attribute a single node to its kernel group."""
        # Try name heuristic first
        matched_group = None
        matched_kernels = []
        confidence = 0.0
        method = "unattributed"
        evidence = {}

        patterns = self.KERNEL_PATTERNS.get(node.op_type, [])

        for group in self.kernel_groups:
            # Check if group name matches op type
            if group.group_name.lower() == node.op_type.lower():
                matched_group = group
                matched_kernels = group.kernel_names
                confidence = 0.8
                method = "name_heuristic"
                evidence["matched_group"] = group.group_name
                break

            # Check pattern matching
            for kernel_name in group.kernel_names:
                for pattern in patterns:
                    if re.search(pattern, kernel_name, re.I):
                        matched_group = group
                        if kernel_name not in matched_kernels:
                            matched_kernels.append(kernel_name)
                        confidence = 0.6
                        method = "name_heuristic"
                        evidence["matched_patterns"] = patterns
                        break

        # Compute attributed time
        attributed_time = 0.0
        if matched_group:
            # Estimate time per op of this type
            op_count = sum(1 for n in self.graph.nodes if n.op_type == node.op_type)
            if op_count > 0:
                attributed_time = matched_group.total_time_ms / op_count
                evidence["estimated_from"] = f"{op_count} ops of type {node.op_type}"

        return AttributionResult(
            node_id=node.node_id,
            node_name=node.name,
            op_type=node.op_type,
            kernel_group_id=matched_group.group_id if matched_group else None,
            kernel_names=matched_kernels,
            attributed_time_ms=attributed_time,
            attribution_method=method,
            confidence=confidence,
            evidence=evidence,
        )

    def _find_unmapped_kernels(self) -> List[str]:
        """Find kernels that weren't attributed to any op."""
        attributed_kernels = set()
        for attr in self.attributions:
            attributed_kernels.update(attr.kernel_names)

        unmapped = []
        for kernel in self.kernels:
            if kernel.kernel_name not in attributed_kernels:
                unmapped.append(kernel.kernel_name)

        return unmapped

    def get_hotspot_ops(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get ops with highest attributed GPU time."""
        sorted_attrs = sorted(self.attributions, key=lambda a: a.attributed_time_ms, reverse=True)

        return [
            {
                "op_type": a.op_type,
                "node_name": a.node_name,
                "attributed_time_ms": a.attributed_time_ms,
                "kernel_count": len(a.kernel_names),
                "confidence": a.confidence,
            }
            for a in sorted_attrs[:top_n]
        ]

    def get_kernel_fragmentation_score(self) -> Dict[str, Any]:
        """
        Compute kernel fragmentation score.
        High fragmentation = many small kernels = launch overhead dominant.
        """
        if not self.kernels:
            return {"score": 0, "assessment": "no_kernels"}

        total_calls = sum(k.calls for k in self.kernels)
        total_time = sum(k.total_time_ms for k in self.kernels)
        avg_duration_us = (total_time * 1000) / total_calls if total_calls > 0 else 0

        # Count microkernels
        microkernel_count = sum(
            k.calls for k in self.kernels if k.avg_time_us < self.MICROKERNEL_THRESHOLD_US
        )
        microkernel_pct = (microkernel_count / total_calls * 100) if total_calls > 0 else 0

        # Fragmentation score (0-100)
        # Based on: microkernel %, avg duration, kernel count per node
        kar = total_calls / len(self.graph.nodes) if self.graph.nodes else 0

        score = min(
            100,
            (
                microkernel_pct * 0.4  # High micro % is bad
                + (100 - min(avg_duration_us, 100)) * 0.3  # Low avg duration is bad
                + min(kar * 10, 30)  # High KAR is bad
            ),
        )

        if score < 20:
            assessment = "low_fragmentation"
        elif score < 50:
            assessment = "moderate_fragmentation"
        else:
            assessment = "high_fragmentation"

        return {
            "score": score,
            "assessment": assessment,
            "avg_kernel_duration_us": avg_duration_us,
            "microkernel_pct": microkernel_pct,
            "kernel_amplification_ratio": kar,
            "recommendation": self._fragmentation_recommendation(score),
        }

    def _fragmentation_recommendation(self, score: float) -> str:
        """Generate recommendation based on fragmentation score."""
        if score < 20:
            return "Kernel launch overhead is well-controlled."
        elif score < 50:
            return "Consider operator fusion to reduce kernel count."
        else:
            return (
                "HIGH LAUNCH OVERHEAD DETECTED. Investigate graph partitioning, "
                "enable operator fusion, or use optimized kernels."
            )
