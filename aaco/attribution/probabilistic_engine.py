# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""
AACO-Ω∞ Probabilistic Attribution Engine

Graph → Partition → Kernel attribution with:
- ONNX graph structure analysis
- Kernel aggregation ratios
- Partition fragmentation index
- Launch tax scoring
- Probabilistic root cause attribution
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AttributionLevel(Enum):
    """Hierarchy of attribution levels."""

    MODEL = "model"
    SUBGRAPH = "subgraph"
    PARTITION = "partition"
    OPERATOR = "operator"
    KERNEL = "kernel"


class BottleneckCategory(Enum):
    """Categories of performance bottlenecks."""

    LAUNCH_BOUND = "launch_bound"
    MEMORY_BOUND = "memory_bound"
    COMPUTE_BOUND = "compute_bound"
    PARTITION_BOUND = "partition_bound"
    SYNC_BOUND = "sync_bound"
    TRANSFER_BOUND = "transfer_bound"
    UNKNOWN = "unknown"


@dataclass
class OperatorNode:
    """ONNX operator node with attribution data."""

    node_id: str = ""
    op_type: str = ""
    name: str = ""
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    input_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    output_shapes: List[Tuple[int, ...]] = field(default_factory=list)
    partition_id: str = ""
    kernel_ids: List[str] = field(default_factory=list)
    kernel_count: int = 0
    total_time_ns: int = 0
    self_time_ns: int = 0
    kernel_amplification_ratio: float = 1.0
    time_per_kernel_ns: float = 0.0


@dataclass
class PartitionNode:
    """Execution partition (subgraph assigned to single EP)."""

    partition_id: str = ""
    execution_provider: str = ""
    operator_ids: List[str] = field(default_factory=list)
    operator_count: int = 0
    kernel_ids: List[str] = field(default_factory=list)
    kernel_count: int = 0
    input_transfers: int = 0
    output_transfers: int = 0
    total_time_ns: int = 0
    launch_time_ns: int = 0
    transfer_time_ns: int = 0
    compute_time_ns: int = 0
    kernel_amplification_ratio: float = 1.0
    partition_efficiency: float = 1.0
    transfer_overhead_pct: float = 0.0


@dataclass
class AttributionResult:
    """Complete attribution result with confidence scores."""

    level: AttributionLevel = AttributionLevel.MODEL
    target_id: str = ""
    target_name: str = ""
    bottleneck_category: BottleneckCategory = BottleneckCategory.UNKNOWN
    confidence: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class AttributionMetrics:
    """Key metrics for attribution analysis."""

    kernel_amplification_ratio: float = 1.0  # KAR
    partition_fragmentation_index: float = 0.0  # PFI
    launch_tax_score: float = 0.0  # LTS
    overall_confidence: float = 0.0


class ProbabilisticAttributionEngine:
    """
    AACO-Ω∞ Probabilistic Attribution Engine

    Maps model structure to kernel execution with confidence scores.
    """

    BOTTLENECK_PATTERNS = {
        BottleneckCategory.LAUNCH_BOUND: {
            "high_kar": True,
            "short_kernels": True,
            "launch_overhead": 0.3,
        },
        BottleneckCategory.MEMORY_BOUND: {
            "high_memory_ratio": True,
            "low_occupancy": True,
        },
        BottleneckCategory.COMPUTE_BOUND: {
            "high_valu_util": True,
            "stable_kernels": True,
        },
        BottleneckCategory.PARTITION_BOUND: {
            "many_partitions": True,
            "transfers_high": True,
        },
    }

    def __init__(self):
        """Initialize attribution engine."""
        self._operators: Dict[str, OperatorNode] = {}
        self._partitions: Dict[str, PartitionNode] = {}
        self._kernel_map: Dict[str, List[str]] = {}
        self._metrics = AttributionMetrics()

    def load_onnx_graph(self, graph_json: Dict[str, Any]) -> None:
        """Load ONNX graph structure."""
        nodes = graph_json.get("nodes", [])

        for node in nodes:
            op = OperatorNode(
                node_id=node.get("id", node.get("name", "")),
                op_type=node.get("op_type", node.get("type", "")),
                name=node.get("name", ""),
                inputs=node.get("inputs", []),
                outputs=node.get("outputs", []),
                partition_id=node.get("partition", ""),
            )
            self._operators[op.node_id] = op

    def load_partition_info(self, partitions: List[Dict[str, Any]]) -> None:
        """Load partition information."""
        for p in partitions:
            partition = PartitionNode(
                partition_id=p.get("id", ""),
                execution_provider=p.get("provider", p.get("ep", "")),
                operator_ids=p.get("operators", []),
            )
            partition.operator_count = len(partition.operator_ids)
            self._partitions[partition.partition_id] = partition

            for op_id in partition.operator_ids:
                if op_id in self._operators:
                    self._operators[op_id].partition_id = partition.partition_id

    def map_kernels_to_operators(
        self,
        kernel_traces: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Map GPU kernels to ONNX operators."""
        operator_kernels: Dict[str, List[str]] = {}

        for trace in kernel_traces:
            kernel_name = trace.get("name", trace.get("kernel_name", ""))
            duration_ns = trace.get("duration_ns", 0)
            matched_op_id = self._match_kernel_to_operator(kernel_name)

            if matched_op_id:
                if matched_op_id not in operator_kernels:
                    operator_kernels[matched_op_id] = []
                operator_kernels[matched_op_id].append(kernel_name)

                if matched_op_id in self._operators:
                    op = self._operators[matched_op_id]
                    op.kernel_ids.append(kernel_name)
                    op.kernel_count += 1
                    op.total_time_ns += duration_ns

        self._kernel_map = operator_kernels
        return operator_kernels

    def _match_kernel_to_operator(self, kernel_name: str) -> Optional[str]:
        """Match kernel name to operator."""
        kernel_lower = kernel_name.lower()
        op_patterns = {
            "gemm": ["gemm", "matmul", "mm_"],
            "conv": ["conv", "winograd"],
            "relu": ["relu"],
            "softmax": ["softmax"],
            "layernorm": ["layernorm", "layer_norm"],
        }

        for op_id, op in self._operators.items():
            op_type_lower = op.op_type.lower()
            for op_key, patterns in op_patterns.items():
                if op_key in op_type_lower:
                    for pattern in patterns:
                        if pattern in kernel_lower:
                            return op_id
        return None

    def compute_metrics(self) -> AttributionMetrics:
        """Compute attribution metrics (KAR, PFI, LTS)."""
        total_operators = len(self._operators)
        total_kernels = sum(op.kernel_count for op in self._operators.values())
        total_time_ns = sum(op.total_time_ns for op in self._operators.values())

        if total_operators > 0:
            self._metrics.kernel_amplification_ratio = total_kernels / total_operators

        if self._partitions:
            num_partitions = len(self._partitions)
            partition_sizes = [p.operator_count for p in self._partitions.values()]
            if partition_sizes:
                import statistics

                variance = statistics.variance(partition_sizes) if len(partition_sizes) > 1 else 0
                self._metrics.partition_fragmentation_index = num_partitions * 0.1 + variance * 0.01

        estimated_launch_overhead_ns = total_kernels * 3000
        if total_time_ns > 0:
            self._metrics.launch_tax_score = estimated_launch_overhead_ns / total_time_ns

        return self._metrics

    def attribute_bottleneck(
        self,
        kernel_stats: Dict[str, Any],
        counter_stats: Optional[Dict[str, Any]] = None,
    ) -> AttributionResult:
        """Attribute performance bottleneck with confidence."""
        result = AttributionResult(level=AttributionLevel.MODEL, target_id="model")
        scores: Dict[BottleneckCategory, float] = {}

        kar = self._metrics.kernel_amplification_ratio
        lts = self._metrics.launch_tax_score

        if kar > 5 and lts > 0.2:
            scores[BottleneckCategory.LAUNCH_BOUND] = 0.8
            result.evidence.append(f"High KAR ({kar:.1f}) suggests kernel fragmentation")

        pfi = self._metrics.partition_fragmentation_index
        if pfi > 0.5 and len(self._partitions) > 3:
            scores[BottleneckCategory.PARTITION_BOUND] = 0.7
            result.evidence.append(f"High PFI ({pfi:.2f}) suggests partition overhead")

        if counter_stats:
            if counter_stats.get("memory_intensity", 0) > 10:
                scores[BottleneckCategory.MEMORY_BOUND] = 0.75
            if counter_stats.get("valu_utilization", 0) > 0.8:
                scores[BottleneckCategory.COMPUTE_BOUND] = 0.85

        if scores:
            best = max(scores.keys(), key=lambda k: scores[k])
            result.bottleneck_category = best
            result.confidence = scores[best]
            result.recommendations = self._get_recommendations(best)

        return result

    def _get_recommendations(self, category: BottleneckCategory) -> List[str]:
        """Get recommendations for bottleneck category."""
        recs = {
            BottleneckCategory.LAUNCH_BOUND: [
                "Enable kernel fusion to reduce launch overhead",
                "Check for unnecessary operator decomposition",
            ],
            BottleneckCategory.MEMORY_BOUND: [
                "Check data layout for cache efficiency",
                "Enable memory-optimized kernels",
            ],
            BottleneckCategory.COMPUTE_BOUND: [
                "Model efficiently utilizing compute",
                "Consider lower precision (FP16/INT8)",
            ],
            BottleneckCategory.PARTITION_BOUND: [
                "Reduce EP transitions",
                "Consider single-EP execution",
            ],
        }
        return recs.get(category, ["Gather more profiling data"])

    def get_hotspot_operators(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get top operators by execution time."""
        operators = sorted(
            self._operators.values(),
            key=lambda op: op.total_time_ns,
            reverse=True,
        )[:top_n]

        total_time = sum(op.total_time_ns for op in self._operators.values())
        return [
            {
                "node_id": op.node_id,
                "op_type": op.op_type,
                "time_ns": op.total_time_ns,
                "time_pct": op.total_time_ns / total_time * 100 if total_time > 0 else 0,
                "kernel_count": op.kernel_count,
            }
            for op in operators
        ]

    def export_attribution(self) -> Dict[str, Any]:
        """Export complete attribution data."""
        return {
            "metrics": {
                "kar": self._metrics.kernel_amplification_ratio,
                "pfi": self._metrics.partition_fragmentation_index,
                "lts": self._metrics.launch_tax_score,
            },
            "operators": len(self._operators),
            "partitions": len(self._partitions),
        }
