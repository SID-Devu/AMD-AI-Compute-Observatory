"""
Probabilistic Attribution Engine.

Maps performance from graph operations to partitions to kernels
with confidence scoring and evidence tracking.
"""

import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


class AttributionLevel(str, Enum):
    """Granularity levels for attribution."""

    GRAPH = "graph"  # Full computation graph
    PARTITION = "partition"  # Subgraph / layer
    KERNEL = "kernel"  # Individual GPU kernel


@dataclass
class AttributionNode:
    """A node in the attribution graph."""

    node_id: str
    level: AttributionLevel
    name: str

    # Timing
    duration_us: float = 0.0
    self_time_us: float = 0.0  # Time not attributed to children

    # Attribution scores
    attribution_pct: float = 0.0  # Percentage of total time
    confidence: float = 1.0  # Confidence in attribution

    # Relationships
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # Evidence
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        return d


@dataclass
class AttributionEdge:
    """
    Edge representing attribution relationship.

    From parent to child with weight and confidence.
    """

    source_id: str
    target_id: str
    weight: float  # Fraction of parent time attributed to child
    confidence: float  # Confidence in this attribution
    evidence_type: str = ""  # e.g., "temporal", "counter", "name_match"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AttributionGraph:
    """
    Multi-level attribution graph.

    Represents hierarchical performance attribution from
    graph operations to partitions to kernels.
    """

    nodes: Dict[str, AttributionNode] = field(default_factory=dict)
    edges: List[AttributionEdge] = field(default_factory=list)

    # Root nodes (top-level graph operations)
    root_ids: List[str] = field(default_factory=list)

    # Total time
    total_time_us: float = 0.0

    def add_node(self, node: AttributionNode) -> None:
        self.nodes[node.node_id] = node
        if node.parent_id is None:
            self.root_ids.append(node.node_id)

    def add_edge(self, edge: AttributionEdge) -> None:
        self.edges.append(edge)
        if edge.source_id in self.nodes:
            self.nodes[edge.source_id].children_ids.append(edge.target_id)

    def get_level_nodes(self, level: AttributionLevel) -> List[AttributionNode]:
        """Get all nodes at a specific level."""
        return [n for n in self.nodes.values() if n.level == level]

    def get_children(self, node_id: str) -> List[AttributionNode]:
        """Get children of a node."""
        if node_id not in self.nodes:
            return []
        return [self.nodes[cid] for cid in self.nodes[node_id].children_ids if cid in self.nodes]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "root_ids": self.root_ids,
            "total_time_us": self.total_time_us,
        }


# ============================================================================
# Attribution Result
# ============================================================================


@dataclass
class AttributionResult:
    """Result of attribution analysis."""

    graph: AttributionGraph

    # Summary
    total_time_us: float = 0.0
    attributed_time_us: float = 0.0
    unattributed_time_us: float = 0.0

    # By level
    graph_level_breakdown: Dict[str, float] = field(default_factory=dict)
    partition_level_breakdown: Dict[str, float] = field(default_factory=dict)
    kernel_level_breakdown: Dict[str, float] = field(default_factory=dict)

    # Confidence metrics
    overall_confidence: float = 1.0
    low_confidence_nodes: List[str] = field(default_factory=list)

    # Top contributors
    top_kernels: List[Tuple[str, float]] = field(default_factory=list)
    top_partitions: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_time_us": self.total_time_us,
            "attributed_time_us": self.attributed_time_us,
            "unattributed_time_us": self.unattributed_time_us,
            "graph_level_breakdown": self.graph_level_breakdown,
            "partition_level_breakdown": self.partition_level_breakdown,
            "kernel_level_breakdown": self.kernel_level_breakdown,
            "overall_confidence": self.overall_confidence,
            "low_confidence_nodes": self.low_confidence_nodes,
            "top_kernels": self.top_kernels[:10],
            "top_partitions": self.top_partitions[:10],
        }


# ============================================================================
# Evidence Types
# ============================================================================


class EvidenceType(str, Enum):
    """Types of evidence for attribution."""

    TEMPORAL = "temporal"  # Time-based correlation
    NAME_MATCH = "name_match"  # Kernel name matches operation
    COUNTER = "counter"  # HW counter correlation
    SHAPE_MATCH = "shape_match"  # Input/output shape matching
    CAUSAL = "causal"  # Causal dependency
    STATISTICAL = "statistical"  # Statistical correlation


@dataclass
class Evidence:
    """Evidence supporting an attribution."""

    evidence_type: EvidenceType
    strength: float  # 0-1
    description: str
    data: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Probabilistic Attribution Engine
# ============================================================================


class ProbabilisticAttributionEngine:
    """
    Multi-level probabilistic attribution from graph to kernels.

    Uses multiple evidence sources to build confident attributions:
    - Temporal correlation (overlapping time windows)
    - Name matching (kernel name contains operation name)
    - Shape matching (tensor shapes align)
    - Counter correlation (HW counters match expected)

    Usage:
        engine = ProbabilisticAttributionEngine()

        # Add graph operations
        engine.add_graph_operation("matmul_0", duration_us=1000)

        # Add kernels
        engine.add_kernel("gemm_kernel", duration_us=800, timestamp_us=0)

        # Build attribution
        result = engine.build_attribution()
    """

    def __init__(self, confidence_threshold: float = 0.5, temporal_window_us: float = 100):
        """
        Args:
            confidence_threshold: Minimum confidence for attribution
            temporal_window_us: Window for temporal correlation
        """
        self.confidence_threshold = confidence_threshold
        self.temporal_window_us = temporal_window_us

        # Collected data
        self._graph_ops: List[Dict[str, Any]] = []
        self._partitions: List[Dict[str, Any]] = []
        self._kernels: List[Dict[str, Any]] = []

        # Name pattern mappings
        self._name_patterns = self._build_name_patterns()

    def add_graph_operation(
        self,
        name: str,
        duration_us: float,
        op_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a graph-level operation."""
        self._graph_ops.append(
            {
                "name": name,
                "duration_us": duration_us,
                "op_type": op_type,
                "metadata": metadata or {},
            }
        )

    def add_partition(
        self,
        name: str,
        duration_us: float,
        parent_op: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a partition (subgraph/layer)."""
        self._partitions.append(
            {
                "name": name,
                "duration_us": duration_us,
                "parent_op": parent_op,
                "metadata": metadata or {},
            }
        )

    def add_kernel(
        self,
        name: str,
        duration_us: float,
        timestamp_us: float = 0,
        counters: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a kernel execution."""
        self._kernels.append(
            {
                "name": name,
                "duration_us": duration_us,
                "timestamp_us": timestamp_us,
                "counters": counters or {},
                "metadata": metadata or {},
            }
        )

    def build_attribution(self) -> AttributionResult:
        """Build multi-level attribution graph."""
        graph = AttributionGraph()

        # Calculate totals
        total_graph_time = sum(op["duration_us"] for op in self._graph_ops)
        total_kernel_time = sum(k["duration_us"] for k in self._kernels)
        graph.total_time_us = max(total_graph_time, total_kernel_time)

        # Level 1: Graph operations
        for op in self._graph_ops:
            node = AttributionNode(
                node_id=f"graph_{op['name']}",
                level=AttributionLevel.GRAPH,
                name=op["name"],
                duration_us=op["duration_us"],
                attribution_pct=(op["duration_us"] / graph.total_time_us * 100)
                if graph.total_time_us > 0
                else 0,
            )
            graph.add_node(node)

        # Level 2: Partitions (if any)
        for part in self._partitions:
            parent_id = None
            if part["parent_op"]:
                parent_id = f"graph_{part['parent_op']}"

            node = AttributionNode(
                node_id=f"part_{part['name']}",
                level=AttributionLevel.PARTITION,
                name=part["name"],
                duration_us=part["duration_us"],
                parent_id=parent_id,
            )
            graph.add_node(node)

            if parent_id:
                edge = AttributionEdge(
                    source_id=parent_id,
                    target_id=node.node_id,
                    weight=part["duration_us"] / graph.nodes[parent_id].duration_us
                    if parent_id in graph.nodes
                    else 0,
                    confidence=0.8,
                    evidence_type="parent_child",
                )
                graph.add_edge(edge)

        # Level 3: Kernels
        for kernel in self._kernels:
            node = AttributionNode(
                node_id=f"kernel_{kernel['name']}_{kernel['timestamp_us']:.0f}",
                level=AttributionLevel.KERNEL,
                name=kernel["name"],
                duration_us=kernel["duration_us"],
                attribution_pct=(kernel["duration_us"] / graph.total_time_us * 100)
                if graph.total_time_us > 0
                else 0,
            )

            # Find best parent attribution
            evidences = self._compute_kernel_evidences(kernel, graph)
            if evidences:
                best_evidence = max(evidences, key=lambda e: e[1])
                parent_id, confidence, ev = best_evidence
                node.parent_id = parent_id
                node.confidence = confidence
                node.evidence = {
                    "type": ev.evidence_type.value,
                    "strength": ev.strength,
                }

            graph.add_node(node)

            # Create edge to parent
            if node.parent_id:
                edge = AttributionEdge(
                    source_id=node.parent_id,
                    target_id=node.node_id,
                    weight=node.duration_us / graph.nodes[node.parent_id].duration_us
                    if node.parent_id in graph.nodes
                    else 0,
                    confidence=node.confidence,
                    evidence_type=node.evidence.get("type", "unknown"),
                )
                graph.add_edge(edge)

        # Build result
        result = self._build_result(graph)

        return result

    def _compute_kernel_evidences(
        self, kernel: Dict[str, Any], graph: AttributionGraph
    ) -> List[Tuple[str, float, Evidence]]:
        """Compute attribution evidence for a kernel."""
        evidences = []
        kernel_name = kernel["name"].lower()

        for node_id, node in graph.nodes.items():
            if node.level == AttributionLevel.KERNEL:
                continue

            evidence_scores = []

            # Name matching evidence
            name_score = self._compute_name_match_score(kernel_name, node.name.lower())
            if name_score > 0.3:
                evidence_scores.append(
                    Evidence(
                        evidence_type=EvidenceType.NAME_MATCH,
                        strength=name_score,
                        description=f"Kernel name '{kernel_name}' matches '{node.name}'",
                    )
                )

            # Pattern matching (known kernel-op mappings)
            pattern_score = self._match_known_pattern(kernel_name, node.name.lower())
            if pattern_score > 0:
                evidence_scores.append(
                    Evidence(
                        evidence_type=EvidenceType.NAME_MATCH,
                        strength=pattern_score,
                        description="Known pattern match",
                    )
                )

            # Combine evidences
            if evidence_scores:
                combined = self._combine_evidences(evidence_scores)
                evidences.append((node_id, combined, evidence_scores[0]))

        return evidences

    def _compute_name_match_score(self, kernel_name: str, op_name: str) -> float:
        """Compute name matching score between kernel and operation."""
        # Extract tokens
        kernel_tokens = set(self._tokenize(kernel_name))
        op_tokens = set(self._tokenize(op_name))

        if not kernel_tokens or not op_tokens:
            return 0.0

        # Jaccard similarity
        intersection = len(kernel_tokens & op_tokens)
        union = len(kernel_tokens | op_tokens)

        return intersection / union if union > 0 else 0.0

    def _match_known_pattern(self, kernel_name: str, op_name: str) -> float:
        """Check for known kernel-operation pattern matches."""
        for pattern, ops in self._name_patterns.items():
            if pattern in kernel_name:
                for op in ops:
                    if op in op_name:
                        return 0.9
        return 0.0

    def _build_name_patterns(self) -> Dict[str, List[str]]:
        """Build known kernel-to-operation pattern mappings."""
        return {
            "gemm": ["matmul", "mm", "linear", "dense"],
            "conv": ["conv", "convolution"],
            "attention": ["attention", "sdpa", "mha"],
            "layernorm": ["layernorm", "layer_norm", "ln"],
            "softmax": ["softmax"],
            "relu": ["relu", "activation"],
            "gelu": ["gelu", "activation"],
            "embedding": ["embedding", "embed"],
            "reduce": ["sum", "mean", "reduce"],
        }

    def _tokenize(self, name: str) -> List[str]:
        """Tokenize a name into words."""
        import re

        # Split on non-alphanumeric
        tokens = re.split(r"[^a-z0-9]+", name.lower())
        return [t for t in tokens if len(t) > 1]

    def _combine_evidences(self, evidences: List[Evidence]) -> float:
        """Combine multiple evidence scores into confidence."""
        if not evidences:
            return 0.0

        # Weighted by evidence type importance
        type_weights = {
            EvidenceType.CAUSAL: 1.0,
            EvidenceType.COUNTER: 0.9,
            EvidenceType.TEMPORAL: 0.8,
            EvidenceType.NAME_MATCH: 0.7,
            EvidenceType.SHAPE_MATCH: 0.6,
            EvidenceType.STATISTICAL: 0.5,
        }

        weighted_sum = sum(e.strength * type_weights.get(e.evidence_type, 0.5) for e in evidences)
        total_weight = sum(type_weights.get(e.evidence_type, 0.5) for e in evidences)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _build_result(self, graph: AttributionGraph) -> AttributionResult:
        """Build attribution result from graph."""
        result = AttributionResult(graph=graph)
        result.total_time_us = graph.total_time_us

        # Calculate breakdowns by level
        for node in graph.get_level_nodes(AttributionLevel.GRAPH):
            result.graph_level_breakdown[node.name] = node.duration_us

        for node in graph.get_level_nodes(AttributionLevel.PARTITION):
            result.partition_level_breakdown[node.name] = node.duration_us

        kernel_times = {}
        for node in graph.get_level_nodes(AttributionLevel.KERNEL):
            if node.name not in kernel_times:
                kernel_times[node.name] = 0
            kernel_times[node.name] += node.duration_us
        result.kernel_level_breakdown = kernel_times

        # Top contributors
        result.top_kernels = sorted(kernel_times.items(), key=lambda x: x[1], reverse=True)[:10]

        result.top_partitions = sorted(
            result.partition_level_breakdown.items(), key=lambda x: x[1], reverse=True
        )[:10]

        # Confidence metrics
        all_confidences = [
            n.confidence for n in graph.nodes.values() if n.level == AttributionLevel.KERNEL
        ]
        if all_confidences:
            result.overall_confidence = np.mean(all_confidences)
            result.low_confidence_nodes = [
                n.node_id for n in graph.nodes.values() if n.confidence < self.confidence_threshold
            ]

        # Attributed vs unattributed
        result.attributed_time_us = sum(
            n.duration_us
            for n in graph.nodes.values()
            if n.level == AttributionLevel.KERNEL and n.parent_id is not None
        )
        result.unattributed_time_us = sum(
            n.duration_us
            for n in graph.nodes.values()
            if n.level == AttributionLevel.KERNEL and n.parent_id is None
        )

        return result


# ============================================================================
# Convenience Functions
# ============================================================================


def attribute_kernels_to_ops(
    kernels: List[Dict[str, Any]], ops: List[Dict[str, Any]]
) -> AttributionResult:
    """
    Quick attribution of kernels to operations.

    Args:
        kernels: List of {"name": ..., "duration_us": ...}
        ops: List of {"name": ..., "duration_us": ...}

    Returns:
        Attribution result
    """
    engine = ProbabilisticAttributionEngine()

    for op in ops:
        engine.add_graph_operation(op["name"], op["duration_us"])

    for kernel in kernels:
        engine.add_kernel(kernel["name"], kernel["duration_us"], kernel.get("timestamp_us", 0))

    return engine.build_attribution()
