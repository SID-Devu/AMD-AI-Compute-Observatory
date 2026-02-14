"""
AACO-SIGMA Graph Analyzer

Analyzes ONNX and other model graph formats.
Provides:
- Graph structure analysis
- Pattern detection
- Critical path identification
- Memory footprint estimation
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from collections import defaultdict
from enum import Enum, auto


class NodeType(Enum):
    """Graph node types."""

    INPUT = auto()
    OUTPUT = auto()
    OPERATOR = auto()
    CONSTANT = auto()
    SUBGRAPH = auto()


@dataclass
class GraphNode:
    """A node in the computation graph."""

    node_id: str
    node_type: NodeType
    op_type: str = ""  # e.g., "Conv", "MatMul", "Relu"
    name: str = ""

    # Inputs/outputs
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Shape info
    output_shapes: List[Tuple[int, ...]] = field(default_factory=list)

    # Execution info (filled after profiling)
    execution_time_ns: int = 0
    kernel_name: str = ""
    kernel_count: int = 0

    # Analysis metadata
    is_compute_heavy: bool = False
    is_memory_heavy: bool = False
    can_be_fused: bool = False


@dataclass
class GraphEdge:
    """An edge (tensor) in the computation graph."""

    edge_id: str
    source_node: str
    target_node: str
    tensor_name: str = ""

    # Tensor info
    dtype: str = ""
    shape: Tuple[int, ...] = ()

    # Size estimation
    size_bytes: int = 0

    @property
    def element_count(self) -> int:
        if self.shape:
            result = 1
            for dim in self.shape:
                result *= dim
            return result
        return 0


@dataclass
class SubgraphPattern:
    """A recognized pattern (subgraph) in the model."""

    pattern_id: str
    pattern_name: str  # e.g., "attention_block", "mlp_block"

    nodes: List[str] = field(default_factory=list)

    # Pattern characteristics
    is_fusible: bool = False
    has_known_optimized_impl: bool = False

    # Expected kernels
    expected_kernel_count: int = 0


@dataclass
class ModelGraph:
    """Representation of a computation graph."""

    name: str = ""

    # Graph structure
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[str, GraphEdge] = field(default_factory=dict)

    # Inputs/outputs
    input_names: List[str] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)

    # Metadata
    opset_version: int = 0
    producer_name: str = ""
    producer_version: str = ""

    # Analysis results
    critical_path: List[str] = field(default_factory=list)
    detected_patterns: List[SubgraphPattern] = field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        """Add a node to the graph."""
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the graph."""
        self.edges[edge.edge_id] = edge

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def get_successors(self, node_id: str) -> List[str]:
        """Get successor node IDs."""
        successors = []
        node = self.nodes.get(node_id)
        if node:
            for output in node.outputs:
                for n in self.nodes.values():
                    if output in n.inputs:
                        successors.append(n.node_id)
        return successors

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get predecessor node IDs."""
        predecessors = []
        node = self.nodes.get(node_id)
        if node:
            for inp in node.inputs:
                for n in self.nodes.values():
                    if inp in n.outputs:
                        predecessors.append(n.node_id)
        return predecessors


class GraphAnalyzer:
    """
    Analyzes computation graphs for performance insights.

    Provides:
    - Graph statistics
    - Critical path analysis
    - Pattern detection
    - Fusion opportunity identification
    """

    # Operator categories
    COMPUTE_HEAVY_OPS = {
        "MatMul",
        "Gemm",
        "Conv",
        "ConvTranspose",
        "MatMulInteger",
        "QAttention",
        "Attention",
    }

    MEMORY_HEAVY_OPS = {
        "Gather",
        "Scatter",
        "GatherElements",
        "ScatterElements",
        "Concat",
        "Split",
        "Slice",
        "Transpose",
        "Reshape",
    }

    # Known patterns (op sequences)
    KNOWN_PATTERNS = {
        "attention_block": ["MatMul", "Softmax", "MatMul"],
        "flash_attention": ["QAttention"],
        "mlp_block": ["MatMul", "Relu|Gelu", "MatMul"],
        "layernorm": [
            "ReduceMean",
            "Sub",
            "Pow",
            "ReduceMean",
            "Add",
            "Sqrt",
            "Div",
            "Mul",
            "Add",
        ],
        "gelu": ["Div", "Erf", "Add", "Mul", "Mul"],  # GELU approximation
        "residual_add": ["Add"],  # Simple but important
    }

    def __init__(self):
        self._graph: Optional[ModelGraph] = None

    def load_onnx(self, model_path: Path) -> ModelGraph:
        """Load ONNX model and create graph representation."""
        try:
            import onnx

            model = onnx.load(str(model_path))
            return self._parse_onnx_model(model)
        except ImportError:
            raise RuntimeError("ONNX not available. Install with: pip install onnx")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

    def _parse_onnx_model(self, model) -> ModelGraph:
        """Parse ONNX model into ModelGraph."""

        graph = ModelGraph(
            name=model.graph.name or "model",
            opset_version=model.opset_import[0].version if model.opset_import else 0,
            producer_name=model.producer_name,
            producer_version=model.producer_version,
        )

        # Parse inputs
        for inp in model.graph.input:
            graph.input_names.append(inp.name)
            node = GraphNode(
                node_id=f"input_{inp.name}",
                node_type=NodeType.INPUT,
                name=inp.name,
                outputs=[inp.name],
            )
            # Get shape if available
            if inp.type.tensor_type.shape.dim:
                shape = tuple(
                    d.dim_value if d.dim_value > 0 else -1 for d in inp.type.tensor_type.shape.dim
                )
                node.output_shapes = [shape]
            graph.add_node(node)

        # Parse operators
        for i, node in enumerate(model.graph.node):
            graph_node = GraphNode(
                node_id=f"op_{i}_{node.op_type}",
                node_type=NodeType.OPERATOR,
                op_type=node.op_type,
                name=node.name or f"{node.op_type}_{i}",
                inputs=list(node.input),
                outputs=list(node.output),
            )

            # Parse attributes
            for attr in node.attribute:
                graph_node.attributes[attr.name] = self._parse_onnx_attribute(attr)

            # Classify node
            graph_node.is_compute_heavy = node.op_type in self.COMPUTE_HEAVY_OPS
            graph_node.is_memory_heavy = node.op_type in self.MEMORY_HEAVY_OPS

            graph.add_node(graph_node)

        # Parse outputs
        for out in model.graph.output:
            graph.output_names.append(out.name)
            node = GraphNode(
                node_id=f"output_{out.name}",
                node_type=NodeType.OUTPUT,
                name=out.name,
                inputs=[out.name],
            )
            graph.add_node(node)

        # Parse initializers (constants/weights)
        for init in model.graph.initializer:
            node = GraphNode(
                node_id=f"const_{init.name}",
                node_type=NodeType.CONSTANT,
                name=init.name,
                outputs=[init.name],
            )
            # Get shape
            node.output_shapes = [tuple(init.dims)]
            graph.add_node(node)

        self._graph = graph
        return graph

    def _parse_onnx_attribute(self, attr) -> Any:
        """Parse ONNX attribute value."""
        import onnx

        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode("utf-8")
        elif attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode("utf-8") for s in attr.strings]
        else:
            return None

    def analyze(self, graph: Optional[ModelGraph] = None) -> Dict[str, Any]:
        """Comprehensive graph analysis."""
        graph = graph or self._graph
        if not graph:
            return {}

        return {
            "statistics": self.get_statistics(graph),
            "critical_path": self.find_critical_path(graph),
            "patterns": self.detect_patterns(graph),
            "fusion_opportunities": self.find_fusion_opportunities(graph),
            "memory_footprint": self.estimate_memory_footprint(graph),
        }

    def get_statistics(self, graph: ModelGraph) -> Dict[str, Any]:
        """Get graph statistics."""
        op_counts: Dict[str, int] = defaultdict(int)

        for node in graph.nodes.values():
            if node.node_type == NodeType.OPERATOR:
                op_counts[node.op_type] += 1

        operators = [n for n in graph.nodes.values() if n.node_type == NodeType.OPERATOR]

        return {
            "total_nodes": len(graph.nodes),
            "operator_nodes": len(operators),
            "input_nodes": len(graph.input_names),
            "output_nodes": len(graph.output_names),
            "op_type_counts": dict(op_counts),
            "compute_heavy_count": sum(1 for n in operators if n.is_compute_heavy),
            "memory_heavy_count": sum(1 for n in operators if n.is_memory_heavy),
            "unique_op_types": len(op_counts),
        }

    def find_critical_path(self, graph: ModelGraph) -> List[str]:
        """Find the critical path through the graph."""
        # Topological sort with longest path tracking
        in_degree: Dict[str, int] = defaultdict(int)
        dist: Dict[str, int] = {}  # Longest distance to node
        parent: Dict[str, str] = {}  # For path reconstruction

        operators = [n for n in graph.nodes.values() if n.node_type == NodeType.OPERATOR]

        for node in operators:
            dist[node.node_id] = 0
            successors = graph.get_successors(node.node_id)
            for succ in successors:
                in_degree[succ] += 1

        # Process nodes in topological order
        queue = [n.node_id for n in operators if in_degree[n.node_id] == 0]

        while queue:
            current = queue.pop(0)
            node = graph.nodes.get(current)
            if not node:
                continue

            # Weight based on compute intensity
            weight = 10 if node.is_compute_heavy else 1

            for succ in graph.get_successors(current):
                new_dist = dist.get(current, 0) + weight
                if new_dist > dist.get(succ, 0):
                    dist[succ] = new_dist
                    parent[succ] = current

                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        # Reconstruct longest path
        if not dist:
            return []

        end_node = max(dist.keys(), key=lambda k: dist[k])
        path = [end_node]

        while end_node in parent:
            end_node = parent[end_node]
            path.append(end_node)

        graph.critical_path = list(reversed(path))
        return graph.critical_path

    def detect_patterns(self, graph: ModelGraph) -> List[SubgraphPattern]:
        """Detect known patterns in the graph."""
        patterns = []
        operators = [n for n in graph.nodes.values() if n.node_type == NodeType.OPERATOR]
        op_sequence = [n.op_type for n in operators]

        for pattern_name, pattern_ops in self.KNOWN_PATTERNS.items():
            # Simple sequence matching
            matches = self._find_sequence_matches(op_sequence, pattern_ops)
            for match_indices in matches:
                pattern = SubgraphPattern(
                    pattern_id=f"{pattern_name}_{len(patterns)}",
                    pattern_name=pattern_name,
                    nodes=[operators[i].node_id for i in match_indices],
                    is_fusible=pattern_name in {"attention_block", "mlp_block", "gelu"},
                )
                patterns.append(pattern)

        graph.detected_patterns = patterns
        return patterns

    def _find_sequence_matches(self, sequence: List[str], pattern: List[str]) -> List[List[int]]:
        """Find pattern matches in sequence (supports | for alternatives)."""
        matches = []

        for i in range(len(sequence) - len(pattern) + 1):
            matched = True
            match_indices = []

            for j, pat in enumerate(pattern):
                alternatives = pat.split("|")
                if sequence[i + j] in alternatives:
                    match_indices.append(i + j)
                else:
                    matched = False
                    break

            if matched:
                matches.append(match_indices)

        return matches

    def find_fusion_opportunities(self, graph: ModelGraph) -> List[Dict[str, Any]]:
        """Find potential kernel fusion opportunities."""
        opportunities = []
        operators = [n for n in graph.nodes.values() if n.node_type == NodeType.OPERATOR]

        # Look for consecutive elementwise operations
        for i, node in enumerate(operators[:-1]):
            next_node = operators[i + 1]

            # Check if next node is only consumer
            if len(graph.get_successors(node.node_id)) == 1:
                # Both lightweight operations
                if not node.is_compute_heavy and not next_node.is_compute_heavy:
                    opportunities.append(
                        {
                            "type": "vertical_fusion",
                            "nodes": [node.node_id, next_node.node_id],
                            "ops": [node.op_type, next_node.op_type],
                            "reason": "Consecutive lightweight operations",
                        }
                    )

        return opportunities

    def estimate_memory_footprint(self, graph: ModelGraph) -> Dict[str, int]:
        """Estimate memory footprint of the graph."""

        total_params = 0
        total_activations = 0
        peak_activation = 0

        # Estimate parameters (constants)
        for node in graph.nodes.values():
            if node.node_type == NodeType.CONSTANT:
                for shape in node.output_shapes:
                    if shape:
                        elements = 1
                        for dim in shape:
                            if dim > 0:
                                elements *= dim
                        total_params += elements * 4  # Assume float32

        # Estimate activations (rough)
        for node in graph.nodes.values():
            if node.output_shapes:
                for shape in node.output_shapes:
                    if shape:
                        elements = 1
                        for dim in shape:
                            if dim > 0:
                                elements *= dim
                        activation_size = elements * 4
                        total_activations += activation_size
                        peak_activation = max(peak_activation, activation_size)

        return {
            "parameters_bytes": total_params,
            "parameters_mb": total_params // (1024 * 1024),
            "total_activations_bytes": total_activations,
            "peak_activation_bytes": peak_activation,
            "peak_activation_mb": peak_activation // (1024 * 1024),
        }
