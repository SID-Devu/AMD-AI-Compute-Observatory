"""
ONNX Graph Extractor
Extracts computational graph metadata from ONNX models for model-to-hardware analysis.
"""

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Single ONNX graph node with metadata."""
    node_id: int
    name: str
    op_type: str
    domain: str
    inputs: List[str]
    outputs: List[str]
    input_shapes: Dict[str, List[int]] = field(default_factory=dict)
    output_shapes: Dict[str, List[int]] = field(default_factory=dict)
    attributes: Dict[str, Any] = field(default_factory=dict)
    estimated_flops: Optional[float] = None
    estimated_bytes: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    """Edge connecting two nodes via a tensor."""
    src_node_id: int
    dst_node_id: int
    tensor_name: str
    tensor_shape: Optional[List[int]] = None
    tensor_dtype: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GraphMetadata:
    """Complete ONNX graph metadata for analysis."""
    model_path: str
    model_hash: str
    ir_version: int
    opset_version: int
    producer_name: str
    producer_version: str
    domain: str
    total_nodes: int
    total_params: int
    total_flops: Optional[float]
    input_info: Dict[str, Dict[str, Any]]
    output_info: Dict[str, Dict[str, Any]]
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    op_counts: Dict[str, int]
    op_categories: Dict[str, List[str]]  # op_type -> category
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["nodes"] = [n.to_dict() for n in self.nodes]
        result["edges"] = [e.to_dict() for e in self.edges]
        return result


class ONNXGraphExtractor:
    """
    Extracts computational graph from ONNX models.
    Provides foundation for model-to-hardware attribution.
    """
    
    # Op type categorization for bottleneck analysis
    OP_CATEGORIES = {
        "compute_heavy": ["MatMul", "Gemm", "Conv", "ConvTranspose", "MatMulInteger"],
        "memory_heavy": ["Concat", "Reshape", "Transpose", "Gather", "Scatter", "Slice"],
        "elementwise": ["Add", "Sub", "Mul", "Div", "Relu", "Sigmoid", "Tanh", "Softmax"],
        "reduction": ["ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "GlobalAveragePool"],
        "attention": ["Attention", "MultiHeadAttention", "QAttention", "SelfAttention"],
        "normalization": ["BatchNormalization", "LayerNormalization", "GroupNormalization"],
        "pooling": ["MaxPool", "AveragePool", "GlobalMaxPool", "GlobalAveragePool"],
        "quantization": ["QuantizeLinear", "DequantizeLinear", "QLinearMatMul", "QLinearConv"],
    }
    
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.graph = None
        self._tensor_shapes: Dict[str, List[int]] = {}
        self._tensor_dtypes: Dict[str, str] = {}
    
    def extract(self) -> GraphMetadata:
        """
        Extract full graph metadata from ONNX model.
        
        Returns:
            GraphMetadata containing all nodes, edges, and analysis.
        """
        try:
            import onnx
            from onnx import numpy_helper, shape_inference
        except ImportError:
            raise ImportError("onnx package required. Install: pip install onnx")
        
        logger.info(f"Loading ONNX model: {self.model_path}")
        self.model = onnx.load(str(self.model_path))
        
        # Run shape inference if possible
        try:
            self.model = shape_inference.infer_shapes(self.model)
        except Exception as e:
            logger.warning(f"Shape inference failed: {e}")
        
        self.graph = self.model.graph
        
        # Extract tensor shapes from value_info
        self._extract_tensor_info()
        
        # Extract nodes
        nodes = self._extract_nodes()
        
        # Extract edges
        edges = self._extract_edges(nodes)
        
        # Compute op statistics
        op_counts = self._compute_op_counts(nodes)
        op_categories = self._categorize_ops(nodes)
        
        # Model info
        model_hash = self._compute_model_hash()
        total_params = self._count_parameters()
        total_flops = self._estimate_flops(nodes)
        
        # Input/output info
        input_info = self._extract_io_info(self.graph.input)
        output_info = self._extract_io_info(self.graph.output)
        
        return GraphMetadata(
            model_path=str(self.model_path),
            model_hash=model_hash,
            ir_version=self.model.ir_version,
            opset_version=self._get_opset_version(),
            producer_name=self.model.producer_name or "unknown",
            producer_version=self.model.producer_version or "unknown",
            domain=self.model.domain or "",
            total_nodes=len(nodes),
            total_params=total_params,
            total_flops=total_flops,
            input_info=input_info,
            output_info=output_info,
            nodes=nodes,
            edges=edges,
            op_counts=op_counts,
            op_categories=op_categories,
        )
    
    def _extract_tensor_info(self) -> None:
        """Extract shape/dtype info from value_info and initializers."""
        import onnx
        
        def extract_shape_dtype(vi):
            shape = []
            dtype = "unknown"
            if vi.type.HasField("tensor_type"):
                tt = vi.type.tensor_type
                dtype = onnx.TensorProto.DataType.Name(tt.elem_type)
                if tt.HasField("shape"):
                    for dim in tt.shape.dim:
                        if dim.HasField("dim_value"):
                            shape.append(dim.dim_value)
                        elif dim.HasField("dim_param"):
                            shape.append(-1)  # Dynamic dimension
                        else:
                            shape.append(-1)
            return shape, dtype
        
        # From inputs
        for inp in self.graph.input:
            shape, dtype = extract_shape_dtype(inp)
            self._tensor_shapes[inp.name] = shape
            self._tensor_dtypes[inp.name] = dtype
        
        # From outputs
        for out in self.graph.output:
            shape, dtype = extract_shape_dtype(out)
            self._tensor_shapes[out.name] = shape
            self._tensor_dtypes[out.name] = dtype
        
        # From value_info (intermediate)
        for vi in self.graph.value_info:
            shape, dtype = extract_shape_dtype(vi)
            self._tensor_shapes[vi.name] = shape
            self._tensor_dtypes[vi.name] = dtype
    
    def _extract_nodes(self) -> List[GraphNode]:
        """Extract all graph nodes with metadata."""
        nodes = []
        
        for idx, node in enumerate(self.graph.node):
            # Extract attributes
            attrs = {}
            for attr in node.attribute:
                attrs[attr.name] = self._parse_attribute(attr)
            
            # Get input/output shapes
            input_shapes = {inp: self._tensor_shapes.get(inp, []) for inp in node.input}
            output_shapes = {out: self._tensor_shapes.get(out, []) for out in node.output}
            
            # Estimate FLOPs for compute ops
            estimated_flops = self._estimate_node_flops(node.op_type, input_shapes, attrs)
            estimated_bytes = self._estimate_node_bytes(node.op_type, input_shapes, output_shapes)
            
            graph_node = GraphNode(
                node_id=idx,
                name=node.name or f"node_{idx}",
                op_type=node.op_type,
                domain=node.domain or "",
                inputs=list(node.input),
                outputs=list(node.output),
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                attributes=attrs,
                estimated_flops=estimated_flops,
                estimated_bytes=estimated_bytes,
            )
            nodes.append(graph_node)
        
        return nodes
    
    def _parse_attribute(self, attr) -> Any:
        """Parse ONNX attribute to Python value."""
        import onnx
        
        if attr.HasField("f"):
            return attr.f
        elif attr.HasField("i"):
            return attr.i
        elif attr.HasField("s"):
            return attr.s.decode("utf-8") if isinstance(attr.s, bytes) else attr.s
        elif attr.floats:
            return list(attr.floats)
        elif attr.ints:
            return list(attr.ints)
        elif attr.strings:
            return [s.decode("utf-8") if isinstance(s, bytes) else s for s in attr.strings]
        else:
            return None
    
    def _extract_edges(self, nodes: List[GraphNode]) -> List[GraphEdge]:
        """Extract graph edges from node connections."""
        # Build output -> node mapping
        output_to_node: Dict[str, int] = {}
        for node in nodes:
            for out in node.outputs:
                output_to_node[out] = node.node_id
        
        edges = []
        for node in nodes:
            for inp in node.inputs:
                if inp in output_to_node:
                    src_node_id = output_to_node[inp]
                    edge = GraphEdge(
                        src_node_id=src_node_id,
                        dst_node_id=node.node_id,
                        tensor_name=inp,
                        tensor_shape=self._tensor_shapes.get(inp),
                        tensor_dtype=self._tensor_dtypes.get(inp),
                    )
                    edges.append(edge)
        
        return edges
    
    def _compute_op_counts(self, nodes: List[GraphNode]) -> Dict[str, int]:
        """Count occurrences of each op type."""
        counts: Dict[str, int] = {}
        for node in nodes:
            op = node.op_type
            counts[op] = counts.get(op, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))
    
    def _categorize_ops(self, nodes: List[GraphNode]) -> Dict[str, List[str]]:
        """Categorize ops for bottleneck analysis."""
        categories: Dict[str, List[str]] = {cat: [] for cat in self.OP_CATEGORIES}
        categories["other"] = []
        
        for node in nodes:
            op = node.op_type
            found = False
            for cat, ops in self.OP_CATEGORIES.items():
                if op in ops:
                    if op not in categories[cat]:
                        categories[cat].append(op)
                    found = True
                    break
            if not found and op not in categories["other"]:
                categories["other"].append(op)
        
        return categories
    
    def _estimate_node_flops(
        self, op_type: str, input_shapes: Dict[str, List[int]], attrs: Dict[str, Any]
    ) -> Optional[float]:
        """Estimate FLOPs for a single node."""
        try:
            shapes = list(input_shapes.values())
            if not shapes or not shapes[0]:
                return None
            
            if op_type in ("MatMul", "Gemm"):
                # For MatMul: [M, K] x [K, N] = 2*M*K*N FLOPs
                if len(shapes) >= 2 and len(shapes[0]) >= 2 and len(shapes[1]) >= 2:
                    a_shape = shapes[0]
                    b_shape = shapes[1]
                    if -1 in a_shape or -1 in b_shape:
                        return None
                    m = a_shape[-2] if len(a_shape) >= 2 else 1
                    k = a_shape[-1]
                    n = b_shape[-1]
                    return float(2 * m * k * n)
            
            elif op_type == "Conv":
                # Convolution FLOPs approximation
                if len(shapes) >= 2:
                    x_shape = shapes[0]  # [N, C, H, W]
                    w_shape = shapes[1]  # [OC, IC, KH, KW]
                    if len(x_shape) >= 4 and len(w_shape) >= 4:
                        if -1 in x_shape or -1 in w_shape:
                            return None
                        n, c, h, w = x_shape[-4:]
                        oc, ic, kh, kw = w_shape[-4:]
                        # Approximate output size
                        oh = h  # Simplified
                        ow = w
                        return float(2 * n * oc * oh * ow * ic * kh * kw)
            
            elif op_type in ("Add", "Sub", "Mul", "Div", "Relu"):
                # Elementwise: 1 FLOP per element
                shape = shapes[0]
                if -1 not in shape:
                    elements = 1
                    for d in shape:
                        elements *= d
                    return float(elements)
            
            return None
        except Exception:
            return None
    
    def _estimate_node_bytes(
        self, op_type: str, input_shapes: Dict[str, List[int]], output_shapes: Dict[str, List[int]]
    ) -> Optional[float]:
        """Estimate memory bytes accessed by a node."""
        try:
            total_bytes = 0
            bytes_per_elem = 4  # Assume float32
            
            for shapes in [input_shapes, output_shapes]:
                for name, shape in shapes.items():
                    if shape and -1 not in shape:
                        elements = 1
                        for d in shape:
                            elements *= d
                        total_bytes += elements * bytes_per_elem
            
            return float(total_bytes) if total_bytes > 0 else None
        except Exception:
            return None
    
    def _compute_model_hash(self) -> str:
        """Compute hash of model for reproducibility."""
        import hashlib
        with open(self.model_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    
    def _count_parameters(self) -> int:
        """Count total parameters in model."""
        import onnx
        from onnx import numpy_helper
        
        total = 0
        for init in self.graph.initializer:
            try:
                arr = numpy_helper.to_array(init)
                total += arr.size
            except Exception:
                pass
        return total
    
    def _estimate_flops(self, nodes: List[GraphNode]) -> Optional[float]:
        """Estimate total FLOPs for model."""
        total = 0
        for node in nodes:
            if node.estimated_flops:
                total += node.estimated_flops
        return total if total > 0 else None
    
    def _get_opset_version(self) -> int:
        """Get ONNX opset version."""
        for opset in self.model.opset_import:
            if opset.domain == "" or opset.domain == "ai.onnx":
                return opset.version
        return 0
    
    def _extract_io_info(self, io_list) -> Dict[str, Dict[str, Any]]:
        """Extract input/output tensor info."""
        info = {}
        for io in io_list:
            info[io.name] = {
                "shape": self._tensor_shapes.get(io.name, []),
                "dtype": self._tensor_dtypes.get(io.name, "unknown"),
            }
        return info
    
    def get_op_summary(self) -> Dict[str, Any]:
        """Get quick summary of ops in model."""
        metadata = self.extract()
        
        # Compute category totals
        category_counts = {}
        for cat, ops in metadata.op_categories.items():
            count = sum(metadata.op_counts.get(op, 0) for op in ops)
            if count > 0:
                category_counts[cat] = count
        
        compute_heavy = category_counts.get("compute_heavy", 0)
        memory_heavy = category_counts.get("memory_heavy", 0)
        
        # Compute-to-memory ratio (higher = more compute intensive)
        cm_ratio = compute_heavy / memory_heavy if memory_heavy > 0 else float("inf")
        
        return {
            "total_nodes": metadata.total_nodes,
            "total_params": metadata.total_params,
            "estimated_gflops": metadata.total_flops / 1e9 if metadata.total_flops else None,
            "op_counts": metadata.op_counts,
            "category_counts": category_counts,
            "compute_memory_ratio": cm_ratio,
            "top_5_ops": list(metadata.op_counts.items())[:5],
        }
    
    def save_to_session(self, session_path: Path) -> None:
        """Save graph metadata to session folder."""
        model_dir = session_path / "model"
        model_dir.mkdir(exist_ok=True)
        
        metadata = self.extract()
        
        # Save full metadata
        with open(model_dir / "graph_metadata.json", "w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        
        # Save nodes as separate file for analysis
        nodes_data = [n.to_dict() for n in metadata.nodes]
        with open(model_dir / "graph_nodes.json", "w") as f:
            json.dump(nodes_data, f, indent=2)
        
        # Save edges
        edges_data = [e.to_dict() for e in metadata.edges]
        with open(model_dir / "graph_edges.json", "w") as f:
            json.dump(edges_data, f, indent=2)
        
        # Save op summary
        summary = self.get_op_summary()
        with open(model_dir / "op_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved graph metadata to {model_dir}")


def extract_graph(model_path: str) -> GraphMetadata:
    """Convenience function to extract graph from model."""
    extractor = ONNXGraphExtractor(model_path)
    return extractor.extract()
