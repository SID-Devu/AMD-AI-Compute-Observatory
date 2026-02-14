"""
AACO-SIGMA Graph to Kernel Mapper

Maps high-level model graph nodes to low-level GPU kernels.
Critical for performance attribution from model ops to hardware execution.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Tuple
from pathlib import Path
from enum import Enum, auto
import hashlib


class MappingConfidence(Enum):
    """Confidence level of a graph-to-kernel mapping."""
    EXACT = auto()       # Debug info confirms mapping
    HIGH = auto()        # Strong heuristic match
    MEDIUM = auto()      # Partial match, some ambiguity
    LOW = auto()         # Weak/inferred mapping
    UNKNOWN = auto()     # Could not determine mapping


@dataclass
class NodeKernelPair:
    """A mapping between a graph node and GPU kernel."""
    
    # Graph node info
    node_id: str
    node_name: str
    node_type: str  # e.g., "MatMul", "Conv", "Softmax"
    
    # Kernel info
    kernel_name: str
    kernel_mangled_name: str = ""
    
    # Mapping metadata
    confidence: MappingConfidence = MappingConfidence.UNKNOWN
    mapping_source: str = ""  # How mapping was determined
    
    # Attribution
    attribution_fraction: float = 1.0  # Fraction of kernel attributed to node
    
    # Debug info
    source_file: str = ""
    source_line: int = 0


@dataclass
class GraphKernelMapping:
    """Complete mapping between model graph and GPU kernels."""
    
    model_name: str = ""
    model_hash: str = ""
    
    # Mappings
    node_to_kernels: Dict[str, List[str]] = field(default_factory=dict)
    kernel_to_nodes: Dict[str, List[str]] = field(default_factory=dict)
    
    # Detailed pairs
    pairs: List[NodeKernelPair] = field(default_factory=list)
    
    # Unmapped
    unmapped_nodes: Set[str] = field(default_factory=set)
    unmapped_kernels: Set[str] = field(default_factory=set)
    
    # Statistics
    total_nodes: int = 0
    total_kernels: int = 0
    mapped_count: int = 0
    
    def get_coverage(self) -> float:
        """Get mapping coverage percentage."""
        if self.total_nodes == 0:
            return 0.0
        return (len(self.node_to_kernels) / self.total_nodes) * 100.0
    
    def get_kernels_for_node(self, node_id: str) -> List[str]:
        """Get kernel names for a graph node."""
        return self.node_to_kernels.get(node_id, [])
    
    def get_nodes_for_kernel(self, kernel_name: str) -> List[str]:
        """Get graph node IDs for a kernel."""
        return self.kernel_to_nodes.get(kernel_name, [])


@dataclass
class MappingResult:
    """Result of a mapping operation."""
    success: bool
    mapping: Optional[GraphKernelMapping] = None
    error_message: str = ""
    warnings: List[str] = field(default_factory=list)


class GraphKernelMapper:
    """
    Maps model graph operations to GPU kernels.
    
    Strategies:
    1. Debug info: Use compiler-emitted source locations
    2. Name matching: Match node names to kernel names
    3. Pattern matching: Use op type to kernel family mapping
    4. Timing correlation: Match execution profiles
    """
    
    # Known op-to-kernel-family mappings
    OP_KERNEL_PATTERNS = {
        # GEMM operations
        "MatMul": ["gemm", "rocblas_sgemm", "rocblas_hgemm", "Cijk"],
        "Gemm": ["gemm", "rocblas_sgemm", "rocblas_hgemm", "Cijk"],
        "FusedMatMul": ["gemm", "rocblas_sgemm", "Cijk"],
        
        # Convolution
        "Conv": ["conv", "miopen", "implicit_gemm", "winograd"],
        "ConvTranspose": ["conv", "miopen"],
        "DepthwiseConv": ["conv", "miopen"],
        
        # Attention
        "Attention": ["attention", "flash", "sdpa"],
        "MultiHeadAttention": ["attention", "mha"],
        "Softmax": ["softmax"],
        
        # Elementwise
        "Add": ["elementwise", "add"],
        "Mul": ["elementwise", "mul"],
        "Relu": ["relu", "activation"],
        "Gelu": ["gelu", "activation"],
        "Sigmoid": ["sigmoid", "activation"],
        
        # Reduction
        "ReduceMean": ["reduce", "mean"],
        "ReduceSum": ["reduce", "sum"],
        "LayerNormalization": ["layernorm", "ln"],
        "BatchNormalization": ["batchnorm", "bn"],
        
        # Memory
        "Transpose": ["transpose", "permute"],
        "Reshape": [],  # Usually no kernel
        "Concat": ["concat"],
        "Split": ["split"],
    }
    
    def __init__(self):
        self._debug_info_cache: Dict[str, Dict[str, str]] = {}
        self._timing_profiles: Dict[str, List[float]] = {}
    
    def map_by_debug_info(self, 
                          graph_nodes: List[Dict[str, Any]],
                          kernel_debug_info: Dict[str, Tuple[str, int]]) -> GraphKernelMapping:
        """
        Map using compiler debug information.
        
        Args:
            graph_nodes: List of graph nodes with 'id', 'name', 'op_type'
            kernel_debug_info: Kernel name -> (source_file, line_number)
        """
        mapping = GraphKernelMapping()
        mapping.total_nodes = len(graph_nodes)
        mapping.total_kernels = len(kernel_debug_info)
        
        for node in graph_nodes:
            node_id = node.get("id", "")
            node_name = node.get("name", "")
            
            # Look for debug info pointing to this node
            for kernel_name, (src_file, src_line) in kernel_debug_info.items():
                # Check if source location references this node
                if node_name in src_file or node_id in src_file:
                    pair = NodeKernelPair(
                        node_id=node_id,
                        node_name=node_name,
                        node_type=node.get("op_type", ""),
                        kernel_name=kernel_name,
                        confidence=MappingConfidence.EXACT,
                        mapping_source="debug_info",
                        source_file=src_file,
                        source_line=src_line,
                    )
                    mapping.pairs.append(pair)
                    
                    if node_id not in mapping.node_to_kernels:
                        mapping.node_to_kernels[node_id] = []
                    mapping.node_to_kernels[node_id].append(kernel_name)
                    
                    if kernel_name not in mapping.kernel_to_nodes:
                        mapping.kernel_to_nodes[kernel_name] = []
                    mapping.kernel_to_nodes[kernel_name].append(node_id)
        
        # Track unmapped
        all_node_ids = {n.get("id") for n in graph_nodes}
        mapping.unmapped_nodes = all_node_ids - set(mapping.node_to_kernels.keys())
        mapping.unmapped_kernels = set(kernel_debug_info.keys()) - set(mapping.kernel_to_nodes.keys())
        mapping.mapped_count = len(mapping.node_to_kernels)
        
        return mapping
    
    def map_by_name(self,
                    graph_nodes: List[Dict[str, Any]], 
                    kernel_names: List[str]) -> GraphKernelMapping:
        """
        Map using name similarity matching.
        """
        mapping = GraphKernelMapping()
        mapping.total_nodes = len(graph_nodes)
        mapping.total_kernels = len(kernel_names)
        
        for node in graph_nodes:
            node_id = node.get("id", "")
            node_name = node.get("name", "").lower()
            op_type = node.get("op_type", "")
            
            # Try direct name match
            for kernel_name in kernel_names:
                kernel_lower = kernel_name.lower()
                
                # Check name containment
                if node_name and node_name in kernel_lower:
                    confidence = MappingConfidence.HIGH
                elif op_type.lower() in kernel_lower:
                    confidence = MappingConfidence.MEDIUM
                else:
                    continue
                
                pair = NodeKernelPair(
                    node_id=node_id,
                    node_name=node.get("name", ""),
                    node_type=op_type,
                    kernel_name=kernel_name,
                    confidence=confidence,
                    mapping_source="name_match",
                )
                mapping.pairs.append(pair)
                
                if node_id not in mapping.node_to_kernels:
                    mapping.node_to_kernels[node_id] = []
                mapping.node_to_kernels[node_id].append(kernel_name)
        
        mapping.mapped_count = len(mapping.node_to_kernels)
        return mapping
    
    def map_by_pattern(self,
                       graph_nodes: List[Dict[str, Any]],
                       kernel_names: List[str]) -> GraphKernelMapping:
        """
        Map using op-type to kernel-family patterns.
        """
        mapping = GraphKernelMapping()
        mapping.total_nodes = len(graph_nodes)
        mapping.total_kernels = len(kernel_names)
        
        for node in graph_nodes:
            node_id = node.get("id", "")
            node_name = node.get("name", "")
            op_type = node.get("op_type", "")
            
            # Get expected kernel patterns for this op type
            patterns = self.OP_KERNEL_PATTERNS.get(op_type, [])
            
            for kernel_name in kernel_names:
                kernel_lower = kernel_name.lower()
                
                for pattern in patterns:
                    if pattern in kernel_lower:
                        pair = NodeKernelPair(
                            node_id=node_id,
                            node_name=node_name,
                            node_type=op_type,
                            kernel_name=kernel_name,
                            confidence=MappingConfidence.MEDIUM,
                            mapping_source=f"pattern:{pattern}",
                        )
                        mapping.pairs.append(pair)
                        
                        if node_id not in mapping.node_to_kernels:
                            mapping.node_to_kernels[node_id] = []
                        if kernel_name not in mapping.node_to_kernels[node_id]:
                            mapping.node_to_kernels[node_id].append(kernel_name)
                        
                        break  # One pattern match per kernel is enough
        
        mapping.mapped_count = len(mapping.node_to_kernels)
        return mapping
    
    def map_by_timing(self,
                      graph_nodes: List[Dict[str, Any]],
                      node_timings: Dict[str, float],
                      kernel_timings: Dict[str, float],
                      tolerance_pct: float = 10.0) -> GraphKernelMapping:
        """
        Map using execution timing correlation.
        
        Matches nodes to kernels with similar execution times.
        """
        mapping = GraphKernelMapping()
        mapping.total_nodes = len(graph_nodes)
        mapping.total_kernels = len(kernel_timings)
        
        for node in graph_nodes:
            node_id = node.get("id", "")
            if node_id not in node_timings:
                continue
            
            node_time = node_timings[node_id]
            
            for kernel_name, kernel_time in kernel_timings.items():
                # Check if times are within tolerance
                if node_time > 0:
                    diff_pct = abs(kernel_time - node_time) / node_time * 100
                    
                    if diff_pct <= tolerance_pct:
                        pair = NodeKernelPair(
                            node_id=node_id,
                            node_name=node.get("name", ""),
                            node_type=node.get("op_type", ""),
                            kernel_name=kernel_name,
                            confidence=MappingConfidence.LOW,
                            mapping_source=f"timing_correlation:{diff_pct:.1f}%",
                        )
                        mapping.pairs.append(pair)
                        
                        if node_id not in mapping.node_to_kernels:
                            mapping.node_to_kernels[node_id] = []
                        mapping.node_to_kernels[node_id].append(kernel_name)
        
        mapping.mapped_count = len(mapping.node_to_kernels)
        return mapping
    
    def merge_mappings(self, *mappings: GraphKernelMapping) -> GraphKernelMapping:
        """
        Merge multiple mapping results, preferring higher confidence.
        """
        merged = GraphKernelMapping()
        
        # Track best confidence per (node_id, kernel_name)
        best_pairs: Dict[Tuple[str, str], NodeKernelPair] = {}
        
        for mapping in mappings:
            merged.total_nodes = max(merged.total_nodes, mapping.total_nodes)
            merged.total_kernels = max(merged.total_kernels, mapping.total_kernels)
            
            for pair in mapping.pairs:
                key = (pair.node_id, pair.kernel_name)
                
                if key not in best_pairs:
                    best_pairs[key] = pair
                else:
                    # Keep higher confidence mapping
                    existing = best_pairs[key]
                    if pair.confidence.value < existing.confidence.value:
                        best_pairs[key] = pair
        
        # Build merged result
        for (node_id, kernel_name), pair in best_pairs.items():
            merged.pairs.append(pair)
            
            if node_id not in merged.node_to_kernels:
                merged.node_to_kernels[node_id] = []
            merged.node_to_kernels[node_id].append(kernel_name)
            
            if kernel_name not in merged.kernel_to_nodes:
                merged.kernel_to_nodes[kernel_name] = []
            merged.kernel_to_nodes[kernel_name].append(node_id)
        
        merged.mapped_count = len(merged.node_to_kernels)
        return merged
    
    def compute_attribution(self, mapping: GraphKernelMapping,
                            kernel_timings: Dict[str, float]) -> Dict[str, float]:
        """
        Compute time attribution from kernels to graph nodes.
        
        Distributes kernel time among mapped nodes.
        """
        node_times: Dict[str, float] = {}
        
        for kernel_name, kernel_time in kernel_timings.items():
            nodes = mapping.get_nodes_for_kernel(kernel_name)
            
            if not nodes:
                continue
            
            # Distribute time equally among mapped nodes
            time_per_node = kernel_time / len(nodes)
            
            for node_id in nodes:
                node_times[node_id] = node_times.get(node_id, 0.0) + time_per_node
        
        return node_times
    
    def get_mapping_report(self, mapping: GraphKernelMapping) -> Dict[str, Any]:
        """Generate a detailed mapping report."""
        
        confidence_dist: Dict[str, int] = {}
        for pair in mapping.pairs:
            conf_name = pair.confidence.name
            confidence_dist[conf_name] = confidence_dist.get(conf_name, 0) + 1
        
        return {
            "total_nodes": mapping.total_nodes,
            "total_kernels": mapping.total_kernels,
            "mapped_nodes": len(mapping.node_to_kernels),
            "mapped_kernels": len(mapping.kernel_to_nodes),
            "coverage_pct": mapping.get_coverage(),
            "confidence_distribution": confidence_dist,
            "unmapped_nodes_count": len(mapping.unmapped_nodes),
            "unmapped_kernels_count": len(mapping.unmapped_kernels),
            "multi_kernel_nodes": sum(
                1 for kernels in mapping.node_to_kernels.values() 
                if len(kernels) > 1
            ),
            "multi_node_kernels": sum(
                1 for nodes in mapping.kernel_to_nodes.values()
                if len(nodes) > 1
            ),
        }
