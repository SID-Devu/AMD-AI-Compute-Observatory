"""
AACO-SIGMA Compiler/Graph Introspection Module

Provides introspection into:
- ONNX model graphs
- Compiler IR (MLIR/LLVM)
- Optimization passes
- Graph-to-kernel mapping
"""

from .graph_analyzer import (
    GraphNode,
    GraphEdge,
    ModelGraph,
    GraphAnalyzer,
    SubgraphPattern,
)

from .compiler_ir import (
    IRModule,
    IRFunction,
    IRBlock,
    IRInstruction,
    CompilerIRReader,
    MLIRReader,
)

from .optimization_tracker import (
    OptimizationPass,
    PassResult,
    OptimizationTracker,
    PassPipelineAnalyzer,
)

from .graph_kernel_mapper import (
    GraphKernelMapping,
    NodeKernelPair,
    GraphKernelMapper,
    MappingResult,
)

__all__ = [
    # Graph analysis
    "GraphNode",
    "GraphEdge",
    "ModelGraph",
    "GraphAnalyzer",
    "SubgraphPattern",
    # Compiler IR
    "IRModule",
    "IRFunction",
    "IRBlock",
    "IRInstruction",
    "CompilerIRReader",
    "MLIRReader",
    # Optimization tracking
    "OptimizationPass",
    "PassResult",
    "OptimizationTracker",
    "PassPipelineAnalyzer",
    # Graph-kernel mapping
    "GraphKernelMapping",
    "NodeKernelPair",
    "GraphKernelMapper",
    "MappingResult",
]
