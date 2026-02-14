"""
AACO Graph Analysis Module
ONNX graph extraction and op-to-kernel mapping for model-to-hardware intelligence.
"""

from aaco.graph.extractor import ONNXGraphExtractor, GraphMetadata
from aaco.graph.mapper import OpKernelMapper, AttributionResult

__all__ = [
    "ONNXGraphExtractor",
    "GraphMetadata",
    "OpKernelMapper",
    "AttributionResult",
]
