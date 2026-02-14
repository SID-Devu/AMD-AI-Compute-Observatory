"""
AACO-SIGMA Attribution Engine

Attributes execution time from GPU kernels back to model layers.
Provides end-to-end performance visibility across the stack.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum, auto

from .counter_model import CounterReading, CounterBasedModel, BottleneckType


class AttributionMethod(Enum):
    """Methods for attributing time."""
    DIRECT = auto()      # 1:1 kernel to layer mapping
    PROPORTIONAL = auto()  # Distribute by operation count
    COUNTER_BASED = auto()  # Use counter data for attribution
    HYBRID = auto()       # Combine multiple methods


@dataclass
class KernelAttribution:
    """Attribution data for a single kernel."""
    kernel_name: str
    duration_ns: int
    
    # Counter data
    counters: Optional[CounterReading] = None
    
    # Attribution to layers
    layer_contributions: Dict[str, float] = field(default_factory=dict)
    
    # Analysis results
    bottleneck: BottleneckType = BottleneckType.UNKNOWN
    efficiency_pct: float = 0.0
    
    # Metadata
    invocation_index: int = 0
    is_fused: bool = False
    fused_ops: List[str] = field(default_factory=list)


@dataclass
class LayerAttribution:
    """Attribution data for a model layer."""
    layer_name: str
    layer_type: str  # e.g., "Linear", "Conv2d", "Attention"
    
    # Time breakdown
    total_time_ns: int = 0
    kernel_times: Dict[str, int] = field(default_factory=dict)
    
    # Compute attribution
    compute_time_ns: int = 0
    memory_time_ns: int = 0
    overhead_time_ns: int = 0
    
    # Bottleneck analysis
    primary_bottleneck: BottleneckType = BottleneckType.UNKNOWN
    bottleneck_breakdown: Dict[str, int] = field(default_factory=dict)
    
    # Efficiency
    efficiency_pct: float = 0.0
    
    # Children (for hierarchical models)
    sublayers: List["LayerAttribution"] = field(default_factory=list)
    
    def get_time_breakdown_pct(self) -> Dict[str, float]:
        """Get percentage breakdown of time."""
        if self.total_time_ns == 0:
            return {}
        
        return {
            "compute": self.compute_time_ns / self.total_time_ns * 100,
            "memory": self.memory_time_ns / self.total_time_ns * 100,
            "overhead": self.overhead_time_ns / self.total_time_ns * 100,
        }


@dataclass
class AttributionResult:
    """Complete attribution result for a model inference."""
    model_name: str
    total_time_ns: int
    
    # Layer attributions
    layers: List[LayerAttribution] = field(default_factory=list)
    
    # Kernel attributions
    kernels: List[KernelAttribution] = field(default_factory=list)
    
    # Summary
    top_layers: List[str] = field(default_factory=list)
    top_kernels: List[str] = field(default_factory=list)
    
    # Overall efficiency
    overall_efficiency_pct: float = 0.0
    primary_bottleneck: BottleneckType = BottleneckType.UNKNOWN
    
    # Warnings/issues
    warnings: List[str] = field(default_factory=list)
    
    def get_layer_time_pct(self) -> Dict[str, float]:
        """Get time percentage per layer."""
        if self.total_time_ns == 0:
            return {}
        
        return {
            layer.layer_name: layer.total_time_ns / self.total_time_ns * 100
            for layer in self.layers
        }


class AttributionEngine:
    """
    Attributes GPU execution time to model layers.
    
    Combines:
    - Kernel timing data
    - Hardware counter readings
    - Graph-to-kernel mappings
    - Layer operation counts
    """
    
    def __init__(self, method: AttributionMethod = AttributionMethod.COUNTER_BASED):
        self.method = method
        self.counter_model = CounterBasedModel()
        
        # Caches
        self._kernel_to_layer: Dict[str, List[str]] = {}
        self._layer_ops: Dict[str, Dict[str, int]] = {}
    
    def set_mapping(self, kernel_to_layer: Dict[str, List[str]]) -> None:
        """Set kernel-to-layer mapping."""
        self._kernel_to_layer = kernel_to_layer
    
    def set_layer_ops(self, layer_ops: Dict[str, Dict[str, int]]) -> None:
        """Set operation counts per layer."""
        self._layer_ops = layer_ops
    
    def attribute(self,
                  kernel_timings: List[Tuple[str, int]],
                  kernel_counters: Optional[Dict[str, CounterReading]] = None,
                  model_name: str = "") -> AttributionResult:
        """
        Attribute kernel times to model layers.
        
        Args:
            kernel_timings: List of (kernel_name, duration_ns)
            kernel_counters: Optional counter readings per kernel
            model_name: Name of the model
            
        Returns:
            AttributionResult with layer and kernel attributions
        """
        result = AttributionResult(
            model_name=model_name,
            total_time_ns=sum(t for _, t in kernel_timings),
        )
        
        # Build layer aggregations
        layer_times: Dict[str, int] = {}
        layer_kernels: Dict[str, Dict[str, int]] = {}
        
        # Process each kernel
        for kernel_name, duration_ns in kernel_timings:
            kernel_attr = KernelAttribution(
                kernel_name=kernel_name,
                duration_ns=duration_ns,
            )
            
            # Add counter analysis if available
            if kernel_counters and kernel_name in kernel_counters:
                counters = kernel_counters[kernel_name]
                kernel_attr.counters = counters
                
                analysis = self.counter_model.analyze(counters)
                kernel_attr.bottleneck = BottleneckType[analysis.get("bottleneck", "UNKNOWN")]
                kernel_attr.efficiency_pct = analysis.get("efficiency_pct", 0.0)
            
            # Attribute to layers
            layers = self._kernel_to_layer.get(kernel_name, [])
            
            if layers:
                # Distribute time among mapped layers
                if self.method == AttributionMethod.PROPORTIONAL:
                    contributions = self._proportional_attribution(
                        kernel_name, layers, duration_ns
                    )
                else:
                    # Equal distribution by default
                    time_per_layer = duration_ns / len(layers)
                    contributions = {layer: time_per_layer for layer in layers}
                
                kernel_attr.layer_contributions = contributions
                
                # Update layer aggregations
                for layer_name, layer_time in contributions.items():
                    layer_times[layer_name] = layer_times.get(layer_name, 0) + int(layer_time)
                    
                    if layer_name not in layer_kernels:
                        layer_kernels[layer_name] = {}
                    layer_kernels[layer_name][kernel_name] = (
                        layer_kernels[layer_name].get(kernel_name, 0) + duration_ns
                    )
            
            result.kernels.append(kernel_attr)
        
        # Build layer attributions
        for layer_name, total_time in layer_times.items():
            layer_attr = LayerAttribution(
                layer_name=layer_name,
                layer_type=self._infer_layer_type(layer_name),
                total_time_ns=total_time,
                kernel_times=layer_kernels.get(layer_name, {}),
            )
            
            # Aggregate bottleneck from kernels
            layer_attr.primary_bottleneck = self._aggregate_bottleneck(
                layer_name, result.kernels
            )
            
            result.layers.append(layer_attr)
        
        # Sort layers by time
        result.layers.sort(key=lambda x: x.total_time_ns, reverse=True)
        result.top_layers = [l.layer_name for l in result.layers[:5]]
        
        # Sort kernels by time
        result.kernels.sort(key=lambda x: x.duration_ns, reverse=True)
        result.top_kernels = [k.kernel_name for k in result.kernels[:5]]
        
        # Overall analysis
        result.overall_efficiency_pct = self._compute_overall_efficiency(result)
        result.primary_bottleneck = self._determine_primary_bottleneck(result)
        
        return result
    
    def _proportional_attribution(self, kernel_name: str,
                                   layers: List[str],
                                   duration_ns: int) -> Dict[str, float]:
        """Attribute kernel time proportionally based on op counts."""
        contributions: Dict[str, float] = {}
        
        # Get op counts for each layer
        total_ops = 0
        layer_ops: Dict[str, int] = {}
        
        for layer in layers:
            ops = self._layer_ops.get(layer, {})
            # Sum all operations
            layer_op_count = sum(ops.values())
            layer_ops[layer] = layer_op_count
            total_ops += layer_op_count
        
        if total_ops == 0:
            # Fall back to equal distribution
            time_per_layer = duration_ns / len(layers)
            return {layer: time_per_layer for layer in layers}
        
        # Proportional distribution
        for layer in layers:
            fraction = layer_ops[layer] / total_ops
            contributions[layer] = duration_ns * fraction
        
        return contributions
    
    def _infer_layer_type(self, layer_name: str) -> str:
        """Infer layer type from name."""
        name_lower = layer_name.lower()
        
        type_patterns = [
            ("attention", "Attention"),
            ("linear", "Linear"),
            ("matmul", "MatMul"),
            ("conv", "Conv"),
            ("norm", "Normalization"),
            ("layernorm", "LayerNorm"),
            ("softmax", "Softmax"),
            ("gelu", "GELU"),
            ("relu", "ReLU"),
            ("embed", "Embedding"),
            ("pool", "Pooling"),
        ]
        
        for pattern, layer_type in type_patterns:
            if pattern in name_lower:
                return layer_type
        
        return "Unknown"
    
    def _aggregate_bottleneck(self, layer_name: str,
                              kernels: List[KernelAttribution]) -> BottleneckType:
        """Determine primary bottleneck for a layer."""
        bottleneck_times: Dict[BottleneckType, int] = {}
        
        for kernel in kernels:
            if layer_name in kernel.layer_contributions:
                time = int(kernel.layer_contributions[layer_name])
                bottleneck_times[kernel.bottleneck] = (
                    bottleneck_times.get(kernel.bottleneck, 0) + time
                )
        
        if not bottleneck_times:
            return BottleneckType.UNKNOWN
        
        # Return bottleneck with most time
        return max(bottleneck_times, key=lambda k: bottleneck_times[k])
    
    def _compute_overall_efficiency(self, result: AttributionResult) -> float:
        """Compute overall efficiency across all kernels."""
        if not result.kernels:
            return 0.0
        
        # Weighted average by duration
        total_weighted = sum(
            k.efficiency_pct * k.duration_ns for k in result.kernels
        )
        total_duration = sum(k.duration_ns for k in result.kernels)
        
        if total_duration == 0:
            return 0.0
        
        return total_weighted / total_duration
    
    def _determine_primary_bottleneck(self, result: AttributionResult) -> BottleneckType:
        """Determine primary bottleneck across all layers."""
        bottleneck_times: Dict[BottleneckType, int] = {}
        
        for layer in result.layers:
            bottleneck_times[layer.primary_bottleneck] = (
                bottleneck_times.get(layer.primary_bottleneck, 0) + layer.total_time_ns
            )
        
        if not bottleneck_times:
            return BottleneckType.UNKNOWN
        
        return max(bottleneck_times, key=lambda k: bottleneck_times[k])
    
    def generate_report(self, result: AttributionResult) -> Dict[str, Any]:
        """Generate a detailed attribution report."""
        return {
            "summary": {
                "model_name": result.model_name,
                "total_time_ms": result.total_time_ns / 1e6,
                "overall_efficiency_pct": result.overall_efficiency_pct,
                "primary_bottleneck": result.primary_bottleneck.name,
            },
            "top_layers": [
                {
                    "name": layer.layer_name,
                    "type": layer.layer_type,
                    "time_ms": layer.total_time_ns / 1e6,
                    "pct": layer.total_time_ns / result.total_time_ns * 100 if result.total_time_ns > 0 else 0,
                    "bottleneck": layer.primary_bottleneck.name,
                }
                for layer in result.layers[:10]
            ],
            "top_kernels": [
                {
                    "name": kernel.kernel_name,
                    "time_ms": kernel.duration_ns / 1e6,
                    "pct": kernel.duration_ns / result.total_time_ns * 100 if result.total_time_ns > 0 else 0,
                    "efficiency_pct": kernel.efficiency_pct,
                }
                for kernel in result.kernels[:10]
            ],
            "warnings": result.warnings,
        }
