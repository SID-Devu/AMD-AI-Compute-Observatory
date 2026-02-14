"""
AACO-SIGMA Code Generator

Generates optimized code variants based on optimization rules.
Supports HIP kernel generation, configuration changes, and graph transforms.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum, auto

from .optimization_rule import RuleAction


class VariantType(Enum):
    """Types of code variants."""

    KERNEL_HIP = auto()  # HIP kernel source
    CONFIG_PYTHON = auto()  # Python configuration
    GRAPH_TRANSFORM = auto()  # Graph transformation
    LAUNCH_CONFIG = auto()  # Kernel launch configuration
    BUILD_FLAG = auto()  # Compiler flags


@dataclass
class CodeVariant:
    """A generated code variant."""

    variant_id: str
    variant_type: VariantType

    # Source
    source_code: str = ""

    # Metadata
    description: str = ""
    expected_improvement_pct: float = 0.0

    # Context
    target_kernel: Optional[str] = None
    target_file: Optional[str] = None

    # Validation
    compilation_status: Optional[str] = None
    correctness_verified: bool = False


@dataclass
class GeneratedCode:
    """Collection of generated code for an optimization."""

    # Variants
    variants: List[CodeVariant] = field(default_factory=list)

    # Summary
    primary_variant: Optional[CodeVariant] = None

    # Instructions
    apply_instructions: str = ""
    rollback_instructions: str = ""


class CodeGenerator:
    """
    Generates optimized code based on optimization rules.

    Templates:
    - HIP kernel optimizations
    - Configuration changes
    - Launch parameter modifications
    """

    def __init__(self):
        self._variant_counter = 0
        self._templates: Dict[str, str] = self._load_templates()

    def generate(self, action: RuleAction, context: Dict[str, Any]) -> GeneratedCode:
        """
        Generate code for an optimization action.
        """
        result = GeneratedCode()

        if action.action_type == "config_change":
            variant = self._generate_config_change(action, context)
            result.variants.append(variant)

        elif action.action_type == "code_transform":
            variants = self._generate_transform(action, context)
            result.variants.extend(variants)

        elif action.action_type == "suggestion":
            variant = self._generate_suggestion(action, context)
            result.variants.append(variant)

        if result.variants:
            result.primary_variant = result.variants[0]
            result.apply_instructions = self._generate_instructions(result.variants)

        return result

    def _generate_config_change(self, action: RuleAction, context: Dict[str, Any]) -> CodeVariant:
        """Generate configuration change code."""
        self._variant_counter += 1

        target = action.target
        new_value = action.new_value

        # Generate Python config code
        if target == "batch_size":
            if new_value == "double":
                code = self._templates["batch_size_double"].format(
                    current=context.get("batch_size", 32)
                )
            else:
                code = f"# Set batch size\nbatch_size = {new_value}"

        elif target == "dtype":
            code = self._templates["dtype_change"].format(
                new_dtype=new_value, model_var=context.get("model_var", "model")
            )

        elif target == "workgroup_size":
            code = self._templates["workgroup_size"].format(size=new_value)

        elif target == "max_registers":
            code = self._templates["max_registers"].format(max_regs=new_value)

        elif target == "tensor_layout":
            code = self._templates["tensor_layout"].format(new_layout=new_value)

        else:
            code = f"# Config change: {target} = {new_value}"

        return CodeVariant(
            variant_id=f"var_{self._variant_counter:04d}",
            variant_type=VariantType.CONFIG_PYTHON,
            source_code=code,
            description=action.description,
        )

    def _generate_transform(self, action: RuleAction, context: Dict[str, Any]) -> List[CodeVariant]:
        """Generate code transform variants."""
        variants = []
        transform = action.transform
        params = action.transform_params

        if transform == "fuse_elementwise":
            variants.append(self._gen_fusion_code(context))

        elif transform == "tile_for_cache":
            tile_size = params.get("tile_size", "auto")
            variants.append(self._gen_tiling_code(context, tile_size))

        elif transform == "use_mfma_intrinsics":
            variants.append(self._gen_mfma_code(context))

        elif transform == "transpose_for_coalescing":
            variants.append(self._gen_transpose_code(context))

        elif transform == "fuse_matmul_bias_activation":
            variants.append(self._gen_matmul_fusion_code(context))

        else:
            # Generic transform stub
            self._variant_counter += 1
            variants.append(
                CodeVariant(
                    variant_id=f"var_{self._variant_counter:04d}",
                    variant_type=VariantType.GRAPH_TRANSFORM,
                    source_code=f"# Transform: {transform}\n# Params: {params}",
                    description=action.description,
                )
            )

        return variants

    def _generate_suggestion(self, action: RuleAction, context: Dict[str, Any]) -> CodeVariant:
        """Generate suggestion as comment."""
        self._variant_counter += 1

        code = f"""# OPTIMIZATION SUGGESTION
# ======================
# {action.description}
#
# Target: {action.target}
# 
# Review the following code and consider applying this optimization:
"""

        return CodeVariant(
            variant_id=f"var_{self._variant_counter:04d}",
            variant_type=VariantType.CONFIG_PYTHON,
            source_code=code,
            description=action.description,
        )

    def _gen_fusion_code(self, context: Dict[str, Any]) -> CodeVariant:
        """Generate kernel fusion code."""
        self._variant_counter += 1

        code = '''# Fused elementwise kernel
import torch
from torch.cuda import amp

def fused_elementwise(x, w1, w2, bias):
    """Fused: x * w1 + x * w2 + bias"""
    with amp.autocast():
        return torch.addcmul(bias, x, w1 + w2)

# Alternative using torch.compile (PyTorch 2.0+)
@torch.compile(mode="reduce-overhead")
def fused_elementwise_compiled(x, w1, w2, bias):
    return x * w1 + x * w2 + bias
'''

        return CodeVariant(
            variant_id=f"var_{self._variant_counter:04d}",
            variant_type=VariantType.GRAPH_TRANSFORM,
            source_code=code,
            description="Fused elementwise operations",
            expected_improvement_pct=20.0,
        )

    def _gen_tiling_code(self, context: Dict[str, Any], tile_size: Any) -> CodeVariant:
        """Generate cache tiling code."""
        self._variant_counter += 1

        if tile_size == "auto":
            # Calculate optimal tile size based on L2 cache
            l2_size_kb = context.get("l2_cache_kb", 8192)
            tile_size = min(256, int((l2_size_kb * 1024 / 4) ** 0.5))

        code = f"""# Cache-optimized tiling
# Tile size: {tile_size} (fits in L2 cache)

TILE_SIZE = {tile_size}

__global__ void tiled_kernel(float* A, float* B, float* C, int N) {{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {{
        // Load tiles into shared memory
        int aRow = by + ty;
        int aCol = tile * TILE_SIZE + tx;
        As[ty][tx] = (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.0f;
        
        int bRow = tile * TILE_SIZE + ty;
        int bCol = bx + tx;
        Bs[ty][tx] = (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        
        __syncthreads();
        
        // Compute tile contribution
        for (int k = 0; k < TILE_SIZE; ++k) {{
            sum += As[ty][k] * Bs[k][tx];
        }}
        
        __syncthreads();
    }}
    
    int row = by + ty;
    int col = bx + tx;
    if (row < N && col < N) {{
        C[row * N + col] = sum;
    }}
}}
"""

        return CodeVariant(
            variant_id=f"var_{self._variant_counter:04d}",
            variant_type=VariantType.KERNEL_HIP,
            source_code=code,
            description=f"Cache-tiled kernel with tile size {tile_size}",
            expected_improvement_pct=20.0,
        )

    def _gen_mfma_code(self, context: Dict[str, Any]) -> CodeVariant:
        """Generate MFMA intrinsics code."""
        self._variant_counter += 1

        code = """// Matrix multiply using AMD MFMA intrinsics
// Requires CDNA architecture (MI100, MI200, MI300)

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_runtime.h>

// MFMA 32x32x8 for FP16
__global__ void mfma_matmul_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // Thread block covers 32x32 output tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * 32;
    int by = blockIdx.y * 32;
    
    // Accumulator registers
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Load A and B fragments
    __half a_frag[8];
    __half b_frag[8];
    
    for (int k = 0; k < K; k += 8) {
        // Load 8 elements from A (row-major)
        for (int i = 0; i < 8; i++) {
            int aRow = by + ty;
            int aCol = k + i;
            a_frag[i] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : __float2half(0.0f);
        }
        
        // Load 8 elements from B (col-major for this tile)
        for (int i = 0; i < 8; i++) {
            int bRow = k + i;
            int bCol = bx + tx;
            b_frag[i] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : __float2half(0.0f);
        }
        
        // MFMA instruction: 32x32x8 FP16 -> FP32
        // amdgcn intrinsic: __builtin_amdgcn_mfma_f32_32x32x8f16
        asm volatile(
            "v_mfma_f32_32x32x8f16 %0, %1, %2, %0"
            : "+v"(acc[0])
            : "v"(*(uint32_t*)&a_frag[0]), "v"(*(uint32_t*)&b_frag[0])
        );
    }
    
    // Store results
    int cRow = by + ty;
    int cCol = bx + tx;
    if (cRow < M && cCol < N) {
        C[cRow * N + cCol] = acc[0];
    }
}
"""

        return CodeVariant(
            variant_id=f"var_{self._variant_counter:04d}",
            variant_type=VariantType.KERNEL_HIP,
            source_code=code,
            description="Matrix multiply using AMD MFMA intrinsics",
            expected_improvement_pct=30.0,
        )

    def _gen_transpose_code(self, context: Dict[str, Any]) -> CodeVariant:
        """Generate memory coalescing transpose."""
        self._variant_counter += 1

        code = '''# Memory-coalesced tensor operations
import torch

def coalesce_for_gemm(tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure tensor is contiguous in memory for coalesced access.
    For GEMM, we want the K dimension contiguous for A,
    and M dimension contiguous for B.
    """
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor

def transpose_for_coalescing(A: torch.Tensor, B: torch.Tensor):
    """
    Transpose tensors if needed for coalesced memory access.
    
    For C = A @ B:
    - A should be row-major (last dim contiguous)
    - B should be column-major for this tile
    """
    # Check strides
    if A.stride(-1) != 1:
        A = A.contiguous()
    
    # B might benefit from transposition
    if B.stride(0) == 1:  # Already column-major
        pass
    else:
        # Create transposed view
        B = B.T.contiguous().T
    
    return A, B

# Using torch.compile for automatic optimization
@torch.compile(mode="reduce-overhead", fullgraph=True)
def optimized_matmul(A, B):
    A, B = transpose_for_coalescing(A, B)
    return torch.matmul(A, B)
'''

        return CodeVariant(
            variant_id=f"var_{self._variant_counter:04d}",
            variant_type=VariantType.CONFIG_PYTHON,
            source_code=code,
            description="Memory coalescing optimization",
            expected_improvement_pct=15.0,
        )

    def _gen_matmul_fusion_code(self, context: Dict[str, Any]) -> CodeVariant:
        """Generate MatMul + Bias + Activation fusion."""
        self._variant_counter += 1

        code = '''# Fused MatMul + Bias + Activation
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedLinear(nn.Module):
    """
    Fused Linear + Bias + Activation for better performance.
    Reduces memory bandwidth by computing in a single kernel.
    """
    
    def __init__(self, in_features, out_features, activation='gelu'):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.activation = activation
        
        # Initialize
        nn.init.kaiming_uniform_(self.weight)
    
    def forward(self, x):
        # Use torch._C._nn.linear_gelu for fused path if available
        if hasattr(torch._C._nn, 'linear_gelu') and self.activation == 'gelu':
            return torch._C._nn.linear_gelu(x, self.weight, self.bias)
        
        # Fallback: Use torch.compile for fusion
        return self._fused_forward(x)
    
    @torch.compile(mode="reduce-overhead")
    def _fused_forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.activation == 'gelu':
            out = F.gelu(out)
        elif self.activation == 'relu':
            out = F.relu(out)
        elif self.activation == 'silu':
            out = F.silu(out)
        return out

# Replace nn.Linear + activation with FusedLinear
def convert_to_fused(model):
    """Convert Linear layers followed by activation to FusedLinear."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Check if next operation is activation
            # (simplified - real implementation would trace graph)
            fused = FusedLinear(
                module.in_features,
                module.out_features,
                activation='gelu'
            )
            fused.weight.data = module.weight.data
            fused.bias.data = module.bias.data
            setattr(model, name, fused)
        else:
            convert_to_fused(module)
    return model
'''

        return CodeVariant(
            variant_id=f"var_{self._variant_counter:04d}",
            variant_type=VariantType.GRAPH_TRANSFORM,
            source_code=code,
            description="Fused MatMul + Bias + Activation",
            expected_improvement_pct=15.0,
        )

    def _generate_instructions(self, variants: List[CodeVariant]) -> str:
        """Generate application instructions."""
        lines = ["# How to Apply These Optimizations", ""]

        for i, variant in enumerate(variants, 1):
            lines.append(f"## Step {i}: {variant.description}")
            lines.append("")

            if variant.variant_type == VariantType.CONFIG_PYTHON:
                lines.append("Add the following to your Python code:")
            elif variant.variant_type == VariantType.KERNEL_HIP:
                lines.append("Replace your kernel with this optimized version:")
            elif variant.variant_type == VariantType.GRAPH_TRANSFORM:
                lines.append("Apply this graph transformation:")

            lines.append("")
            lines.append("```")
            lines.append(
                variant.source_code[:500] + ("..." if len(variant.source_code) > 500 else "")
            )
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def _load_templates(self) -> Dict[str, str]:
        """Load code generation templates."""
        return {
            "batch_size_double": """# Double batch size for better utilization
# Previous: batch_size = {current}
batch_size = {current} * 2
print(f"Increased batch size to {{batch_size}}")
""",
            "dtype_change": """# Change precision to {new_dtype}
{model_var} = {model_var}.to(torch.{new_dtype})
# Enable autocast for mixed precision
with torch.autocast(device_type='cuda', dtype=torch.{new_dtype}):
    output = {model_var}(input)
""",
            "workgroup_size": """# Set optimal workgroup size
__launch_bounds__({size})
__global__ void optimized_kernel(...) {{
    // Kernel body
}}
""",
            "max_registers": """# Limit register usage for better occupancy
__launch_bounds__(256, {max_regs})
__global__ void register_limited_kernel(...) {{
    // Kernel body  
}}
""",
            "tensor_layout": """# Change tensor layout to {new_layout}
# For convolutions, {new_layout} often provides better vectorization
tensor = tensor.to(memory_format=torch.channels_last)  # For NCHW->NHWC
# Or use contiguous() after transposing
""",
        }
