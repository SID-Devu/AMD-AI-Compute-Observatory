"""
AACO-SIGMA Compiler IR Reader

Reads and analyzes compiler intermediate representations.
Supports MLIR and LLVM IR for AMD GPU compilation analysis.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
from pathlib import Path
from enum import Enum, auto


class IRLevel(Enum):
    """IR abstraction level."""

    HIGH = auto()  # High-level ops (ONNX, TF, etc.)
    MID = auto()  # Mid-level (MLIR dialects)
    LOW = auto()  # Low-level (LLVM IR)
    MACHINE = auto()  # Machine code (GCN/CDNA ISA)


@dataclass
class IRInstruction:
    """A single IR instruction."""

    opcode: str
    result: str = ""
    operands: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Source location
    source_line: int = 0
    source_file: str = ""

    # Analysis metadata
    is_memory_op: bool = False
    is_compute_op: bool = False
    is_control_flow: bool = False


@dataclass
class IRBlock:
    """A basic block in IR."""

    block_id: str
    label: str = ""

    instructions: List[IRInstruction] = field(default_factory=list)

    # Control flow
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)

    # Metadata
    is_entry: bool = False
    is_exit: bool = False


@dataclass
class IRFunction:
    """A function in IR."""

    name: str
    mangled_name: str = ""

    # Signature
    return_type: str = ""
    parameters: List[Tuple[str, str]] = field(default_factory=list)  # (name, type)

    # Body
    blocks: List[IRBlock] = field(default_factory=list)

    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)

    # GPU-specific
    is_kernel: bool = False
    workgroup_size: Tuple[int, int, int] = (1, 1, 1)

    # Statistics
    instruction_count: int = 0
    register_pressure: int = 0
    shared_memory_bytes: int = 0

    def get_instruction_counts(self) -> Dict[str, int]:
        """Count instruction types."""
        counts: Dict[str, int] = {}
        for block in self.blocks:
            for inst in block.instructions:
                counts[inst.opcode] = counts.get(inst.opcode, 0) + 1
        return counts


@dataclass
class IRModule:
    """A complete IR module."""

    name: str = ""
    level: IRLevel = IRLevel.MID

    functions: List[IRFunction] = field(default_factory=list)
    globals: Dict[str, Any] = field(default_factory=dict)

    # Source info
    source_file: str = ""
    target_triple: str = ""  # e.g., "amdgcn-amd-amdhsa"

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_kernels(self) -> List[IRFunction]:
        """Get GPU kernel functions."""
        return [f for f in self.functions if f.is_kernel]


class CompilerIRReader:
    """
    Base class for compiler IR readers.
    """

    def read_file(self, path: Path) -> IRModule:
        """Read IR from file."""
        raise NotImplementedError

    def read_string(self, content: str) -> IRModule:
        """Read IR from string."""
        raise NotImplementedError

    def detect_format(self, content: str) -> str:
        """Detect IR format from content."""
        if "module {" in content or "@" in content and "func" in content:
            return "mlir"
        elif "define" in content and "@" in content:
            return "llvm"
        elif ".amdgcn" in content or "s_" in content:
            return "gcn"
        return "unknown"


class MLIRReader(CompilerIRReader):
    """
    Reads MLIR (Multi-Level IR) files.

    Supports dialects:
    - std: Standard dialect
    - gpu: GPU dialect
    - linalg: Linear algebra operations
    - tensor: Tensor operations
    - memref: Memory reference operations
    """

    # MLIR operation patterns
    FUNC_PATTERN = re.compile(r"func(?:\.func)?\s+@(\w+)\s*\(([^)]*)\)\s*(?:->)?([^{]*)\s*{")
    OP_PATTERN = re.compile(
        r'(%\w+)\s*=\s*"?(\w+(?:\.\w+)?)"?\s*(?:\(([^)]*)\))?\s*(?::)?([^}\n]*)'
    )
    GPU_KERNEL_PATTERN = re.compile(r"gpu\.launch_func\s+@(\w+)::@(\w+)")
    BLOCK_PATTERN = re.compile(r"\^(\w+)(?:\(([^)]*)\))?:")

    def read_file(self, path: Path) -> IRModule:
        """Read MLIR from file."""
        with open(path) as f:
            content = f.read()
        return self.read_string(content)

    def read_string(self, content: str) -> IRModule:
        """Parse MLIR string into IRModule."""
        module = IRModule(level=IRLevel.MID)

        # Parse functions
        for match in self.FUNC_PATTERN.finditer(content):
            func_name = match.group(1)
            params_str = match.group(2)
            return_type = match.group(3).strip() if match.group(3) else ""

            func = IRFunction(
                name=func_name,
                return_type=return_type,
            )

            # Parse parameters
            if params_str:
                for param in params_str.split(","):
                    param = param.strip()
                    parts = param.split(":")
                    if len(parts) == 2:
                        func.parameters.append((parts[0].strip(), parts[1].strip()))

            # Check if GPU kernel
            if "gpu.func" in content or f"@{func_name}" in content:
                # Look for gpu.func marker
                kernel_marker = f"gpu.func @{func_name}"
                if kernel_marker in content:
                    func.is_kernel = True

            # Extract function body
            func_start = match.end()
            func_body = self._extract_function_body(content, func_start)

            # Parse blocks and instructions
            func.blocks = self._parse_blocks(func_body)
            func.instruction_count = sum(len(b.instructions) for b in func.blocks)

            module.functions.append(func)

        # Parse target triple
        triple_match = re.search(r'llvm\.target_triple\s*=\s*"([^"]+)"', content)
        if triple_match:
            module.target_triple = triple_match.group(1)

        # Parse module name
        name_match = re.search(r"module\s+@(\w+)", content)
        if name_match:
            module.name = name_match.group(1)

        return module

    def _extract_function_body(self, content: str, start: int) -> str:
        """Extract function body by matching braces."""
        depth = 1
        end = start

        for i in range(start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        return content[start:end]

    def _parse_blocks(self, body: str) -> List[IRBlock]:
        """Parse basic blocks from function body."""
        blocks = []

        # Split by block labels
        block_splits = self.BLOCK_PATTERN.split(body)

        if not block_splits or len(block_splits) < 2:
            # Single implicit entry block
            block = IRBlock(block_id="entry", is_entry=True)
            block.instructions = self._parse_instructions(body)
            return [block]

        # Parse each block
        i = 0
        while i < len(block_splits):
            if i == 0:
                # Content before first block (entry)
                if block_splits[0].strip():
                    block = IRBlock(block_id="entry", is_entry=True)
                    block.instructions = self._parse_instructions(block_splits[0])
                    blocks.append(block)
                i += 1
            else:
                block_label = block_splits[i]
                block_splits[i + 1] if i + 1 < len(block_splits) else ""
                block_body = block_splits[i + 2] if i + 2 < len(block_splits) else ""

                block = IRBlock(block_id=block_label, label=f"^{block_label}")
                block.instructions = self._parse_instructions(block_body)
                blocks.append(block)

                i += 3

        return blocks

    def _parse_instructions(self, block_body: str) -> List[IRInstruction]:
        """Parse instructions from block body."""
        instructions = []

        for line in block_body.split("\n"):
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            match = self.OP_PATTERN.match(line)
            if match:
                result = match.group(1)
                opcode = match.group(2)
                operands_str = match.group(3) or ""

                inst = IRInstruction(
                    opcode=opcode,
                    result=result,
                    operands=[o.strip() for o in operands_str.split(",") if o.strip()],
                )

                # Classify instruction
                inst.is_memory_op = any(
                    mem_op in opcode.lower() for mem_op in ["load", "store", "memref", "alloc"]
                )
                inst.is_compute_op = any(
                    comp_op in opcode.lower()
                    for comp_op in ["add", "mul", "div", "sub", "matmul", "conv"]
                )
                inst.is_control_flow = any(
                    cf_op in opcode.lower() for cf_op in ["br", "cond_br", "return", "call"]
                )

                instructions.append(inst)

        return instructions

    def analyze_kernel(self, func: IRFunction) -> Dict[str, Any]:
        """Analyze a GPU kernel function."""
        inst_counts = func.get_instruction_counts()

        # Count memory vs compute ops
        memory_ops = sum(
            count
            for op, count in inst_counts.items()
            if any(m in op.lower() for m in ["load", "store", "memref"])
        )

        compute_ops = sum(
            count
            for op, count in inst_counts.items()
            if any(c in op.lower() for c in ["add", "mul", "div", "matmul"])
        )

        return {
            "name": func.name,
            "is_kernel": func.is_kernel,
            "instruction_count": func.instruction_count,
            "block_count": len(func.blocks),
            "memory_ops": memory_ops,
            "compute_ops": compute_ops,
            "compute_to_memory_ratio": compute_ops / memory_ops if memory_ops > 0 else float("inf"),
            "instruction_breakdown": inst_counts,
        }
