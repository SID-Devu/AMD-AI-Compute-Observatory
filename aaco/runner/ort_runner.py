"""
ONNX Runtime Inference Runner
Executes models with configurable backends and collects per-iteration metrics.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np

from aaco.core.schema import InferenceResult, InferenceIteration
from aaco.core.utils import get_monotonic_ns, ns_to_ms

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for an inference run."""

    model_path: str
    backend: str  # "cpu", "migraphx", "rocm", "cuda", etc.
    batch_size: int = 1
    warmup_iterations: int = 10
    measure_iterations: int = 100
    input_shapes: Optional[Dict[str, List[int]]] = None
    input_dtype: str = "float32"
    device_id: int = 0
    fp16_enable: bool = False
    extra_ep_config: Optional[Dict[str, Any]] = None


class ORTRunner:
    """
    ONNX Runtime inference runner with multi-backend support.
    Handles session creation, warmup, measurement, and metrics collection.
    """

    BACKEND_PROVIDERS = {
        "cpu": "CPUExecutionProvider",
        "migraphx": "MIGraphXExecutionProvider",
        "rocm": "ROCMExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
        "dml": "DmlExecutionProvider",
    }

    def __init__(self, config: RunConfig):
        self.config = config
        self.session = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.input_shapes: Dict[str, List[int]] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize ONNX Runtime session with configured backend."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

        # Get provider name
        provider = self.BACKEND_PROVIDERS.get(
            self.config.backend.lower(), f"{self.config.backend}ExecutionProvider"
        )

        # Check available providers
        available = ort.get_available_providers()
        logger.info(f"Available providers: {available}")

        if provider not in available:
            logger.warning(f"{provider} not available, falling back to CPU")
            provider = "CPUExecutionProvider"

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 0  # Use default
        sess_options.inter_op_num_threads = 0

        # Provider-specific options
        provider_options = self._get_provider_options(provider)

        # Create session
        logger.info(f"Creating session with {provider}")
        self.session = ort.InferenceSession(
            self.config.model_path,
            sess_options=sess_options,
            providers=[provider] if not provider_options else [(provider, provider_options)],
        )

        # Get input/output info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Determine input shapes
        self._resolve_input_shapes()

        self._initialized = True
        logger.info(f"Session initialized: inputs={self.input_names}, outputs={self.output_names}")

    def _get_provider_options(self, provider: str) -> Optional[Dict[str, Any]]:
        """Get provider-specific options."""
        options = {}

        if provider == "MIGraphXExecutionProvider":
            options["device_id"] = self.config.device_id
            if self.config.fp16_enable:
                options["fp16_enable"] = True
        elif provider == "ROCMExecutionProvider":
            options["device_id"] = self.config.device_id
        elif provider == "CUDAExecutionProvider":
            options["device_id"] = self.config.device_id
            if self.config.fp16_enable:
                options["cudnn_conv_use_max_workspace"] = "1"
        elif provider == "TensorrtExecutionProvider":
            options["device_id"] = self.config.device_id
            if self.config.fp16_enable:
                options["trt_fp16_enable"] = True

        # Merge extra config
        if self.config.extra_ep_config:
            options.update(self.config.extra_ep_config)

        return options if options else None

    def _resolve_input_shapes(self) -> None:
        """Resolve input shapes from config or model metadata."""
        if self.config.input_shapes:
            self.input_shapes = self.config.input_shapes
            return

        # Try to infer from model
        for inp in self.session.get_inputs():
            shape = inp.shape
            # Replace dynamic dims with batch size or defaults
            resolved_shape = []
            for i, dim in enumerate(shape):
                if isinstance(dim, str) or dim is None or dim < 0:
                    if i == 0:
                        resolved_shape.append(self.config.batch_size)
                    else:
                        resolved_shape.append(128)  # Default for sequence length etc.
                else:
                    if i == 0:
                        resolved_shape.append(self.config.batch_size)
                    else:
                        resolved_shape.append(dim)

            self.input_shapes[inp.name] = resolved_shape

    def _create_inputs(self) -> Dict[str, np.ndarray]:
        """Create input tensors for inference."""
        inputs = {}

        dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "int64": np.int64,
            "int32": np.int32,
            "uint8": np.uint8,
        }

        for inp in self.session.get_inputs():
            name = inp.name
            shape = self.input_shapes.get(name, [1])

            # Determine dtype
            ort_type = inp.type
            if "float16" in ort_type or self.config.input_dtype == "float16":
                dtype = np.float16
            elif "int64" in ort_type:
                dtype = np.int64
            elif "int32" in ort_type:
                dtype = np.int32
            else:
                dtype = dtype_map.get(self.config.input_dtype, np.float32)

            # Create random input
            if np.issubdtype(dtype, np.integer):
                inputs[name] = np.random.randint(0, 100, size=shape, dtype=dtype)
            else:
                inputs[name] = np.random.randn(*shape).astype(dtype)

        return inputs

    def run_single(self, inputs: Dict[str, np.ndarray]) -> Tuple[List[np.ndarray], float]:
        """Run a single inference and return outputs + latency in ms."""
        start = get_monotonic_ns()
        outputs = self.session.run(self.output_names, inputs)
        end = get_monotonic_ns()
        latency_ms = ns_to_ms(end - start)
        return outputs, latency_ms

    def run(
        self,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> Tuple[InferenceResult, List[InferenceIteration]]:
        """
        Run full inference benchmark with warmup and measurement.

        Args:
            progress_callback: Optional callback(iteration, total, latency_ms)

        Returns:
            Tuple of (InferenceResult summary, List of per-iteration data)
        """
        if not self._initialized:
            self.initialize()

        inputs = self._create_inputs()
        iterations: List[InferenceIteration] = []
        latencies: List[float] = []

        total_iters = self.config.warmup_iterations + self.config.measure_iterations

        # Warmup phase
        logger.info(f"Running {self.config.warmup_iterations} warmup iterations...")
        for i in range(self.config.warmup_iterations):
            t_start = get_monotonic_ns()
            _, latency = self.run_single(inputs)
            t_end = get_monotonic_ns()

            iterations.append(
                InferenceIteration(
                    iter_idx=i,
                    t_start_ns=t_start,
                    t_end_ns=t_end,
                    latency_ms=latency,
                    phase="warmup",
                )
            )

            if progress_callback:
                progress_callback(i + 1, total_iters, latency)

        # Measurement phase
        logger.info(f"Running {self.config.measure_iterations} measurement iterations...")
        for i in range(self.config.measure_iterations):
            t_start = get_monotonic_ns()
            _, latency = self.run_single(inputs)
            t_end = get_monotonic_ns()

            latencies.append(latency)
            iterations.append(
                InferenceIteration(
                    iter_idx=self.config.warmup_iterations + i,
                    t_start_ns=t_start,
                    t_end_ns=t_end,
                    latency_ms=latency,
                    phase="measure",
                )
            )

            if progress_callback:
                progress_callback(self.config.warmup_iterations + i + 1, total_iters, latency)

        # Compute summary statistics
        result = InferenceResult.from_latencies(latencies, warmup=self.config.warmup_iterations)

        logger.info(
            f"Inference complete: p50={result.p50_ms:.2f}ms, "
            f"p99={result.p99_ms:.2f}ms, "
            f"throughput={result.throughput_samples_per_sec:.1f} samples/s"
        )

        return result, iterations

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata for session recording."""
        if not self._initialized:
            self.initialize()

        info = {
            "path": self.config.model_path,
            "inputs": {},
            "outputs": {},
        }

        for inp in self.session.get_inputs():
            info["inputs"][inp.name] = {
                "shape": inp.shape,
                "type": inp.type,
            }

        for out in self.session.get_outputs():
            info["outputs"][out.name] = {
                "shape": out.shape,
                "type": out.type,
            }

        return info

    def cleanup(self) -> None:
        """Release session resources."""
        if self.session:
            del self.session
            self.session = None
        self._initialized = False


def run_inference(
    model_path: str,
    backend: str = "cpu",
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 100,
    input_shapes: Optional[Dict[str, List[int]]] = None,
    **kwargs,
) -> Tuple[InferenceResult, List[InferenceIteration]]:
    """
    Convenience function to run inference benchmark.

    Args:
        model_path: Path to ONNX model
        backend: Backend name (cpu, migraphx, rocm, cuda, etc.)
        batch_size: Input batch size
        warmup: Number of warmup iterations
        iterations: Number of measurement iterations
        input_shapes: Optional explicit input shapes
        **kwargs: Additional config options

    Returns:
        Tuple of (InferenceResult, List[InferenceIteration])
    """
    config = RunConfig(
        model_path=model_path,
        backend=backend,
        batch_size=batch_size,
        warmup_iterations=warmup,
        measure_iterations=iterations,
        input_shapes=input_shapes,
        **kwargs,
    )

    runner = ORTRunner(config)
    try:
        return runner.run()
    finally:
        runner.cleanup()
