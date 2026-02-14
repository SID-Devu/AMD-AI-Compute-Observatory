"""
Token-level LLM Profiler
Detailed profiling for LLM inference workloads with prefill/decode separation.
"""

import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TokenTiming:
    """Timing for a single token generation."""

    token_idx: int
    t_start_ns: int
    t_end_ns: int
    latency_ms: float
    phase: str  # "prefill" or "decode"
    cumulative_latency_ms: float
    token_id: Optional[int] = None


@dataclass
class PhaseSummary:
    """Summary statistics for a generation phase."""

    phase: str
    token_count: int
    total_time_ms: float
    mean_latency_ms: float
    p50_latency_ms: float
    p90_latency_ms: float
    p99_latency_ms: float
    std_latency_ms: float
    tokens_per_sec: float
    first_token_latency_ms: Optional[float] = None  # For decode phase


@dataclass
class LLMProfileResult:
    """Complete LLM profiling result."""

    model_name: str
    prompt_length: int
    generation_length: int
    total_tokens: int
    total_time_ms: float

    # Phase summaries
    prefill: PhaseSummary
    decode: PhaseSummary

    # Key metrics
    time_to_first_token_ms: float  # TTFT
    tokens_per_second: float  # Overall
    prefill_tokens_per_sec: float
    decode_tokens_per_sec: float

    # Per-token data
    token_timings: List[TokenTiming]

    # Drift analysis
    latency_drift_pct: float  # How much decode latency increases over time
    decode_stability: float  # 1 - CoV

    # Bottleneck indicators
    prefill_bound: bool  # Is prefill taking disproportionate time?
    decode_memory_pressure: bool  # Is decode slowing due to KV cache growth?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "prompt_length": self.prompt_length,
            "generation_length": self.generation_length,
            "total_tokens": self.total_tokens,
            "total_time_ms": self.total_time_ms,
            "prefill": asdict(self.prefill),
            "decode": asdict(self.decode),
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "tokens_per_second": self.tokens_per_second,
            "prefill_tokens_per_sec": self.prefill_tokens_per_sec,
            "decode_tokens_per_sec": self.decode_tokens_per_sec,
            "latency_drift_pct": self.latency_drift_pct,
            "decode_stability": self.decode_stability,
            "prefill_bound": self.prefill_bound,
            "decode_memory_pressure": self.decode_memory_pressure,
            "token_timings": [asdict(t) for t in self.token_timings],
        }


class LLMProfiler:
    """
    Profiles LLM inference at token granularity.

    Captures:
    - Prefill phase (processing prompt)
    - Decode phase (generating tokens)
    - Per-token latency
    - TTFT (Time To First Token)
    - Latency drift over decode
    """

    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self.token_timings: List[TokenTiming] = []
        self._start_time_ns: int = 0
        self._prefill_tokens: int = 0
        self._decode_tokens: int = 0
        self._in_prefill: bool = True
        self._cumulative_ms: float = 0

    def start_session(self, prompt_length: int) -> None:
        """Start a new profiling session."""
        self.token_timings = []
        self._start_time_ns = time.perf_counter_ns()
        self._prefill_tokens = prompt_length
        self._decode_tokens = 0
        self._in_prefill = True
        self._cumulative_ms = 0
        logger.debug(f"Started LLM profiling session, prompt_length={prompt_length}")

    def record_prefill(self, t_start_ns: int, t_end_ns: int, token_count: int) -> None:
        """Record prefill phase timing."""
        latency_ms = (t_end_ns - t_start_ns) / 1e6
        self._cumulative_ms += latency_ms

        # Record as single prefill "token" event
        timing = TokenTiming(
            token_idx=0,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            latency_ms=latency_ms,
            phase="prefill",
            cumulative_latency_ms=self._cumulative_ms,
        )
        self.token_timings.append(timing)
        self._prefill_tokens = token_count
        self._in_prefill = False
        logger.debug(f"Prefill recorded: {latency_ms:.2f}ms for {token_count} tokens")

    def record_token(
        self, t_start_ns: int, t_end_ns: int, token_id: Optional[int] = None
    ) -> TokenTiming:
        """Record single decode token timing."""
        latency_ms = (t_end_ns - t_start_ns) / 1e6
        self._cumulative_ms += latency_ms
        self._decode_tokens += 1

        timing = TokenTiming(
            token_idx=self._decode_tokens,
            t_start_ns=t_start_ns,
            t_end_ns=t_end_ns,
            latency_ms=latency_ms,
            phase="decode",
            cumulative_latency_ms=self._cumulative_ms,
            token_id=token_id,
        )
        self.token_timings.append(timing)
        return timing

    def finish_session(self) -> LLMProfileResult:
        """Complete session and compute statistics."""
        # Separate phases
        prefill_timings = [t for t in self.token_timings if t.phase == "prefill"]
        decode_timings = [t for t in self.token_timings if t.phase == "decode"]

        # Compute phase summaries
        prefill_summary = self._compute_phase_summary(
            "prefill", prefill_timings, self._prefill_tokens
        )
        decode_summary = self._compute_phase_summary("decode", decode_timings, len(decode_timings))

        # Key metrics
        total_time_ms = self._cumulative_ms
        total_tokens = self._prefill_tokens + self._decode_tokens

        ttft = prefill_summary.total_time_ms if prefill_timings else 0
        overall_tps = (total_tokens / (total_time_ms / 1000)) if total_time_ms > 0 else 0

        # Drift analysis for decode
        drift_pct = self._compute_latency_drift(decode_timings)
        stability = self._compute_decode_stability(decode_timings)

        # Bottleneck indicators
        prefill_pct = prefill_summary.total_time_ms / total_time_ms if total_time_ms > 0 else 0
        prefill_bound = prefill_pct > 0.4  # Prefill > 40% of total time

        # Memory pressure: significant slowdown in late decode
        memory_pressure = drift_pct > 20  # >20% slowdown over decode

        return LLMProfileResult(
            model_name=self.model_name,
            prompt_length=self._prefill_tokens,
            generation_length=self._decode_tokens,
            total_tokens=total_tokens,
            total_time_ms=total_time_ms,
            prefill=prefill_summary,
            decode=decode_summary,
            time_to_first_token_ms=ttft,
            tokens_per_second=overall_tps,
            prefill_tokens_per_sec=prefill_summary.tokens_per_sec,
            decode_tokens_per_sec=decode_summary.tokens_per_sec,
            token_timings=self.token_timings,
            latency_drift_pct=drift_pct,
            decode_stability=stability,
            prefill_bound=prefill_bound,
            decode_memory_pressure=memory_pressure,
        )

    def _compute_phase_summary(
        self, phase: str, timings: List[TokenTiming], token_count: int
    ) -> PhaseSummary:
        """Compute summary statistics for a phase."""
        if not timings:
            return PhaseSummary(
                phase=phase,
                token_count=0,
                total_time_ms=0,
                mean_latency_ms=0,
                p50_latency_ms=0,
                p90_latency_ms=0,
                p99_latency_ms=0,
                std_latency_ms=0,
                tokens_per_sec=0,
            )

        latencies = np.array([t.latency_ms for t in timings])
        total_ms = float(np.sum(latencies))

        # For prefill, we have one measurement for all tokens
        # For decode, each timing is one token
        effective_count = token_count if phase == "prefill" else len(timings)
        tps = (effective_count / (total_ms / 1000)) if total_ms > 0 else 0

        first_token = timings[0].latency_ms if phase == "decode" and timings else None

        return PhaseSummary(
            phase=phase,
            token_count=effective_count,
            total_time_ms=total_ms,
            mean_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p90_latency_ms=float(np.percentile(latencies, 90)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            std_latency_ms=float(np.std(latencies)),
            tokens_per_sec=tps,
            first_token_latency_ms=first_token,
        )

    def _compute_latency_drift(self, decode_timings: List[TokenTiming]) -> float:
        """
        Compute latency drift over decode sequence.
        Compares first 20% vs last 20% of decode.
        """
        if len(decode_timings) < 10:
            return 0.0

        n = len(decode_timings)
        early = decode_timings[: n // 5]
        late = decode_timings[-n // 5 :]

        early_mean = np.mean([t.latency_ms for t in early])
        late_mean = np.mean([t.latency_ms for t in late])

        drift = ((late_mean - early_mean) / early_mean * 100) if early_mean > 0 else 0
        return float(drift)

    def _compute_decode_stability(self, decode_timings: List[TokenTiming]) -> float:
        """Compute decode stability (1 - coefficient of variation)."""
        if len(decode_timings) < 2:
            return 1.0

        latencies = [t.latency_ms for t in decode_timings]
        mean = np.mean(latencies)
        std = np.std(latencies)

        cov = std / mean if mean > 0 else 0
        return float(1 - min(cov, 1))

    def get_decode_latency_histogram(self, bins: int = 20) -> Dict[str, Any]:
        """Get histogram of decode token latencies."""
        decode_timings = [t for t in self.token_timings if t.phase == "decode"]
        if not decode_timings:
            return {"bins": [], "counts": []}

        latencies = [t.latency_ms for t in decode_timings]
        counts, bin_edges = np.histogram(latencies, bins=bins)

        return {
            "bins": [float(b) for b in bin_edges[:-1]],
            "counts": [int(c) for c in counts],
            "mean": float(np.mean(latencies)),
            "median": float(np.median(latencies)),
        }

    def get_token_latency_series(self) -> Dict[str, List[float]]:
        """Get time series of token latencies for plotting."""
        decode_timings = [t for t in self.token_timings if t.phase == "decode"]

        return {
            "token_idx": [t.token_idx for t in decode_timings],
            "latency_ms": [t.latency_ms for t in decode_timings],
            "cumulative_ms": [t.cumulative_latency_ms for t in decode_timings],
        }


class LLMBenchmarkRunner:
    """
    Runs LLM inference benchmarks with detailed token-level profiling.
    """

    def __init__(
        self,
        inference_fn: Callable[[List[int]], List[int]],
        tokenizer_fn: Optional[Callable[[str], List[int]]] = None,
        model_name: str = "unknown",
    ):
        """
        Args:
            inference_fn: Function that takes input tokens, returns output tokens
            tokenizer_fn: Optional function to tokenize strings
            model_name: Name for reporting
        """
        self.inference_fn = inference_fn
        self.tokenizer_fn = tokenizer_fn
        self.model_name = model_name
        self.profiler = LLMProfiler(model_name)

    def run_benchmark(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        warmup_runs: int = 2,
    ) -> List[LLMProfileResult]:
        """
        Run benchmark on multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Max tokens to generate per prompt
            warmup_runs: Number of warmup runs

        Returns:
            List of LLMProfileResult for each prompt.
        """
        results = []

        # Warmup
        if prompts and warmup_runs > 0:
            logger.info(f"Running {warmup_runs} warmup iterations...")
            warmup_prompt = prompts[0]
            for _ in range(warmup_runs):
                self._run_single(warmup_prompt, max_new_tokens, record=False)

        # Actual runs
        for i, prompt in enumerate(prompts):
            logger.info(f"Benchmarking prompt {i + 1}/{len(prompts)}")
            result = self._run_single(prompt, max_new_tokens, record=True)
            results.append(result)

        return results

    def _run_single(
        self, prompt: str, max_new_tokens: int, record: bool = True
    ) -> Optional[LLMProfileResult]:
        """Run single inference with profiling."""
        # Tokenize
        if self.tokenizer_fn:
            input_ids = self.tokenizer_fn(prompt)
        else:
            # Dummy tokenization for testing
            input_ids = list(range(len(prompt.split())))

        self.profiler.start_session(len(input_ids))

        # Simulated inference with token-by-token profiling
        # In real usage, this would hook into the inference loop
        prefill_start = time.perf_counter_ns()

        # This is a simplified simulation - real implementation would
        # hook into the actual inference loop

        try:
            # Call the actual inference
            output_ids = self.inference_fn(input_ids)

            prefill_end = time.perf_counter_ns()
            self.profiler.record_prefill(prefill_start, prefill_end, len(input_ids))

            # In a real scenario, we'd have per-token timing from the inference loop
            # Here we simulate based on total output
            if len(output_ids) > len(input_ids):
                new_tokens = output_ids[len(input_ids) :]
                avg_time_per_token = 10.0  # Simulated - would come from real timing

                for i, token_id in enumerate(new_tokens[:max_new_tokens]):
                    t_start = time.perf_counter_ns()
                    # Simulate token generation latency
                    time.sleep(avg_time_per_token / 1000)
                    t_end = time.perf_counter_ns()

                    self.profiler.record_token(t_start, t_end, token_id)

            if record:
                return self.profiler.finish_session()
            return None

        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None


def analyze_llm_results(results: List[LLMProfileResult]) -> Dict[str, Any]:
    """
    Aggregate analysis across multiple LLM profile results.

    Args:
        results: List of profiling results

    Returns:
        Aggregated statistics and insights.
    """
    if not results:
        return {"error": "No results to analyze"}

    # Aggregate metrics
    ttfts = [r.time_to_first_token_ms for r in results]
    tps_values = [r.tokens_per_second for r in results]
    decode_tps = [r.decode_tokens_per_sec for r in results]
    drifts = [r.latency_drift_pct for r in results]

    return {
        "num_runs": len(results),
        "ttft": {
            "mean_ms": float(np.mean(ttfts)),
            "p50_ms": float(np.percentile(ttfts, 50)),
            "p99_ms": float(np.percentile(ttfts, 99)),
        },
        "tokens_per_second": {
            "mean": float(np.mean(tps_values)),
            "p50": float(np.percentile(tps_values, 50)),
            "min": float(np.min(tps_values)),
            "max": float(np.max(tps_values)),
        },
        "decode_performance": {
            "mean_tps": float(np.mean(decode_tps)),
            "stability": float(np.mean([r.decode_stability for r in results])),
        },
        "latency_drift": {
            "mean_pct": float(np.mean(drifts)),
            "max_pct": float(np.max(drifts)),
        },
        "bottleneck_summary": {
            "prefill_bound_runs": sum(1 for r in results if r.prefill_bound),
            "memory_pressure_runs": sum(1 for r in results if r.decode_memory_pressure),
        },
    }
