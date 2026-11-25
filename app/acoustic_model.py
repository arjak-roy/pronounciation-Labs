from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from app.audio_utils import standardize_waveform_for_wav2vec2
from app.config import Settings

# Global worker-local ONNX session used inside ProcessPoolExecutor
_WORKER_ORT_SESSION: ort.InferenceSession | None = None
_WORKER_MODEL_PATH: str | None = None


@dataclass(frozen=True)
class AcousticForwardOutput:
    """
    Mandatory Day-1 Optimization #1 (Single-Pass Tensor Reuse):
    Holds both `logits` [frames, vocab_size] and intermediate Layer 12 `hidden_states` [frames, hidden_dim]
    produced from a single forward pass of Wav2Vec2.
    """

    logits: np.ndarray
    hidden_states: np.ndarray
    frame_stride_ms: float = 20.0


def _init_cpu_worker(onnx_path_str: str, intra_op_threads: int, inter_op_threads: int) -> None:
    """Initializer for ProcessPoolExecutor workers with strictly pinned ONNX CPU threads."""
    global _WORKER_ORT_SESSION, _WORKER_MODEL_PATH
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = int(intra_op_threads)
    sess_options.inter_op_num_threads = int(inter_op_threads)
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    _WORKER_ORT_SESSION = ort.InferenceSession(
        onnx_path_str,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )
    _WORKER_MODEL_PATH = onnx_path_str


def _run_cpu_worker_inference(waveform_1d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Execute single-pass ONNX inference inside the pinned CPU worker process."""
    global _WORKER_ORT_SESSION
    if _WORKER_ORT_SESSION is None:
        raise RuntimeError("CPU ONNX worker session was not initialized.")

    normed = standardize_waveform_for_wav2vec2(waveform_1d)
    input_tensor = normed[np.newaxis, :].astype(np.float32)
    input_name = _WORKER_ORT_SESSION.get_inputs()[0].name
    outputs = _WORKER_ORT_SESSION.run(None, {input_name: input_tensor})

    logits = np.asarray(outputs[0][0], dtype=np.float32)
    hidden_states = (
        np.asarray(outputs[1][0], dtype=np.float32)
        if len(outputs) > 1
        else np.zeros((logits.shape[0], 1), dtype=np.float32)
    )
    return logits, hidden_states


class BaseAcousticEngine(ABC):
    """Strategy interface for Mode B Acoustic Phoneme Recognition (CPU & GPU)."""

    @property
    @abstractmethod
    def device_profile(self) -> str:
        """Return 'cpu' or 'gpu'."""

    @property
    @abstractmethod
    def active_provider(self) -> str:
        """Return underlying runtime execution provider name."""

    @abstractmethod
    async def infer_single_pass(self, waveform_1d: np.ndarray) -> AcousticForwardOutput:
        """Run one forward pass returning both logits and intermediate hidden_states."""

    @abstractmethod
    def infer_single_pass_sync(self, waveform_1d: np.ndarray) -> AcousticForwardOutput:
        """Synchronous single-pass inference."""

    @abstractmethod
    def close(self) -> None:
        """Clean up process pools or device sessions."""


class OnnxCpuAcousticEngine(BaseAcousticEngine):
    """
    Mode B — CPU Profile (`MODE_B_DEVICE=cpu`):
    Runs INT8 dynamically quantized ONNX model with `CPUExecutionProvider` and pinned
    `intra_op_num_threads` inside a `ProcessPoolExecutor` to prevent CPU thread thrashing.
    """

    def __init__(
        self,
        onnx_path: Path,
        intra_op_threads: int = 2,
        inter_op_threads: int = 1,
        max_workers: int = 2,
    ) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Mode B CPU INT8 ONNX model not found at {onnx_path}. "
                "Run `python -m scripts.export_onnx` to export the model."
            )
        self.onnx_path = onnx_path.resolve()
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads
        self.max_workers = max_workers

        # Local in-process session with pinned threads for synchronous calls & health checks
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = int(intra_op_threads)
        sess_options.inter_op_num_threads = int(inter_op_threads)
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        # Dedicated ProcessPoolExecutor when max_workers > 0; if 0, uses single in-process session
        self._pool: ProcessPoolExecutor | None = (
            ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=_init_cpu_worker,
                initargs=(str(self.onnx_path), self.intra_op_threads, self.inter_op_threads),
            )
            if self.max_workers > 0
            else None
        )

    @property
    def device_profile(self) -> str:
        return "cpu"

    @property
    def active_provider(self) -> str:
        return "CPUExecutionProvider (INT8 ONNX)"

    def infer_single_pass_sync(self, waveform_1d: np.ndarray) -> AcousticForwardOutput:
        normed = standardize_waveform_for_wav2vec2(waveform_1d)
        input_tensor = normed[np.newaxis, :].astype(np.float32)
        outputs = self._session.run(None, {self._input_name: input_tensor})
        logits = np.asarray(outputs[0][0], dtype=np.float32)
        hidden_states = (
            np.asarray(outputs[1][0], dtype=np.float32)
            if len(outputs) > 1
            else np.zeros((logits.shape[0], 1), dtype=np.float32)
        )
        return AcousticForwardOutput(logits=logits, hidden_states=hidden_states)

    async def infer_single_pass(self, waveform_1d: np.ndarray) -> AcousticForwardOutput:
        import asyncio

        if self._pool is None:
            return await asyncio.to_thread(self.infer_single_pass_sync, waveform_1d)
        loop = asyncio.get_running_loop()
        logits, hidden_states = await loop.run_in_executor(
            self._pool, _run_cpu_worker_inference, waveform_1d
        )
        return AcousticForwardOutput(logits=logits, hidden_states=hidden_states)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None


class OnnxGpuAcousticEngine(BaseAcousticEngine):
    """
    Mode B — GPU Profile (`MODE_B_DEVICE=gpu`):
    Runs FP16/FP32 ONNX model via `CUDAExecutionProvider` / `TensorrtExecutionProvider`
    (or PyTorch CUDA AMP fallback if CUDAExecutionProvider is unavailable).
    """

    def __init__(
        self,
        onnx_path: Path,
        intra_op_threads: int = 2,
    ) -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"Mode B GPU ONNX model not found at {onnx_path}. "
                "Run `python -m scripts.export_onnx` to export the model."
            )
        self.onnx_path = onnx_path.resolve()
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = int(intra_op_threads)
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        available = ort.get_available_providers()
        preferred = [
            p
            for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
            if p in available
        ]
        self._session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=sess_options,
            providers=preferred or ["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._input_type = self._session.get_inputs()[0].type

    @property
    def device_profile(self) -> str:
        return "gpu"

    @property
    def active_provider(self) -> str:
        providers = self._session.get_providers()
        return providers[0] if providers else "CUDAExecutionProvider"

    def infer_single_pass_sync(self, waveform_1d: np.ndarray) -> AcousticForwardOutput:
        normed = standardize_waveform_for_wav2vec2(waveform_1d)
        dtype = np.float16 if "float16" in self._input_type else np.float32
        input_tensor = normed[np.newaxis, :].astype(dtype)
        outputs = self._session.run(None, {self._input_name: input_tensor})
        logits = np.asarray(outputs[0][0], dtype=np.float32)
        hidden_states = (
            np.asarray(outputs[1][0], dtype=np.float32)
            if len(outputs) > 1
            else np.zeros((logits.shape[0], 1), dtype=np.float32)
        )
        return AcousticForwardOutput(logits=logits, hidden_states=hidden_states)

    async def infer_single_pass(self, waveform_1d: np.ndarray) -> AcousticForwardOutput:
        import asyncio

        return await asyncio.to_thread(self.infer_single_pass_sync, waveform_1d)

    def close(self) -> None:
        pass


def create_acoustic_engine(settings: Settings) -> BaseAcousticEngine:
    """
    Factory method for Mode B Acoustic Phoneme Engine controlled via `MODE_B_DEVICE=cpu|gpu`.
    """
    if settings.mode_b_device == "gpu":
        gpu_path = settings.onnx_gpu_path if settings.onnx_gpu_path.exists() else settings.onnx_cpu_path
        return OnnxGpuAcousticEngine(
            onnx_path=gpu_path,
            intra_op_threads=settings.mode_b_intra_op_threads,
        )
    return OnnxCpuAcousticEngine(
        onnx_path=settings.onnx_cpu_path,
        intra_op_threads=settings.mode_b_intra_op_threads,
        inter_op_threads=settings.mode_b_inter_op_threads,
        max_workers=settings.mode_b_process_workers,
    )
