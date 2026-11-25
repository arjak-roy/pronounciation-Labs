from __future__ import annotations

import logging

import numpy as np
from faster_whisper import WhisperModel

from app.stt_engines import STT_LANG_MAP, BaseSTTEngine

logger = logging.getLogger("voice_pronunciation_engine.mode_a_gpu")


class WhisperLiveGpuEngine(BaseSTTEngine):
    """
    Mode A — GPU Profile (`MODE_A_DEVICE=gpu`):
    Uses `faster-whisper` (`large-v3-turbo` FP16/INT8 on CUDA, with graceful fallback to CPU INT8
    if CUDA runtime libraries are not installed on a dev laptop).
    """

    def __init__(
        self,
        model_id: str = "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        engine_variant: str = "whisper",
        fallback_cpu_model: str = "Systran/faster-whisper-small",
    ) -> None:
        self.engine_variant = engine_variant
        self.model_id = model_id
        try:
            self._whisper = WhisperModel(
                model_id,
                device="cuda",
                compute_type="int8_float16",
            )
            self._active_device = "cuda"
        except Exception as exc:
            logger.warning(
                "CUDA CTranslate2 unavailable (%s); falling back to INT8 CPU model %s.",
                exc,
                fallback_cpu_model,
            )
            self.model_id = fallback_cpu_model
            self._whisper = WhisperModel(
                fallback_cpu_model,
                device="cpu",
                compute_type="int8",
                cpu_threads=4,
            )
            self._active_device = "cpu-fallback"

    @property
    def engine_name(self) -> str:
        return f"gpu-{self.engine_variant} ({self.model_id} on {self._active_device})"

    @property
    def device_profile(self) -> str:
        return "gpu"

    def transcribe_sync(self, waveform_1d: np.ndarray, language: str) -> tuple[str, float]:
        if waveform_1d.size < 1600:
            return "", 0.0

        iso_lang = STT_LANG_MAP.get(language.lower(), "en")
        segments, _ = self._whisper.transcribe(
            waveform_1d,
            language=iso_lang,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
        )
        texts: list[str] = []
        probs: list[float] = []
        for seg in segments:
            t = seg.text.strip()
            if t:
                texts.append(t)
                probs.append(float(np.clip(np.exp(seg.avg_logprob), 0.1, 0.99)))

        if not texts:
            return "", 0.0
        return " ".join(texts).strip(), round(float(np.mean(probs)), 2)
