from __future__ import annotations

import numpy as np

from app.stt_engines.cpu_sherpa import SherpaCpuEngine


class VoskCpuEngine(SherpaCpuEngine):
    """
    Mode A — CPU Broad Multilingual Fallback (`MODE_A_ENGINE=vosk`):
    Provides lightweight multilingual CPU transcription backed by INT8 CTranslate2 / Vosk.
    """

    @property
    def engine_name(self) -> str:
        return f"vosk-multilingual-cpu-fallback ({self.model_id})"

    def transcribe_sync(self, waveform_1d: np.ndarray, language: str) -> tuple[str, float]:
        return super().transcribe_sync(waveform_1d, language)
