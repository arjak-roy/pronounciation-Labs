from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import numpy as np

from app.config import Settings
from app.schemas import TranscribeEvent

# Normalize ISO language variants (e.g. 'en-us' -> 'en') for Whisper / STT engines
STT_LANG_MAP: dict[str, str] = {
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "hi": "hi",
    "ja": "ja",
    "zh": "zh",
    "ko": "ko",
    "ru": "ru",
    "ar": "ar",
}


class BaseSTTEngine(ABC):
    """Strategy interface for Mode A Real-Time Live STT (CPU & GPU)."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Identifier of active STT engine."""

    @property
    @abstractmethod
    def device_profile(self) -> str:
        """Return 'cpu' or 'gpu'."""

    @abstractmethod
    def transcribe_sync(self, waveform_1d: np.ndarray, language: str) -> tuple[str, float]:
        """Transcribe 16kHz mono float32 waveform and return (transcript, confidence)."""

    async def transcribe_async(self, waveform_1d: np.ndarray, language: str) -> tuple[str, float]:
        """Non-blocking async wrapper around transcribe_sync."""
        return await asyncio.to_thread(self.transcribe_sync, waveform_1d, language)

    def create_session(self, language: str = "en") -> StreamingSTTSession:
        """Create a per-WebSocket streaming session with VAD endpointing."""
        return StreamingSTTSession(engine=self, language=language)


class StreamingSTTSession:
    """
    Stateful Mode A WebSocket streaming accumulator:
    - Accepts 100ms-200ms 16kHz 16-bit mono PCM chunks (`linear16`)
    - Applies server-side VAD energy gating (drops pure silence frames)
    - Emits rolling `type="interim"` events every ~550ms of active speech
    - Emits stabilized `type="final"` events on natural silence pause (~750ms) or 15s max window
    """

    def __init__(
        self,
        engine: BaseSTTEngine,
        language: str = "en",
        sample_rate: int = 16000,
        interim_interval_sec: float = 0.35,
        silence_endpoint_sec: float = 0.55,
        max_utterance_sec: float = 15.0,
    ) -> None:
        self.engine = engine
        self.language = language
        self.sample_rate = sample_rate
        self.interim_step_samples = int(interim_interval_sec * sample_rate)
        self.silence_endpoint_samples = int(silence_endpoint_sec * sample_rate)
        self.max_utterance_samples = int(max_utterance_sec * sample_rate)

        self._buffer = np.zeros(0, dtype=np.float32)
        self._samples_since_last_interim = 0
        self._trailing_silent_samples = 0
        self._has_active_speech = False
        self._last_interim_text = ""
        self._interim_in_flight = False

    async def ingest_pcm16_bytes(self, pcm16_bytes: bytes) -> list[TranscribeEvent]:
        """Process a binary frame of 16kHz 16-bit mono PCM and return 0 or more TranscribeEvents."""
        if not pcm16_bytes or len(pcm16_bytes) < 2:
            return []

        usable_len = len(pcm16_bytes) - (len(pcm16_bytes) % 2)
        pcm16 = np.frombuffer(pcm16_bytes[:usable_len], dtype=np.int16)
        chunk_f32 = pcm16.astype(np.float32) / 32768.0

        rms = float(np.sqrt(np.mean(chunk_f32 * chunk_f32) + 1e-12))
        is_voiced = rms >= 0.012

        events: list[TranscribeEvent] = []

        if is_voiced:
            self._has_active_speech = True
            self._trailing_silent_samples = 0
            self._buffer = np.concatenate([self._buffer, chunk_f32])
            self._samples_since_last_interim += len(chunk_f32)
        elif self._has_active_speech:
            # Keep a short trailing context for clean final consonant decoding
            self._trailing_silent_samples += len(chunk_f32)
            if self._trailing_silent_samples <= int(0.35 * self.sample_rate):
                self._buffer = np.concatenate([self._buffer, chunk_f32])
        else:
            # Pure silence before speech starts -> drop frame (Mandatory Optimization #4)
            return []

        # Check if we hit natural breath/utterance end OR 15-second max window
        if self._has_active_speech and (
            self._trailing_silent_samples >= self.silence_endpoint_samples
            or len(self._buffer) >= self.max_utterance_samples
        ):
            final_ev = await self.flush()
            if final_ev is not None:
                events.append(final_ev)
            return events

        # Emit rolling interim event if enough new voiced audio arrived and no decode is in flight
        if (
            self._has_active_speech
            and not self._interim_in_flight
            and self._samples_since_last_interim >= self.interim_step_samples
            and len(self._buffer) >= int(0.28 * self.sample_rate)
        ):
            self._samples_since_last_interim = 0
            self._interim_in_flight = True
            try:
                # Cap rolling interim window to latest 6s so latency stays ultra-low
                window = self._buffer[-int(6.0 * self.sample_rate) :]
                text, conf = await self.engine.transcribe_async(window, self.language)
            finally:
                self._interim_in_flight = False

            if text and text != self._last_interim_text:
                self._last_interim_text = text
                events.append(
                    TranscribeEvent(
                        type="interim",
                        transcript=text,
                        language=self.language,
                        confidence=conf,
                    )
                )

        return events

    async def flush(self) -> TranscribeEvent | None:
        """Force-finalize current buffered utterance and reset buffer."""
        if not self._has_active_speech or len(self._buffer) < int(0.15 * self.sample_rate):
            self.reset()
            return None

        audio_to_decode = self._buffer
        self.reset()
        text, conf = await self.engine.transcribe_async(audio_to_decode, self.language)
        if not text:
            return None
        return TranscribeEvent(
            type="final",
            transcript=text,
            language=self.language,
            confidence=conf,
        )

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._samples_since_last_interim = 0
        self._trailing_silent_samples = 0
        self._has_active_speech = False
        self._last_interim_text = ""


def create_stt_engine(settings: Settings) -> BaseSTTEngine:
    """
    Factory for Mode A STT engine controlled via `MODE_A_DEVICE=cpu|gpu` and `MODE_A_ENGINE`.
    """
    if settings.mode_a_device == "gpu":
        from app.stt_engines.gpu_whisperlive import WhisperLiveGpuEngine

        return WhisperLiveGpuEngine(
            model_id=settings.mode_a_gpu_model,
            engine_variant=settings.mode_a_engine,
            fallback_cpu_model=settings.mode_a_cpu_model,
        )

    if settings.mode_a_engine == "vosk":
        from app.stt_engines.cpu_vosk import VoskCpuEngine

        return VoskCpuEngine(
            fallback_whisper_model=settings.mode_a_cpu_model,
            cpu_threads=settings.mode_a_cpu_threads,
        )

    from app.stt_engines.cpu_sherpa import SherpaCpuEngine

    return SherpaCpuEngine(
        sherpa_dir=settings.mode_a_sherpa_dir,
        fallback_whisper_model=settings.mode_a_cpu_model,
        cpu_threads=settings.mode_a_cpu_threads,
        force_whisper=(settings.mode_a_engine == "whisper"),
        download_root=settings.mode_b_models_dir / "whisper",
    )
