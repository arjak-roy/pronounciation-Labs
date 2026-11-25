from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import sherpa_onnx

from app.schemas import TranscribeEvent
from app.stt_engines import BaseSTTEngine, StreamingSTTSession

logger = logging.getLogger("voice_pronunciation_engine.mode_a_sherpa")


class SherpaStreamingSession(StreamingSTTSession):
    """
    True frame-synchronous causal streaming session backed by `sherpa_onnx.OnlineRecognizer`
    (Streaming Zipformer INT8 Transducer). Decodes each 100ms chunk in ~5-12ms on CPU.
    """

    def __init__(
        self,
        engine: SherpaCpuEngine,
        recognizer: sherpa_onnx.OnlineRecognizer,
        language: str = "en",
        sample_rate: int = 16000,
    ) -> None:
        super().__init__(engine=engine, language=language, sample_rate=sample_rate)
        self._recognizer = recognizer
        self._stream = self._recognizer.create_stream()
        self._last_partial: str = ""

    async def ingest_pcm16_bytes(self, pcm16_bytes: bytes) -> list[TranscribeEvent]:
        if not pcm16_bytes or len(pcm16_bytes) < 2:
            return []

        usable_len = len(pcm16_bytes) - (len(pcm16_bytes) % 2)
        pcm16 = np.frombuffer(pcm16_bytes[:usable_len], dtype=np.int16)
        samples_f32 = (pcm16.astype(np.float32) / 32768.0).tolist()

        self._stream.accept_waveform(self.sample_rate, samples_f32)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        raw_result = self._recognizer.get_result(self._stream)
        text = (raw_result if isinstance(raw_result, str) else getattr(raw_result, "text", str(raw_result))).strip()

        events: list[TranscribeEvent] = []
        if text and text != self._last_partial:
            self._last_partial = text
            events.append(
                TranscribeEvent(
                    type="interim",
                    transcript=text,
                    language=self.language,
                    confidence=0.95,
                )
            )

        if self._recognizer.is_endpoint(self._stream):
            if text:
                events.append(
                    TranscribeEvent(
                        type="final",
                        transcript=text,
                        language=self.language,
                        confidence=0.96,
                    )
                )
            self._recognizer.reset(self._stream)
            self._last_partial = ""

        return events

    async def flush(self) -> TranscribeEvent | None:
        # Feed tail padding so the transducer flushes the final word
        tail_padding = [0.0] * int(0.45 * self.sample_rate)
        self._stream.accept_waveform(self.sample_rate, tail_padding)
        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)

        raw_result = self._recognizer.get_result(self._stream)
        text = (raw_result if isinstance(raw_result, str) else getattr(raw_result, "text", str(raw_result))).strip()
        self.reset()
        if not text:
            return None
        return TranscribeEvent(
            type="final",
            transcript=text,
            language=self.language,
            confidence=0.96,
        )

    def reset(self) -> None:
        super().reset()
        self._stream = self._recognizer.create_stream()
        self._last_partial = ""


class SherpaCpuEngine(BaseSTTEngine):
    """
    Mode A — Primary CPU Streaming Engine (`MODE_A_DEVICE=cpu`, `MODE_A_ENGINE=sherpa`):
    Uses `Sherpa-ONNX` Streaming Zipformer INT8 Transducer for sub-15ms real-time chunk decoding.
    """

    def __init__(
        self,
        sherpa_dir: Path = Path("models/sherpa-en"),
        fallback_whisper_model: str = "Systran/faster-whisper-small",
        cpu_threads: int = 2,
    ) -> None:
        self.sherpa_dir = sherpa_dir
        self.cpu_threads = cpu_threads
        self.model_id = fallback_whisper_model

        encoder_path = sherpa_dir / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
        decoder_path = sherpa_dir / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
        joiner_path = sherpa_dir / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
        tokens_path = sherpa_dir / "tokens.txt"

        if encoder_path.exists() and decoder_path.exists() and joiner_path.exists() and tokens_path.exists():
            logger.info("Loading Sherpa-ONNX Streaming Zipformer INT8 from %s ...", sherpa_dir)
            self._recognizer: sherpa_onnx.OnlineRecognizer | None = (
                sherpa_onnx.OnlineRecognizer.from_transducer(
                    tokens=str(tokens_path),
                    encoder=str(encoder_path),
                    decoder=str(decoder_path),
                    joiner=str(joiner_path),
                    num_threads=cpu_threads,
                    sample_rate=16000,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    enable_endpoint_detection=True,
                    rule1_min_trailing_silence=2.0,
                    rule2_min_trailing_silence=0.7,
                    rule3_min_utterance_length=15.0,
                )
            )
            self._whisper = None
        else:
            from faster_whisper import WhisperModel

            logger.info("Sherpa-ONNX files not found; loading faster-whisper (%s)...", fallback_whisper_model)
            self._recognizer = None
            self._whisper = WhisperModel(
                fallback_whisper_model,
                device="cpu",
                compute_type="int8",
                cpu_threads=cpu_threads,
            )

    @property
    def engine_name(self) -> str:
        if self._recognizer is not None:
            return "Sherpa-ONNX Streaming Zipformer INT8 (Real-Time Causal)"
        return f"faster-whisper-cpu-int8 ({self.model_id})"

    @property
    def device_profile(self) -> str:
        return "cpu"

    def create_session(self, language: str = "en") -> StreamingSTTSession:
        if self._recognizer is not None:
            return SherpaStreamingSession(
                engine=self,
                recognizer=self._recognizer,
                language=language,
                sample_rate=16000,
            )
        return super().create_session(language=language)

    def transcribe_sync(self, waveform_1d: np.ndarray, language: str) -> tuple[str, float]:
        if waveform_1d.size < 800:
            return "", 0.0

        if self._recognizer is not None:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, waveform_1d.astype(np.float32).tolist())
            tail = [0.0] * 6400
            stream.accept_waveform(16000, tail)
            stream.input_finished()
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
            raw = self._recognizer.get_result(stream)
            text = (raw if isinstance(raw, str) else getattr(raw, "text", str(raw))).strip()
            return text, (0.95 if text else 0.0)

        if self._whisper is not None:
            segments, _ = self._whisper.transcribe(
                waveform_1d,
                language="en",
                beam_size=1,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,
                without_timestamps=True,
            )
            texts = [seg.text.strip() for seg in segments if seg.text.strip()]
            return " ".join(texts), (0.90 if texts else 0.0)

        return "", 0.0
