from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from app.schemas import TranscribeEvent
from app.stt_engines import STT_LANG_MAP, BaseSTTEngine, StreamingSTTSession

logger = logging.getLogger("voice_pronunciation_engine.mode_a_sherpa")

NEMOTRON_DEFAULT_REPO = (
    "csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11"
)
NEMOTRON_FILES = (
    "encoder.int8.onnx",
    "decoder.int8.onnx",
    "joiner.int8.onnx",
    "tokens.txt",
)

# NVIDIA Nemotron 3.5 ASR Streaming 0.6B supported language-locales
# (mapped from our UI language selector codes to Nemotron's prompt_dictionary keys).
NEMOTRON_LANG_MAP: dict[str, str] = {
    "de": "de-DE",
    "en": "en-US",
    "en-us": "en-US",
    "en-gb": "en-GB",
    "es": "es-ES",
    "fr": "fr-FR",
    "it": "it-IT",
    "pt": "pt-BR",
    "nl": "nl-NL",
    "pl": "pl-PL",
    "sv": "sv-SE",
    "tr": "tr-TR",
    "ru": "ru-RU",
    "hi": "hi-IN",
    "vi": "vi-VN",
    "uk": "uk-UA",
    "el": "el-GR",
    "ar": "ar-AR",
    "ja": "ja-JP",
    "zh": "zh-CN",
    "ko": "ko-KR",
}

_LANG_TAG_RE = re.compile(r"\s*<[a-z]{2,3}(?:-[A-Z]{2})?>\s*$")

# Short 1-word script indicators steer Whisper's BPE decoder into the target Unicode block
# for languages not in Nemotron 3.5's 40 locales (such as Bengali `bn`).
WHISPER_SCRIPT_PROMPTS: dict[str, str] = {
    "bn": "বাংলা",
    "hi": "हिन्दी",
    "ta": "தமிழ்",
    "te": "తెలుగు",
    "mr": "मराठी",
    "gu": "ગુજરાતી",
    "ur": "اردو",
    "ar": "العربية",
}

INDIC_AND_MULTIBYTE_LANGS = frozenset(
    {"bn", "hi", "ta", "te", "mr", "gu", "ur", "ar", "ja", "zh", "ko", "ru", "el"}
)


def ensure_nemotron_onnx_files(
    nemotron_dir: Path,
    repo_id: str = NEMOTRON_DEFAULT_REPO,
) -> bool:
    """Ensure the 4 INT8 ONNX artifacts for Nemotron 3.5 Streaming 0.6B are present locally."""
    nemotron_dir.mkdir(parents=True, exist_ok=True)
    if all((nemotron_dir / fn).exists() for fn in NEMOTRON_FILES):
        return True

    # Reuse decoder.int8.onnx, joiner.int8.onnx, and tokens.txt from previous 560ms dir if present
    legacy_dir = nemotron_dir.parent / "nemotron-3.5-0.6b-int8"
    if legacy_dir.exists() and legacy_dir != nemotron_dir:
        for shared_fn in ("decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
            src = legacy_dir / shared_fn
            dst = nemotron_dir / shared_fn
            if src.exists() and not dst.exists():
                shutil.copyfile(src, dst)

    from huggingface_hub import hf_hub_download

    for fn in NEMOTRON_FILES:
        dest = nemotron_dir / fn
        if not dest.exists():
            logger.info("Downloading Nemotron 3.5 0.6B INT8 artifact %s from %s ...", fn, repo_id)
            hf_hub_download(
                repo_id=repo_id,
                filename=fn,
                local_dir=str(nemotron_dir),
            )
    return True


class SherpaStreamingSession(StreamingSTTSession):
    """
    True frame-synchronous causal streaming session backed by `sherpa_onnx.OnlineRecognizer`
    (NVIDIA Nemotron 3.5 ASR Streaming 0.6B 160ms INT8 Cache-Aware FastConformer-RNNT).
    """

    def __init__(
        self,
        engine: SherpaCpuEngine,
        recognizer: Any,
        language: str = "de",
        sample_rate: int = 16000,
        nemotron_lang: str | None = None,
    ) -> None:
        super().__init__(engine=engine, language=language, sample_rate=sample_rate)
        self._recognizer = recognizer
        self._nemotron_lang = nemotron_lang
        self._stream = self._create_configured_stream()
        self._last_partial: str = ""
        self._lock = asyncio.Lock()

    def _create_configured_stream(self) -> Any:
        stream = self._recognizer.create_stream()
        if self._nemotron_lang:
            try:
                stream.set_option("language", self._nemotron_lang)
            except Exception as exc:
                logger.debug("Could not set stream language option %s: %s", self._nemotron_lang, exc)
        return stream

    def _clean_text(self, raw_result: Any) -> str:
        text = (
            raw_result
            if isinstance(raw_result, str)
            else getattr(raw_result, "text", str(raw_result))
        ).strip()
        return _LANG_TAG_RE.sub("", text).strip()

    def _ingest_and_decode_sync(self, samples_f32: np.ndarray) -> tuple[str, bool]:
        self._stream.accept_waveform(self.sample_rate, samples_f32)
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        text = self._clean_text(self._recognizer.get_result(self._stream))
        is_ep = bool(self._recognizer.is_endpoint(self._stream))
        if is_ep:
            self._recognizer.reset(self._stream)
        return text, is_ep

    def _flush_sync(self) -> str:
        # 200ms tail padding flushes the 160ms cache-aware transducer window immediately
        tail_padding = np.zeros(int(0.20 * self.sample_rate), dtype=np.float32)
        self._stream.accept_waveform(self.sample_rate, tail_padding)
        self._stream.input_finished()
        while self._recognizer.is_ready(self._stream):
            self._recognizer.decode_stream(self._stream)
        text = self._clean_text(self._recognizer.get_result(self._stream))
        self.reset()
        return text

    async def ingest_pcm16_bytes(self, pcm16_bytes: bytes) -> list[TranscribeEvent]:
        if not pcm16_bytes or len(pcm16_bytes) < 2:
            return []

        usable_len = len(pcm16_bytes) - (len(pcm16_bytes) % 2)
        pcm16 = np.frombuffer(pcm16_bytes[:usable_len], dtype=np.int16)
        samples_f32 = np.ascontiguousarray(pcm16.astype(np.float32) / 32768.0, dtype=np.float32)

        async with self._lock:
            text, is_ep = await asyncio.to_thread(self._ingest_and_decode_sync, samples_f32)

        events: list[TranscribeEvent] = []
        if text and text != self._last_partial:
            self._last_partial = text
            events.append(
                TranscribeEvent(
                    type="interim",
                    transcript=text,
                    language=self.language,
                    confidence=0.96,
                )
            )

        if is_ep:
            if text:
                events.append(
                    TranscribeEvent(
                        type="final",
                        transcript=text,
                        language=self.language,
                        confidence=0.97,
                    )
                )
            self._last_partial = ""

        return events

    async def flush(self) -> TranscribeEvent | None:
        async with self._lock:
            text = await asyncio.to_thread(self._flush_sync)
        if not text:
            return None
        return TranscribeEvent(
            type="final",
            transcript=text,
            language=self.language,
            confidence=0.97,
        )

    def reset(self) -> None:
        super().reset()
        self._stream = self._create_configured_stream()
        self._last_partial = ""


class SherpaCpuEngine(BaseSTTEngine):
    """
    Mode A — Primary Streaming & Utterance STT Engine:
    - Default (`MODE_A_ENGINE=nemotron`): NVIDIA Nemotron 3.5 ASR Streaming 0.6B 160ms INT8
      (`nvidia/nemotron-3.5-asr-streaming-0.6b` via `sherpa-onnx` cache-aware FastConformer-RNNT).
    - Automatically falls back to `faster-whisper-small` INT8 for languages outside Nemotron 3.5's
      40 locales (such as Bengali `bn`) or when `force_whisper=True`.
    """

    def __init__(
        self,
        sherpa_dir: Path = Path("models/sherpa-en"),
        nemotron_dir: Path = Path("models/nemotron-3.5-0.6b-160ms-int8"),
        nemotron_repo: str = NEMOTRON_DEFAULT_REPO,
        engine_variant: str = "nemotron",
        fallback_whisper_model: str = "Systran/faster-whisper-small",
        cpu_threads: int = 2,
        force_whisper: bool = False,
        download_root: Path | None = None,
    ) -> None:
        self.sherpa_dir = sherpa_dir
        self.nemotron_dir = nemotron_dir
        self.nemotron_repo = nemotron_repo
        self.engine_variant = engine_variant
        self.cpu_threads = max(int(cpu_threads), os.cpu_count() or 2)
        self.model_id = fallback_whisper_model
        self.download_root = download_root or Path("models/whisper")
        self._recognizer: Any = None
        self._is_nemotron: bool = False
        self._whisper: Any = None

        if not force_whisper and engine_variant == "nemotron":
            try:
                import sherpa_onnx

                ensure_nemotron_onnx_files(nemotron_dir=nemotron_dir, repo_id=nemotron_repo)
                encoder_path = nemotron_dir / "encoder.int8.onnx"
                decoder_path = nemotron_dir / "decoder.int8.onnx"
                joiner_path = nemotron_dir / "joiner.int8.onnx"
                tokens_path = nemotron_dir / "tokens.txt"

                logger.info(
                    "Loading NVIDIA Nemotron 3.5 ASR Streaming 0.6B (160ms INT8) from %s (threads=%d)...",
                    nemotron_dir,
                    self.cpu_threads,
                )
                self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                    tokens=str(tokens_path),
                    encoder=str(encoder_path),
                    decoder=str(decoder_path),
                    joiner=str(joiner_path),
                    num_threads=self.cpu_threads,
                    sample_rate=16000,
                    decoding_method="greedy_search",
                    enable_endpoint_detection=True,
                    rule1_min_trailing_silence=1.4,
                    rule2_min_trailing_silence=0.45,
                    rule3_min_utterance_length=15.0,
                    provider="cpu",
                )
                self._is_nemotron = True
                logger.info("NVIDIA Nemotron 3.5 ASR Streaming 0.6B (160ms INT8) ready.")
                return
            except Exception as exc:
                logger.warning(
                    "Nemotron 3.5 0.6B initialization failed (%s); falling back to faster-whisper (%s).",
                    exc,
                    fallback_whisper_model,
                )

        encoder_path = sherpa_dir / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
        decoder_path = sherpa_dir / "decoder-epoch-99-avg-1-chunk-16-left-128.onnx"
        joiner_path = sherpa_dir / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"
        tokens_path = sherpa_dir / "tokens.txt"

        if (
            not force_whisper
            and encoder_path.exists()
            and decoder_path.exists()
            and joiner_path.exists()
            and tokens_path.exists()
        ):
            import sherpa_onnx

            logger.info("Loading Sherpa-ONNX Streaming Zipformer INT8 from %s ...", sherpa_dir)
            self._recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=str(tokens_path),
                encoder=str(encoder_path),
                decoder=str(decoder_path),
                joiner=str(joiner_path),
                num_threads=self.cpu_threads,
                sample_rate=16000,
                feature_dim=80,
                decoding_method="greedy_search",
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=2.0,
                rule2_min_trailing_silence=0.7,
                rule3_min_utterance_length=15.0,
            )
        else:
            self._get_or_load_whisper()

    def _get_or_load_whisper(self) -> Any:
        if self._whisper is None:
            from faster_whisper import WhisperModel

            self.download_root.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Loading multilingual faster-whisper INT8 (%s) on CPU (threads=%d, cache=%s)...",
                self.model_id,
                self.cpu_threads,
                self.download_root,
            )
            self._whisper = WhisperModel(
                self.model_id,
                device="cpu",
                compute_type="int8",
                cpu_threads=self.cpu_threads,
                download_root=str(self.download_root),
            )
        return self._whisper

    @property
    def engine_name(self) -> str:
        if self._recognizer is not None and self._is_nemotron:
            return "nvidia/nemotron-3.5-asr-streaming-0.6b (160ms sherpa-onnx INT8)"
        if self._recognizer is not None:
            return "Sherpa-ONNX Streaming Zipformer INT8 (Real-Time Causal)"
        return f"faster-whisper-cpu-int8 ({self.model_id})"

    @property
    def device_profile(self) -> str:
        return "cpu"

    def _supports_nemotron_lang(self, language: str) -> bool:
        return language.lower() in NEMOTRON_LANG_MAP

    def create_session(self, language: str = "de") -> StreamingSTTSession:
        if self._recognizer is not None:
            if self._is_nemotron:
                if self._supports_nemotron_lang(language):
                    return SherpaStreamingSession(
                        engine=self,
                        recognizer=self._recognizer,
                        language=language,
                        sample_rate=16000,
                        nemotron_lang=NEMOTRON_LANG_MAP[language.lower()],
                    )
                # Non-Nemotron language (e.g. Bengali `bn`) uses buffered Whisper session
                return super().create_session(language=language)
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

        use_transducer = self._recognizer is not None and (
            not self._is_nemotron or self._supports_nemotron_lang(language)
        )
        if use_transducer:
            stream = self._recognizer.create_stream()
            if self._is_nemotron:
                nemotron_lang = NEMOTRON_LANG_MAP.get(language.lower(), "auto")
                try:
                    stream.set_option("language", nemotron_lang)
                except Exception as exc:
                    logger.debug("Could not set stream language option %s: %s", nemotron_lang, exc)

            stream.accept_waveform(16000, np.ascontiguousarray(waveform_1d, dtype=np.float32))
            # 200ms tail silence flushes the 160ms right-context chunk in Nemotron 3.5
            tail = np.zeros(int(0.20 * 16000), dtype=np.float32)
            stream.accept_waveform(16000, tail)
            stream.input_finished()
            while self._recognizer.is_ready(stream):
                self._recognizer.decode_stream(stream)
            raw = self._recognizer.get_result(stream)
            text = (raw if isinstance(raw, str) else getattr(raw, "text", str(raw))).strip()
            text = _LANG_TAG_RE.sub("", text).strip()
            return text, (0.96 if text else 0.0)

        whisper = self._get_or_load_whisper()
        if whisper is not None:
            iso_lang = STT_LANG_MAP.get(language.lower(), "de")
            is_multibyte = iso_lang in INDIC_AND_MULTIBYTE_LANGS
            use_silero_vad = waveform_1d.size >= 8000
            script_prompt = WHISPER_SCRIPT_PROMPTS.get(iso_lang)
            segments, _ = whisper.transcribe(
                waveform_1d,
                language=iso_lang,
                initial_prompt=script_prompt,
                beam_size=2 if is_multibyte else 1,
                best_of=2 if is_multibyte else 1,
                temperature=(0.0, 0.2) if is_multibyte else 0.0,
                condition_on_previous_text=False,
                vad_filter=use_silero_vad,
                vad_parameters=dict(
                    threshold=0.38,
                    min_speech_duration_ms=100,
                    min_silence_duration_ms=250,
                    speech_pad_ms=120,
                ),
                no_speech_threshold=0.62,
                log_prob_threshold=-1.2 if is_multibyte else -1.0,
                compression_ratio_threshold=4.2 if is_multibyte else 2.4,
                without_timestamps=True,
            )
            texts: list[str] = []
            probs: list[float] = []
            for seg in segments:
                if getattr(seg, "no_speech_prob", 0.0) > 0.72:
                    continue
                t = seg.text.strip()
                if t:
                    texts.append(t)
                    probs.append(float(np.clip(np.exp(seg.avg_logprob), 0.1, 0.99)))
            if not texts:
                return "", 0.0
            return " ".join(texts), round(float(np.mean(probs)), 2)

        return "", 0.0
