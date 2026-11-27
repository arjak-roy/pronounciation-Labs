from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.stt_engines import StreamingSTTSession
from app.stt_engines.cpu_sherpa import (
    NEMOTRON_LANG_MAP,
    SherpaCpuEngine,
    SherpaStreamingSession,
)


class _DummyStream:
    def __init__(self) -> None:
        self.options: dict[str, str] = {}
        self.accepted_samples: int = 0
        self.finished: bool = False

    def set_option(self, key: str, value: str) -> None:
        self.options[key] = value

    def accept_waveform(self, sample_rate: int, samples: np.ndarray | list[float]) -> None:
        assert sample_rate == 16000
        self.accepted_samples += len(samples)

    def input_finished(self) -> None:
        self.finished = True


class _DummyNemotronRecognizer:
    def __init__(self, transcript: str = "Guten Morgen! <de-DE>") -> None:
        self.transcript = transcript
        self.last_stream: _DummyStream | None = None
        self._decoded_once: bool = False

    def create_stream(self) -> _DummyStream:
        self.last_stream = _DummyStream()
        self._decoded_once = False
        return self.last_stream

    def is_ready(self, stream: _DummyStream) -> bool:
        if not self._decoded_once and stream.accepted_samples > 0:
            return True
        return False

    def decode_stream(self, stream: _DummyStream) -> None:
        self._decoded_once = True

    def get_result(self, stream: _DummyStream) -> str:
        return self.transcript

    def ys_probs(self, stream: _DummyStream) -> list[float]:
        return [float(np.log(0.9))] * 3

    def is_endpoint(self, stream: _DummyStream) -> bool:
        return False

    def reset(self, stream: _DummyStream) -> None:
        self._decoded_once = False


def test_default_settings_use_nemotron_0_6b() -> None:
    s = Settings()
    assert s.mode_a_engine == "nemotron"
    assert "nemotron-3.5-asr-streaming-0.6b" in s.mode_a_nemotron_repo


def test_nemotron_transcribe_sets_language_and_strips_tag() -> None:
    engine = SherpaCpuEngine.__new__(SherpaCpuEngine)
    engine._recognizer = _DummyNemotronRecognizer("Guten Morgen, wie geht es Ihnen? <de-DE>")
    engine._is_nemotron = True
    engine._whisper = None

    assert engine.engine_name == "nvidia/nemotron-3.5-asr-streaming-0.6b (560ms sherpa-onnx INT8)"
    waveform = np.ones(16000, dtype=np.float32) * 0.1
    text, conf = engine.transcribe_sync(waveform, language="de")

    assert text == "Guten Morgen, wie geht es Ihnen?"
    assert conf == 0.9
    assert engine._recognizer.last_stream is not None
    assert engine._recognizer.last_stream.options.get("language") == "de-DE"
    assert engine._recognizer.last_stream.finished is True


@pytest.mark.asyncio
async def test_nemotron_streaming_session_sets_option_and_flushes() -> None:
    dummy_rec = _DummyNemotronRecognizer("Hello world. <en-US>")
    engine = SherpaCpuEngine.__new__(SherpaCpuEngine)
    engine._recognizer = dummy_rec
    engine._is_nemotron = True
    engine._whisper = None

    session = engine.create_session(language="en")
    assert isinstance(session, SherpaStreamingSession)
    assert dummy_rec.last_stream is not None
    assert dummy_rec.last_stream.options.get("language") == NEMOTRON_LANG_MAP["en"]

    pcm16 = (np.ones(1600, dtype=np.float32) * 1000).astype(np.int16).tobytes()
    events = await session.ingest_pcm16_bytes(pcm16)
    assert len(events) == 1
    assert events[0].type == "interim"
    assert events[0].transcript == "Hello world."

    final_ev = await session.flush()
    assert final_ev is not None
    assert final_ev.type == "final"
    assert final_ev.transcript == "Hello world."


def test_nemotron_auto_language_mode() -> None:
    dummy_rec = _DummyNemotronRecognizer("Bonjour tout le monde. <fr-FR>")
    engine = SherpaCpuEngine.__new__(SherpaCpuEngine)
    engine._recognizer = dummy_rec
    engine._is_nemotron = True
    engine._whisper = None

    text, conf = engine.transcribe_sync(np.ones(16000, dtype=np.float32) * 0.1, language="auto")
    assert text == "Bonjour tout le monde."
    assert conf == 0.9
    assert dummy_rec.last_stream is not None
    assert dummy_rec.last_stream.options.get("language") == "auto"



def test_zipformer_fallback_is_english_only() -> None:
    engine = SherpaCpuEngine.__new__(SherpaCpuEngine)
    engine._recognizer = _DummyNemotronRecognizer("hello")
    engine._is_nemotron = False
    engine._whisper = None

    assert isinstance(engine.create_session(language="en"), SherpaStreamingSession)
    de_session = engine.create_session(language="de")
    assert not isinstance(de_session, SherpaStreamingSession)
    assert isinstance(de_session, StreamingSTTSession)


def test_cpu_threads_capped_by_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.stt_engines.cpu_sherpa as mod

    monkeypatch.setattr(mod.SherpaCpuEngine, "_get_or_load_whisper", lambda self: None)
    engine = mod.SherpaCpuEngine(
        engine_variant="none", cpu_threads=2, sherpa_dir=Path("does-not-exist")
    )
    assert engine.cpu_threads == min(2, os.cpu_count() or 2)
