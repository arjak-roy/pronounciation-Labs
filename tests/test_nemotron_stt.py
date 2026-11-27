from __future__ import annotations

import numpy as np
import pytest

from app.config import Settings
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

    def accept_waveform(self, sample_rate: int, samples: list[float]) -> None:
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

    assert engine.engine_name == "nvidia/nemotron-3.5-asr-streaming-0.6b (sherpa-onnx INT8)"
    waveform = np.ones(16000, dtype=np.float32) * 0.1
    text, conf = engine.transcribe_sync(waveform, language="de")

    assert text == "Guten Morgen, wie geht es Ihnen?"
    assert conf == 0.96
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
