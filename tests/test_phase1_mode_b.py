from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.audio_utils import (
    load_and_normalize_audio,
    split_at_breath_pauses,
    standardize_waveform_for_wav2vec2,
)
from app.config import Settings
from app.g2p import G2PService
from app.main import create_app


def synthesize_speech_wav_bytes(text: str, sample_rate: int = 16000) -> bytes:
    """
    Generate real spoken 16kHz 16-bit mono WAV audio using Windows SAPI SpeechSynthesizer
    (or fallback harmonic speech-like formant synthesis on headless systems).
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        ps_script = (
            "Add-Type -AssemblyName System.Speech; "
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo("
            f"{sample_rate}, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
            "[System.Speech.AudioFormat.AudioChannel]::Mono); "
            f"$synth.SetOutputToWaveFile('{tmp_path}', $fmt); "
            f"$synth.Speak('{text.replace(chr(39), '')}'); "
            "$synth.Dispose();"
        )
        res = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True,
            timeout=15,
            check=False,
        )
        if res.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 1024:
            return tmp_path.read_bytes()
    except Exception:
        pass
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass

    # Fallback voiced waveform if PowerShell SAPI is unavailable
    duration_s = max(1.2, 0.35 * len(text.split()))
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False, dtype=np.float32)
    waveform = 0.3 * np.sin(2 * np.pi * 140.0 * t) + 0.15 * np.sin(2 * np.pi * 650.0 * t)
    buf = io.BytesIO()
    sf.write(buf, waveform, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def test_audio_normalization_and_breath_chunking() -> None:
    """Verify 16kHz mono float32 normalization, silence trimming, and >5s breath-pause splitting."""
    sr = 16000
    # Create 7.2s audio with a silent breath pause around 3.4s - 3.7s
    t1 = np.linspace(0, 3.4, int(sr * 3.4), dtype=np.float32)
    part1 = 0.4 * np.sin(2 * np.pi * 220.0 * t1)
    pause = np.zeros(int(sr * 0.3), dtype=np.float32)
    t2 = np.linspace(0, 3.5, int(sr * 3.5), dtype=np.float32)
    part2 = 0.4 * np.sin(2 * np.pi * 260.0 * t2)
    leading_silence = np.zeros(int(sr * 0.25), dtype=np.float32)

    full = np.concatenate([leading_silence, part1, pause, part2])
    buf = io.BytesIO()
    sf.write(buf, full, sr, subtype="PCM_16", format="WAV")

    samples, leading_ms = load_and_normalize_audio(buf.getvalue(), target_sr=sr, trim_silence=True)
    assert samples.dtype == np.float32
    assert leading_ms >= 150

    standardized = standardize_waveform_for_wav2vec2(samples)
    assert abs(float(np.mean(standardized))) < 1e-4
    assert abs(float(np.std(standardized)) - 1.0) < 1e-2

    chunks = split_at_breath_pauses(samples, sample_rate=sr, max_chunk_seconds=5.0)
    assert len(chunks) == 2
    # Breath split should land inside the 3.4s - 3.7s low-energy pause window
    assert 3200 <= chunks[0].end_ms <= 3850


def test_g2p_ipa_sanitization_and_strict_vocab_mapping() -> None:
    """
    Coding Guideline #5:
    Verify espeak-ng IPA output strips suprasegmentals (`ˈ`, `ˌ`, `ː`) and maps strictly
    to valid Wav2Vec2 tokenizer IDs without ever producing `<unk>` or invalid indices.
    """
    settings = Settings()
    g2p = G2PService.from_tokenizer_dir(settings.mode_b_tokenizer_dir)

    targets = g2p.phonemize_text("She thought about the pronunciation!", language="en")
    assert len(targets) == 5

    for wt in targets:
        assert "ˈ" not in wt.sanitized_ipa
        assert "ˌ" not in wt.sanitized_ipa
        assert len(wt.token_ids) == len(wt.phonemes)
        assert len(wt.token_ids) > 0
        for tid in wt.token_ids:
            assert tid in g2p.id_to_token
            assert tid != g2p.unk_id
            assert tid != g2p.blank_id


def test_end_to_end_assess_endpoint_real_onnx() -> None:
    """
    Phase 1 End-to-End Integration Test:
    Tests `POST /api/v1/pronunciation/assess` and `POST /assess` against the real exported
    Wav2Vec2 ONNX model (`logits` + `hidden_states` single-pass output), `torchaudio.functional.forced_align`,
    and `panphon` articulatory error diagnostics.
    """
    settings = Settings(mode_b_device="cpu", mode_b_process_workers=1)
    app = create_app(settings=settings)

    wav_thought = synthesize_speech_wav_bytes("thought")
    wav_mispronounced = synthesize_speech_wav_bytes("taught")

    with TestClient(app) as client:
        tester_page = client.get("/tester")
        assert tester_page.status_code == 200
        assert "Voice & Pronunciation" in tester_page.text

        health = client.get("/health")
        assert health.status_code == 200
        health_data = health.json()
        assert health_data["mode_b"]["ready"] is True
        assert "CPUExecutionProvider" in health_data["mode_b"]["provider"]

        # 1. Assess matching speech on /api/v1/pronunciation/assess
        resp1 = client.post(
            "/api/v1/pronunciation/assess",
            data={"reference_text": "thought", "language": "en"},
            files={"audio": ("thought.wav", wav_thought, "audio/wav")},
        )
        assert resp1.status_code == 200, resp1.text
        body1 = resp1.json()
        assert 0 <= body1["overall_score"] <= 100
        assert set(body1["metrics"].keys()) == {"accuracy", "fluency", "prosody_similarity"}
        assert len(body1["words"]) == 1
        word1 = body1["words"][0]
        assert word1["word"] == "thought"
        assert word1["end_ms"] > word1["start_ms"]
        assert len(word1["phonemes"]) >= 2

        # 2. Assess intentionally substituted initial consonant ("taught" vs "thought") on /assess alias
        resp2 = client.post(
            "/assess",
            data={"reference_text": "thought", "language": "en"},
            files={"audio": ("taught.wav", wav_mispronounced, "audio/wav")},
        )
        assert resp2.status_code == 200, resp2.text
        body2 = resp2.json()
        word2 = body2["words"][0]
        assert word2["word"] == "thought"
        # "taught" (/tɔːt/) vs "thought" (/θɔːt/) should score lower than accurate "thought"
        assert word2["score"] <= word1["score"]


def test_single_pass_tensor_reuse_and_gpu_profile_toggle() -> None:
    """
    Mandatory Day-1 Optimization #1 & Guideline #1:
    Verify that a single forward pass produces both `logits` [T, V] and Layer 12 `hidden_states` [T, D],
    and that `MODE_B_DEVICE=gpu` cleanly initializes `OnnxGpuAcousticEngine` behind the identical contract.
    """
    from app.acoustic_model import create_acoustic_engine

    wav_bytes = synthesize_speech_wav_bytes("pronunciation")
    samples, _ = load_and_normalize_audio(wav_bytes, target_sr=16000)

    # Test CPU engine single-pass dual outputs
    cpu_settings = Settings(mode_b_device="cpu", mode_b_process_workers=1)
    cpu_engine = create_acoustic_engine(cpu_settings)
    try:
        out_cpu = cpu_engine.infer_single_pass_sync(samples)
        assert out_cpu.logits.ndim == 2
        assert out_cpu.hidden_states.ndim == 2
        assert out_cpu.logits.shape[0] == out_cpu.hidden_states.shape[0]
        assert out_cpu.hidden_states.shape[1] >= 768  # Layer 12 embedding dimension (1024 for lv-60)
    finally:
        cpu_engine.close()

    # Test GPU profile toggle (`MODE_B_DEVICE=gpu`)
    gpu_settings = Settings(mode_b_device="gpu")
    gpu_engine = create_acoustic_engine(gpu_settings)
    try:
        assert gpu_engine.device_profile == "gpu"
        out_gpu = gpu_engine.infer_single_pass_sync(samples)
        assert out_gpu.logits.shape == out_cpu.logits.shape
        assert out_gpu.hidden_states.shape == out_cpu.hidden_states.shape
    finally:
        gpu_engine.close()

