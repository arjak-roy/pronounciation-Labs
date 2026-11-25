from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


@dataclass(frozen=True)
class AudioChunk:
    """A slice of normalized 16kHz mono float32 audio with its offset in the original clip."""

    samples: np.ndarray
    start_sample: int
    end_sample: int
    sample_rate: int = 16000

    @property
    def start_ms(self) -> int:
        return int(round(self.start_sample * 1000.0 / self.sample_rate))

    @property
    def end_ms(self) -> int:
        return int(round(self.end_sample * 1000.0 / self.sample_rate))

    @property
    def duration_seconds(self) -> float:
        return float(len(self.samples)) / float(self.sample_rate)


def load_and_normalize_audio(
    audio_bytes: bytes,
    target_sr: int = 16000,
    trim_silence: bool = True,
    silence_db_threshold: float = -42.0,
) -> tuple[np.ndarray, int]:
    """
    Decode WAV/PCM bytes and normalize strictly to 16kHz mono float32 in [-1.0, 1.0].

    Returns:
        Tuple of (normalized_mono_float32_samples, leading_trimmed_ms).
    """
    if not audio_bytes:
        raise ValueError("Uploaded audio payload is empty.")

    try:
        with io.BytesIO(audio_bytes) as buf:
            samples, sr = sf.read(buf, dtype="float32", always_2d=True)
    except Exception:
        # Fallback for raw 16-bit mono PCM @ target_sr if no WAV header is present
        if len(audio_bytes) % 2 != 0:
            raise ValueError("Invalid audio format: expected 16kHz mono WAV or 16-bit PCM.")
        pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
        samples = (pcm16.astype(np.float32) / 32768.0)[:, np.newaxis]
        sr = target_sr

    # Downmix multi-channel to mono
    mono: np.ndarray = np.mean(samples, axis=1, dtype=np.float32)

    # Resample to 16kHz if needed
    if sr != target_sr and len(mono) > 0:
        g = gcd(int(sr), int(target_sr))
        up = target_sr // g
        down = sr // g
        mono = resample_poly(mono, up, down).astype(np.float32)

    leading_trimmed_ms = 0
    if trim_silence and len(mono) > int(0.1 * target_sr):
        mono, leading_trimmed_samples = trim_leading_trailing_silence(
            mono, sample_rate=target_sr, threshold_db=silence_db_threshold
        )
        leading_trimmed_ms = int(round(leading_trimmed_samples * 1000.0 / target_sr))

    # Peak-safe normalization into [-1.0, 1.0]
    peak = float(np.max(np.abs(mono))) if len(mono) > 0 else 0.0
    if peak > 1.0:
        mono = mono / peak

    return np.ascontiguousarray(mono, dtype=np.float32), leading_trimmed_ms


def standardize_waveform_for_wav2vec2(waveform: np.ndarray) -> np.ndarray:
    """
    Apply zero-mean, unit-variance normalization required by Wav2Vec2 feature extractor.
    """
    if waveform.size == 0:
        return waveform.astype(np.float32)
    mean = float(np.mean(waveform))
    var = float(np.var(waveform))
    normed = (waveform - mean) / float(np.sqrt(var + 1e-7))
    return np.ascontiguousarray(normed, dtype=np.float32)


def trim_leading_trailing_silence(
    samples: np.ndarray,
    sample_rate: int = 16000,
    threshold_db: float = -42.0,
    frame_ms: int = 20,
    pad_ms: int = 60,
) -> tuple[np.ndarray, int]:
    """
    Trim leading and trailing silence using short-time RMS energy gating.
    Returns (trimmed_samples, leading_samples_removed).
    """
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    if len(samples) <= frame_len * 2:
        return samples, 0

    num_frames = len(samples) // frame_len
    framed = samples[: num_frames * frame_len].reshape(num_frames, frame_len)
    rms = np.sqrt(np.mean(np.square(framed), axis=1) + 1e-12)
    peak_rms = float(np.max(rms))
    if peak_rms < 1e-5:
        return samples, 0

    db = 20.0 * np.log10(rms / peak_rms + 1e-12)
    active_indices = np.where(db > threshold_db)[0]
    if active_indices.size == 0:
        return samples, 0

    pad_samples = int(sample_rate * pad_ms / 1000)
    start_sample = max(0, int(active_indices[0] * frame_len) - pad_samples)
    end_sample = min(len(samples), int((active_indices[-1] + 1) * frame_len) + pad_samples)
    return samples[start_sample:end_sample], start_sample


def split_at_breath_pauses(
    samples: np.ndarray,
    sample_rate: int = 16000,
    max_chunk_seconds: float = 5.0,
    min_chunk_seconds: float = 1.5,
) -> Sequence[AudioChunk]:
    """
    Mandatory Day-1 Optimization #4:
    Split pronunciation clips > max_chunk_seconds at natural breath pauses (low-energy valleys)
    to prevent O(N^2) self-attention slowdowns on long paragraphs.
    """
    max_samples = int(max_chunk_seconds * sample_rate)
    min_samples = int(min_chunk_seconds * sample_rate)
    total_samples = len(samples)

    if total_samples <= max_samples:
        return [AudioChunk(samples=samples, start_sample=0, end_sample=total_samples, sample_rate=sample_rate)]

    chunks: list[AudioChunk] = []
    window_ms = 20
    frame_len = max(1, int(sample_rate * window_ms / 1000))
    cursor = 0

    while cursor < total_samples:
        remaining = total_samples - cursor
        if remaining <= max_samples:
            chunks.append(
                AudioChunk(
                    samples=samples[cursor:total_samples],
                    start_sample=cursor,
                    end_sample=total_samples,
                    sample_rate=sample_rate,
                )
            )
            break

        # Search for lowest-energy breath pause between [cursor + min_samples, cursor + max_samples]
        search_start = cursor + min_samples
        search_end = min(total_samples, cursor + max_samples)
        segment = samples[search_start:search_end]
        num_frames = len(segment) // frame_len

        if num_frames <= 2:
            split_point = search_end
        else:
            framed = segment[: num_frames * frame_len].reshape(num_frames, frame_len)
            energies = np.mean(np.square(framed), axis=1)
            # Smooth over 5 frames (~100ms breath pause window)
            kernel = np.ones(min(5, len(energies)), dtype=np.float32)
            smoothed = np.convolve(energies, kernel / len(kernel), mode="valid")
            min_frame_idx = int(np.argmin(smoothed))
            split_point = search_start + (min_frame_idx * frame_len)

        chunks.append(
            AudioChunk(
                samples=samples[cursor:split_point],
                start_sample=cursor,
                end_sample=split_point,
                sample_rate=sample_rate,
            )
        )
        cursor = split_point

    return chunks
