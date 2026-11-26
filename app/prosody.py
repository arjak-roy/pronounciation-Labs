from __future__ import annotations

import numpy as np

from app.g2p import IPA_VOWEL_CHARS
from app.schemas import (
    ConnectedSpeechReport,
    PronunciationMetrics,
    WordAssessment,
    WordLinkDiagnostic,
)

UNSTRESSED_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "to",
        "of",
        "in",
        "on",
        "at",
        "for",
        "and",
        "or",
        "but",
        "is",
        "am",
        "are",
        "was",
        "were",
        "it",
        "as",
        "by",
    }
)


def extract_f0_pitch_contour(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    frame_ms: int = 30,
    hop_ms: int = 15,
    fmin_hz: float = 75.0,
    fmax_hz: float = 420.0,
) -> np.ndarray:
    """
    Extract fundamental frequency (F0 in Hz) across voiced frames using normalized
    autocorrelation on 16kHz mono float32 audio.
    """
    frame_len = int(sample_rate * frame_ms / 1000)
    hop_len = int(sample_rate * hop_ms / 1000)
    if waveform.size < frame_len:
        return np.array([], dtype=np.float32)

    min_lag = max(1, int(sample_rate / fmax_hz))
    max_lag = min(frame_len - 1, int(sample_rate / fmin_hz))
    if max_lag <= min_lag:
        return np.array([], dtype=np.float32)

    f0_values: list[float] = []
    window = np.hanning(frame_len).astype(np.float32)

    for start in range(0, len(waveform) - frame_len + 1, hop_len):
        frame = waveform[start : start + frame_len] * window
        energy = float(np.mean(frame * frame))
        if energy < 1e-4:
            continue

        frame_zm = frame - np.mean(frame)
        norm = float(np.sum(frame_zm * frame_zm)) + 1e-12
        # FFT-based fast autocorrelation
        n_fft = 1 << (2 * frame_len - 1).bit_length()
        spec = np.fft.rfft(frame_zm, n=n_fft)
        acf = np.fft.irfft(spec * np.conj(spec), n=n_fft)[:frame_len] / norm

        search_region = acf[min_lag : max_lag + 1]
        if search_region.size == 0:
            continue
        best_idx = int(np.argmax(search_region))
        peak_corr = float(search_region[best_idx])
        if peak_corr >= 0.35:
            lag = min_lag + best_idx
            f0_values.append(float(sample_rate) / float(lag))

    return np.asarray(f0_values, dtype=np.float32)


def compute_stress_contrast_score(
    words: list[WordAssessment],
    waveform: np.ndarray | None,
    sample_rate: int = 16000,
) -> float:
    """
    Evaluate English stress-timed rhythm by verifying that content words have longer
    per-phoneme duration and/or stronger acoustic energy than unstressed function words.
    """
    if len(words) < 2:
        return 0.90

    content_durations: list[float] = []
    function_durations: list[float] = []
    content_rms: list[float] = []
    function_rms: list[float] = []

    for w in words:
        dur_per_phoneme = max(10.0, float(w.end_ms - w.start_ms)) / max(1, len(w.phonemes))
        is_func = w.word.lower() in UNSTRESSED_FUNCTION_WORDS
        if is_func:
            function_durations.append(dur_per_phoneme)
        else:
            content_durations.append(dur_per_phoneme)

        if waveform is not None and waveform.size > 0:
            s_idx = max(0, int(w.start_ms * sample_rate / 1000))
            e_idx = min(len(waveform), max(s_idx + 1, int(w.end_ms * sample_rate / 1000)))
            seg = waveform[s_idx:e_idx]
            if seg.size > 0:
                rms = float(np.sqrt(np.mean(seg * seg) + 1e-12))
                if is_func:
                    function_rms.append(rms)
                else:
                    content_rms.append(rms)

    if not content_durations or not function_durations:
        return 0.90

    dur_ratio = float(np.mean(content_durations)) / max(1.0, float(np.mean(function_durations)))
    rms_ratio = (
        float(np.mean(content_rms)) / max(1e-6, float(np.mean(function_rms)))
        if (content_rms and function_rms)
        else 1.15
    )

    # Natural stress-timed speech has content words >= 1.05x duration/energy of function words
    score_dur = float(np.clip(dur_ratio / 1.15, 0.65, 1.0))
    score_rms = float(np.clip(rms_ratio / 1.15, 0.65, 1.0))
    return 0.55 * score_dur + 0.45 * score_rms


def compute_dtw_cosine_similarity(
    user_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
) -> float:
    """
    Compute Dynamic Time Warping (DTW) cosine similarity between Layer 12 hidden_states
    from user speech and cached native reference embeddings.
    """
    if user_embeddings.size == 0 or reference_embeddings.size == 0:
        return 0.80

    u_norm = user_embeddings / (np.linalg.norm(user_embeddings, axis=-1, keepdims=True) + 1e-8)
    r_norm = reference_embeddings / (
        np.linalg.norm(reference_embeddings, axis=-1, keepdims=True) + 1e-8
    )

    sim_matrix = np.matmul(u_norm, r_norm.T)
    cost_matrix = np.clip(1.0 - sim_matrix, 0.0, 2.0)

    n, m = cost_matrix.shape
    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = cost_matrix[i - 1, j - 1]
            dtw[i, j] = c + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    mean_cost = float(dtw[n, m] / max(1, n + m))
    similarity = float(np.clip(1.0 - mean_cost, 0.0, 1.0))
    return similarity


def compute_pronunciation_metrics(
    words: list[WordAssessment],
    hidden_states: np.ndarray,
    total_duration_ms: int,
    waveform: np.ndarray | None = None,
    sample_rate: int = 16000,
    native_hidden_embeddings: np.ndarray | None = None,
) -> tuple[int, PronunciationMetrics]:
    """
    Compute overall pronunciation score and sub-metrics (`accuracy`, `fluency`, `prosody_similarity`)
    using:
    1. Phoneme-weighted word GOP accuracy
    2. Speech rate & inter-word pause fluency
    3. F0 pitch intonation dynamics + content/function word stress contrast + Layer 12 coherence
    """
    if not words:
        empty_metrics = PronunciationMetrics(accuracy=0, fluency=0, prosody_similarity=0)
        return 0, empty_metrics

    # 1. Accuracy: weighted average of word scores by phoneme count
    weights = [max(1, len(w.phonemes)) for w in words]
    accuracy = int(
        np.clip(
            round(float(np.average([w.score for w in words], weights=weights))),
            0,
            100,
        )
    )

    # 2. Fluency: evaluate unnatural inter-word pauses (>250ms) and articulation speech rate
    unnatural_pause_ms = 0
    for i in range(1, len(words)):
        gap_ms = max(0, words[i].start_ms - words[i - 1].end_ms)
        if gap_ms > 250:
            unnatural_pause_ms += gap_ms - 250

    active_span_ms = max(1, words[-1].end_ms - words[0].start_ms)
    pause_penalty_ratio = min(1.0, float(unnatural_pause_ms) / float(max(500, active_span_ms)))

    total_phonemes = sum(len(w.phonemes) for w in words)
    phonemes_per_sec = total_phonemes / max(0.25, total_duration_ms / 1000.0)
    rate_score = (
        1.0
        if 5.0 <= phonemes_per_sec <= 18.0
        else max(0.5, 1.0 - abs(phonemes_per_sec - 11.0) * 0.05)
    )

    fluency = int(
        np.clip(
            round(100.0 * (0.70 * (1.0 - pause_penalty_ratio * 0.65) + 0.30 * rate_score)),
            0,
            100,
        )
    )

    # 3. Prosody Similarity: combine F0 pitch contour dynamism, stress-timed contrast, and Layer 12
    if native_hidden_embeddings is not None and native_hidden_embeddings.size > 0:
        dtw_sim = compute_dtw_cosine_similarity(hidden_states, native_hidden_embeddings)
        prosody_similarity = int(np.clip(round(dtw_sim * 100.0), 0, 100))
    else:
        # 3a. Pitch (F0) intonation dynamism
        pitch_score = 0.88
        if waveform is not None and waveform.size > 0:
            f0 = extract_f0_pitch_contour(waveform, sample_rate=sample_rate)
            if f0.size >= 4:
                f0_cv = float(np.std(f0) / (np.mean(f0) + 1e-6))
                # Natural expressive speech has F0 coefficient of variation ~ 0.04 to 0.30
                pitch_score = 0.96 if 0.03 <= f0_cv <= 0.35 else 0.78

        # 3b. Content vs Function word stress contrast
        stress_score = compute_stress_contrast_score(words, waveform, sample_rate=sample_rate)

        # 3c. Layer 12 contextual trajectory coherence
        transition_coherence = 0.85
        if hidden_states.ndim == 2 and hidden_states.shape[0] > 2 and hidden_states.shape[1] > 1:
            norms = np.linalg.norm(hidden_states, axis=-1, keepdims=True) + 1e-8
            unit_h = hidden_states / norms
            adj_cos = np.sum(unit_h[:-1] * unit_h[1:], axis=-1)
            transition_coherence = float(np.clip(np.mean(adj_cos), 0.0, 1.0))

        prosody_raw = (
            0.35 * accuracy
            + 0.25 * (pitch_score * 100.0)
            + 0.25 * (stress_score * 100.0)
            + 0.15 * (transition_coherence * 100.0)
        )
        prosody_similarity = int(np.clip(round(prosody_raw), 0, 100))

    overall_score = int(
        np.clip(
            round(0.60 * accuracy + 0.25 * fluency + 0.15 * prosody_similarity),
            0,
            100,
        )
    )

    metrics = PronunciationMetrics(
        accuracy=accuracy,
        fluency=fluency,
        prosody_similarity=prosody_similarity,
    )
    return overall_score, metrics


def analyze_connected_speech(
    words: list[WordAssessment],
    waveform: np.ndarray | None = None,
    sample_rate: int = 16000,
) -> ConnectedSpeechReport:
    """
    Analyze word-to-word connected speech linking (C->V liaison, V->V glide linking,
    consonant gemination/elision) and sentence F0 intonation trajectory.
    """
    links: list[WordLinkDiagnostic] = []
    smooth_count = 0
    choppy_count = 0

    for i in range(len(words) - 1):
        w1 = words[i]
        w2 = words[i + 1]
        gap_ms = max(0, w2.start_ms - w1.end_ms)

        end_ch = w1.expected_ipa[-1:] if w1.expected_ipa else ""
        start_ch = w2.expected_ipa[:1] if w2.expected_ipa else ""
        end_is_vowel = bool(end_ch and end_ch in IPA_VOWEL_CHARS)
        start_is_vowel = bool(start_ch and start_ch in IPA_VOWEL_CHARS)

        if not end_is_vowel and start_is_vowel:
            link_type = "consonant_to_vowel"
        elif end_is_vowel and start_is_vowel:
            link_type = "vowel_to_vowel"
        elif not end_is_vowel and not start_is_vowel and end_ch == start_ch:
            link_type = "consonant_cluster"
        else:
            link_type = "normal"

        if gap_ms <= 90:
            status = "smooth_link"
            smooth_count += 1
            tip = (
                f"Smooth C→V link: '{w1.word}‿{w2.word}' connected seamlessly ({gap_ms}ms)."
                if link_type == "consonant_to_vowel"
                else None
            )
        elif gap_ms <= 190:
            status = "natural_pause"
            tip = (
                f"Try linking the final /{end_ch}/ of '{w1.word}' directly into the vowel of '{w2.word}' ('{w1.word}‿{w2.word}')."
                if link_type == "consonant_to_vowel"
                else None
            )
        else:
            status = "choppy_pause"
            choppy_count += 1
            tip = (
                f"Choppy break ({gap_ms}ms) between '{w1.word}' and '{w2.word}' — blend '{w1.word}‿{w2.word}' without stopping airflow."
            )

        links.append(
            WordLinkDiagnostic(
                from_word=w1.word,
                to_word=w2.word,
                link_type=link_type,
                gap_ms=gap_ms,
                status=status,
                coaching_tip=tip,
            )
        )

    total_links = max(1, len(links))
    linking_score = int(
        np.clip(
            round(100.0 * (smooth_count + 0.65 * (len(links) - smooth_count - choppy_count)) / total_links),
            0,
            100,
        )
    ) if links else 100

    # Downsample F0 contour to <= 24 points for compact UI rendering
    contour_list: list[float] = []
    intonation_type = "expressive"
    if waveform is not None and waveform.size > 0:
        f0 = extract_f0_pitch_contour(waveform, sample_rate=sample_rate)
        if f0.size >= 4:
            step = max(1, f0.size // 24)
            contour_list = [round(float(x), 1) for x in f0[::step][:24]]
            f0_cv = float(np.std(f0) / (np.mean(f0) + 1e-6))
            head_mean = float(np.mean(f0[: max(2, f0.size // 3)]))
            tail_mean = float(np.mean(f0[-max(2, f0.size // 3) :]))
            if f0_cv < 0.03:
                intonation_type = "flat_monotone"
            elif tail_mean < head_mean * 0.92:
                intonation_type = "falling_statement"
            elif tail_mean > head_mean * 1.08:
                intonation_type = "rising_question"
            else:
                intonation_type = "expressive"

    if choppy_count > 0:
        summary_tip = f"Detected {choppy_count} choppy pause(s) between words. Practice linking consonant-to-vowel boundaries ('‿') in one continuous breath."
    elif intonation_type == "flat_monotone":
        summary_tip = "Your word linking is smooth, but your pitch stayed flat. Raise your pitch slightly on stressed content words."
    else:
        summary_tip = "Natural connected speech flow and expressive intonation contour!"

    return ConnectedSpeechReport(
        linking_score=linking_score,
        smooth_links_count=smooth_count,
        choppy_Pause_count=choppy_count,
        intonation_contour_hz=contour_list,
        intonation_type=intonation_type,
        summary_tip=summary_tip,
        links=links,
    )
