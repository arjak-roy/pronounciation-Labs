from __future__ import annotations

import builtins
import io
import pathlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import panphon.distance
import torch
import torchaudio.functional as F_audio

from app.g2p import WordPhonemeTarget
from app.schemas import (
    PhonemeAssessment,
    SyllableStressItem,
    WordAssessment,
    WordStressDiagnostic,
)

# Native allophone & dialectal equivalence sets (e.g., US/UK vowel mergers, American alveolar flap,
# unstressed schwa variants). Crucially, distinct learner error pairs (/θ/ vs /t,s/, /ʃ/ vs /s/,
# /ɹ/ vs /l/, /v/ vs /w/, /iː/ vs /ɪ/) are NEVER in the same equivalence set.
ALLOPHONE_EQUIVALENCE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"ɔː", "ɔ", "ɑː", "ɑ", "ɒ"}),  # cot-caught / open-back vowel family
    frozenset({"ə", "ɐ", "ʌ", "ɜ", "ɨ", "ᵻ"}),  # unstressed schwa / central vowel family
    frozenset({"ɜː", "ɚ", "ɝ", "ɜ"}),  # rhotic mid-central vowels
    frozenset({"iː", "i"}),  # tense high-front vowel
    frozenset({"ɪ", "ɨ", "ᵻ"}),  # lax high-front vowel
    frozenset({"uː", "u", "ʉ"}),  # tense high-back vowel
    frozenset({"e", "ɛ", "eɪ"}),  # mid-front vowel family
    frozenset({"o", "oʊ", "əʊ"}),  # mid-back diphthong family
    frozenset({"aɪ", "ɑɪ", "ae"}),  # PRICE diphthong variants
    frozenset({"aʊ", "ɑʊ"}),  # MOUTH diphthong variants
    frozenset({"t", "ɾ", "ʔ", "tʰ"}),  # alveolar stop + American flap / glottal allophones
    frozenset({"d", "ɾ"}),  # voiced alveolar stop + flap
    frozenset({"ɡ", "g"}),  # unicode velar stop variants
    frozenset({"ɹ", "r", "ɻ", "ɚ"}),  # rhotic approximant variants
    frozenset({"l", "ɫ"}),  # clear and dark L allophones
)

UNRELEASED_FINAL_STOPS: frozenset[str] = frozenset({"t", "d", "p", "b", "k", "ɡ", "g"})


@contextmanager
def _force_utf8_text_open() -> Iterator[None]:
    """
    Ensure `panphon` internal CSV/YAML resource reads (`files('panphon').joinpath(...).open()`)
    always use `encoding='utf-8'` on Windows where default locale encoding is `cp1252`.
    """
    orig_builtin_open = builtins.open
    orig_io_open = io.open
    orig_path_open = pathlib.Path.open

    def _utf8_open(file, mode="r", buffering=-1, encoding=None, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "b" not in mode and encoding is None:
            encoding = "utf-8"
        return orig_builtin_open(file, mode, buffering, encoding, *args, **kwargs)

    def _utf8_path_open(self, mode="r", buffering=-1, encoding=None, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "b" not in mode and encoding is None:
            encoding = "utf-8"
        return orig_path_open(self, mode, buffering, encoding, *args, **kwargs)

    builtins.open = _utf8_open  # type: ignore[assignment]
    io.open = _utf8_open  # type: ignore[assignment]
    pathlib.Path.open = _utf8_path_open  # type: ignore[method-assign]
    try:
        yield
    finally:
        builtins.open = orig_builtin_open  # type: ignore[assignment]
        io.open = orig_io_open  # type: ignore[assignment]
        pathlib.Path.open = orig_path_open  # type: ignore[method-assign]


@dataclass(frozen=True)
class AlignedPhonemeSpan:
    """Frame span and calibrated GOP score for a single aligned target phoneme."""

    phoneme: str
    token_id: int
    start_frame: int
    end_frame: int
    gop_score: int


class ForcedAlignerGOP:
    """
    Mode B Forced Alignment, Goodness of Pronunciation (GOP), and Conditional Error Diagnostics:
    - Uses `torchaudio.functional.forced_align` on single-pass `logits` log-posteriors.
    - Pools probability mass across acceptable allophones in the 392-token vocabulary (Group Log-Sum-Exp GOP).
    - Filters boundary coarticulation transitions via Top-K core frame weighting.
    - Applies Mandatory Day-1 Optimization #3 (Conditional Error Diagnostics):
      Words with `score > gop_pass_threshold` (85) are marked `correct` and skip
      Needleman-Wunsch alignment and `panphon` articulatory distance lookups.
    """

    def __init__(
        self,
        id_to_token: dict[int, str],
        blank_id: int = 0,
        gop_pass_threshold: float = 85.0,
        gop_warn_threshold: float = 60.0,
        frame_stride_ms: float = 20.0,
    ) -> None:
        self.id_to_token = id_to_token
        self.token_to_id: dict[str, int] = {tok: idx for idx, tok in id_to_token.items()}
        self.blank_id = blank_id
        self.gop_pass_threshold = gop_pass_threshold
        self.gop_warn_threshold = gop_warn_threshold
        self.frame_stride_ms = frame_stride_ms
        with _force_utf8_text_open():
            self._panphon_dst = panphon.distance.Distance()
        self._articulatory_cache: dict[tuple[str, str], float] = {}
        self._allophone_token_ids_cache: dict[int, list[int]] = {}

    @staticmethod
    def compute_log_probs(logits: np.ndarray) -> np.ndarray:
        """Compute numerically stable log-softmax across vocabulary dimension [frames, vocab_size]."""
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        log_sum_exp = np.log(np.sum(exp_x, axis=-1, keepdims=True) + 1e-12)
        return (shifted - log_sum_exp).astype(np.float32)

    @staticmethod
    def is_acceptable_allophone(expected_p: str, heard_p: str) -> bool:
        """Return True if `heard_p` is an acceptable native allophone or dialectal variant of `expected_p`."""
        if expected_p == heard_p:
            return True
        if expected_p.replace("ː", "") == heard_p.replace("ː", "") and expected_p not in ("iː", "ɪ"):
            return True
        for group in ALLOPHONE_EQUIVALENCE_GROUPS:
            if expected_p in group and heard_p in group:
                return True
        return False

    def _get_equivalent_token_ids(self, token_id: int) -> list[int]:
        """
        Return all token IDs in the 392-token Wav2Vec2 vocabulary that represent the same
        phoneme or acceptable native allophones so split probability mass is pooled accurately.
        """
        if token_id in self._allophone_token_ids_cache:
            return self._allophone_token_ids_cache[token_id]

        base_sym = self.id_to_token.get(token_id, "")
        equiv_symbols: set[str] = {base_sym}
        for group in ALLOPHONE_EQUIVALENCE_GROUPS:
            if base_sym in group:
                equiv_symbols.update(group)

        # Also include length-stripped base vowel if not a minimal tense/lax pair
        if "ː" in base_sym and base_sym != "iː":
            equiv_symbols.add(base_sym.replace("ː", ""))

        ids = sorted(
            {
                self.token_to_id[sym]
                for sym in equiv_symbols
                if sym in self.token_to_id and self.token_to_id[sym] != self.blank_id
            }
        )
        if token_id not in ids:
            ids.append(token_id)

        self._allophone_token_ids_cache[token_id] = ids
        return ids

    def _articulatory_substitution_cost(self, expected_p: str, heard_p: str) -> float:
        """
        Compute normalized articulatory feature distance using panphon's 24-feature matrix.
        Returns 0.0 for identical phonemes or acceptable native allophones.
        """
        if self.is_acceptable_allophone(expected_p, heard_p):
            return 0.0
        key = (expected_p, heard_p)
        if key in self._articulatory_cache:
            return self._articulatory_cache[key]

        try:
            dist = float(self._panphon_dst.feature_edit_distance(expected_p, heard_p))
            cost = float(np.clip(0.25 + dist * 1.5, 0.25, 1.35))
        except Exception:
            cost = 1.0

        self._articulatory_cache[key] = cost
        return cost

    def describe_articulatory_diff(self, expected_p: str, heard_p: str) -> str:
        """
        Inspect panphon's 24 articulatory features (`voi`, `cont`, `cor`, `ant`, `distr`,
        `lab`, `hi`, `lo`, `back`, `round`, `tense`, `nas`, `lat`) for `expected_p` vs `heard_p`
        and return a clear, learner-friendly physical mouth/tongue instruction.
        """
        if self.is_acceptable_allophone(expected_p, heard_p):
            return "Mouth & tongue placement matched naturally."

        try:
            exp_seg = self._panphon_dst.fm.fts(expected_p) or (
                self._panphon_dst.fm.fts(expected_p[0]) if expected_p else None
            )
            hrd_seg = self._panphon_dst.fm.fts(heard_p) or (
                self._panphon_dst.fm.fts(heard_p[0]) if heard_p else None
            )
        except Exception:
            exp_seg = None
            hrd_seg = None

        if exp_seg is None or hrd_seg is None:
            return f"Produced /{heard_p}/ instead of target /{expected_p}/ — isolate /{expected_p}/ slowly."

        exp_d = exp_seg if isinstance(exp_seg, dict) else getattr(exp_seg, "data", {})
        hrd_d = hrd_seg if isinstance(hrd_seg, dict) else getattr(hrd_seg, "data", {})
        cues: list[str] = []

        # 1. Manner of articulation (Continuant: stop vs fricative)
        if exp_d.get("cont", 0) == 1 and hrd_d.get("cont", 0) == -1:
            cues.append(
                f"Airflow was blocked completely (/{heard_p}/ stop); keep a narrow gap so air flows continuously for /{expected_p}/"
            )
        elif exp_d.get("cont", 0) == -1 and hrd_d.get("cont", 0) == 1:
            cues.append(
                f"Air leaked continuously (/{heard_p}/ fricative); make a crisp complete seal then release for /{expected_p}/"
            )

        # 2. Place of articulation (Dental/Distributed vs Alveolar, Labial, Palatal)
        if expected_p in ("θ", "ð") and heard_p in ("t", "d", "s", "z"):
            cues.append(
                "Move your tongue tip slightly forward so it rests lightly between your upper and lower front teeth"
            )
        elif expected_p in ("ʃ", "ʒ") and heard_p in ("s", "z"):
            cues.append(
                "Pull your tongue tip slightly back toward the roof of your mouth and round your lips slightly ('shh')"
            )
        elif expected_p in ("s", "z") and heard_p in ("ʃ", "ʒ", "θ", "t"):
            cues.append(
                "Keep your tongue tip just behind your upper front teeth (without touching between teeth) for a crisp 'sss'"
            )
        elif exp_d.get("lab", 0) == 1 and hrd_d.get("lab", 0) != 1:
            cues.append(f"Engage your lips more for labial /{expected_p}/")

        # 3. Voicing (Vocal cord vibration)
        if exp_d.get("voi", 0) == 1 and hrd_d.get("voi", 0) == -1:
            cues.append(
                f"Turn on your vocal cords (feel your throat buzz) to voice /{expected_p}/ instead of voiceless /{heard_p}/"
            )
        elif exp_d.get("voi", 0) == -1 and hrd_d.get("voi", 0) == 1:
            cues.append(
                f"Whisper the sound with pure breath (no throat vibration) for unvoiced /{expected_p}/"
            )

        # 4. Lateral vs Rhotic (/l/ vs /ɹ/)
        if exp_d.get("lat", 0) == -1 and hrd_d.get("lat", 0) == 1:
            cues.append(
                "Curl or bunch your tongue tip slightly back so it does NOT touch the roof of your mouth"
            )
        elif exp_d.get("lat", 0) == 1 and hrd_d.get("lat", 0) == -1:
            cues.append(
                "Touch the tip of your tongue firmly to the ridge right behind your upper front teeth for /l/"
            )

        # 5. Vowel height, backness, rounding & tenseness
        if exp_d.get("syl", 0) == 1 and hrd_d.get("syl", 0) == 1:
            if exp_d.get("lo", 0) == 1 and hrd_d.get("lo", 0) != 1:
                cues.append(f"Drop your jaw slightly wider for open vowel /{expected_p}/")
            elif exp_d.get("hi", 0) == 1 and hrd_d.get("hi", 0) != 1:
                cues.append(f"Raise your tongue higher toward the roof of your mouth for /{expected_p}/")
            if exp_d.get("round", 0) == 1 and hrd_d.get("round", 0) != 1:
                cues.append(f"Round your lips into an oval circle for /{expected_p}/")
            elif exp_d.get("round", 0) == -1 and hrd_d.get("round", 0) == 1:
                cues.append(f"Unround and spread your lips slightly for /{expected_p}/")
            if exp_d.get("tense", 0) == 1 and hrd_d.get("tense", 0) == -1:
                cues.append(f"Hold /{expected_p}/ slightly longer and with firmer lip/tongue tension")

        if not cues:
            return f"You produced /{heard_p}/ instead of /{expected_p}/ — adjust tongue placement toward /{expected_p}/."
        return " · ".join(cues) + "."

    def _evaluate_word_stress(
        self,
        wt: WordPhonemeTarget,
        aligned_phonemes: list[AlignedPhonemeSpan],
    ) -> WordStressDiagnostic:
        """
        Evaluate multi-syllable lexical stress by comparing acoustic syllable duration (ms)
        and phoneme GOP prominence against `espeak-ng`'s expected primary stress syllable.
        """
        if not wt.syllables or len(wt.syllables) <= 1:
            dur = int(
                round(
                    sum(max(1, p.end_frame - p.start_frame) for p in aligned_phonemes)
                    * self.frame_stride_ms
                )
            )
            avg_s = (
                int(round(float(np.mean([p.gop_score for p in aligned_phonemes]))))
                if aligned_phonemes
                else 100
            )
            return WordStressDiagnostic(
                syllable_count=1,
                expected_primary_index=0,
                detected_peak_index=0,
                stress_correct=True,
                stress_pattern_display=f"ˈ{wt.sanitized_ipa}",
                coaching_tip=None,
                syllables=[
                    SyllableStressItem(
                        syllable_ipa=wt.sanitized_ipa,
                        stress="primary",
                        duration_ms=dur,
                        score=avg_s,
                    )
                ],
            )

        syl_items: list[SyllableStressItem] = []
        prominence_scores: list[float] = []
        expected_primary_idx = 0

        for idx, syl in enumerate(wt.syllables):
            if syl.stress == "primary":
                expected_primary_idx = idx
            valid_indices = [i for i in syl.phoneme_indices if 0 <= i < len(aligned_phonemes)]
            if valid_indices:
                spans = [aligned_phonemes[i] for i in valid_indices]
                dur_ms = int(
                    round(
                        sum(max(1, s.end_frame - s.start_frame) for s in spans)
                        * self.frame_stride_ms
                    )
                )
                syl_gop = int(round(float(np.mean([s.gop_score for s in spans]))))
            else:
                dur_ms = 40
                syl_gop = 75

            syl_items.append(
                SyllableStressItem(
                    syllable_ipa=syl.ipa,
                    stress=syl.stress,
                    duration_ms=dur_ms,
                    score=syl_gop,
                )
            )
            # Prominence combines acoustic duration and clarity
            prominence_scores.append(dur_ms * (0.5 + 0.5 * (syl_gop / 100.0)))

        detected_peak_idx = int(np.argmax(prominence_scores))
        # Stress is considered accurate if primary syllable is the peak OR within 80% of peak prominence
        expected_prom = prominence_scores[expected_primary_idx]
        max_prom = max(prominence_scores) if prominence_scores else 1.0
        stress_ok = (detected_peak_idx == expected_primary_idx) or (expected_prom >= 0.78 * max_prom)

        pattern_parts: list[str] = []
        for idx, item in enumerate(syl_items):
            if idx == expected_primary_idx:
                pattern_parts.append(f"ˈ{item.syllable_ipa.upper()}")
            elif item.stress == "secondary":
                pattern_parts.append(f"ˌ{item.syllable_ipa}")
            else:
                pattern_parts.append(item.syllable_ipa)
        pattern_str = " · ".join(pattern_parts)

        tip: str | None = None
        if not stress_ok:
            target_syl = syl_items[expected_primary_idx].syllable_ipa
            tip = (
                f"Put stronger emphasis and longer vowel length on syllable #{expected_primary_idx + 1} "
                f"(/{target_syl}/ in {pattern_str}) while keeping unstressed syllables shorter."
            )
        else:
            target_syl = syl_items[expected_primary_idx].syllable_ipa
            tip = f"Good lexical stress on syllable #{expected_primary_idx + 1} (/{target_syl}/ in {pattern_str})."

        return WordStressDiagnostic(
            syllable_count=len(syl_items),
            expected_primary_index=expected_primary_idx,
            detected_peak_index=detected_peak_idx,
            stress_correct=stress_ok,
            stress_pattern_display=pattern_str,
            coaching_tip=tip,
            syllables=syl_items,
        )

    def _calibrate_gop_score(
        self,
        log_probs: np.ndarray,
        token_id: int,
        start_frame: int,
        end_frame: int,
        is_word_final: bool = False,
    ) -> int:
        """
        Compute Goodness of Pronunciation (GOP) score (0-100) with:
        1. Allophone Group Log-Sum-Exp pooling across the 392-token vocabulary
        2. Center / Top-70% frame weighting to filter out boundary coarticulation transitions
        3. Word-final unreleased stop lookahead (+2 frames)
        """
        num_frames = log_probs.shape[0]
        phoneme_sym = self.id_to_token.get(token_id, "")
        extra_lookahead = 2 if (is_word_final and phoneme_sym in UNRELEASED_FINAL_STOPS) else 1
        s_f = max(0, start_frame - 1)
        e_f = min(num_frames, max(start_frame + 1, end_frame + extra_lookahead))
        span = log_probs[s_f:e_f]
        if span.shape[0] == 0:
            return 0

        # 1. Pool probability mass across target phoneme + acceptable allophone tokens
        equiv_ids = self._get_equivalent_token_ids(token_id)
        group_lps = span[:, equiv_ids]
        max_lp = np.max(group_lps, axis=-1, keepdims=True)
        pooled_frame_lps = (
            np.squeeze(max_lp, axis=-1)
            + np.log(np.sum(np.exp(group_lps - max_lp), axis=-1) + 1e-12)
        )
        pooled_frame_lps = np.minimum(pooled_frame_lps, 0.0)

        # 2. Filter boundary coarticulation transitions by focusing on the top-70% core frames
        k = max(1, int(np.ceil(len(pooled_frame_lps) * 0.70)))
        top_k_lps = np.sort(pooled_frame_lps)[-k:]
        peak_target_lp = float(np.max(pooled_frame_lps))
        core_mean_lp = float(np.mean(top_k_lps))

        # 3. Strongest non-blank, non-allophone competitor in the core span
        competitor_span = np.copy(span)
        if 0 <= self.blank_id < competitor_span.shape[1]:
            competitor_span[:, self.blank_id] = -1e9
        for eid in equiv_ids:
            if 0 <= eid < competitor_span.shape[1]:
                competitor_span[:, eid] = -1e9
        best_competitor_lp = float(np.max(competitor_span))

        # Log-Posterior Ratio (positive when pooled target beats all non-allophone competitors)
        lpr = min(0.0, peak_target_lp - best_competitor_lp)

        posterior_prob = float(np.exp(np.clip(peak_target_lp, -20.0, 0.0)))
        lpr_component = float(np.exp(np.clip(lpr * 0.32, -10.0, 0.0)))
        core_component = float(np.exp(np.clip(core_mean_lp * 0.14, -10.0, 0.0)))

        raw_score = 100.0 * (0.55 * lpr_component + 0.25 * posterior_prob + 0.20 * core_component)

        # Word-final unreleased stop allowance: in natural speech, final /t, d, p, k/ often
        # end in glottal/oral closure (high blank posterior following a voiced vowel)
        if is_word_final and phoneme_sym in UNRELEASED_FINAL_STOPS and raw_score < 78.0:
            blank_prob = float(np.mean(np.exp(span[:, self.blank_id]))) if 0 <= self.blank_id < span.shape[1] else 0.0
            if blank_prob > 0.75:
                raw_score = max(raw_score, 82.0)

        return int(np.clip(round(raw_score), 0, 100))

    def _forced_align_tokens(
        self,
        log_probs: np.ndarray,
        flat_token_ids: list[int],
    ) -> list[tuple[int, int]]:
        """
        Run `torchaudio.functional.forced_align` using pooled allophone emissions so dialectal
        allophones align cleanly to their expected target phoneme positions.
        """
        num_frames = log_probs.shape[0]
        num_tokens = len(flat_token_ids)
        if num_tokens == 0:
            return []

        repeats = sum(
            1 for i in range(1, num_tokens) if flat_token_ids[i] == flat_token_ids[i - 1]
        )
        min_required_frames = num_tokens + repeats

        if num_frames >= min_required_frames:
            try:
                # Boost target emission column with pooled allophone log-probs so forced_align
                # snaps accurately onto American flaps / vowel allophones
                boosted_lp = np.copy(log_probs)
                for tid in set(flat_token_ids):
                    equiv_ids = self._get_equivalent_token_ids(tid)
                    if len(equiv_ids) > 1:
                        boosted_lp[:, tid] = np.max(log_probs[:, equiv_ids], axis=-1)

                emissions = torch.from_numpy(boosted_lp).unsqueeze(0)
                targets = torch.tensor([flat_token_ids], dtype=torch.int32)
                aligned_tokens, _ = F_audio.forced_align(
                    log_probs=emissions,
                    targets=targets,
                    blank=self.blank_id,
                )
                frame_labels = aligned_tokens[0].tolist()

                spans: list[tuple[int, int]] = []
                target_idx = 0
                t = 0
                while t < num_frames and target_idx < num_tokens:
                    if frame_labels[t] == flat_token_ids[target_idx]:
                        start_f = t
                        while t < num_frames and frame_labels[t] == flat_token_ids[target_idx]:
                            t += 1
                        spans.append((start_f, t))
                        target_idx += 1
                    else:
                        t += 1

                if len(spans) == num_tokens:
                    return spans
            except Exception:
                pass

        spans_fallback: list[tuple[int, int]] = []
        for idx in range(num_tokens):
            start_f = int(round(idx * max(1, num_frames - 1) / max(1, num_tokens)))
            end_f = max(
                start_f + 1,
                int(round((idx + 1) * num_frames / max(1, num_tokens))),
            )
            spans_fallback.append((min(start_f, max(0, num_frames - 1)), min(end_f, num_frames)))
        return spans_fallback

    def decode_ctc_window(
        self,
        log_probs: np.ndarray,
        start_frame: int,
        end_frame: int,
    ) -> list[str]:
        """
        CTC greedy collapse on `log_probs[start_frame:end_frame]` to obtain the acoustic
        phoneme sequence (`heard_ipa`) without running a second model pass.
        """
        s = max(0, start_frame)
        e = min(log_probs.shape[0], max(s + 1, end_frame))
        window = log_probs[s:e]
        if window.shape[0] == 0:
            return []

        pred_ids = np.argmax(window, axis=-1).tolist()
        collapsed: list[str] = []
        prev_id: int | None = None

        for tid in pred_ids:
            if tid != prev_id and tid != self.blank_id:
                tok = self.id_to_token.get(int(tid), "")
                if tok and tok not in ("<pad>", "<s>", "</s>", "<unk>", "|", " "):
                    collapsed.append(tok)
            prev_id = tid
        return collapsed

    def align_expected_and_heard_phonemes(
        self,
        expected_spans: list[AlignedPhonemeSpan],
        heard_phonemes: list[str],
    ) -> list[PhonemeAssessment]:
        """
        Needleman-Wunsch global sequence alignment weighted by `panphon` 24-feature articulatory
        distance matrix + native allophone equivalence rules.
        """
        n = len(expected_spans)
        m = len(heard_phonemes)
        del_cost = 0.90
        ins_cost = 0.90

        dp = np.zeros((n + 1, m + 1), dtype=np.float32)
        backtrack: list[list[str]] = [["" for _ in range(m + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            dp[i, 0] = i * del_cost
            backtrack[i][0] = "D"
        for j in range(1, m + 1):
            dp[0, j] = j * ins_cost
            backtrack[0][j] = "I"

        for i in range(1, n + 1):
            exp_p = expected_spans[i - 1].phoneme
            for j in range(1, m + 1):
                hrd_p = heard_phonemes[j - 1]
                sub_cost = self._articulatory_substitution_cost(exp_p, hrd_p)
                cost_match = dp[i - 1, j - 1] + sub_cost
                cost_del = dp[i - 1, j] + del_cost
                cost_ins = dp[i, j - 1] + ins_cost

                if cost_match <= cost_del and cost_match <= cost_ins:
                    dp[i, j] = cost_match
                    backtrack[i][j] = "M"
                elif cost_del <= cost_ins:
                    dp[i, j] = cost_del
                    backtrack[i][j] = "D"
                else:
                    dp[i, j] = cost_ins
                    backtrack[i][j] = "I"

        i, j = n, m
        diagnostics: list[PhonemeAssessment] = []
        while i > 0 or j > 0:
            op = backtrack[i][j]
            if op == "M":
                span = expected_spans[i - 1]
                hrd_p = heard_phonemes[j - 1]
                art_dist = self._articulatory_substitution_cost(span.phoneme, hrd_p)
                is_allophone_or_passed = (
                    self.is_acceptable_allophone(span.phoneme, hrd_p)
                    or (span.gop_score >= self.gop_pass_threshold and art_dist <= 0.35)
                )
                if is_allophone_or_passed:
                    diagnostics.append(
                        PhonemeAssessment(
                            expected=span.phoneme,
                            heard=span.phoneme,
                            gop_score=max(span.gop_score, 85),
                            error=None,
                        )
                    )
                else:
                    adjusted_gop = min(
                        span.gop_score,
                        int(max(5, round(span.gop_score * (1.0 - min(0.75, art_dist * 0.55))))),
                    )
                    diagnostics.append(
                        PhonemeAssessment(
                            expected=span.phoneme,
                            heard=hrd_p,
                            gop_score=adjusted_gop,
                            error="substitution",
                            articulatory_diff=self.describe_articulatory_diff(span.phoneme, hrd_p),
                        )
                    )
                i -= 1
                j -= 1
            elif op == "D":
                span = expected_spans[i - 1]
                # If a word-final stop was unreleased but scored >= 78 via closure GOP, treat as accurate
                if i == n and span.phoneme in UNRELEASED_FINAL_STOPS and span.gop_score >= 78:
                    diagnostics.append(
                        PhonemeAssessment(
                            expected=span.phoneme,
                            heard=span.phoneme,
                            gop_score=span.gop_score,
                            error=None,
                        )
                    )
                else:
                    diagnostics.append(
                        PhonemeAssessment(
                            expected=span.phoneme,
                            heard=None,
                            gop_score=min(span.gop_score, 25),
                            error="deletion",
                            articulatory_diff=f"Missing /{span.phoneme}/ sound — make sure to articulate /{span.phoneme}/ clearly instead of skipping it.",
                        )
                    )
                i -= 1
            else:
                hrd_p = heard_phonemes[j - 1]
                diagnostics.append(
                    PhonemeAssessment(
                        expected=None,
                        heard=hrd_p,
                        gop_score=20,
                        error="insertion",
                        articulatory_diff=f"Extra /{hrd_p}/ sound inserted — transition smoothly without adding /{hrd_p}/.",
                    )
                )
                j -= 1

        diagnostics.reverse()
        return diagnostics

    def assess_words(
        self,
        logits: np.ndarray,
        word_targets: list[WordPhonemeTarget],
        chunk_offset_ms: int = 0,
    ) -> list[WordAssessment]:
        """
        Align expected word phonemes against single-pass `logits`, compute pooled-allophone GOP
        scores, and apply Mandatory Day-1 Optimization #3 (Conditional Error Diagnostics).
        """
        if not word_targets or logits.shape[0] == 0:
            return []

        log_probs = self.compute_log_probs(logits)
        flat_token_ids: list[int] = []
        word_slices: list[tuple[int, int]] = []

        for wt in word_targets:
            start_idx = len(flat_token_ids)
            flat_token_ids.extend(wt.token_ids)
            word_slices.append((start_idx, len(flat_token_ids)))

        all_spans = self._forced_align_tokens(log_probs, flat_token_ids)
        num_frames = log_probs.shape[0]

        word_assessments: list[WordAssessment] = []
        for word_idx, (wt, (s_idx, e_idx)) in enumerate(zip(word_targets, word_slices)):
            token_spans = all_spans[s_idx:e_idx]
            if not token_spans:
                continue

            aligned_phonemes: list[AlignedPhonemeSpan] = []
            num_p = len(wt.phonemes)
            for p_idx, (p_sym, tok_id, (sf_idx, ef_idx)) in enumerate(
                zip(wt.phonemes, wt.token_ids, token_spans)
            ):
                is_final = p_idx == num_p - 1
                gop = self._calibrate_gop_score(
                    log_probs, tok_id, sf_idx, ef_idx, is_word_final=is_final
                )
                aligned_phonemes.append(
                    AlignedPhonemeSpan(
                        phoneme=p_sym,
                        token_id=tok_id,
                        start_frame=sf_idx,
                        end_frame=ef_idx,
                        gop_score=gop,
                    )
                )

            word_start_frame = aligned_phonemes[0].start_frame
            word_end_frame = aligned_phonemes[-1].end_frame
            start_ms = chunk_offset_ms + int(round(word_start_frame * self.frame_stride_ms))
            end_ms = chunk_offset_ms + int(
                round(max(word_start_frame + 1, word_end_frame) * self.frame_stride_ms)
            )

            word_gop = int(round(float(np.mean([p.gop_score for p in aligned_phonemes]))))
            stress_diag = self._evaluate_word_stress(wt, aligned_phonemes)

            # Mandatory Day-1 Optimization #3:
            # If a word's average GOP score exceeds `85/100`, mark it `correct` and skip
            # Needleman-Wunsch alignment and `panphon` lookups for that word!
            if word_gop > self.gop_pass_threshold:
                phoneme_items = [
                    PhonemeAssessment(
                        expected=p.phoneme,
                        heard=p.phoneme,
                        gop_score=p.gop_score,
                        error=None,
                    )
                    for p in aligned_phonemes
                ]
                word_assessments.append(
                    WordAssessment(
                        word=wt.word,
                        score=word_gop,
                        status="correct",
                        expected_ipa=wt.sanitized_ipa,
                        heard_ipa=wt.sanitized_ipa,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        phonemes=phoneme_items,
                        word_stress=stress_diag,
                    )
                )
                continue

            prev_end = all_spans[word_slices[word_idx - 1][1] - 1][1] if word_idx > 0 else 0
            next_start = (
                all_spans[word_slices[word_idx + 1][0]][0]
                if word_idx + 1 < len(word_slices)
                else num_frames
            )
            win_start = max(prev_end, word_start_frame - 1)
            win_end = min(next_start, word_end_frame + 1)
            heard_list = self.decode_ctc_window(log_probs, win_start, win_end)

            if not heard_list:
                for p in aligned_phonemes:
                    span_slice = np.copy(
                        log_probs[p.start_frame : max(p.start_frame + 1, p.end_frame)]
                    )
                    if 0 <= self.blank_id < span_slice.shape[1]:
                        span_slice[:, self.blank_id] = -1e9
                    top_id = int(np.argmax(np.mean(span_slice, axis=0)))
                    tok = self.id_to_token.get(top_id, p.phoneme)
                    if tok not in ("<pad>", "<s>", "</s>", "<unk>", "|", " "):
                        heard_list.append(tok)

            phoneme_diagnostics = self.align_expected_and_heard_phonemes(
                aligned_phonemes, heard_list
            )
            heard_ipa_str = "".join(p.heard for p in phoneme_diagnostics if p.heard is not None)

            final_word_score = (
                int(round(float(np.mean([p.gop_score for p in phoneme_diagnostics]))))
                if phoneme_diagnostics
                else word_gop
            )

            if not heard_ipa_str or final_word_score < 20:
                status = "omitted" if not heard_ipa_str else "mispronounced"
            elif final_word_score >= 80 and all(
                p.error is None for p in phoneme_diagnostics
            ):
                status = "correct"
            elif final_word_score >= self.gop_warn_threshold:
                status = "warning"
            else:
                status = "mispronounced"

            word_assessments.append(
                WordAssessment(
                    word=wt.word,
                    score=final_word_score,
                    status=status,
                    expected_ipa=wt.sanitized_ipa,
                    heard_ipa=heard_ipa_str,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    phonemes=phoneme_diagnostics,
                    word_stress=stress_diag,
                )
            )

        return word_assessments
