"""Aligner / GOP behaviour on synthetic logits. No model files needed."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from app.aligner_gop import ForcedAlignerGOP
from app.g2p import SyllableTarget, WordPhonemeTarget

TOKENS = ["<pad>", "θ", "ɔː", "t", "s", "ɪ", "k", "p", "f", "ə", "ʃ"]
ID = {tok: i for i, tok in enumerate(TOKENS)}
ID_TO_TOKEN = dict(enumerate(TOKENS))
BLANK = 0


def logits_for(labels: list[int], strength: float = 6.0) -> np.ndarray:
    """One 20 ms frame per label; the labelled token gets a strong logit."""
    lg = np.zeros((len(labels), len(TOKENS)), dtype=np.float32)
    for i, lab in enumerate(labels):
        lg[i, lab] = strength
    return lg


def word(text: str, phonemes: list[str]) -> WordPhonemeTarget:
    return WordPhonemeTarget(
        word=text,
        raw_ipa="".join(phonemes),
        sanitized_ipa="".join(phonemes),
        phonemes=phonemes,
        token_ids=[ID[p] for p in phonemes],
        syllables=[
            SyllableTarget(ipa="".join(phonemes), stress="primary",
                           phoneme_indices=list(range(len(phonemes))))
        ],
    )


@pytest.fixture(scope="module")
def aligner() -> ForcedAlignerGOP:
    return ForcedAlignerGOP(id_to_token=ID_TO_TOKEN, blank_id=BLANK)


THOUGHT = word("thought", ["θ", "ɔː", "t"])
# frames: blank blank [θ x3] blank [ɔː x4] blank [t x2] blank
SAID_THOUGHT = [0, 0, 1, 1, 1, 0, 2, 2, 2, 2, 0, 3, 3, 0]
SAID_TAUGHT = [0, 0, 3, 3, 3, 0, 2, 2, 2, 2, 0, 3, 3, 0]


def test_clean_word_takes_fast_path(aligner):
    w = aligner.assess_words(logits_for(SAID_THOUGHT), [THOUGHT])[0]
    assert w.status == "correct"
    assert w.score > 85
    assert w.heard_ipa == w.expected_ipa
    assert all(p.error is None for p in w.phonemes)


def test_swapped_consonant_is_flagged_and_word_is_mispronounced(aligner):
    w = aligner.assess_words(logits_for(SAID_TAUGHT), [THOUGHT])[0]
    assert w.heard_ipa == "tɔːt"
    first = w.phonemes[0]
    assert (first.expected, first.heard, first.error) == ("θ", "t", "substitution")
    assert first.gop_score < 30
    assert first.articulatory_diff
    # one clearly wrong sound must not be averaged into a mere "warning"
    assert w.status == "mispronounced"


def test_one_bad_sound_cannot_hide_in_a_long_word(aligner):
    phonemes = ["t", "ɪ", "k", "s", "p", "ɔː", "f", "ə", "ʃ", "θ"]
    labels = [BLANK]
    for p in phonemes[:-1]:
        labels += [ID[p]] * 3 + [BLANK]
    labels += [ID["t"]] * 3 + [BLANK]  # last slot is acoustically /t/, target is /θ/
    w = aligner.assess_words(logits_for(labels), [word("longword", phonemes)])[0]
    # mean of the sound scores stays above the pass threshold...
    assert np.mean([p.gop_score for p in w.phonemes]) > 80
    # ...but the word must still not be reported as correct
    assert w.status != "correct"
    assert any(p.error == "substitution" and p.expected == "θ" for p in w.phonemes)


def test_alignment_fallback_is_logged(aligner, caplog):
    # 2 frames cannot hold 3 target tokens, so the even-spacing fallback is used
    with caplog.at_level(logging.WARNING, logger="voice_pronunciation_engine"):
        spans = aligner._forced_align_tokens(
            ForcedAlignerGOP.compute_log_probs(logits_for([1, 3])), [1, 2, 3]
        )
    assert len(spans) == 3
    assert any("even spacing" in r.getMessage() for r in caplog.records)
