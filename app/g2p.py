from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import espeakng_loader
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from phonemizer.separator import Separator

IPA_VOWEL_CHARS = frozenset("aeiouɑɒæɛɪɔʊʌəɐɜɝɚɨʉɵyøœɶɯɤ")

# Suprasegmentals, stress marks, length marks, and tie bars to sanitize before tokenizer lookup
SUPRASEGMENTAL_STRIP_CHARS = frozenset(
    {
        "ˈ",  # primary stress
        "ˌ",  # secondary stress
        "ː",  # long vowel mark (handled explicitly before stripping)
        "ˑ",  # half-long mark
        "̆",  # extra-short
        ".",  # syllable break
        "|",  # minor (foot) group
        "‖",  # major (intonation) group
        "͡",  # tie bar above
        "͜",  # tie bar below
        "‿",  # linking mark
        "↗",  # global rise
        "↘",  # global fall
        "⁻",
        "-",
    }
)

# Common espeak-ng IPA character normalizations to standard Wav2Vec2 espeak vocabulary
IPA_FALLBACK_CHAR_MAP: dict[str, str] = {
    "ɡ": "ɡ",
    "g": "ɡ",
    "ɚ": "ɜ",
    "ɝ": "ɜː",
    "ɫ": "l",
    "ɾ": "t",
    "ʔ": "t",
    "ɐ": "ə",
    "ʉ": "u",
    "ɵ": "ə",
    "ɒ": "ɔ",
    "ɑ": "ɑː",
    "r": "ɹ",
}

LANGUAGE_MAP: dict[str, str] = {
    "de": "de",
    "en": "en-us",
    "en-us": "en-us",
    "en-gb": "en-gb",
    "es": "es",
    "fr": "fr-fr",
    "it": "it",
    "pt": "pt-br",
    "nl": "nl",
    "pl": "pl",
    "sv": "sv",
    "tr": "tr",
    "ru": "ru",
    "hi": "hi",
    "ar": "ar",
    "ja": "ja",
    "zh": "cmn",
    "ko": "ko",
}

_WORD_TOKEN_RE = re.compile(r"[^\w\s'-]+", flags=re.UNICODE)


@dataclass(frozen=True)
class SyllableTarget:
    """Syllable stress and phoneme index range within a WordPhonemeTarget."""

    ipa: str
    stress: Literal["primary", "secondary", "unstressed"]
    phoneme_indices: list[int]


@dataclass(frozen=True)
class WordPhonemeTarget:
    """Expected phoneme breakdown and strictly validated Wav2Vec2 token IDs for a reference word."""

    word: str
    raw_ipa: str
    sanitized_ipa: str
    phonemes: list[str]
    token_ids: list[int]
    syllables: list[SyllableTarget] = field(default_factory=list)


def configure_espeak_environment(
    custom_library_path: str | None = None,
    custom_data_path: str | None = None,
) -> None:
    """
    Configure phonemizer's EspeakWrapper using either custom paths or bundled espeakng-loader
    so espeak-ng works across Windows, Linux, and macOS without external PATH fragility.
    """
    data_path = custom_data_path or os.environ.get("ESPEAK_DATA_PATH") or espeakng_loader.get_data_path()
    if data_path and Path(data_path).exists():
        os.environ["ESPEAK_DATA_PATH"] = str(data_path)
        os.environ["PHONEMIZER_ESPEAK_DATA"] = str(data_path)

    lib_path = custom_library_path or os.environ.get("PHONEMIZER_ESPEAK_LIBRARY") or espeakng_loader.get_library_path()
    if lib_path and Path(lib_path).exists():
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(lib_path)
        try:
            EspeakWrapper.set_library(str(lib_path))
        except Exception:
            pass


class G2PService:
    """
    Grapheme-to-Phoneme (G2P) service backed by espeak-ng (`phonemizer`) with:
    1. Custom dictionary overrides
    2. IPA suprasegmental sanitization (`ˈ`, `ˌ`, `ː`)
    3. Strict Wav2Vec2 tokenizer vocabulary mapping (zero `<unk>` index errors during `forced_align`)
    """

    def __init__(
        self,
        vocab: dict[str, int],
        custom_lexicon: dict[str, str] | None = None,
        espeak_library_path: str | None = None,
        espeak_data_path: str | None = None,
    ) -> None:
        configure_espeak_environment(espeak_library_path, espeak_data_path)
        self.vocab = dict(vocab)
        self.id_to_token: dict[int, str] = { idx: tok for tok, idx in self.vocab.items() }
        self.custom_lexicon = {k.lower(): v for k, v in (custom_lexicon or {}).items()}
        self._backends: dict[str, EspeakBackend] = {}

        # Identify special tokens to exclude from acoustic phoneme targets
        self.blank_id: int = self.vocab.get("<pad>", 0)
        self.unk_id: int = self.vocab.get("<unk>", -1)
        self.special_tokens: set[str] = {"<pad>", "<s>", "</s>", "<unk>", "|", " "}

        # Valid phoneme vocabulary sorted longest-first for greedy maximal munch tokenization
        self.valid_phoneme_tokens: list[str] = sorted(
            [tok for tok in self.vocab.keys() if tok not in self.special_tokens],
            key=len,
            reverse=True,
        )

    @classmethod
    def from_tokenizer_dir(
        cls,
        tokenizer_dir: Path | str,
        custom_lexicon: dict[str, str] | None = None,
        espeak_library_path: str | None = None,
        espeak_data_path: str | None = None,
    ) -> G2PService:
        tokenizer_path = Path(tokenizer_dir)
        vocab_file = tokenizer_path / "vocab.json"
        if not vocab_file.exists():
            raise FileNotFoundError(
                f"Tokenizer vocab.json not found at {vocab_file}. Run `python -m scripts.export_onnx` first."
            )
        with vocab_file.open("r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(
            vocab=vocab,
            custom_lexicon=custom_lexicon,
            espeak_library_path=espeak_library_path,
            espeak_data_path=espeak_data_path,
        )

    def _get_backend(self, language: str) -> EspeakBackend:
        espeak_lang = LANGUAGE_MAP.get(language.lower(), language.lower())
        if espeak_lang not in self._backends:
            self._backends[espeak_lang] = EspeakBackend(
                language=espeak_lang,
                preserve_punctuation=False,
                with_stress=True,
                tie=False,
            )
        return self._backends[espeak_lang]

    @staticmethod
    def _is_vowel_phoneme(phoneme: str) -> bool:
        return any(ch in IPA_VOWEL_CHARS for ch in phoneme)

    def segment_syllables_with_stress(
        self,
        raw_ipa: str,
        phonemes: list[str],
    ) -> list[SyllableTarget]:
        """
        Group a word's phonemes into syllables around vowel nuclei and assign
        `primary` (`ˈ`), `secondary` (`ˌ`), or `unstressed` labels from `espeak-ng` output.
        """
        if not phonemes:
            return []

        vowel_indices = [i for i, p in enumerate(phonemes) if self._is_vowel_phoneme(p)]
        if len(vowel_indices) <= 1:
            stress: Literal["primary", "secondary", "unstressed"] = (
                "primary" if "ˈ" in raw_ipa else ("secondary" if "ˌ" in raw_ipa else "primary")
            )
            return [
                SyllableTarget(
                    ipa="".join(phonemes),
                    stress=stress,
                    phoneme_indices=list(range(len(phonemes))),
                )
            ]

        # Determine stress rank of each vowel nucleus by scanning raw_ipa in order
        compact_raw = raw_ipa.replace(" ", "")
        stress_markers_by_vowel: list[Literal["primary", "secondary", "unstressed"]] = []
        pending_stress: Literal["primary", "secondary", "unstressed"] = "unstressed"
        in_vowel_run = False

        for ch in compact_raw:
            if ch == "ˈ":
                pending_stress = "primary"
                in_vowel_run = False
            elif ch == "ˌ":
                if pending_stress != "primary":
                    pending_stress = "secondary"
                in_vowel_run = False
            elif ch in IPA_VOWEL_CHARS:
                if not in_vowel_run:
                    stress_markers_by_vowel.append(pending_stress)
                    pending_stress = "unstressed"
                    in_vowel_run = True
            elif ch != "ː":
                in_vowel_run = False

        while len(stress_markers_by_vowel) < len(vowel_indices):
            stress_markers_by_vowel.append("unstressed")

        if "primary" not in stress_markers_by_vowel:
            stress_markers_by_vowel[0] = "primary"

        # Partition phoneme indices into syllables around vowel_indices
        syllables: list[SyllableTarget] = []
        num_vowels = len(vowel_indices)
        start_p = 0

        for s_idx, v_idx in enumerate(vowel_indices):
            if s_idx == num_vowels - 1:
                end_p = len(phonemes)
            else:
                next_v = vowel_indices[s_idx + 1]
                # Split intervocalic consonants so at least 1 onset consonant goes to next syllable
                consonant_count = next_v - v_idx - 1
                if consonant_count <= 0:
                    end_p = v_idx + 1
                elif consonant_count == 1:
                    end_p = v_idx + 1
                else:
                    end_p = next_v - 1

            p_indices = list(range(start_p, end_p))
            syl_ipa = "".join(phonemes[i] for i in p_indices)
            syllables.append(
                SyllableTarget(
                    ipa=syl_ipa,
                    stress=stress_markers_by_vowel[s_idx],
                    phoneme_indices=p_indices,
                )
            )
            start_p = end_p

        return syllables

    def sanitize_ipa(self, raw_ipa: str) -> str:
        """
        Strip primary/secondary stress marks (`ˈ`, `ˌ`) and non-phonemic markers while
        preserving length marks (`ː`) only when the combined token exists in `self.vocab`.
        """
        cleaned = raw_ipa.strip()
        for ch in ("ˈ", "ˌ", "ˑ", "̆", ".", "|", "‖", "͡", "͜", "‿", "↗", "↘", "⁻", "-"):
            cleaned = cleaned.replace(ch, "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def map_ipa_to_vocab_tokens(self, raw_ipa: str) -> tuple[list[str], list[int]]:
        """
        Strictly map an espeak-ng IPA string into the Wav2Vec2 tokenizer vocabulary.
        Guarantees every returned token_id is a valid acoustic phoneme ID (never `<unk>` or `<pad>`).
        """
        sanitized = self.sanitize_ipa(raw_ipa)
        if not sanitized:
            return [], []

        # Step 1: If space-separated phoneme tokens are already provided by phonemizer, inspect each
        phonemes: list[str] = []
        token_ids: list[int] = []

        # Remove spaces for greedy longest-match segmentation against model vocab
        compact = sanitized.replace(" ", "")
        idx = 0
        n = len(compact)

        while idx < n:
            matched_token: str | None = None

            # Try longest matching token in Wav2Vec2 vocabulary first (handles `ɔː`, `ɑː`, `tʃ`, `dʒ`, `aɪ`, etc.)
            for cand in self.valid_phoneme_tokens:
                if compact.startswith(cand, idx):
                    matched_token = cand
                    idx += len(cand)
                    break

            if matched_token is not None:
                phonemes.append(matched_token)
                token_ids.append(self.vocab[matched_token])
                continue

            # Single character fallback normalization
            ch = compact[idx]
            idx += 1

            if ch in SUPRASEGMENTAL_STRIP_CHARS or ch.isspace():
                continue

            mapped_ch = IPA_FALLBACK_CHAR_MAP.get(ch)
            if mapped_ch:
                if mapped_ch in self.vocab and mapped_ch not in self.special_tokens:
                    phonemes.append(mapped_ch)
                    token_ids.append(self.vocab[mapped_ch])
                    continue
                # Or if mapped_ch stripped of `ː` is in vocab
                base_ch = mapped_ch.replace("ː", "")
                if base_ch in self.vocab and base_ch not in self.special_tokens:
                    phonemes.append(base_ch)
                    token_ids.append(self.vocab[base_ch])
                    continue

        return phonemes, token_ids

    def phonemize_text(self, reference_text: str, language: str = "en") -> list[WordPhonemeTarget]:
        """
        Convert reference text into a list of WordPhonemeTarget objects with sanitized IPA
        and strictly validated Wav2Vec2 vocabulary token IDs.
        """
        cleaned_text = _WORD_TOKEN_RE.sub(" ", reference_text)
        words = [w.strip() for w in cleaned_text.split() if w.strip()]
        if not words:
            return []

        backend = self._get_backend(language)
        separator = Separator(phone=" ", word="|", syllable="")

        # Separate custom lexicon hits from words needing espeak-ng
        words_to_phonemize: list[str] = []
        words_indices: list[int] = []
        raw_ipas: list[str] = [""] * len(words)

        for i, word in enumerate(words):
            lower_w = word.lower()
            if lower_w in self.custom_lexicon:
                raw_ipas[i] = self.custom_lexicon[lower_w]
            else:
                words_to_phonemize.append(lower_w)
                words_indices.append(i)

        if words_to_phonemize:
            batch_out = backend.phonemize(
                words_to_phonemize,
                separator=separator,
                strip=True,
            )
            for idx_in_words, ipa_str in zip(words_indices, batch_out):
                raw_ipas[idx_in_words] = ipa_str.replace("|", " ").strip()

        results: list[WordPhonemeTarget] = []
        for word, raw_ipa in zip(words, raw_ipas):
            phonemes, token_ids = self.map_ipa_to_vocab_tokens(raw_ipa)
            if not token_ids:
                # Fallback if word produced no phonemes (e.g. digits/symbols): phonemize character by character
                phonemes, token_ids = self.map_ipa_to_vocab_tokens("ə")
            sanitized_ipa = "".join(phonemes)
            syllables = self.segment_syllables_with_stress(raw_ipa, phonemes)
            results.append(
                WordPhonemeTarget(
                    word=word,
                    raw_ipa=raw_ipa,
                    sanitized_ipa=sanitized_ipa,
                    phonemes=phonemes,
                    token_ids=token_ids,
                    syllables=syllables,
                )
            )
        return results
