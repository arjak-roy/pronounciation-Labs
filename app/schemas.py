from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PhonemeAssessment(BaseModel):
    """Phoneme-level GOP score and articulatory diagnostic."""

    model_config = ConfigDict(extra="forbid")

    expected: str | None = Field(
        default=None, description="Expected IPA phoneme symbol (null for insertion errors)"
    )
    heard: str | None = Field(
        default=None, description="Acoustic IPA phoneme heard (null for deletion errors)"
    )
    gop_score: int = Field(..., ge=0, le=100, description="Goodness of Pronunciation score (0-100)")
    error: Literal["substitution", "deletion", "insertion"] | None = Field(
        default=None, description="Phoneme error classification, or null when accurate"
    )
    articulatory_diff: str | None = Field(
        default=None,
        description="Dynamic panphon articulatory feature diff explaining what the mouth/tongue did vs how to fix it",
    )


class SyllableStressItem(BaseModel):
    """Per-syllable stress and duration breakdown for multi-syllable words."""

    model_config = ConfigDict(extra="forbid")

    syllable_ipa: str
    stress: Literal["primary", "secondary", "unstressed"] = "unstressed"
    duration_ms: int = 0
    score: int = 100


class WordStressDiagnostic(BaseModel):
    """Multi-syllable lexical stress diagnostic for a word."""

    model_config = ConfigDict(extra="forbid")

    syllable_count: int = 1
    expected_primary_index: int = 0
    detected_peak_index: int = 0
    stress_correct: bool = True
    stress_pattern_display: str = ""
    coaching_tip: str | None = None
    syllables: list[SyllableStressItem] = Field(default_factory=list)


class WordLinkDiagnostic(BaseModel):
    """Connected-speech linking analysis between word[i] and word[i+1]."""

    model_config = ConfigDict(extra="forbid")

    from_word: str
    to_word: str
    link_type: Literal["consonant_to_vowel", "consonant_cluster", "vowel_to_vowel", "normal"]
    gap_ms: int
    status: Literal["smooth_link", "natural_pause", "choppy_pause"]
    coaching_tip: str | None = None


class ConnectedSpeechReport(BaseModel):
    """Sentence-level connected speech, word linking, and intonation summary."""

    model_config = ConfigDict(extra="forbid")

    linking_score: int = Field(default=100, ge=0, le=100)
    smooth_links_count: int = 0
    choppy_Pause_count: int = 0
    intonation_contour_hz: list[float] = Field(default_factory=list)
    intonation_type: Literal["falling_statement", "rising_question", "flat_monotone", "expressive"] = "expressive"
    summary_tip: str = ""
    links: list[WordLinkDiagnostic] = Field(default_factory=list)


class WordAssessment(BaseModel):
    """Word-level pronunciation assessment with timestamps and phoneme breakdown."""

    model_config = ConfigDict(extra="forbid")

    word: str = Field(..., description="Reference word text")
    score: int = Field(..., ge=0, le=100, description="Word GOP pronunciation score (0-100)")
    status: Literal["correct", "warning", "mispronounced", "omitted"] = Field(
        ..., description="Diagnostic status classification"
    )
    expected_ipa: str = Field(..., description="Sanitized expected IPA transcription")
    heard_ipa: str = Field(..., description="Decoded acoustic IPA transcription")
    start_ms: int = Field(..., ge=0, description="Word start timestamp in milliseconds")
    end_ms: int = Field(..., ge=0, description="Word end timestamp in milliseconds")
    phonemes: list[PhonemeAssessment] = Field(
        default_factory=list, description="Granular per-phoneme breakdown"
    )
    word_stress: WordStressDiagnostic | None = Field(
        default=None, description="Multi-syllable lexical stress diagnostic"
    )


class PronunciationMetrics(BaseModel):
    """High-level pronunciation sub-scores."""

    model_config = ConfigDict(extra="forbid")

    accuracy: int = Field(..., ge=0, le=100, description="Acoustic phoneme accuracy score (0-100)")
    fluency: int = Field(..., ge=0, le=100, description="Speech rate and pause continuity score (0-100)")
    prosody_similarity: int = Field(
        ..., ge=0, le=100, description="Prosodic / embedding similarity score (0-100)"
    )


class PronunciationAssessResponse(BaseModel):
    """Unified Mode B REST API response contract for POST /api/v1/pronunciation/assess."""

    model_config = ConfigDict(extra="forbid")

    overall_score: int = Field(..., ge=0, le=100, description="Composite pronunciation score (0-100)")
    metrics: PronunciationMetrics
    words: list[WordAssessment]
    connected_speech: ConnectedSpeechReport | None = Field(
        default=None, description="Connected speech linking and intonation contour report"
    )


class TranscribeEvent(BaseModel):
    """Unified Mode A WebSocket server->client JSON message contract."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["interim", "final"]
    transcript: str
    language: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class FreeSpeechAssessResponse(BaseModel):
    """
    Combined Mode A (Low-Latency STT) + Mode B (Pronunciation Auto-Correction) response
    for unscripted 10-15 second free-speaking intervals.
    """

    model_config = ConfigDict(extra="forbid")

    detected_transcript: str = Field(..., description="Intended speech transcribed by Mode A STT")
    stt_confidence: float = Field(..., ge=0.0, le=1.0, description="Mode A STT confidence")
    overall_score: int = Field(..., ge=0, le=100)
    metrics: PronunciationMetrics
    words: list[WordAssessment]
    connected_speech: ConnectedSpeechReport | None = Field(
        default=None, description="Connected speech linking and intonation contour report"
    )


class LibrarySyllablePreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    syllable_ipa: str
    stress: Literal["primary", "secondary", "unstressed"] = "unstressed"


class LibraryWordPreview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    word: str
    expected_ipa: str
    phonemes: list[str] = Field(default_factory=list)
    syllables: list[LibrarySyllablePreview] = Field(default_factory=list)


class LibraryItemCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1, max_length=300)
    item_type: Literal["word", "phrase"] | None = None
    language: str = Field(default="en")
    difficulty: Literal["Beginner", "Intermediate", "Advanced"] = "Beginner"
    category: str = Field(default="Custom")
    focus_phoneme: str = Field(default="")
    meaning_or_tip: str = Field(default="")


class LibraryItemUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    item_type: Literal["word", "phrase"] | None = None
    language: str | None = None
    difficulty: Literal["Beginner", "Intermediate", "Advanced"] | None = None
    category: str | None = None
    focus_phoneme: str | None = None
    meaning_or_tip: str | None = None


class LibraryItemResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    text: str
    item_type: Literal["word", "phrase"]
    language: str
    difficulty: str
    category: str
    focus_phoneme: str
    meaning_or_tip: str
    expected_ipa: str
    words_breakdown: list[LibraryWordPreview] = Field(default_factory=list)
    best_score: int = 0
    attempts_count: int = 0
    last_practiced_at: str | None = None
    created_at: str


class WeakPhonemeStat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phoneme: str
    error_count: int


class PracticeAttemptSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    item_id: int | None = None
    reference_text: str
    language: str
    overall_score: int
    accuracy_score: int
    fluency_score: int
    prosody_score: int
    weak_phonemes: list[str] = Field(default_factory=list)
    created_at: str


class LibraryStatsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_items: int
    words_count: int
    phrases_count: int
    practiced_count: int
    mastered_count: int
    total_attempts: int
    average_score: int
    weak_phonemes: list[WeakPhonemeStat] = Field(default_factory=list)
    recent_attempts: list[PracticeAttemptSummary] = Field(default_factory=list)
