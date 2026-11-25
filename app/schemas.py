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
