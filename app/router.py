from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse

from app.acoustic_model import BaseAcousticEngine
from app.aligner_gop import ForcedAlignerGOP
from app.audio_utils import load_and_normalize_audio, split_at_breath_pauses
from app.config import Settings
from app.db import PronunciationLibraryDB
from app.g2p import G2PService
from app.prosody import analyze_connected_speech, compute_pronunciation_metrics
from app.schemas import (
    FreeSpeechAssessResponse,
    LibraryItemCreate,
    LibraryItemResponse,
    LibraryItemUpdate,
    LibraryStatsResponse,
    PronunciationAssessResponse,
)
from app.stt_engines import BaseSTTEngine

router = APIRouter()
logger = logging.getLogger("voice_pronunciation_engine")
_STATIC_HTML_PATH = Path(__file__).resolve().parent / "static" / "index.html"
_REACT_HTML_PATH = Path(__file__).resolve().parent / "static" / "react-app" / "index.html"


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_transitvox_react_app() -> HTMLResponse:
    """Serve the production TransitVox React App at /, falling back to HTML tester if not built."""
    if _REACT_HTML_PATH.exists():
        return HTMLResponse(content=_REACT_HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content=_STATIC_HTML_PATH.read_text(encoding="utf-8"))


@router.get("/tester", response_class=HTMLResponse, summary="Browser HTML Testing Client")
async def serve_html_tester() -> HTMLResponse:
    """Serve the built-in browser HTML testing client for Mode A & Mode B."""
    return HTMLResponse(content=_STATIC_HTML_PATH.read_text(encoding="utf-8"))


@router.get("/api/v1/pronunciation/phonemize", summary="Preview Expected Word-by-Word IPA for Shadowing")
async def preview_phonemize(
    request: Request,
    text: str,
    language: str = "en",
) -> dict[str, object]:
    """Return word-by-word expected IPA transcription using espeak-ng for the Shadowing UI."""
    g2p: G2PService | None = getattr(request.app.state, "g2p", None)
    if g2p is None:
        raise HTTPException(status_code=503, detail="G2P service not initialized.")
    targets = g2p.phonemize_text(reference_text=text, language=language)
    return {
        "text": text,
        "language": language,
        "words": [
            {
                "word": t.word,
                "expected_ipa": t.sanitized_ipa,
                "phonemes": t.phonemes,
                "syllables": [
                    {"syllable_ipa": s.ipa, "stress": s.stress} for s in t.syllables
                ],
            }
            for t in targets
        ],
    }


@router.get("/health")
async def health_check(request: Request) -> dict[str, object]:
    """Liveness and active Mode A / Mode B profile status."""
    settings: Settings = request.app.state.settings
    acoustic_engine: BaseAcousticEngine | None = getattr(
        request.app.state, "acoustic_engine", None
    )
    stt_engine: BaseSTTEngine | None = getattr(request.app.state, "stt_engine", None)
    g2p: G2PService | None = getattr(request.app.state, "g2p", None)
    library_db: PronunciationLibraryDB | None = getattr(request.app.state, "library_db", None)
    return {
        "status": "ok",
        "mode_a": {
            "device": settings.mode_a_device,
            "engine": settings.mode_a_engine,
            "ready": stt_engine is not None,
            "provider": stt_engine.engine_name if stt_engine else None,
        },
        "mode_b": {
            "device": settings.mode_b_device,
            "model": settings.mode_b_model,
            "ready": acoustic_engine is not None and g2p is not None,
            "provider": acoustic_engine.active_provider if acoustic_engine else None,
            "vocab_size": len(g2p.vocab) if g2p else 0,
        },
        "library_db": {
            "ready": library_db is not None,
            "path": str(settings.database_path),
        },
    }


def _require_library_db(request: Request) -> PronunciationLibraryDB:
    library_db: PronunciationLibraryDB | None = getattr(request.app.state, "library_db", None)
    if library_db is None:
        raise HTTPException(status_code=503, detail="Library database not initialized.")
    return library_db


@router.get(
    "/api/v1/library/items",
    response_model=list[LibraryItemResponse],
    summary="List Words & Phrases from the SQLite Library",
)
async def list_library_items(
    request: Request,
    language: str | None = None,
    item_type: str | None = None,
    difficulty: str | None = None,
    category: str | None = None,
    search: str | None = None,
) -> list[LibraryItemResponse]:
    db = _require_library_db(request)
    return db.list_items(
        language=language,
        item_type=item_type,
        difficulty=difficulty,
        category=category,
        search=search,
    )


@router.post(
    "/api/v1/library/items",
    response_model=LibraryItemResponse,
    summary="Create a Word or Phrase with Automatic espeak-ng IPA & Syllable Breakdown",
)
async def create_library_item(
    request: Request,
    payload: LibraryItemCreate,
) -> LibraryItemResponse:
    db = _require_library_db(request)
    return db.create_item(payload)


@router.put(
    "/api/v1/library/items/{item_id}",
    response_model=LibraryItemResponse,
    summary="Update a Word or Phrase in the Library",
)
async def update_library_item(
    request: Request,
    item_id: int,
    payload: LibraryItemUpdate,
) -> LibraryItemResponse:
    db = _require_library_db(request)
    updated = db.update_item(item_id, payload)
    if updated is None:
        raise HTTPException(status_code=404, detail="Library item not found.")
    return updated


@router.delete(
    "/api/v1/library/items/{item_id}",
    summary="Delete a Word or Phrase from the Library",
)
async def delete_library_item(
    request: Request,
    item_id: int,
) -> dict[str, object]:
    db = _require_library_db(request)
    deleted = db.delete_item(item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Library item not found.")
    return {"deleted": True, "id": item_id}


@router.get(
    "/api/v1/library/stats",
    response_model=LibraryStatsResponse,
    summary="Learner Progress, Weak Phonemes & Attempt History",
)
async def get_library_stats(
    request: Request,
    language: str | None = None,
) -> LibraryStatsResponse:
    db = _require_library_db(request)
    return db.get_stats(language=language)


async def _score_samples_against_text(
    settings: Settings,
    acoustic_engine: BaseAcousticEngine,
    g2p: G2PService,
    aligner: ForcedAlignerGOP,
    samples: np.ndarray,
    leading_trimmed_ms: int,
    reference_text: str,
    language: str,
) -> PronunciationAssessResponse:
    word_targets = g2p.phonemize_text(reference_text=reference_text, language=language)
    if not word_targets:
        raise HTTPException(
            status_code=422,
            detail="Could not extract valid IPA phoneme targets from reference_text.",
        )

    chunks = split_at_breath_pauses(
        samples=samples,
        sample_rate=settings.sample_rate,
        max_chunk_seconds=settings.mode_b_max_chunk_seconds,
    )

    logits_list: list[np.ndarray] = []
    hidden_list: list[np.ndarray] = []

    for chunk in chunks:
        forward_out = await acoustic_engine.infer_single_pass(chunk.samples)
        logits_list.append(forward_out.logits)
        hidden_list.append(forward_out.hidden_states)

    combined_logits = (
        np.concatenate(logits_list, axis=0) if len(logits_list) > 1 else logits_list[0]
    )
    combined_hidden = (
        np.concatenate(hidden_list, axis=0) if len(hidden_list) > 1 else hidden_list[0]
    )

    words_assessed = aligner.assess_words(
        logits=combined_logits,
        word_targets=word_targets,
        chunk_offset_ms=leading_trimmed_ms,
    )

    total_duration_ms = int(round(len(samples) * 1000.0 / settings.sample_rate))
    overall_score, metrics = compute_pronunciation_metrics(
        words=words_assessed,
        hidden_states=combined_hidden,
        total_duration_ms=total_duration_ms,
        waveform=samples,
        sample_rate=settings.sample_rate,
    )
    connected = analyze_connected_speech(
        words=words_assessed,
        waveform=samples,
        sample_rate=settings.sample_rate,
    )

    return PronunciationAssessResponse(
        overall_score=overall_score,
        metrics=metrics,
        words=words_assessed,
        connected_speech=connected,
    )


async def _handle_pronunciation_assess(
    request: Request,
    audio: UploadFile,
    reference_text: str,
    language: str,
    sentence_id: str | None,
) -> PronunciationAssessResponse:
    settings: Settings = request.app.state.settings
    acoustic_engine: BaseAcousticEngine | None = getattr(
        request.app.state, "acoustic_engine", None
    )
    g2p: G2PService | None = getattr(request.app.state, "g2p", None)
    aligner: ForcedAlignerGOP | None = getattr(request.app.state, "aligner", None)

    if acoustic_engine is None or g2p is None or aligner is None:
        raise HTTPException(
            status_code=503,
            detail="Mode B Pronunciation Engine is not initialized.",
        )

    if not reference_text or not reference_text.strip():
        raise HTTPException(status_code=422, detail="reference_text must not be empty.")

    audio_bytes = await audio.read()
    try:
        samples, leading_trimmed_ms = load_and_normalize_audio(
            audio_bytes=audio_bytes,
            target_sr=settings.sample_rate,
            trim_silence=settings.mode_b_trim_silence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if samples.size < int(0.08 * settings.sample_rate):
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio clip is too short (< 80ms of active speech).",
        )

    result = await _score_samples_against_text(
        settings=settings,
        acoustic_engine=acoustic_engine,
        g2p=g2p,
        aligner=aligner,
        samples=samples,
        leading_trimmed_ms=leading_trimmed_ms,
        reference_text=reference_text,
        language=language,
    )

    library_db: PronunciationLibraryDB | None = getattr(request.app.state, "library_db", None)
    if library_db is not None:
        weak_phonemes: list[str] = []
        for w in result.words:
            for p in w.phonemes:
                if p.error is not None and p.expected:
                    weak_phonemes.append(p.expected)
        parsed_item_id: int | None = None
        if sentence_id and sentence_id.isdigit():
            parsed_item_id = int(sentence_id)
        try:
            library_db.record_attempt(
                reference_text=reference_text,
                language=language,
                overall_score=result.overall_score,
                accuracy_score=result.metrics.accuracy,
                fluency_score=result.metrics.fluency,
                prosody_score=result.metrics.prosody_similarity,
                weak_phonemes=weak_phonemes,
                item_id=parsed_item_id,
            )
        except Exception:
            # The learner still gets their result; the lost attempt must at least be visible.
            logger.exception("record_attempt failed; attempt not saved")

    return result


@router.post(
    "/api/v1/pronunciation/assess",
    response_model=PronunciationAssessResponse,
    summary="Mode B: On-Demand Pronunciation Assessment",
)
async def assess_pronunciation_v1(
    request: Request,
    audio: UploadFile = File(..., description="16kHz, 16-bit, mono WAV audio file"),
    reference_text: str = Form(..., description="Expected text spoken by the user"),
    language: str = Form(default="en", description="ISO language code (e.g., en, es, fr, de, hi)"),
    sentence_id: str | None = Form(default=None, description="Optional lesson sentence cache ID"),
) -> PronunciationAssessResponse:
    """Unified Mode B REST endpoint for word- and phoneme-level pronunciation assessment."""
    return await _handle_pronunciation_assess(
        request=request,
        audio=audio,
        reference_text=reference_text,
        language=language,
        sentence_id=sentence_id,
    )


@router.post(
    "/assess",
    response_model=PronunciationAssessResponse,
    summary="Mode B Alias: Single-Pass Pronunciation Assessment",
)
async def assess_pronunciation_short(
    request: Request,
    audio: UploadFile = File(..., description="16kHz, 16-bit, mono WAV audio file"),
    reference_text: str = Form(..., description="Expected text spoken by the user"),
    language: str = Form(default="en", description="ISO language code"),
    sentence_id: str | None = Form(default=None, description="Optional lesson sentence cache ID"),
) -> PronunciationAssessResponse:
    """Short alias for POST /api/v1/pronunciation/assess."""
    return await _handle_pronunciation_assess(
        request=request,
        audio=audio,
        reference_text=reference_text,
        language=language,
        sentence_id=sentence_id,
    )


@router.post(
    "/api/v1/stt/transcribe",
    summary="Mode A: Upload Audio File for Multilingual Whisper Transcription",
)
async def transcribe_uploaded_audio(
    request: Request,
    audio: UploadFile = File(..., description="16kHz mono WAV audio file"),
    language: str = Form(default="de", description="ISO language code (de, en, en-gb, es, fr, hi)"),
) -> dict[str, object]:
    """Transcribe an uploaded audio file using Mode A (`faster-whisper-small` INT8)."""
    settings: Settings = request.app.state.settings
    stt_engine: BaseSTTEngine | None = getattr(request.app.state, "stt_engine", None)
    if stt_engine is None:
        raise HTTPException(status_code=503, detail="Mode A STT engine is not initialized.")

    audio_bytes = await audio.read()
    try:
        samples, _ = load_and_normalize_audio(
            audio_bytes=audio_bytes,
            target_sr=settings.sample_rate,
            trim_silence=settings.mode_b_trim_silence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if samples.size < int(0.10 * settings.sample_rate):
        raise HTTPException(status_code=400, detail="Audio clip is too short (< 100ms).")

    transcript, confidence = await stt_engine.transcribe_async(samples, language=language)
    duration_ms = int(round(samples.size * 1000.0 / settings.sample_rate))
    return {
        "transcript": transcript,
        "language": language,
        "confidence": confidence,
        "duration_ms": duration_ms,
    }


@router.post(
    "/api/v1/pronunciation/free-assess",
    response_model=FreeSpeechAssessResponse,
    summary="Mode A + Mode B: 10-15s Unscripted Free-Speech STT + Pronunciation Auto-Correction",
)
async def assess_free_speech(
    request: Request,
    audio: UploadFile = File(..., description="16kHz mono WAV (10-15s free speech)"),
    language: str = Form(default="en", description="Explicit ISO language code"),
) -> FreeSpeechAssessResponse:
    """
    Transcribe unscripted 10-15s audio using Mode A low-latency STT to infer intended text,
    then score the exact same audio through Mode B (Wav2Vec2 + forced_align + panphon)
    to highlight mispronounced words automatically.
    """
    settings: Settings = request.app.state.settings
    stt_engine: BaseSTTEngine | None = getattr(request.app.state, "stt_engine", None)
    acoustic_engine: BaseAcousticEngine | None = getattr(
        request.app.state, "acoustic_engine", None
    )
    g2p: G2PService | None = getattr(request.app.state, "g2p", None)
    aligner: ForcedAlignerGOP | None = getattr(request.app.state, "aligner", None)

    if stt_engine is None or acoustic_engine is None or g2p is None or aligner is None:
        raise HTTPException(status_code=503, detail="Mode A or Mode B engine not initialized.")

    audio_bytes = await audio.read()
    try:
        samples, leading_trimmed_ms = load_and_normalize_audio(
            audio_bytes=audio_bytes,
            target_sr=settings.sample_rate,
            trim_silence=settings.mode_b_trim_silence,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if samples.size < int(0.15 * settings.sample_rate):
        raise HTTPException(status_code=400, detail="Audio clip is too short (< 150ms).")

    detected_text, stt_conf = await stt_engine.transcribe_async(samples, language=language)
    if not detected_text:
        raise HTTPException(
            status_code=422,
            detail="No clear speech detected in the uploaded audio clip.",
        )

    assessed = await _score_samples_against_text(
        settings=settings,
        acoustic_engine=acoustic_engine,
        g2p=g2p,
        aligner=aligner,
        samples=samples,
        leading_trimmed_ms=leading_trimmed_ms,
        reference_text=detected_text,
        language=language,
    )

    return FreeSpeechAssessResponse(
        detected_transcript=detected_text,
        stt_confidence=stt_conf,
        overall_score=assessed.overall_score,
        metrics=assessed.metrics,
        words=assessed.words,
        connected_speech=assessed.connected_speech,
    )


@router.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket, lang: str = "en") -> None:
    """
    Mode A: WebSocket `/ws/transcribe?lang={code}`
    - Client -> Server: Binary frames (`16kHz, 16-bit, mono PCM` in 100ms-200ms chunks)
                        or JSON control message `{"type": "flush"}` / `{"type": "reset"}`.
    - Server -> Client: JSON `{"type": "interim"|"final", "transcript": "...", "language": "en", "confidence": 0.96}`
    """
    await websocket.accept()
    if not lang or lang.lower() == "auto":
        await websocket.send_json(
            {
                "error": "Explicit `lang` query parameter is required (never use language='auto')."
            }
        )
        await websocket.close(code=1008)
        return

    stt_engine: BaseSTTEngine | None = getattr(websocket.app.state, "stt_engine", None)
    if stt_engine is None:
        await websocket.send_json({"error": "Mode A STT engine is not initialized."})
        await websocket.close(code=1011)
        return

    session = stt_engine.create_session(language=lang)
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            if "bytes" in msg and msg["bytes"] is not None:
                events = await session.ingest_pcm16_bytes(msg["bytes"])
                for ev in events:
                    await websocket.send_json(ev.model_dump())
            elif "text" in msg and msg["text"] is not None:
                try:
                    payload = json.loads(msg["text"])
                except ValueError:
                    payload = {"type": msg["text"].strip().lower()}

                cmd = payload.get("type")
                if cmd == "flush":
                    final_ev = await session.flush()
                    if final_ev is not None:
                        await websocket.send_json(final_ev.model_dump())
                elif cmd == "reset":
                    session.reset()
                elif cmd == "close":
                    final_ev = await session.flush()
                    if final_ev is not None:
                        await websocket.send_json(final_ev.model_dump())
                    await websocket.close(code=1000)
                    break
    except WebSocketDisconnect:
        pass
