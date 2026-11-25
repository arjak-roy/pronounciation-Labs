from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.acoustic_model import create_acoustic_engine
from app.aligner_gop import ForcedAlignerGOP
from app.config import Settings, get_settings
from app.g2p import G2PService
from app.router import router
from app.stt_engines import create_stt_engine

logger = logging.getLogger("voice_pronunciation_engine")
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_CHARACTERS_DIR = _STATIC_DIR / "characters"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Coding Guideline #3:
    Load all models once at FastAPI `lifespan` startup; run CPU inference inside a
    `ProcessPoolExecutor` with pinned `intra_op_num_threads`.
    """
    settings: Settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings

    logger.info(
        "Starting Voice & Pronunciation Engine (MODE_B_DEVICE=%s, MODE_B_MODEL=%s)",
        settings.mode_b_device,
        settings.mode_b_model,
    )

    g2p = G2PService.from_tokenizer_dir(
        tokenizer_dir=settings.mode_b_tokenizer_dir,
        espeak_library_path=settings.espeak_library_path,
        espeak_data_path=settings.espeak_data_path,
    )
    acoustic_engine = create_acoustic_engine(settings)
    aligner = ForcedAlignerGOP(
        id_to_token=g2p.id_to_token,
        blank_id=g2p.blank_id,
        gop_pass_threshold=settings.mode_b_gop_pass_threshold,
        gop_warn_threshold=settings.mode_b_gop_warn_threshold,
    )

    app.state.g2p = g2p
    app.state.acoustic_engine = acoustic_engine
    app.state.aligner = aligner
    app.state.stt_engine = create_stt_engine(settings) if settings.mode_a_enabled else None

    logger.info(
        "Mode B Engine ready: provider=%s (vocab_size=%d) | Mode A enabled=%s",
        acoustic_engine.active_provider,
        len(g2p.vocab),
        settings.mode_a_enabled,
    )

    try:
        yield
    finally:
        acoustic_engine.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="100% Open-Source Voice & Pronunciation Engine",
        version="0.1.0",
        description=(
            "Mode A (Real-Time Live STT) & Mode B (On-Demand Acoustic Pronunciation Assessment) "
            "with CPU (ONNX INT8) and GPU (ONNX FP16 / CUDA) execution profiles."
        ),
        lifespan=lifespan,
    )
    if settings is not None:
        app.state.settings = settings
    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    if _CHARACTERS_DIR.exists():
        app.mount("/characters", StaticFiles(directory=str(_CHARACTERS_DIR)), name="characters")
    app.include_router(router)
    return app


app = create_app()
