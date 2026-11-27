from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Unified configuration for Mode A (Live STT) and Mode B (Pronunciation Engine)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Mode A: Real-Time Live STT (NVIDIA Nemotron 3.5 ASR Streaming 0.6B INT8 by default) ---
    mode_a_enabled: bool = Field(
        default=True, validation_alias="MODE_A_ENABLED"
    )
    mode_a_device: Literal["cpu", "gpu"] = Field(
        default="cpu", validation_alias="MODE_A_DEVICE"
    )
    mode_a_engine: Literal["nemotron", "sherpa", "vosk", "qwen", "whisper"] = Field(
        default="nemotron", validation_alias="MODE_A_ENGINE"
    )
    mode_a_nemotron_repo: str = Field(
        default="csukuangfj2/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-560ms-int8-2026-06-11",
        validation_alias="MODE_A_NEMOTRON_REPO",
    )
    mode_a_nemotron_dir: Path = Field(
        default=Path("models/nemotron-3.5-0.6b-int8"),
        validation_alias="MODE_A_NEMOTRON_DIR",
    )
    mode_a_sherpa_dir: Path = Field(
        default=Path("models/sherpa-en"),
        validation_alias="MODE_A_SHERPA_DIR",
    )
    mode_a_cpu_model: str = Field(
        default="Systran/faster-whisper-small",
        validation_alias="MODE_A_CPU_MODEL",
    )
    mode_a_gpu_model: str = Field(
        default="mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        validation_alias="MODE_A_GPU_MODEL",
    )
    mode_a_cpu_threads: int = Field(
        default=2, validation_alias="MODE_A_CPU_THREADS"
    )

    # --- Mode B: On-Demand Pronunciation Engine ---
    mode_b_device: Literal["cpu", "gpu"] = Field(
        default="cpu", validation_alias="MODE_B_DEVICE"
    )
    mode_b_model: str = Field(
        default="facebook/wav2vec2-lv-60-espeak-cv-ft",
        validation_alias="MODE_B_MODEL",
        description="HuggingFace model ID (e.g. facebook/wav2vec2-xlsr-53-espeak-cv-ft or facebook/wav2vec2-lv-60-espeak-cv-ft)",
    )

    # --- Mode B ONNX & Tokenizer Artifact Paths ---
    mode_b_models_dir: Path = Field(
        default=Path("models"), validation_alias="MODE_B_MODELS_DIR"
    )
    mode_b_onnx_cpu_filename: str = Field(
        default="wav2vec2_int8.onnx", validation_alias="MODE_B_ONNX_CPU_FILENAME"
    )
    mode_b_onnx_gpu_filename: str = Field(
        default="wav2vec2_fp16.onnx", validation_alias="MODE_B_ONNX_GPU_FILENAME"
    )
    mode_b_tokenizer_dir: Path = Field(
        default=Path("models/tokenizer"), validation_alias="MODE_B_TOKENIZER_DIR"
    )
    mode_b_hidden_layer_index: int = Field(
        default=12, validation_alias="MODE_B_HIDDEN_LAYER_INDEX"
    )

    # --- CPU Thread Pinning & Process Pool ---
    mode_b_intra_op_threads: int = Field(
        default=2, validation_alias="MODE_B_INTRA_OP_THREADS"
    )
    mode_b_inter_op_threads: int = Field(
        default=1, validation_alias="MODE_B_INTER_OP_THREADS"
    )
    mode_b_process_workers: int = Field(
        default=0, validation_alias="MODE_B_PROCESS_WORKERS"
    )

    # --- GOP Calibration & Conditional Diagnostics ---
    mode_b_gop_pass_threshold: float = Field(
        default=85.0,
        validation_alias="MODE_B_GOP_PASS_THRESHOLD",
        description="Words scoring above this threshold skip Needleman-Wunsch & panphon diagnostics.",
    )
    mode_b_gop_warn_threshold: float = Field(
        default=60.0,
        validation_alias="MODE_B_GOP_WARN_THRESHOLD",
    )

    # --- Audio Standards & VAD Chunking ---
    sample_rate: int = Field(default=16000, validation_alias="SAMPLE_RATE")
    mode_b_max_chunk_seconds: float = Field(
        default=5.0,
        validation_alias="MODE_B_MAX_CHUNK_SECONDS",
        description="Split pronunciation clips longer than this at natural breath pauses.",
    )
    mode_b_trim_silence: bool = Field(
        default=True, validation_alias="MODE_B_TRIM_SILENCE"
    )

    # --- SQLite Word & Phrase Library Database ---
    database_path: Path = Field(
        default=Path("data/pronunciation_lab.db"),
        validation_alias="DATABASE_PATH",
    )

    # --- Optional Custom espeak-ng Library Override ---
    espeak_library_path: str | None = Field(
        default=None, validation_alias="ESPEAK_LIBRARY_PATH"
    )
    espeak_data_path: str | None = Field(
        default=None, validation_alias="ESPEAK_DATA_PATH"
    )

    @property
    def onnx_cpu_path(self) -> Path:
        return self.mode_b_models_dir / self.mode_b_onnx_cpu_filename

    @property
    def onnx_gpu_path(self) -> Path:
        return self.mode_b_models_dir / self.mode_b_onnx_gpu_filename


def get_settings() -> Settings:
    return Settings()
