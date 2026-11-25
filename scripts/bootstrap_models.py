from __future__ import annotations

import logging
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from app.config import get_settings
from scripts.export_onnx import export_wav2vec2_dual_onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bootstrap_models")

SHERPA_REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
SHERPA_FILES = (
    "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
    "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
    "tokens.txt",
)


def ensure_sherpa_model(sherpa_dir: Path) -> None:
    sherpa_dir.mkdir(parents=True, exist_ok=True)
    for fn in SHERPA_FILES:
        dest = sherpa_dir / fn
        if not dest.exists():
            logger.info("Downloading Mode A Sherpa-ONNX artifact: %s ...", fn)
            cached = hf_hub_download(repo_id=SHERPA_REPO, filename=fn)
            shutil.copyfile(cached, dest)
    logger.info("Mode A Sherpa-ONNX model ready at %s", sherpa_dir)


def ensure_models() -> None:
    settings = get_settings()
    if settings.mode_a_enabled:
        ensure_sherpa_model(settings.mode_a_sherpa_dir)

    vocab_file = settings.mode_b_tokenizer_dir / "vocab.json"
    if settings.onnx_cpu_path.exists() and vocab_file.exists():
        logger.info("Mode B ONNX model already present at %s", settings.onnx_cpu_path)
        return

    logger.info("Exporting Mode B ONNX model (%s)...", settings.mode_b_model)
    export_wav2vec2_dual_onnx(
        model_id=settings.mode_b_model,
        output_dir=settings.mode_b_models_dir,
        tokenizer_dir=settings.mode_b_tokenizer_dir,
        cpu_filename=settings.mode_b_onnx_cpu_filename,
        gpu_filename=settings.mode_b_onnx_gpu_filename,
        hidden_layer_index=settings.mode_b_hidden_layer_index,
        export_gpu_fp16=(settings.mode_b_device == "gpu"),
    )


if __name__ == "__main__":
    ensure_models()
