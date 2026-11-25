from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import onnx
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoConfig, AutoProcessor, Wav2Vec2ForCTC

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("export_onnx")


class DualOutputWav2Vec2Wrapper(nn.Module):
    """
    Mandatory Day-1 Optimization #1 (Single-Pass Tensor Reuse):
    Wraps `Wav2Vec2ForCTC` to output BOTH:
      1. `logits`: [batch, frames, vocab_size] for CTC decoding + forced_align GOP
      2. `hidden_states`: [batch, frames, hidden_dim] from Layer `hidden_layer_index` (Layer 12)
         for DTW accent/prosody similarity.
    """

    def __init__(self, model: Wav2Vec2ForCTC, hidden_layer_index: int = 12) -> None:
        super().__init__()
        self.model = model
        self.hidden_layer_index = hidden_layer_index

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model(
            input_values,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits
        all_hidden = outputs.hidden_states
        layer_idx = min(self.hidden_layer_index, len(all_hidden) - 1)
        hidden_state = all_hidden[layer_idx]
        return logits, hidden_state


def export_tokenizer_vocab(model_id: str, tokenizer_dir: Path) -> dict[str, int]:
    """Download and persist the exact Wav2Vec2 IPA tokenizer vocabulary (`vocab.json`)."""
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading processor/tokenizer for %s ...", model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.save_pretrained(str(tokenizer_dir))

    vocab = processor.tokenizer.get_vocab()
    vocab_file = tokenizer_dir / "vocab.json"
    with vocab_file.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    logger.info("Saved tokenizer vocabulary (%d tokens) to %s", len(vocab), vocab_file)
    return vocab


def convert_onnx_to_fp16(fp32_onnx_path: Path, fp16_onnx_path: Path) -> None:
    """
    Convert exported FP32 ONNX graph weights to FP16 for GPU execution (`MODE_B_DEVICE=gpu`)
    using onnxruntime's built-in float16 converter while keeping input/output interfaces in float32.
    """
    try:
        from onnxruntime.transformers.float16 import convert_float_to_float16

        model_fp32 = onnx.load(str(fp32_onnx_path), load_external_data=True)
        model_fp16 = convert_float_to_float16(
            model_fp32,
            keep_io_types=True,
        )
        onnx.save(model_fp16, str(fp16_onnx_path))
        logger.info(
            "Saved GPU FP16 ONNX model to %s (%.1f MB)",
            fp16_onnx_path,
            fp16_onnx_path.stat().st_size / (1024 * 1024),
        )
    except Exception as exc:
        logger.warning(
            "FP16 conversion fallback (%s); copying FP32 graph to %s for GPU execution.",
            exc,
            fp16_onnx_path,
        )
        shutil.copyfile(fp32_onnx_path, fp16_onnx_path)


def export_wav2vec2_dual_onnx(
    model_id: str = "facebook/wav2vec2-lv-60-espeak-cv-ft",
    output_dir: Path = Path("models"),
    tokenizer_dir: Path = Path("models/tokenizer"),
    cpu_filename: str = "wav2vec2_int8.onnx",
    gpu_filename: str = "wav2vec2_fp16.onnx",
    hidden_layer_index: int = 12,
    opset_version: int = 17,
    export_gpu_fp16: bool = True,
) -> tuple[Path, Path]:
    """
    Download `Wav2Vec2ForCTC` IPA model, export Dual-Output (`logits` + `hidden_states`) ONNX graph,
    and produce:
      1. INT8 dynamically quantized ONNX model (`wav2vec2_int8.onnx`) for CPU profile
      2. FP16 ONNX model (`wav2vec2_fp16.onnx`) for GPU profile
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    export_tokenizer_vocab(model_id, tokenizer_dir)

    logger.info("Loading acoustic phoneme model %s ...", model_id)
    config = AutoConfig.from_pretrained(model_id)
    config.output_hidden_states = True
    base_model = Wav2Vec2ForCTC.from_pretrained(model_id, config=config)
    base_model.eval()

    wrapper = DualOutputWav2Vec2Wrapper(base_model, hidden_layer_index=hidden_layer_index)
    wrapper.eval()

    fp32_onnx_path = output_dir / "wav2vec2_fp32.onnx"
    cpu_int8_path = output_dir / cpu_filename
    gpu_fp16_path = output_dir / gpu_filename

    # 2 seconds of dummy 16kHz mono float32 audio
    dummy_input = torch.randn(1, 32000, dtype=torch.float32)

    logger.info("Exporting Dual-Output FP32 ONNX graph to %s ...", fp32_onnx_path)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            str(fp32_onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_values"],
            output_names=["logits", "hidden_states"],
            dynamic_axes={
                "input_values": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "frame_length"},
                "hidden_states": {0: "batch_size", 1: "frame_length"},
            },
            dynamo=False,
        )

    logger.info("Running INT8 dynamic quantization for CPU profile -> %s ...", cpu_int8_path)
    quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(cpu_int8_path),
        weight_type=QuantType.QInt8,
    )
    logger.info(
        "CPU INT8 ONNX export complete (%.1f MB)",
        cpu_int8_path.stat().st_size / (1024 * 1024),
    )

    if export_gpu_fp16:
        logger.info("Generating GPU FP16 ONNX model -> %s ...", gpu_fp16_path)
        convert_onnx_to_fp16(fp32_onnx_path, gpu_fp16_path)

    # Remove intermediate FP32 graph to save disk space
    if fp32_onnx_path.exists():
        fp32_onnx_path.unlink()
    ext_data = Path(str(fp32_onnx_path) + ".data")
    if ext_data.exists():
        ext_data.unlink()

    return cpu_int8_path, gpu_fp16_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Wav2Vec2-espeak IPA model to Dual-Output INT8 (CPU) and FP16 (GPU) ONNX."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="facebook/wav2vec2-lv-60-espeak-cv-ft",
        help="HuggingFace model ID (e.g., facebook/wav2vec2-xlsr-53-espeak-cv-ft or facebook/wav2vec2-lv-60-espeak-cv-ft)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("models/tokenizer"))
    parser.add_argument("--hidden-layer", type=int, default=12)
    args = parser.parse_args()

    export_wav2vec2_dual_onnx(
        model_id=args.model_id,
        output_dir=args.output_dir,
        tokenizer_dir=args.tokenizer_dir,
        hidden_layer_index=args.hidden_layer,
    )


if __name__ == "__main__":
    main()
