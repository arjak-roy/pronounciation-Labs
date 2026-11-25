# 100% Open-Source Voice & Pronunciation Engine

Self-hosted voice platform supporting **Mode A (Real-Time Live STT)** and **Mode B (On-Demand Pronunciation Assessment)** with **CPU (`ONNX INT8`)** and **GPU (`ONNX FP16 / CUDA`)** execution profiles behind unified API contracts.

---

## Architecture Highlights (Phase 1 — Mode B)

- **Strict Acoustic Phoneme Recognition:** Uses `Wav2Vec2` fine-tuned on IPA (`facebook/wav2vec2-lv-60-espeak-cv-ft` or `facebook/wav2vec2-xlsr-53-espeak-cv-ft`) + `torchaudio.functional.forced_align` (never standard LM-backed STT models that auto-correct mispronunciations).
- **Single-Pass Tensor Reuse (Optimization #1):** `scripts/export_onnx.py` wraps `Wav2Vec2ForCTC` to output both `logits` (`[frames, vocab_size]`) and Layer 12 `hidden_states` (`[frames, hidden_dim]`) in a single forward pass.
- **Conditional Error Diagnostics (Optimization #3):** Word-level Goodness of Pronunciation (GOP) scores are computed first. Words with `score > 85` are marked `correct` and skip Needleman-Wunsch alignment and `panphon` 24-feature articulatory lookups.
- **Breath-Pause Chunking (Optimization #4):** Clips longer than `5.0s` are automatically split at natural low-energy breath pauses to prevent $O(N^2)$ self-attention slowdowns.
- **Strict IPA Vocabulary Mapping:** `app/g2p.py` sanitizes `espeak-ng` output (`ˈ`, `ˌ`, `ː`, tie bars) and maps every phoneme strictly to the exported `Wav2Vec2` `vocab.json` so `<unk>` tokens never trigger index errors in `forced_align`.
- **Pinned CPU Process Pool:** CPU inference runs inside a `ProcessPoolExecutor` with `sess_options.intra_op_num_threads` pinned to prevent thread thrashing.

---

## Quick Start (Local CPU / GPU)

### 1. Install Dependencies
```powershell
cd C:\Projects\voice-pronunciation-engine
uv venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### 2. Export Dual-Output ONNX Models (INT8 CPU + FP16 GPU)
```powershell
# Fast English IPA model (~95 MB INT8 ONNX):
.\.venv\Scripts\python.exe -m scripts.export_onnx --model-id facebook/wav2vec2-lv-60-espeak-cv-ft

# Or Multilingual XLSR-53 IPA model (~300 MB INT8 ONNX):
.\.venv\Scripts\python.exe -m scripts.export_onnx --model-id facebook/wav2vec2-xlsr-53-espeak-cv-ft
```

### 3. Start the FastAPI Server
```powershell
# CPU Profile (Default)
$env:MODE_B_DEVICE="cpu"
.\.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000

# GPU Profile
$env:MODE_B_DEVICE="gpu"
.\.venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000
```

### 4. Run Tests
```powershell
.\.venv\Scripts\pytest.exe -v
```
