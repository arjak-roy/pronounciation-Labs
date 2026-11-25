from __future__ import annotations

import io
import json
from pathlib import Path

import soundfile as sf
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def main() -> None:
    sample_wav = Path("samples/mispronounced_sample.wav")
    if not sample_wav.exists():
        from tests.test_phase1_mode_b import synthesize_speech_wav_bytes

        sample_wav.parent.mkdir(exist_ok=True)
        sample_wav.write_bytes(synthesize_speech_wav_bytes("She can see the tea"))

    wav_bytes = sample_wav.read_bytes()
    samples_int16, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16")
    raw_pcm16_bytes = samples_int16.tobytes()

    settings = Settings(mode_a_device="cpu", mode_b_device="cpu", mode_b_process_workers=1)
    app = create_app(settings=settings)

    with TestClient(app) as client:
        print("=== 1. Testing Mode A WebSocket (/ws/transcribe?lang=en) ===")
        with client.websocket_connect("/ws/transcribe?lang=en") as ws:
            # Stream in 200ms (3200 samples = 6400 bytes) 16kHz 16-bit mono PCM chunks
            chunk_size = 6400
            for offset in range(0, len(raw_pcm16_bytes), chunk_size):
                ws.send_bytes(raw_pcm16_bytes[offset : offset + chunk_size])

            # Flush utterance at end of stream
            ws.send_text(json.dumps({"type": "close"}))
            while True:
                try:
                    msg = ws.receive_json()
                    print("WebSocket Event:", json.dumps(msg, ensure_ascii=False))
                except Exception:
                    break

        print("\n=== 2. Testing Combined Mode A + Mode B (/api/v1/pronunciation/free-assess) ===")
        resp = client.post(
            "/api/v1/pronunciation/free-assess",
            data={"language": "en"},
            files={"audio": (sample_wav.name, wav_bytes, "audio/wav")},
        )
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
