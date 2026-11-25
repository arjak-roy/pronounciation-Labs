from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.test_phase1_mode_b import synthesize_speech_wav_bytes


def main() -> None:
    samples_dir = Path("samples")
    samples_dir.mkdir(exist_ok=True)

    correct_wav = samples_dir / "correct_sample.wav"
    mispronounced_wav = samples_dir / "mispronounced_sample.wav"

    reference_text = "She can see the sea"
    # 1. Accurate pronunciation of "She can see the sea"
    correct_wav.write_bytes(synthesize_speech_wav_bytes("She can see the sea"))
    # 2. Mispronounced: saying "tea" (/tiː/) instead of "sea" (/siː/)
    mispronounced_wav.write_bytes(synthesize_speech_wav_bytes("She can see the tea"))

    print(f"Generated sample WAVs in: {samples_dir.resolve()}")
    print(f"  1. {correct_wav.name} (spoken: 'She can see the sea')")
    print(f"  2. {mispronounced_wav.name} (spoken: 'She can see the tea' vs reference 'She can see the sea')\n")

    settings = Settings(mode_b_device="cpu", mode_b_process_workers=1)
    app = create_app(settings=settings)

    with TestClient(app) as client:
        print("=== 1. GET /health ===")
        health_resp = client.get("/health")
        print(json.dumps(health_resp.json(), indent=2))

        print(f"\n=== 2. POST /api/v1/pronunciation/assess (Correct: '{reference_text}') ===")
        resp_ok = client.post(
            "/api/v1/pronunciation/assess",
            data={"reference_text": reference_text, "language": "en"},
            files={"audio": (correct_wav.name, correct_wav.read_bytes(), "audio/wav")},
        )
        print(json.dumps(resp_ok.json(), indent=2, ensure_ascii=False))

        print("\n=== 3. POST /api/v1/pronunciation/assess (Mispronounced: said 'tea' instead of 'sea') ===")
        resp_err = client.post(
            "/api/v1/pronunciation/assess",
            data={"reference_text": reference_text, "language": "en"},
            files={"audio": (mispronounced_wav.name, mispronounced_wav.read_bytes(), "audio/wav")},
        )
        print(json.dumps(resp_err.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
