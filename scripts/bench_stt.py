"""
Real-time Mode A STT benchmark.

Streams WAV files into the live STT session at wall-clock pace (like a browser mic would) and
reports what "smooth" means in numbers:

  rtf            compute time / audio time (must stay well below 1.0, per concurrent stream)
  first_partial  ms from speech onset to the first interim transcript
  max_lag        worst ms the stream fell behind real time (>0 = audio backing up)
  final_latency  ms from end of audio to the final transcript
  wer            word error rate, when --ref is given

Usage (run it ON the target box, e.g. the t2.medium):
  python -m scripts.bench_stt test_de.wav --lang de --ref "Alles hat ein Ende ..."
  python -m scripts.bench_stt test_de.wav --lang de --streams 2      # concurrency check
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

import numpy as np

from app.audio_utils import load_and_normalize_audio
from app.config import Settings
from app.stt_engines import BaseSTTEngine, create_stt_engine

SR = 16000


def wer(ref: str, hyp: str) -> float:
    norm = lambda t: re.sub(r"[^\w\s]", "", t.lower()).split()  # noqa: E731
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(len(r), 1)


def speech_onset_sec(samples: np.ndarray, frame: int = 320) -> float:
    """First 20ms frame clearly above the clip's noise floor."""
    n = len(samples) // frame
    if n == 0:
        return 0.0
    rms = np.sqrt(np.mean(samples[: n * frame].reshape(n, frame) ** 2, axis=1))
    thresh = max(0.01, float(np.percentile(rms, 10)) * 4)
    hits = np.nonzero(rms > thresh)[0]
    return float(hits[0] * frame / SR) if hits.size else 0.0


async def run_stream(engine: BaseSTTEngine, samples: np.ndarray, lang: str, chunk_ms: int) -> dict:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    step = int(SR * chunk_ms / 1000) * 2
    session = engine.create_session(language=lang)

    finals: list[str] = []
    first_partial = None
    compute = 0.0
    max_lag = 0.0
    start = time.perf_counter()
    for i, off in enumerate(range(0, len(pcm), step)):
        due = start + i * chunk_ms / 1000
        now = time.perf_counter()
        if now < due:
            await asyncio.sleep(due - now)
        max_lag = max(max_lag, time.perf_counter() - due)
        t0 = time.perf_counter()
        events = await session.ingest_pcm16_bytes(pcm[off : off + step])
        compute += time.perf_counter() - t0
        for ev in events:
            if ev.type == "interim" and first_partial is None:
                first_partial = time.perf_counter() - start
            elif ev.type == "final":
                finals.append(ev.transcript)

    audio_end = time.perf_counter()
    t0 = time.perf_counter()
    final_ev = await session.flush()
    compute += time.perf_counter() - t0
    if final_ev is not None:
        finals.append(final_ev.transcript)

    dur = len(samples) / SR
    onset = speech_onset_sec(samples)
    return {
        "transcript": " ".join(finals),
        "rtf": compute / dur,
        "first_partial_ms": None if first_partial is None else (first_partial - onset) * 1000,
        "max_lag_ms": max_lag * 1000,
        "final_latency_ms": (time.perf_counter() - audio_end) * 1000,
    }


def fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f}"


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wavs", nargs="+", type=Path)
    ap.add_argument("--lang", default="de")
    ap.add_argument("--ref", default=None, help="Reference text (single-file runs) for WER")
    ap.add_argument("--chunk-ms", type=int, default=160, help="Client frame size")
    ap.add_argument("--streams", type=int, default=1, help="Concurrent sessions per file")
    ap.add_argument("--threads", type=int, default=None, help="Override MODE_A_CPU_THREADS")
    args = ap.parse_args()

    overrides = {"mode_a_cpu_threads": args.threads} if args.threads else {}
    settings = Settings(**overrides)
    t0 = time.perf_counter()
    engine = create_stt_engine(settings)
    print(f"engine: {engine.engine_name}  (load {time.perf_counter() - t0:.1f}s, "
          f"threads={getattr(engine, 'cpu_threads', '?')})")

    for wav in args.wavs:
        samples, _ = load_and_normalize_audio(wav.read_bytes(), target_sr=SR, trim_silence=False)
        results = await asyncio.gather(
            *(run_stream(engine, samples, args.lang, args.chunk_ms) for _ in range(args.streams))
        )
        print(f"\n{wav.name}  ({len(samples) / SR:.1f}s audio, {args.streams} stream(s), "
              f"{args.chunk_ms}ms frames)")
        for n, r in enumerate(results, 1):
            line = (f"  [{n}] rtf={r['rtf']:.2f}  first_partial={fmt(r['first_partial_ms'])}ms  "
                    f"max_lag={fmt(r['max_lag_ms'])}ms  final_latency={fmt(r['final_latency_ms'])}ms")
            if args.ref:
                line += f"  wer={wer(args.ref, r['transcript']):.1%}"
            print(line)
        print(f"  transcript: {results[0]['transcript']!r}")

    try:
        import resource

        print(f"\npeak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
