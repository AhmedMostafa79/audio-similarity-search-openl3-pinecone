"""Generate a few tiny synthetic WAV clips for the public demo.
Usage:
    python scripts/generate_sample_audio.py --out data/sample_audio --n 3
"""
import argparse
import os
import numpy as np
import soundfile as sf


def synth_wave(kind: str, sr: int, seconds: float):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    if kind == "sine":
        return 0.3 * np.sin(2 * np.pi * 440 * t)
    if kind == "chord":
        freqs = [220, 330, 440]
        return 0.25 * sum(np.sin(2 * np.pi * f * t) for f in freqs) / len(freqs)
    if kind == "noise":
        return 0.15 * np.random.randn(len(t))
    return 0.1 * np.sin(2 * np.pi * 550 * t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/sample_audio")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    kinds = ["sine", "chord", "noise"]
    for i in range(args.n):
        kind = kinds[i % len(kinds)]
        y = synth_wave(kind, args.sr, args.seconds)
        fname = f"sample_{kind}_{i}.wav"
        path = os.path.join(args.out, fname)
        sf.write(path, y, args.sr)
        print("Wrote", path)


if __name__ == "__main__":
    main()
