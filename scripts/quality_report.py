#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comprehensive audio-quality audit for the speaker_a training data.

Metrics per file: duration, loudness, clipping, silence ratio, low-band energy
(accompaniment proxy), spectral centroid/flatness, SNR estimate. Speaker
consistency via WavLM speaker embeddings. Slice-level stats for the 312 clips.
Outputs JSON + a 4-panel PNG report.
"""
from __future__ import annotations

import glob
import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import librosa  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoFeatureExtractor, WavLMForXVector  # noqa: E402

NEW_DIR = "/mnt/workspace/dataset_speaker_a_v2_raw"
OLD_DIR = "/mnt/workspace/dataset_speaker_a_raw"
SLICE_DIR = "/mnt/workspace/rvc-src/logs/speaker_a_v2/0_gt_wavs"
ASR_LIST = "/mnt/workspace/gptsovits-exp2/asr.list"
OUT_JSON = "/mnt/workspace/training/quality_report.json"
OUT_PNG = "/mnt/workspace/training/quality_report.png"

SR = 16000
FRAME = 2048
HOP = 512


def analyze(path: str) -> dict:
    audio, _ = librosa.load(path, sr=SR, mono=True)
    duration = len(audio) / SR
    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    rms_db = 20 * np.log10(rms + 1e-12)
    peak = float(np.max(np.abs(audio)) + 1e-12)
    peak_db = 20 * np.log10(peak)
    clip_pct = float(np.mean(np.abs(audio) > 0.985) * 100.0)

    frames = []
    for start in range(0, max(1, len(audio) - FRAME), HOP):
        frames.append(float(np.sqrt(np.mean(audio[start:start + FRAME] ** 2) + 1e-12)))
    frames = np.array(frames) if frames else np.array([0.0])
    quiet_threshold = max(1e-4, rms * 0.08)
    silence_ratio = float(np.mean(frames < quiet_threshold))
    noise_floor = float(np.percentile(frames, 10))
    snr_db = 20 * np.log10((rms + 1e-12) / (noise_floor + 1e-12))

    spec_sum, count = None, 0
    for start in range(0, max(1, len(audio) - FRAME), HOP * 8):
        chunk = audio[start:start + FRAME]
        if len(chunk) < FRAME:
            break
        mag = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        spec_sum = mag if spec_sum is None else spec_sum + mag
        count += 1
    freqs = np.fft.rfftfreq(FRAME, d=1.0 / SR)
    mag = spec_sum / max(1, count)
    low_ratio = float(mag[freqs < 120].sum() / (mag.sum() + 1e-9))
    centroid = float((freqs * mag).sum() / (mag.sum() + 1e-9))
    flatness = float(np.exp(np.mean(np.log(mag + 1e-9))) / (np.mean(mag) + 1e-9))

    return {
        "file": os.path.basename(path),
        "duration": round(duration, 2),
        "rms_db": round(rms_db, 1),
        "peak_db": round(peak_db, 1),
        "clip_pct": round(clip_pct, 3),
        "silence_ratio": round(silence_ratio, 3),
        "snr_db": round(snr_db, 1),
        "low_ratio": round(low_ratio, 3),
        "centroid_hz": round(centroid, 0),
        "flatness": round(flatness, 4),
    }


def embed(model, extractor, path: str):
    audio, _ = librosa.load(path, sr=16000, mono=True)
    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return F.normalize(outputs.embeddings, dim=-1).squeeze(0)


def main() -> int:
    new_files = sorted(glob.glob(os.path.join(NEW_DIR, "*.wav")))
    old_files = sorted(glob.glob(os.path.join(OLD_DIR, "*.wav")))
    print("new files:", len(new_files), "| old files:", len(old_files))

    rows = []
    for index, path in enumerate(new_files, start=1):
        rows.append(analyze(path))
        if index % 8 == 0:
            print("analyzed", index, "/", len(new_files))

    print("loading WavLM speaker model")
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()

    embs = {}
    for path in new_files + old_files:
        embs[path] = embed(model, extractor, path)
    reference = F.normalize(torch.stack(list(embs.values())).mean(dim=0), dim=-1)
    for index, path in enumerate(new_files):
        rows[index]["speaker_sim"] = round(float(torch.dot(reference, embs[path])), 3)
    old_sims = [float(torch.dot(reference, embs[path])) for path in old_files]

    slice_rows = []
    for path in sorted(glob.glob(os.path.join(SLICE_DIR, "*.wav"))):
        audio, _ = librosa.load(path, sr=SR, mono=True)
        slice_rows.append({
            "duration": round(len(audio) / SR, 2),
            "rms_db": round(20 * np.log10(float(np.sqrt(np.mean(audio ** 2) + 1e-12)) + 1e-12), 1),
        })

    asr_lengths = []
    if os.path.isfile(ASR_LIST):
        with open(ASR_LIST, encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("|")
                if len(parts) >= 4:
                    asr_lengths.append(len(parts[3]))

    report = {
        "new_files": rows,
        "old_speaker_similarity": {
            "mean": round(float(np.mean(old_sims)), 3),
            "min": round(float(np.min(old_sims)), 3),
        },
        "reference_similarity_self": round(float(np.mean([float(torch.dot(reference, embs[p])) for p in new_files])), 3),
        "slices": {
            "count": len(slice_rows),
            "duration_mean": round(float(np.mean([r["duration"] for r in slice_rows])), 2) if slice_rows else 0,
            "duration_min": round(float(np.min([r["duration"] for r in slice_rows])), 2) if slice_rows else 0,
            "duration_max": round(float(np.max([r["duration"] for r in slice_rows])), 2) if slice_rows else 0,
            "rms_mean": round(float(np.mean([r["rms_db"] for r in slice_rows])), 1) if slice_rows else 0,
        },
        "asr_text_lengths": {
            "count": len(asr_lengths),
            "mean": round(float(np.mean(asr_lengths)), 1) if asr_lengths else 0,
            "min": int(np.min(asr_lengths)) if asr_lengths else 0,
            "max": int(np.max(asr_lengths)) if asr_lengths else 0,
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    # ---------- chart ----------
    durations = [r["duration"] for r in rows]
    rms_vals = [r["rms_db"] for r in rows]
    clips = [r["clip_pct"] for r in rows]
    sims = [r["speaker_sim"] for r in rows]
    low_vals = [r["low_ratio"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0][0]
    ax.bar(range(len(rows)), durations, color="#4C9EEB")
    ax.set_title("New dataset: duration per file (s)")
    ax.set_xlabel("file index")
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.bar(range(len(rows)), rms_vals, color="#F2A93B", label="RMS dBFS")
    ax.plot(range(len(rows)), [-30] * len(rows), "r--", label="-30 dB floor")
    ax.set_title("Loudness per file (RMS dBFS)")
    ax.set_xlabel("file index")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.bar(range(len(rows)), sims, color="#5CB85C")
    ax.axhline(0.90, color="r", linestyle="--", label="0.90 threshold")
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Speaker similarity to speaker_a centroid")
    ax.set_xlabel("file index")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    if slice_rows:
        ax.hist([r["duration"] for r in slice_rows], bins=20, color="#9B59B6")
    ax.set_title("Training slices: duration distribution (312 clips)")
    ax.set_xlabel("seconds")
    ax.grid(alpha=0.3)

    fig.suptitle("Speaker_a data quality audit", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, dpi=120)
    print("saved", OUT_JSON, "and", OUT_PNG)

    print("\n=== per-file ===")
    print("%-4s %-6s %-6s %-6s %-6s %-6s %-6s %s" % ("idx", "dur", "rms", "clip%", "sil", "snr", "sim", "low"))
    for i, r in enumerate(rows):
        print("%-4d %-6.1f %-6.1f %-6.2f %-6.2f %-6.1f %-6.3f %.3f" % (
            i + 1, r["duration"], r["rms_db"], r["clip_pct"], r["silence_ratio"],
            r["snr_db"], r["speaker_sim"], r["low_ratio"]))
    print("\n=== summary ===")
    print("total duration: %.1f min" % (sum(durations) / 60.0))
    print("rms range: %.1f .. %.1f dBFS" % (min(rms_vals), max(rms_vals)))
    print("max clipping: %.3f%%" % max(clips))
    print("speaker sim: mean %.3f | min %.3f" % (float(np.mean(sims)), min(sims)))
    print("low-band ratio: mean %.3f | max %.3f" % (float(np.mean(low_vals)), max(low_vals)))
    print("old-data speaker sim: mean %.3f" % report["old_speaker_similarity"]["mean"])
    print("slices:", report["slices"])
    print("asr text lengths:", report["asr_text_lengths"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
