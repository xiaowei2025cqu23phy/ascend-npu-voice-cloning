#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit a folder of local audio/video files for RVC training suitability.

Reports per file: duration, RMS, silence ratio, low-band energy ratio (a proxy
for accompaniment/bass) and stereo correlation (music is usually wide, a solo
voice is close to mono).
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
EXTS = {".mp4", ".mkv", ".webm", ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", *args],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def extract(path: Path, target: Path, channels: int) -> None:
    proc = run_ffmpeg(["-y", "-i", str(path), "-ac", str(channels), "-ar", "16000",
                       "-sample_fmt", "s16", str(target)])
    if proc.returncode != 0 or not target.exists():
        raise RuntimeError(proc.stderr.decode("utf-8", "replace")[-300:])


def read_wav(path: Path):
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels)
    return data, rate


def analyze(path: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        mono_path = Path(tmp) / "mono.wav"
        stereo_path = Path(tmp) / "stereo.wav"
        extract(path, mono_path, 1)
        extract(path, stereo_path, 2)
        mono, rate = read_wav(mono_path)
        stereo, _ = read_wav(stereo_path)
    mono = mono.flatten()
    duration = len(mono) / rate
    if duration <= 0:
        raise ValueError("empty audio")

    frame = 2048
    hop = 1024
    rms_values = []
    for start in range(0, max(1, len(mono) - frame), hop):
        chunk = mono[start:start + frame]
        rms_values.append(float(np.sqrt(np.mean(chunk ** 2) + 1e-12)))
    rms_values = np.array(rms_values)
    overall_rms = float(np.sqrt(np.mean(mono ** 2) + 1e-12))
    threshold = max(1e-4, overall_rms * 0.1)
    silence_ratio = float(np.mean(rms_values < threshold)) if len(rms_values) else 1.0

    # average spectrum
    spec_sum = None
    count = 0
    for start in range(0, max(1, len(mono) - frame), hop * 4):
        chunk = mono[start:start + frame]
        if len(chunk) < frame:
            break
        window = np.hanning(len(chunk))
        mag = np.abs(np.fft.rfft(chunk * window))
        spec_sum = mag if spec_sum is None else spec_sum + mag
        count += 1
    freqs = np.fft.rfftfreq(frame, d=1.0 / rate)
    mean_mag = spec_sum / max(1, count)
    low_ratio = float(mean_mag[freqs < 120].sum() / (mean_mag.sum() + 1e-9))
    centroid = float((freqs * mean_mag).sum() / (mean_mag.sum() + 1e-9))

    # stereo correlation (music is wide, voice is centered)
    if stereo.ndim == 2 and stereo.shape[1] == 2:
        left, right = stereo[:, 0], stereo[:, 1]
        denom = (np.std(left) * np.std(right)) + 1e-12
        correlation = float(np.mean((left - left.mean()) * (right - right.mean())) / denom)
        side_ratio = float(np.sqrt(np.mean(((left - right) / 2.0) ** 2) + 1e-12) /
                           (np.sqrt(np.mean(((left + right) / 2.0) ** 2) + 1e-12) + 1e-12))
    else:
        correlation = 1.0
        side_ratio = 0.0

    return {
        "duration": round(duration, 2),
        "rms_db": round(20 * math.log10(overall_rms + 1e-12), 1),
        "silence_ratio": round(silence_ratio, 3),
        "low_ratio": round(low_ratio, 3),
        "centroid_hz": round(centroid, 0),
        "correlation": round(correlation, 3),
        "side_ratio": round(side_ratio, 3),
    }


def main() -> int:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    files = sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    )
    if not files:
        print("no media files found in", folder)
        return 2
    print("auditing %d files in %s" % (len(files), folder))
    print()
    header = "%-6s %-7s %-8s %-6s %-8s %-7s %-6s %s" % (
        "dur", "rmsdB", "silence", "low", "centroid", "corr", "side", "file")
    print(header)
    print("-" * len(header))
    results = []
    for index, path in enumerate(files, start=1):
        try:
            stats = analyze(path)
        except Exception as error:  # noqa: BLE001
            print("FAIL  %s : %s" % (path.name[:60], str(error)[:120]))
            continue
        results.append({"file": str(path), **stats})
        print("%-6.1f %-7.1f %-8.2f %-6.2f %-8.0f %-7.2f %-6.2f %s" % (
            stats["duration"], stats["rms_db"], stats["silence_ratio"],
            stats["low_ratio"], stats["centroid_hz"], stats["correlation"],
            stats["side_ratio"], path.name[:52]))
    if results:
        total = sum(item["duration"] for item in results)
        print()
        print("files: %d | total duration: %.1f s (%.1f min)" % (len(results), total, total / 60.0))
        print("mean low_ratio (bass/accompaniment proxy): %.3f" % (
            sum(item["low_ratio"] for item in results) / len(results)))
        print("mean side_ratio (music width proxy): %.3f" % (
            sum(item["side_ratio"] for item in results) / len(results)))
        report_dir = Path(__file__).resolve().parent
        (report_dir / "local_dataset_report.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("report: %s" % (report_dir / "local_dataset_report.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
