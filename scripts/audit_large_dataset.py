#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast quality audit for a large folder of video/audio files.

Dedupes by SHA1, decodes audio to a temp mono 16k wav, measures duration and
quality metrics, then writes JSON and prints a summary.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
EXTS = {".mp4", ".mkv", ".webm", ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}
SR = 16000
FRAME = 2048
HOP = 512


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(path: Path, target: Path) -> bool:
    proc = subprocess.run(
        [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
         "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", str(target)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return proc.returncode == 0 and target.exists()


def read_wav(path: Path):
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, rate


def analyze(audio: np.ndarray, rate: int) -> dict:
    duration = len(audio) / rate
    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    rms_db = 20 * np.log10(rms + 1e-12)
    peak = float(np.max(np.abs(audio)) + 1e-12)
    peak_db = 20 * np.log10(peak)
    clip_pct = float(np.mean(np.abs(audio) > 0.985) * 100.0)

    frame_rms = []
    for start in range(0, max(1, len(audio) - FRAME), HOP):
        frame_rms.append(float(np.sqrt(np.mean(audio[start:start + FRAME] ** 2) + 1e-12)))
    frame_rms = np.array(frame_rms) if frame_rms else np.array([0.0])
    silence = float(np.mean(frame_rms < max(1e-4, rms * 0.08)))
    noise = float(np.percentile(frame_rms, 10))
    snr = 20 * np.log10((rms + 1e-12) / (noise + 1e-12))

    spec_sum, count = None, 0
    for start in range(0, max(1, len(audio) - FRAME), HOP * 8):
        chunk = audio[start:start + FRAME]
        if len(chunk) < FRAME:
            break
        mag = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        spec_sum = mag if spec_sum is None else spec_sum + mag
        count += 1
    freqs = np.fft.rfftfreq(FRAME, d=1.0 / rate)
    mag = spec_sum / max(1, count)
    low = float(mag[freqs < 120].sum() / (mag.sum() + 1e-9))
    centroid = float((freqs * mag).sum() / (mag.sum() + 1e-9))
    flatness = float(np.exp(np.mean(np.log(mag + 1e-9))) / (np.mean(mag) + 1e-9))
    return {
        "duration": round(duration, 2),
        "rms_db": round(rms_db, 1),
        "peak_db": round(peak_db, 1),
        "clip_pct": round(clip_pct, 3),
        "silence_ratio": round(silence, 3),
        "snr_db": round(snr, 1),
        "low_ratio": round(low, 3),
        "centroid_hz": round(centroid, 0),
        "flatness": round(flatness, 4),
    }


def main() -> int:
    source = Path(sys.argv[1])
    out_json = Path(sys.argv[2])

    media = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTS]
    print("media files found:", len(media), flush=True)

    seen = {}
    unique = []
    for index, path in enumerate(media, start=1):
        digest = sha1(path)
        if digest in seen:
            print("DUP", path.name[:50], "==", seen[digest][:50], flush=True)
            continue
        seen[digest] = path.name
        unique.append(path)
    print("unique by content:", len(unique), flush=True)

    rows = []
    by_name = {}
    with tempfile.TemporaryDirectory() as tmp:
        for index, path in enumerate(unique, start=1):
            wav = Path(tmp) / "a.wav"
            if not decode(path, wav):
                print("DECODE_FAIL", path.name[:60], flush=True)
                continue
            audio, rate = read_wav(wav)
            if len(audio) < rate * 1.0:
                print("TOO_SHORT", path.name[:60], flush=True)
                continue
            row = analyze(audio, rate)
            row["file"] = path.name
            row["size_mb"] = round(path.stat().st_size / 1048576, 2)
            rows.append(row)
            by_name[path.name] = row
            if index % 10 == 0:
                print("progress %d/%d" % (index, len(unique)), flush=True)

    payload = {
        "source": str(source),
        "media_files": len(media),
        "unique_files": len(unique),
        "usable_files": len(rows),
        "total_duration": round(sum(r["duration"] for r in rows), 1),
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    durations = sorted(r["duration"] for r in rows)
    rms_vals = [r["rms_db"] for r in rows]
    print("\n=== summary ===")
    print("usable files: %d | total %.1f min (%.1f h)" % (
        len(rows), sum(durations) / 60.0, sum(durations) / 3600.0))
    if durations:
        print("duration: min %.1fs | median %.1fs | max %.1fs" % (
            durations[0], durations[len(durations) // 2], durations[-1]))
        print("rms: mean %.1f | min %.1f | max %.1f dBFS" % (
            float(np.mean(rms_vals)), min(rms_vals), max(rms_vals)))
        print("clipping>0.1%%: %d files | silence>30%%: %d files" % (
            sum(1 for r in rows if r["clip_pct"] > 0.1),
            sum(1 for r in rows if r["silence_ratio"] > 0.3)))
        print("low_ratio>0.10: %d files | centroid mean %.0f Hz" % (
            sum(1 for r in rows if r["low_ratio"] > 0.10),
            float(np.mean([r["centroid_hz"] for r in rows]))))
        for flag, predicate in (
            ("LOW SNR (<15dB)", lambda r: r["snr_db"] < 15),
            ("VERY QUIET (<-25dBFS)", lambda r: r["rms_db"] < -25),
            ("CLIPPING (>0.5%)", lambda r: r["clip_pct"] > 0.5),
            ("ACCOMPANIMENT RISK (low>0.12)", lambda r: r["low_ratio"] > 0.12),
        ):
            bad = [r for r in rows if predicate(r)]
            print("%s: %d" % (flag, len(bad)))
            for r in bad[:6]:
                print("   ", r["file"][:60], "| dur %.0fs rms %.1f clip %.2f%% snr %.0f low %.3f" % (
                    r["duration"], r["rms_db"], r["clip_pct"], r["snr_db"], r["low_ratio"]))
    print("json:", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
