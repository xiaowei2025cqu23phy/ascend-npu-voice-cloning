#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dedupe a folder of media files and extract 48 kHz mono WAVs for training."""
from __future__ import annotations

import hashlib
import subprocess
import sys
import wave
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
EXTS = {".mp4", ".mkv", ".webm", ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac"}


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception:  # noqa: BLE001
        return 0.0


def main() -> int:
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)

    seen = {}
    unique = []
    for path in sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTS):
        digest = sha1(path)
        if digest in seen:
            print("DUP ", path.name, "==", seen[digest])
            continue
        seen[digest] = path.name
        unique.append(path)

    print("unique media files:", len(unique))
    total = 0.0
    ok = 0
    for index, path in enumerate(unique, start=1):
        out = target / ("%03d.wav" % index)
        proc = subprocess.run(
            [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-ac", "1", "-ar", "48000", "-sample_fmt", "s16", str(out)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode == 0 and out.exists():
            duration = wav_duration(out)
            total += duration
            ok += 1
            print("OK  %02d %6.1fs  %s" % (index, duration, path.name[:58]))
        else:
            print("FAIL", path.name, proc.stderr.decode("utf-8", "replace")[-120:])
    print("\nextracted %d files | total %.1f s (%.1f min) -> %s" % (ok, total, total / 60.0, target))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
