#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract the quality-filtered speaker_a files to 32 kHz mono WAVs for upload."""
from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
LIST = Path(r"<PROJECT_ROOT>\training\speaker_a_clean_files.txt")
SOURCE = Path(r"<DATA_DIR>\speaker_a")
TARGET = Path(r"<PROJECT_ROOT>\local-wavs\speaker_a_clean")
SR = 32000


def duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception:  # noqa: BLE001
        return 0.0


def main() -> int:
    names = [line.strip() for line in LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    TARGET.mkdir(parents=True, exist_ok=True)
    total = 0.0
    ok = 0
    for index, name in enumerate(names, start=1):
        source = SOURCE / name
        if not source.is_file():
            print("MISSING", name[:70])
            continue
        out = TARGET / ("%03d.wav" % index)
        proc = subprocess.run(
            [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
             "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", str(out)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if proc.returncode != 0 or not out.exists():
            print("FAIL", name[:70])
            continue
        seconds = duration(out)
        total += seconds
        ok += 1
        if index % 10 == 0:
            print("progress %d/%d | %.1f min" % (index, len(names), total / 60.0), flush=True)
    print("\nextracted %d files | total %.1f s (%.1f min) -> %s" % (ok, total, total / 60.0, TARGET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
