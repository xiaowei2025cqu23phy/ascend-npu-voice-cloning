#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run ASR in quality mode (beam 5 + VAD) and keep confidence for filtering."""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time

MODEL_DIR = "/mnt/workspace/pretrained_models/faster-whisper"
OUT_JSONL = "/mnt/workspace/gptsovits-exp3/asr_quality.jsonl"
OUT_LIST = "/mnt/workspace/gptsovits-exp3/asr_quality.list"

MIN_LOGPROB = -1.0
MIN_CHARS = 4
MAX_CHARS = 80


def clean_text(text: str) -> str:
    return text.replace("|", " ").replace("\n", " ").strip()


def suspicious(text: str) -> bool:
    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return True
    if re.search(r"(.)\1{5,}", text):
        return True
    if len(set(text)) <= 2:
        return True
    return False


def main() -> int:
    wav_dir = sys.argv[1]
    spk = sys.argv[2] if len(sys.argv) > 2 else "speaker_a"
    paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    print("clips:", len(paths), flush=True)

    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8",
                         download_root=MODEL_DIR, cpu_threads=8)
    started = time.time()
    kept = 0
    with open(OUT_JSONL, "w", encoding="utf-8") as handle:
        for index, path in enumerate(paths, start=1):
            try:
                segments, info = model.transcribe(
                    path, language="zh", beam_size=5, vad_filter=True,
                    condition_on_previous_text=False,
                )
                segments = list(segments)
                text = clean_text("".join(seg.text for seg in segments))
                logprob = (
                    sum(seg.avg_logprob for seg in segments) / len(segments)
                    if segments else -9.0
                )
                duration = info.duration if info and info.duration else 0.0
                record = {
                    "wav": os.path.abspath(path),
                    "text": text,
                    "avg_logprob": round(float(logprob), 3),
                    "duration": round(float(duration), 2),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if logprob >= MIN_LOGPROB and not suspicious(text):
                    kept += 1
            except Exception as error:  # noqa: BLE001
                print("FAIL", os.path.basename(path), error, flush=True)
            if index % 25 == 0:
                elapsed = time.time() - started
                print("progress %d/%d | %.1f min | kept %d" % (
                    index, len(paths), elapsed / 60.0, kept), flush=True)

    # Build the filtered list for 1a.
    lines = []
    with open(OUT_JSONL, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            text = record["text"]
            if record["avg_logprob"] >= MIN_LOGPROB and not suspicious(text):
                lines.append("%s|%s|ZH|%s" % (record["wav"], spk, text))
    with open(OUT_LIST, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("kept:", len(lines), "/", len(paths), "->", OUT_LIST, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
