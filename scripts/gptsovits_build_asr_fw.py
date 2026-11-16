#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the GPT-SoVITS ASR list with faster-whisper (much faster than transformers)."""
from __future__ import annotations

import glob
import os
import sys

MODEL_DIR = "/mnt/workspace/pretrained_models/faster-whisper"


def main() -> int:
    wav_dir = sys.argv[1]
    out_path = sys.argv[2]
    speaker = sys.argv[3] if len(sys.argv) > 3 else "speaker_a"
    paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    print("slices:", len(paths), flush=True)

    from faster_whisper import WhisperModel

    model = WhisperModel("small", device="cpu", compute_type="int8",
                         download_root=MODEL_DIR, cpu_threads=8)
    lines = []
    for index, path in enumerate(paths, start=1):
        try:
            segments, _ = model.transcribe(path, language="zh", beam_size=1, vad_filter=False)
            text = "".join(segment.text for segment in segments).strip()
            text = text.replace("|", " ").replace("\n", " ")
            if not text:
                continue
            lines.append("%s|%s|ZH|%s" % (os.path.abspath(path), speaker, text))
        except Exception as error:  # noqa: BLE001
            print("FAIL", os.path.basename(path), error, flush=True)
        if index % 20 == 0:
            print("progress %d/%d" % (index, len(paths)), flush=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("asr lines:", len(lines), "->", out_path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
