#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply quality filters to the audited speaker_a dataset and report the clean subset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPORT = Path(r"<PROJECT_ROOT>\training\quality_report_new.json")
OUT_FILES = Path(r"<PROJECT_ROOT>\training\speaker_a_clean_files.txt")

VLOG_KEYWORDS = ["随拍", "对镜", "自拍", "哈喽", "低质量片段", "图文", "日常", "花絮", "vlog", "VLOG"]
MAX_LOW = 0.12
MIN_RMS = -25.0
MAX_CLIP = 0.5
MAX_SILENCE = 0.45
MIN_DURATION = 3.0


def main() -> int:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = data["rows"]

    kept, dropped = [], []
    for row in rows:
        reasons = []
        name = row["file"]
        if any(keyword in name for keyword in VLOG_KEYWORDS):
            reasons.append("vlog/non-song")
        if row["duration"] < MIN_DURATION:
            reasons.append("too short")
        if row["rms_db"] < MIN_RMS:
            reasons.append("too quiet")
        if row["clip_pct"] > MAX_CLIP:
            reasons.append("clipping")
        if row["silence_ratio"] > MAX_SILENCE:
            reasons.append("too much silence")
        if row["low_ratio"] > MAX_LOW:
            reasons.append("accompaniment risk")
        if reasons:
            row["_reasons"] = reasons
            dropped.append(row)
        else:
            kept.append(row)

    kept_duration = sum(r["duration"] for r in kept)
    dropped_duration = sum(r["duration"] for r in dropped)
    print("total: %d files / %.1f min" % (len(rows), (kept_duration + dropped_duration) / 60.0))
    print("kept : %d files / %.1f min" % (len(kept), kept_duration / 60.0))
    print("dropped: %d files / %.1f min" % (len(dropped), dropped_duration / 60.0))

    print("\n=== dropped files ===")
    for row in sorted(dropped, key=lambda r: -r["duration"]):
        print("%-5.0fs rms %-6.1f low %-5.2f sil %-4.2f | %-38s | %s" % (
            row["duration"], row["rms_db"], row["low_ratio"], row["silence_ratio"],
            ",".join(row["_reasons"]), row["file"][:52]))

    OUT_FILES.write_text("\n".join(r["file"] for r in kept) + "\n", encoding="utf-8")
    print("\nclean file list ->", OUT_FILES)
    print("kept duration hours: %.2f h" % (kept_duration / 3600.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
