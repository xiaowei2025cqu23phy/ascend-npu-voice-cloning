#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the quality audit results for the enlarged speaker_a dataset."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(r"<PROJECT_ROOT>\training")
REPORT = ROOT / "quality_report_new.json"
OUT = ROOT / "quality_report.png"

VLOG_KEYWORDS = ["随拍", "对镜", "自拍", "哈喽", "低质量片段", "图文", "日常", "花絮", "vlog", "VLOG"]


def is_kept(row: dict) -> bool:
    if any(k in row["file"] for k in VLOG_KEYWORDS):
        return False
    if row["duration"] < 3.0:
        return False
    if row["rms_db"] < -25.0:
        return False
    if row["clip_pct"] > 0.5:
        return False
    if row["silence_ratio"] > 0.45:
        return False
    if row["low_ratio"] > 0.12:
        return False
    return True


def main() -> int:
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = data["rows"]
    kept = [r for r in rows if is_kept(r)]
    dropped = [r for r in rows if not is_kept(r)]
    kept_min = sum(r["duration"] for r in kept) / 60.0
    dropped_min = sum(r["duration"] for r in dropped) / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    ax = axes[0][0]
    ax.scatter([r["duration"] for r in kept], [r["rms_db"] for r in kept],
               s=18, c="#2E8B57", label="kept (%d / %.1f min)" % (len(kept), kept_min))
    ax.scatter([r["duration"] for r in dropped], [r["rms_db"] for r in dropped],
               s=28, c="#C0392B", marker="x", label="dropped (%d / %.1f min)" % (len(dropped), dropped_min))
    ax.axhline(-25, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("RMS (dBFS)")
    ax.set_title("Loudness vs duration")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    ax.scatter([r["low_ratio"] for r in kept], [r["snr_db"] for r in kept], s=18, c="#2E8B57")
    ax.scatter([r["low_ratio"] for r in dropped], [r["snr_db"] for r in dropped], s=28, c="#C0392B", marker="x")
    ax.axvline(0.12, color="r", linestyle="--", linewidth=1)
    ax.set_xlabel("low-band energy ratio (accompaniment proxy)")
    ax.set_ylabel("SNR estimate (dB)")
    ax.set_title("Accompaniment risk vs SNR")
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    ax.hist([r["duration"] for r in kept], bins=25, color="#5B8FF9")
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("files")
    ax.set_title("Duration distribution (kept files)")
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    labels = ["clean kept\n%.1f min" % kept_min, "dropped\n%.1f min" % dropped_min]
    ax.bar(labels, [kept_min, dropped_min], color=["#2E8B57", "#C0392B"])
    ax.set_ylabel("minutes")
    ax.set_title("Total duration after quality filter")
    for i, value in enumerate([kept_min, dropped_min]):
        ax.text(i, value + 0.5, "%.1f min" % value, ha="center", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Speaker_a data quality audit: %d files / %.1f min" % (
        data["unique_files"], data["total_duration"] / 60.0), fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT, dpi=120)
    print("saved", OUT)
    print("kept: %d files / %.1f min | dropped: %d files / %.1f min" % (
        len(kept), kept_min, len(dropped), dropped_min))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
