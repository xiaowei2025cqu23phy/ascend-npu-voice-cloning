#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build an RVC training filelist for one experiment directory.

RVC v2 (with f0) expects one line per slice:

    <0_gt_wavs/xxx.wav>|<3_feature768/xxx.npy>|<2a_f0/xxx.wav.npy>|<2b-f0nsf/xxx.wav.npy>|<speaker_id>

Usage:
    python make_filelist.py logs/exp_name
"""
from __future__ import annotations

import os
import sys


def main(argv: list[str]) -> int:
    exp = argv[1] if len(argv) > 1 else "logs/exp"
    wav_dir = os.path.join(exp, "0_gt_wavs")
    if not os.path.isdir(wav_dir):
        print("missing", wav_dir, file=sys.stderr)
        return 2
    items = sorted(name for name in os.listdir(wav_dir) if name.endswith(".wav"))
    lines = []
    missing = 0
    for name in items:
        stem = os.path.splitext(name)[0]
        wav = os.path.abspath(os.path.join(wav_dir, name))
        feat = os.path.abspath(os.path.join(exp, "3_feature768", stem + ".npy"))
        f0a = os.path.abspath(os.path.join(exp, "2a_f0", name + ".npy"))
        f0b = os.path.abspath(os.path.join(exp, "2b-f0nsf", name + ".npy"))
        if all(os.path.exists(path) for path in (wav, feat, f0a, f0b)):
            lines.append("%s|%s|%s|%s|0" % (wav, feat, f0a, f0b))
        else:
            missing += 1
    target = os.path.join(exp, "filelist.txt")
    with open(target, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("filelist:", target, "| usable:", len(lines), "| incomplete:", missing)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
