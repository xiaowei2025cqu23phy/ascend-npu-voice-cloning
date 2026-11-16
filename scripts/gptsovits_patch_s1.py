#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load trusted s1 (GPT) training weights under torch 2.7 (weights_only default)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "GPT-SoVITS"
MARKER = "# NPU_S1_LOAD_PATCHED"
TARGETS = {
    "GPT_SoVITS/AR/models/t2s_lightning_module.py": [
        ('torch.load(pretrained_s1, map_location="cpu")', 'torch.load(pretrained_s1, map_location="cpu", weights_only=False)'),
    ],
    "GPT_SoVITS/AR/data/dataset.py": [
        ('torch.load(path_bert, map_location="cpu")', 'torch.load(path_bert, map_location="cpu", weights_only=False)'),
    ],
}


def main() -> int:
    for relative, replacements in TARGETS.items():
        path = ROOT / relative
        if not path.is_file():
            print("missing:", path, file=sys.stderr)
            continue
        text = path.read_text(encoding="utf-8")
        if MARKER in text:
            print("already patched:", path)
            continue
        changed = 0
        for old, new in replacements:
            if old in text:
                text = text.replace(old, new)
                changed += 1
        if changed:
            path.write_text(MARKER + "\n" + text, encoding="utf-8")
            print("patched %d call(s): %s" % (changed, path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
