#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Allow loading trusted local GPT-SoVITS weights under torch 2.7 weights_only."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "GPT-SoVITS"
MARKER = "# NPU_INFER_LOAD_PATCHED"
TARGETS = {
    "GPT_SoVITS/inference_webui.py": [
        ('torch.load(sovits_path, map_location="cpu")', 'torch.load(sovits_path, map_location="cpu", weights_only=False)'),
        ('torch.load(gpt_path, map_location="cpu")', 'torch.load(gpt_path, map_location="cpu", weights_only=False)'),
    ],
    "GPT_SoVITS/TTS_infer_pack/TTS.py": [
        ('torch.load(weights_path, map_location=self.configs.device)',
         'torch.load(weights_path, map_location=self.configs.device, weights_only=False)'),
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
