#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Allow RVC training checkpoints to load under torch >= 2.6."""
from __future__ import annotations

import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "rvc-src" / "train" / "utils.py"
OLD = 'torch.load(checkpoint_path, map_location="cpu")'
NEW = 'torch.load(checkpoint_path, map_location="cpu", weights_only=False)'


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    if "# NPU_CKPT_PATCHED" in text:
        print("already patched:", PATH)
        return 0
    count = text.count(OLD)
    if count == 0:
        print("anchor not found", file=sys.stderr)
        return 2
    PATH.write_text("# NPU_CKPT_PATCHED\n" + text.replace(OLD, NEW), encoding="utf-8")
    print("patched %d call(s) in %s" % (count, PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
