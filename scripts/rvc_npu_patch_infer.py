#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch the RVC inference path for Ascend NPU (idempotent).

Two fixes:
1. parselmouth is optional on aarch64 (only the "pm" F0 method needs it).
2. RMVPE contains fp32 GRU ops that the NPU rejects, so run RMVPE on CPU
   while keeping the rest of the pipeline on npu:0.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT.parent / "rvc-src" / "infer" / "vc" / "pipeline.py"


def main() -> int:
    text = PIPELINE.read_text(encoding="utf-8")
    if "# NPU_INFER_PATCHED" in text:
        print("already patched:", PIPELINE)
        return 0
    text = text.replace(
        "import parselmouth\n",
        'try:\n    import parselmouth\nexcept Exception:  # aarch64: only "pm" needs it\n    parselmouth = None\n',
        1,
    )
    old = '''                self.model_rmvpe = RMVPE(
                    "%s/rmvpe.pt" % os.environ["rmvpe_root"],
                    is_half=self.is_half,
                    device=self.device,
                )'''
    new = '''                _rmvpe_device = "cpu" if str(self.device).startswith("npu") else self.device
                logger.info("rmvpe device: %s", _rmvpe_device)
                self.model_rmvpe = RMVPE(
                    "%s/rmvpe.pt" % os.environ["rmvpe_root"],
                    is_half=(self.is_half and _rmvpe_device != "cpu"),
                    device=_rmvpe_device,
                )'''
    if old not in text:
        print("RMVPE anchor not found", file=sys.stderr)
        return 2
    text = text.replace(old, new, 1)
    PIPELINE.write_text("# NPU_INFER_PATCHED\n" + text, encoding="utf-8")
    print("patched:", PIPELINE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
