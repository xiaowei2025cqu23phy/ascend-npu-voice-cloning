#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the s2 training config for the larger speaker_a_v2 dataset (continue from best)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/mnt/workspace/GPT-SoVITS")
TEMPLATE = ROOT / "GPT_SoVITS" / "configs" / "s2.json"
OUT = Path("/mnt/workspace/gptsovits-exp2/tmp_s2.json")
EXP_DIR = Path("/mnt/workspace/gptsovits-exp2/speaker_a")
PRETRAINED = ROOT / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained"
CONTINUE_FROM = ROOT / "SoVITS_weights" / "speaker_a_e8_s408.pth"

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def main() -> int:
    data = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    data["train"]["batch_size"] = BATCH
    data["train"]["epochs"] = EPOCHS
    data["train"]["fp16_run"] = False
    data["train"]["log_interval"] = 5
    data["train"]["save_every_epoch"] = 2
    data["train"]["if_save_latest"] = 1
    data["train"]["if_save_every_weights"] = 1
    data["train"]["pretrained_s2G"] = str(CONTINUE_FROM) if CONTINUE_FROM.is_file() else str(PRETRAINED / "s2G2333k.pth")
    data["train"]["pretrained_s2D"] = str(PRETRAINED / "s2D2333k.pth")
    data["train"]["gpu_numbers"] = "0"
    data["data"]["exp_dir"] = str(EXP_DIR)
    data["s2_ckpt_dir"] = str(EXP_DIR / "logs_s2")
    data["save_weight_dir"] = str(ROOT / "SoVITS_weights")
    data["name"] = "speaker_a_v2"
    data["version"] = "v2"
    data["model"]["version"] = "v2"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", OUT, "| epochs:", EPOCHS, "batch:", BATCH, "| s2G init:", data["train"]["pretrained_s2G"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
