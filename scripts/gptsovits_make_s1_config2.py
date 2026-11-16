#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the s1 training config for the larger speaker_a_v2 dataset (continue from best e2)."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path("/mnt/workspace/GPT-SoVITS")
TEMPLATE = ROOT / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml"
OUT = Path("/mnt/workspace/gptsovits-exp2/tmp_s1.yaml")
EXP_DIR = Path("/mnt/workspace/gptsovits-exp2/speaker_a")
CONTINUE_FROM = ROOT / "GPT_weights" / "speaker_a-e2.ckpt"

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def main() -> int:
    data = yaml.load(TEMPLATE.read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    data["train"]["batch_size"] = BATCH
    data["train"]["epochs"] = EPOCHS
    data["train"]["precision"] = "32"
    data["train"]["save_every_n_epoch"] = 1
    data["train"]["if_save_every_weights"] = 1
    data["train"]["if_save_latest"] = 1
    data["train"]["if_dpo"] = 0
    data["train"]["half_weights_save_dir"] = "GPT_weights"
    data["train"]["exp_name"] = "speaker_a_v2"
    data["pretrained_s1"] = str(CONTINUE_FROM) if CONTINUE_FROM.is_file() else ""
    data["train_semantic_path"] = str(EXP_DIR / "6-name2semantic.tsv")
    data["train_phoneme_path"] = str(EXP_DIR / "2-name2text.txt")
    data["output_dir"] = str(EXP_DIR / "logs_s1")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False), encoding="utf-8")
    print("wrote", OUT, "| epochs:", EPOCHS, "batch:", BATCH, "| s1 init:", data["pretrained_s1"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
