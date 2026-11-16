#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Patch RVC's DataLoader for containers with a small /dev/shm.

RVC's default DataLoader uses num_workers=4 + prefetch_factor=8, which needs a
large shared-memory segment and dies with "Bus error" on 64 MB /dev/shm
containers. Workers are switched off.
"""
from __future__ import annotations

import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "rvc-src" / "train" / "train.py"
OLD = '''    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )'''
NEW = '''    train_loader = DataLoader(
        train_dataset,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=False,
    )'''


def main() -> int:
    text = PATH.read_text(encoding="utf-8")
    if "# NPU_DATALOADER_PATCHED" in text:
        print("already patched:", PATH)
        return 0
    if OLD not in text:
        print("DataLoader anchor not found", file=sys.stderr)
        return 2
    PATH.write_text("# NPU_DATALOADER_PATCHED\n" + text.replace(OLD, NEW, 1), encoding="utf-8")
    print("patched:", PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
