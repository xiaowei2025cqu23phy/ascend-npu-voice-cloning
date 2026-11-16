#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ascend NPU / torch 2.7 patches for GPT-SoVITS s2 (SoVITS) training.

1. s2_train.py: run on npu:0 when torch_npu is available (the non-CUDA branch
   already moves models with .to(device)).
2. DataLoader: num_workers=0 for 64 MB /dev/shm containers.
3. torch.load: weights_only=False for trusted checkpoints.
4. matplotlib >= 3.10 removed FigureCanvasAgg.tostring_rgb().
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "GPT-SoVITS"


def patch_s2() -> int:
    path = ROOT / "GPT_SoVITS" / "s2_train.py"
    text = path.read_text(encoding="utf-8")
    if "# NPU_S2_PATCHED" in text:
        print("s2_train already patched")
        return 0
    old_device = 'device = "cpu"  # cuda以外的设备，等mps优化后加入'
    new_device = (
        'device = "cpu"\n'
        'try:\n'
        '    import torch_npu  # noqa: F401\n'
        '    if hasattr(torch, "npu") and torch.npu.is_available():\n'
        '        device = "npu:0"\n'
        'except Exception:\n'
        '    pass'
    )
    if old_device not in text:
        print("device anchor not found", file=sys.stderr)
        return 2
    text = text.replace(old_device, new_device, 1)
    old_loader = '''    train_loader = DataLoader(
        train_dataset,
        num_workers=6,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=4,
    )'''
    new_loader = '''    train_loader = DataLoader(
        train_dataset,
        num_workers=0,
        shuffle=False,
        pin_memory=False,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=False,
    )'''
    if old_loader not in text:
        print("loader anchor not found", file=sys.stderr)
        return 3
    text = text.replace(old_loader, new_loader, 1)
    text = text.replace(
        'torch.load(hps.train.pretrained_s2G, map_location="cpu")',
        'torch.load(hps.train.pretrained_s2G, map_location="cpu", weights_only=False)',
    )
    text = text.replace(
        'torch.load(hps.train.pretrained_s2D, map_location="cpu")',
        'torch.load(hps.train.pretrained_s2D, map_location="cpu", weights_only=False)',
    )
    path.write_text("# NPU_S2_PATCHED\n" + text, encoding="utf-8")
    print("patched:", path)
    return 0


def patch_utils() -> int:
    path = ROOT / "GPT_SoVITS" / "utils.py"
    text = path.read_text(encoding="utf-8")
    if "# NPU_CKPT_PATCHED" in text:
        print("utils already patched")
        return 0
    old = 'checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")'
    if old not in text:
        print("utils anchor not found", file=sys.stderr)
        return 2
    text = text.replace(old, 'checkpoint_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)', 1)
    path.write_text("# NPU_CKPT_PATCHED\n" + text, encoding="utf-8")
    print("patched:", path)
    return 0


def patch_plot() -> int:
    path = ROOT / "GPT_SoVITS" / "utils.py"
    text = path.read_text(encoding="utf-8")
    if "# NPU_PLOT_PATCHED" in text:
        print("plot helper already patched")
        return 0
    old = 'data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")'
    new = 'data = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()'
    if old not in text:
        print("plot anchor not found", file=sys.stderr)
        return 2
    path.write_text("# NPU_PLOT_PATCHED\n" + text.replace(old, new, 1), encoding="utf-8")
    print("patched plot helper:", path)
    return 0


if __name__ == "__main__":
    code = patch_s2()
    code = patch_utils() or code
    code = patch_plot() or code
    sys.exit(code)
