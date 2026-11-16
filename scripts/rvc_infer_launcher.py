"""Launch the RVC CLI with PyTorch >= 2.6 checkpoint-loading compatibility.

The fine-tuned RVC weights pickle references `getattr`, which torch 2.7's
weights_only default rejects; this launcher relaxes checkpoint loading only for
trusted local model files.
"""
import os
import sys

ROOT = os.environ.get("RVC_ROOT", "/workspace/rvc-src")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch  # noqa: E402

_original_load = torch.load


def _load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


torch.load = _load

from infer.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
