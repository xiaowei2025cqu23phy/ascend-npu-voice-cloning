#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the GPT-SoVITS output against speaker_a's reference speaker embedding."""
from __future__ import annotations

import glob
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import librosa  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoFeatureExtractor, WavLMForXVector  # noqa: E402

MODEL_ID = "microsoft/wavlm-base-plus-sv"


def embed(model, extractor, path):
    wav, _ = librosa.load(path, sr=16000, mono=True)
    inputs = extractor(wav, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return F.normalize(outputs.embeddings, dim=-1).squeeze(0)


def main() -> int:
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = WavLMForXVector.from_pretrained(MODEL_ID).eval()

    old_refs = sorted(glob.glob("/mnt/workspace/dataset_speaker_a_raw/*.wav"))
    new_refs = sorted(glob.glob("/mnt/workspace/dataset_speaker_a_v3_raw/*.wav"))
    refs = old_refs[:8] + new_refs[::max(1, len(new_refs) // 12)][:12]
    print("reference clips:", len(refs))
    speaker_a = F.normalize(torch.stack([embed(model, extractor, p) for p in refs]).mean(dim=0), dim=-1)

    candidates = [
        ("old 17.7min best", "/mnt/workspace/gptsovits_demo_best/output.wav"),
        ("best gpt + speaker_a_v3 s2", "/mnt/workspace/ab3/best_gpt_speaker_a_v3_s2/output.wav"),
        ("speaker_a_v3c e1 + speaker_a_v3 s2", "/mnt/workspace/ab3c/speaker_a_v3c_e1_speaker_a_v3_s2/output.wav"),
        ("speaker_a_v3c e2 + speaker_a_v3 s2", "/mnt/workspace/ab3c/speaker_a_v3c_e2_speaker_a_v3_s2/output.wav"),
        ("speaker_a_v3c e3 + speaker_a_v3 s2", "/mnt/workspace/ab3c/speaker_a_v3c_e3_speaker_a_v3_s2/output.wav"),
        ("reference clip (speaker_a)", "/mnt/workspace/demo_src/ref_speaker_a8.wav"),
    ]
    print("%-28s %s" % ("file", "cosine to speaker_a"))
    print("-" * 45)
    for label, path in candidates:
        if not os.path.isfile(path):
            print("%-28s (missing)" % label)
            continue
        print("%-28s %.3f" % (label, float(torch.dot(speaker_a, embed(model, extractor, path)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
