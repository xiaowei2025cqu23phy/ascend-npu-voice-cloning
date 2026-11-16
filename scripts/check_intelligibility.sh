#!/usr/bin/env bash
# Transcribe several synthesized outputs to compare intelligibility.
set -u
source /home/developer/Ascend/cann-9.0.0/set_env.sh 2>/dev/null || true
# shellcheck disable=SC1091
source /mnt/workspace/gptsovits-venv/bin/activate
SP_GOMP=$(ls /mnt/workspace/gptsovits-venv/lib/python3.11/site-packages/sentencepiece.libs/libgomp-*.so* 2>/dev/null | head -n 1 || true)
export LD_PRELOAD="${SP_GOMP:-/lib/aarch64-linux-gnu/libgomp.so.1}${LD_PRELOAD:+:$LD_PRELOAD}"
export HF_ENDPOINT=https://hf-mirror.com

python - <<'PY'
import wave

from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8",
                     download_root="/mnt/workspace/pretrained_models/faster-whisper",
                     cpu_threads=8)
files = [
    ("zero-shot", "/mnt/workspace/gptsovits_demo/output.wav"),
    ("s2 only (D1)", "/mnt/workspace/gptsovits_demo_ft/output.wav"),
    ("s1+s2 (D1) old best", "/mnt/workspace/gptsovits_demo_best/output.wav"),
    ("speaker_a_v2 e1+e10", "/mnt/workspace/ab_test/new_e1_new_e10/output.wav"),
    ("speaker_a_v3 s2 + old gpt", "/mnt/workspace/ab3/best_gpt_speaker_a_v3_s2/output.wav"),
    ("speaker_a_v3c e3 + speaker_a_v3 s2", "/mnt/workspace/ab3c/speaker_a_v3c_e3_speaker_a_v3_s2/output.wav"),
]
for label, path in files:
    try:
        with wave.open(path, "rb") as handle:
            duration = handle.getnframes() / float(handle.getframerate())
        segments, _ = model.transcribe(path, language="zh", beam_size=5, vad_filter=False)
        text = "".join(seg.text for seg in segments).strip()
        print("%-28s | %.2fs | %s" % (label, duration, text))
    except Exception as error:  # noqa: BLE001
        print(label, "FAIL", error)
PY
