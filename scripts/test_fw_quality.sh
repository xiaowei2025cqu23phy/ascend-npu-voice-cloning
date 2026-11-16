#!/usr/bin/env bash
# Measure faster-whisper quality-mode (beam5 + VAD) throughput and confidence.
set -u
source /home/developer/Ascend/cann-9.0.0/set_env.sh 2>/dev/null || true
# shellcheck disable=SC1091
source /mnt/workspace/gptsovits-venv/bin/activate
SP_GOMP=$(ls /mnt/workspace/gptsovits-venv/lib/python3.11/site-packages/sentencepiece.libs/libgomp-*.so* 2>/dev/null | head -n 1 || true)
export LD_PRELOAD="${SP_GOMP:-/lib/aarch64-linux-gnu/libgomp.so.1}${LD_PRELOAD:+:$LD_PRELOAD}"
export HF_ENDPOINT=https://hf-mirror.com

python - <<'PY'
import glob
import time

from faster_whisper import WhisperModel

files = sorted(glob.glob("/mnt/workspace/rvc-src/logs/speaker_a_v3/0_gt_wavs/*.wav"))[200:210]
model = WhisperModel("small", device="cpu", compute_type="int8",
                     download_root="/mnt/workspace/pretrained_models/faster-whisper",
                     cpu_threads=8)

for beam, vad in ((5, True), (5, False), (3, True)):
    t0 = time.time()
    texts = []
    lps = []
    for path in files:
        segments, _ = model.transcribe(path, language="zh", beam_size=beam,
                                       vad_filter=vad, condition_on_previous_text=False)
        segments = list(segments)
        text = "".join(seg.text for seg in segments).strip()
        texts.append(text)
        if segments:
            lps.append(sum(seg.avg_logprob for seg in segments) / len(segments))
    elapsed = time.time() - t0
    audio = 10 * 3.7
    print("beam=%d vad=%s | 10 clips in %.1fs | %.1fx realtime | lp=%.2f | %s" % (
        beam, vad, elapsed, audio / elapsed, (sum(lps) / len(lps)) if lps else 0, texts[0][:18]))
PY
