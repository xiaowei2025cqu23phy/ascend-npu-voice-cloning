#!/usr/bin/env bash
# End-to-end pipeline: assemble 71-min dataset -> slice -> ASR -> features ->
# s2 (NPU) and s1 (CPU) incremental training -> inference -> similarity check.
set -eu
set -o pipefail
PROGRESS=/mnt/workspace/bigrun.log
say() { echo "[$(date '+%H:%M:%S')] $*"; }

source /home/developer/Ascend/cann-9.0.0/set_env.sh

########################################
say "STEP 1/7 assemble uploaded dataset"
cd /mnt/workspace
cat part_* > speaker_a_v3_raw.tar.gz
sha256sum speaker_a_v3_raw.tar.gz
rm -f part_*
rm -rf dataset_speaker_a_v3_raw speaker_a_clean
tar -xzf speaker_a_v3_raw.tar.gz
mv speaker_a_clean dataset_speaker_a_v3_raw
rm -f speaker_a_v3_raw.tar.gz
python3 -c "import os; print('raw wavs:', len(os.listdir('dataset_speaker_a_v3_raw')))"

########################################
say "STEP 2/7 slice (48k, 3.7s)"
# shellcheck disable=SC1091
source /mnt/workspace/rvc-venv/bin/activate
GOMP=$(ls /mnt/workspace/rvc-venv/lib/python3.11/site-packages/scikit_learn.libs/libgomp-*.so* 2>/dev/null | head -n 1 || true)
[ -n "$GOMP" ] && export LD_PRELOAD="$GOMP${LD_PRELOAD:+:$LD_PRELOAD}"
cd /mnt/workspace/rvc-src
rm -rf logs/speaker_a_v3
mkdir -p logs/speaker_a_v3
python -m train.preprocess /mnt/workspace/dataset_speaker_a_v3_raw 48000 4 logs/speaker_a_v3 True 3.7 2>&1 | tail -n 4
python -c "import os; print('slices:', len(os.listdir('logs/speaker_a_v3/0_gt_wavs')))"
deactivate || true

########################################
say "STEP 3/7 ASR (faster-whisper)"
# shellcheck disable=SC1091
source /mnt/workspace/gptsovits-venv/bin/activate
SP_GOMP=$(ls /mnt/workspace/gptsovits-venv/lib/python3.11/site-packages/sentencepiece.libs/libgomp-*.so* 2>/dev/null | head -n 1 || true)
export LD_PRELOAD="${SP_GOMP:-/lib/aarch64-linux-gnu/libgomp.so.1}${LD_PRELOAD:+:$LD_PRELOAD}"
export HF_ENDPOINT=https://hf-mirror.com
EXP_ROOT=/mnt/workspace/gptsovits-exp3
EXP_NAME=speaker_a
EXP_DIR="$EXP_ROOT/$EXP_NAME"
WAV_DIR=/mnt/workspace/rvc-src/logs/speaker_a_v3/0_gt_wavs
rm -rf "$EXP_DIR"
mkdir -p "$EXP_DIR"
python /mnt/workspace/training/gptsovits_build_asr_fw.py "$WAV_DIR" "$EXP_ROOT/asr.list" 2>&1 | tail -n 6

########################################
say "STEP 4/7 features (1a/1b/1c)"
cd /mnt/workspace/GPT-SoVITS
export PYTHONPATH="$PWD/GPT_SoVITS:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export inp_text="$EXP_ROOT/asr.list"
export inp_wav_dir="$WAV_DIR"
export exp_name="$EXP_NAME"
export opt_dir="$EXP_DIR"
export bert_pretrained_dir="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
export is_half="False"
export i_part="0"
export all_parts="1"
python GPT_SoVITS/prepare_datasets/1-get-text.py 2>&1 | tail -n 5
if [ -f "$EXP_DIR/2-name2text-0.txt" ]; then
  cat "$EXP_DIR/2-name2text-0.txt" > "$EXP_DIR/2-name2text.txt"
  rm -f "$EXP_DIR/2-name2text-0.txt"
fi
export cnhubert_base_dir="GPT_SoVITS/pretrained_models/chinese-hubert-base"
python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py 2>&1 | tail -n 4
export pretrained_s2G="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
export s2config_path="GPT_SoVITS/configs/s2.json"
python GPT_SoVITS/prepare_datasets/3-get-semantic.py 2>&1 | tail -n 4
if [ -f "$EXP_DIR/6-name2semantic-0.tsv" ]; then
  {
    printf 'item_name\tsemantic_audio\n'
    cat "$EXP_DIR/6-name2semantic-0.tsv"
  } > "$EXP_DIR/6-name2semantic.tsv"
  rm -f "$EXP_DIR/6-name2semantic-0.tsv"
fi
python - <<'PY'
import os
exp = os.environ["opt_dir"]
for name in ("2-name2text.txt", "6-name2semantic.tsv"):
    path = os.path.join(exp, name)
    print(name, "lines:", len(open(path, encoding="utf8").read().strip().split("\n")) if os.path.exists(path) else "MISSING")
print("cnhubert:", len(os.listdir(os.path.join(exp, "4-cnhubert"))))
print("wav32k:", len(os.listdir(os.path.join(exp, "5-wav32k"))))
PY

########################################
say "STEP 5/7 s2 training (NPU, 6 epochs, continue from speaker_a_v2_e10)"
python - "$EXP_DIR" "$EXP_ROOT" <<'PY'
import json
import sys
from pathlib import Path

exp_dir, exp_root = sys.argv[1], sys.argv[2]
root = Path("/mnt/workspace/GPT-SoVITS")
data = json.loads((root / "GPT_SoVITS" / "configs" / "s2.json").read_text(encoding="utf-8"))
continue_from = root / "SoVITS_weights" / "speaker_a_v2_e10_s770.pth"
data["train"]["batch_size"] = 8
data["train"]["epochs"] = 6
data["train"]["fp16_run"] = False
data["train"]["log_interval"] = 10
data["train"]["save_every_epoch"] = 2
data["train"]["if_save_latest"] = 1
data["train"]["if_save_every_weights"] = 1
data["train"]["pretrained_s2G"] = str(continue_from) if continue_from.is_file() else str(root / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s2G2333k.pth")
data["train"]["pretrained_s2D"] = str(root / "GPT_SoVITS" / "pretrained_models" / "gsv-v2final-pretrained" / "s2D2333k.pth")
data["train"]["gpu_numbers"] = "0"
data["data"]["exp_dir"] = exp_dir
data["s2_ckpt_dir"] = exp_dir + "/logs_s2"
data["save_weight_dir"] = str(root / "SoVITS_weights")
data["name"] = "speaker_a_v3"
data["version"] = "v2"
data["model"]["version"] = "v2"
out = Path(exp_root) / "tmp_s2.json"
out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("s2 config:", out, "| init:", data["train"]["pretrained_s2G"])
PY
python /mnt/workspace/training/gptsovits_patch_s2.py
python GPT_SoVITS/s2_train.py --config "$EXP_ROOT/tmp_s2.json" 2>&1 | tail -n 6

########################################
say "STEP 6/7 s1 training (CPU, 1 epoch, continue from speaker_a_v2-e1)"
python - "$EXP_DIR" "$EXP_ROOT" <<'PY'
import sys
from pathlib import Path

import yaml

exp_dir, exp_root = sys.argv[1], sys.argv[2]
root = Path("/mnt/workspace/GPT-SoVITS")
data = yaml.load((root / "GPT_SoVITS" / "configs" / "s1longer-v2.yaml").read_text(encoding="utf-8"), Loader=yaml.FullLoader)
continue_from = root / "GPT_weights" / "speaker_a_v2-e1.ckpt"
data["train"]["batch_size"] = 8
data["train"]["epochs"] = 1
data["train"]["precision"] = "32"
data["train"]["save_every_n_epoch"] = 1
data["train"]["if_save_every_weights"] = 1
data["train"]["if_save_latest"] = 1
data["train"]["if_dpo"] = 0
data["train"]["half_weights_save_dir"] = "GPT_weights"
data["train"]["exp_name"] = "speaker_a_v3"
data["pretrained_s1"] = str(continue_from) if continue_from.is_file() else ""
data["train_semantic_path"] = exp_dir + "/6-name2semantic.tsv"
data["train_phoneme_path"] = exp_dir + "/2-name2text.txt"
data["output_dir"] = exp_dir + "/logs_s1"
out = Path(exp_root) / "tmp_s1.yaml"
out.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False), encoding="utf-8")
print("s1 config:", out, "| init:", data["pretrained_s1"])
PY
python /mnt/workspace/training/gptsovits_patch_s1.py
python GPT_SoVITS/s1_train.py --config_file "$EXP_ROOT/tmp_s1.yaml" 2>&1 | tail -n 6

########################################
say "STEP 7/7 inference + similarity check"
python /mnt/workspace/training/gptsovits_patch_infer.py
mkdir -p "$EXP_ROOT/demo"
GPT=$(ls -1 GPT_weights/speaker_a_v3-*.ckpt 2>/dev/null | sort -V | tail -n 1)
SOVITS=$(ls -1 SoVITS_weights/speaker_a_v3_e*.pth 2>/dev/null | sort -V | tail -n 1)
echo "final gpt: $GPT | sovits: $SOVITS"
python GPT_SoVITS/inference_cli.py \
  --gpt_model "$GPT" \
  --sovits_model "$SOVITS" \
  --ref_audio /mnt/workspace/demo_src/ref_speaker_a8.wav \
  --ref_text /mnt/workspace/demo_src/ref_speaker_a8.txt \
  --ref_language 中文 \
  --target_text /mnt/workspace/demo_src/target_speaker_a.txt \
  --target_language 中文 \
  --output_path "$EXP_ROOT/demo" 2>&1 | tail -n 6
ls -la "$EXP_ROOT/demo" || true
say "ALL DONE"
