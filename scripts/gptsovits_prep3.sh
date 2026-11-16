#!/usr/bin/env bash
# GPT-SoVITS dataset prep for the larger speaker_a_v2 dataset (312 slices).
set -u
source /home/developer/Ascend/cann-9.0.0/set_env.sh 2>/dev/null || true
# shellcheck disable=SC1091
source /mnt/workspace/gptsovits-venv/bin/activate
SP_GOMP=$(ls /mnt/workspace/gptsovits-venv/lib/python3.11/site-packages/sentencepiece.libs/libgomp-*.so* 2>/dev/null | head -n 1 || true)
export LD_PRELOAD="${SP_GOMP:-/lib/aarch64-linux-gnu/libgomp.so.1}${LD_PRELOAD:+:$LD_PRELOAD}"
export HF_ENDPOINT=https://hf-mirror.com
cd /mnt/workspace/GPT-SoVITS || exit 1
export PYTHONPATH="$PWD/GPT_SoVITS:$PWD${PYTHONPATH:+:$PYTHONPATH}"

EXP_ROOT=/mnt/workspace/gptsovits-exp2
EXP_NAME=speaker_a
EXP_DIR="$EXP_ROOT/$EXP_NAME"
WAV_DIR=/mnt/workspace/rvc-src/logs/speaker_a_v2/0_gt_wavs
rm -rf "$EXP_DIR"
mkdir -p "$EXP_DIR"

python -c "import os; print('wavs:', len([n for n in os.listdir('$WAV_DIR') if n.endswith('.wav')]))"

echo "===== 1) ASR list ====="
rm -f "$EXP_ROOT/asr.list"
python /mnt/workspace/training/gptsovits_build_asr.py "$WAV_DIR" "$EXP_ROOT/asr.list" 2>&1 | tail -n 5

echo "===== 2) 1a: text + BERT ====="
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
python -c "import os; p='$EXP_DIR/2-name2text.txt'; print('name2text:', len(open(p,encoding='utf8').read().strip().split(chr(10))) if os.path.exists(p) else 'MISSING')"

echo "===== 3) 1b: HuBERT + wav32k ====="
export cnhubert_base_dir="GPT_SoVITS/pretrained_models/chinese-hubert-base"
python GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py 2>&1 | tail -n 4
python -c "import os; print('cnhubert:', len(os.listdir('$EXP_DIR/4-cnhubert')) if os.path.isdir('$EXP_DIR/4-cnhubert') else 'MISSING', '| wav32k:', len(os.listdir('$EXP_DIR/5-wav32k')) if os.path.isdir('$EXP_DIR/5-wav32k') else 'MISSING')"

echo "===== 4) 1c: semantic tokens ====="
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
python -c "import os; p='$EXP_DIR/6-name2semantic.tsv'; print('semantic:', len(open(p,encoding='utf8').read().strip().split(chr(10))) if os.path.exists(p) else 'MISSING')"

echo "=== DONE ==="
