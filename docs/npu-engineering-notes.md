# RVC 昇腾 NPU 训练指南（CANNLab 实测打通）

> 实测日期：2026-11-08 · 环境：CANNLab NPU A2（Ascend 910B3 ×1，HBM 64 GB）
> 目标：在昇腾 NPU 上完成「数据准备 → 特征提取 → RVC 训练 → 变声推理」全流程
> 当前状态：**全链路已跑通**（用 61 秒烟雾数据训练 2 epoch + 推理出声）

---

## 1. 环境与位置

| 项 | 值 |
|---|---|
| 云平台 | CANNLab（https://gitcode.com/org/cann/cannlab/environment） |
| 机型/模板 | NPU A2 · `cann_9.0.0 py3.11-A2-arm-20261106`（910B3，16 vCPU，32 GiB） |
| CANN | 9.0.0，`/home/developer/Ascend/cann-9.0.0` |
| Python | 3.11.4；torch 2.7.1+cpu；torch-npu 2.7.1.post4（镜像预装） |
| RVC 仓库 | gitee 镜像 `zhyqieqie/rvc` → `/mnt/workspace/rvc-src`（已去 fairseq，HuBERT 用 transformers） |
| Python 环境 | `/mnt/workspace/rvc-venv`（`--system-site-packages`，复用系统 torch） |
| 预训练模型 | HF `lj1995/VoiceConversionWebUI`（经 hf-mirror.com 下载）→ `assets/`，共 1.6 GB |
| 训练实验 | `/mnt/workspace/rvc-src/logs/fish-smoke` |
| 训练产物 | `assets/weights/fish-smoke.pth`（推理模型，57 MB） |
| 推理示例 | `/mnt/workspace/rvc_demo/demo_npu.wav`（NPU 推理输出） |

⚠️ 持久化只保留 `/mnt/workspace` 和 `/home`；不用时**记得关机**（卡时额度）。

---

## 2. 一次性准备（已在本环境完成，重建时照做）

```bash
# 1) 拉仓库（gitee 镜像，避免 GitHub 不通）
git clone --depth 1 https://gitee.com/zhyqieqie/rvc.git /mnt/workspace/rvc-src

# 2) venv（复用系统 torch，不覆盖 CANN 镜像）
python3 -m venv --system-site-packages /mnt/workspace/rvc-venv
source /mnt/workspace/rvc-venv/bin/activate
pip install "numpy<2" librosa soundfile faiss-cpu av ffmpeg-python PyYAML \
  scikit-learn tensorboard torchfcpe "transformers==4.49.0" onnxruntime tqdm \
  matplotlib einops coloredlogs networkx local-attention hyper-connections

# 3) 预训练模型（HF 被墙 → 用 hf-mirror）
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("lj1995/VoiceConversionWebUI",
                  allow_patterns=["hubert_base/*", "rmvpe.pt", "pretrained_v2/*"],
                  local_dir="/mnt/workspace/rvc-src/assets")
PY
# HF 仓库根目录的 rmvpe.pt 要移到 RVC 期望位置
mkdir -p /mnt/workspace/rvc-src/assets/rmvpe
mv /mnt/workspace/rvc-src/assets/rmvpe.pt /mnt/workspace/rvc-src/assets/rmvpe/rmvpe.pt

# 4) 给 venv 提供 ffmpeg CLI（RVC CPU 音频加载走 ffmpeg）
FF=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
ln -sf "$FF" /mnt/workspace/rvc-venv/bin/ffmpeg

# 5) 应用 NPU 补丁（脚本见 training/rvc_npu_patch.py、rvc_npu_patch_infer.py）
python training/rvc_npu_patch.py
python training/rvc_npu_patch_infer.py
```

### NPU 补丁改了什么

| 文件 | 改动 |
|---|---|
| `configs/config.py` | 识别 `torch.npu`，设备选 `npu:0`、精度 fp32、显存/名称走 NPU |
| `train/train.py` | `.cuda()` → `.to(npu:0)`；`torch.cuda.*` → NPU 兼容包装；fp32 下 AMP 用 no-op |
| `train/dataset/extract_f0.py` | `parselmouth` 惰性导入；新增 `npu` 模式 |
| `train/dataset/extract_hubert_feature.py` | 设备选择支持 `npu:0` |
| `infer/vc/pipeline.py` | `parselmouth` 惰性导入；NPU 上 RMVPE 自动退回 CPU |

### 环境坑（都已踩平）

- **sklearn/libgomp 静态 TLS**：启动前 `export LD_PRELOAD=$(ls .../scikit_learn.libs/libgomp-*.so*)`。
- **parselmouth 在 aarch64 上要源码编译 Praat**：放弃安装；F0 用 RMVPE/CPU。
- **RMVPE 内含 GRU，fp32 内核在 NPU 不支持**（`DynamicGRUV2 ... DT_FLOAT not supported`）→ 预处理阶段 F0 用 CPU（或将来改 fp16）。
- **torch 2.7 默认 `weights_only=True`**：README 里的推理 launcher 会临时放宽，仅用于可信本地模型。
- **路径**：必须从仓库根目录用 `python -m train.xxx` 运行（直接跑脚本会命中 `train` 包名循环导入）。

---

## 3. 每次训练的命令流

```bash
source /home/developer/Ascend/cann-9.0.0/set_env.sh
source /mnt/workspace/rvc-venv/bin/activate
cd /mnt/workspace/rvc-src
export LD_PRELOAD=$(ls /mnt/workspace/rvc-venv/lib/python3.11/site-packages/scikit_learn.libs/libgomp-*.so* | head -n1)
EXP=logs/my-singer
mkdir -p "$EXP"; cp configs/v2/48k.json "$EXP/config.json"

# 1) 切片（CPU；输入是 44.1k/48k 原始人声）
python -m train.preprocess /mnt/workspace/dataset_raw 48000 4 "$EXP" True 3.7

# 2) F0 提取（CPU 跑 rmvpe；NPU 暂不可用）
python -m train.dataset.extract_f0 cpu "$EXP" 4 rmvpe

# 3) HuBERT 特征（NPU）
python -m train.dataset.extract_hubert_feature npu 1 0 "$EXP" v2 false

# 4) 生成 filelist（格式：wav|3_feature768/xxx.npy|2a_f0/xxx.wav.npy|2b-f0nsf/xxx.wav.npy|0）
python training/make_filelist.py "$EXP"

# 5) 训练（NPU，fp32）
python -m train.train -e my-singer -sr 48k -f0 1 -bs 4 -g 0 -te 200 -se 10 \
  -pg assets/pretrained_v2/f0G48k.pth -pd assets/pretrained_v2/f0D48k.pth \
  -l 1 -c 0 -sw 1 -v v2

# 6) 推理（NPU；F0 自动走 CPU）
python /mnt/workspace/rvc_infer_launcher.py \
  --model my-singer.pth --input test.wav --output out.wav --index-rate 0
```

> 真实训练量参考：**10–30 分钟干净单人干声**（有授权），100–200 epoch；
> 910B3 上单 epoch 比 61 秒烟雾数据长很多，建议先 5–10 epoch 看 loss 曲线再决定总轮数。
> 训练完可再用 `train_index.py` 生成检索 index（faiss，CPU）提升相似度。

---

## 4. 实测结果（2026-11-08）

| 步骤 | 结果 |
|---|---|
| 切片 | 10 段原始音频 → 12 段训练切片（0_gt_wavs / 1_16k_wavs） |
| F0 | 12/12（CPU RMVPE） |
| HuBERT 特征 | **12/12（NPU）**，输出 (frames, 768) |
| 训练 | 2 epoch：约 32s + 16s；loss 正常记录（loss_disc/gen/fm/mel/kl） |
| 成品模型 | `assets/weights/fish-smoke.pth`（57 MB） |
| 推理 | **Status: Success**，Features 0.21s / F0 4.90s（CPU）/ Synthesis **0.65s（NPU）** |

> 注意：烟雾模型只训了 2 epoch、数据是 61 秒 TTS 语音，音色效果不代表最终质量；
> 质量取决于真实授权数据 + 足够 epoch。

---

## 5. 下一步

1. 提供**有授权的**目标歌手人声（10–30 分钟，干声、无伴奏、无混响）。
2. 本机用 `training/prepare_dataset.py` 清洗切片 → 上传 `/mnt/workspace/dataset_raw`。
3. 按第 3 节命令训练（先 5–10 epoch 验证 loss）。
4. 训练完在 NPU 推理试听，再决定是否加 index / 继续训练。

---

## 6. speaker_a（speaker_a）数据集实测记录（2026-11-08）

数据来源：本地 `speaker_a_20260624_042439`（21 个 MP4，81 MB，多为「无声卡」翻唱/清唱）。

| 项 | 结果 |
|---|---|
| 素材体检 | 21 段 / 11.7 分钟；单声道、无伴奏（低频能量≈0.035、立体声侧信号≈0.045） |
| 抽音轨 | 21/21 → 48 kHz 单声道 WAV（`dataset_speaker_a_raw`） |
| 切片 | 207 个训练片段（RVC preprocess） |
| F0 | CPU RMVPE（6 进程） |
| HuBERT | NPU，207/207 |
| 训练 | **30 epoch，约 45–50 s/epoch**（bs=4），loss 正常（mel 27–31，disc≈3） |
| 成品 | `rvc-src/assets/weights/speaker_a.pth`（57.5 MB），另有 e10/e20/e30 快照 |
| 推理 | NPU：Features 0.2s / F0 5–7s(CPU) / Synthesis 0.7s |
| Demo | `rvc_demo/speaker_a_taiwan_out.wav`、`speaker_a_kangkang_out.wav`、`speaker_a_huihui_out.wav` |

### 新踩的坑

- **容器 `/dev/shm` 只有 64 MB**：RVC 默认 `DataLoader(num_workers=4, prefetch_factor=8)` 会报
  `DataLoader worker ... Bus error`。补丁 `rvc_npu_patch_dataloader.py` 改成 `num_workers=0` 后正常。
- **相对路径**：`infer.cli` 会 `chdir` 到仓库根目录，`--input/--output` 用**绝对路径**。
- 训练中断后可直接续训：同一个 `-e speaker_a` 目录会从最新 checkpoint 自动恢复；
  想加轮数就改 `-te 100` 重新跑同一条命令。

### 继续提升音色相似度

1. 加 epoch：30 → 80~100（约 45 s/epoch，910B3 上 1 小时能跑 80 epoch）。
2. 数据更多更干净：目标 10–30 分钟干声（当前仅 11.7 分钟，且来自短视频压缩音频）。
3. 训练完成后用 `train_index.py` 生成检索 index（faiss，CPU），推理时 `--index-rate 0.75`。
4. 对比 e10/e20/e30 快照，选听觉上最像的一版。

> ⚠️ 授权：speaker_a是真人声音，本模型仅用于本地个人研究；对外分享模型/翻唱作品前需取得本人授权。

---

## 7. 100 epoch + index 最终结果（2026-11-08 晚）

| 项 | 结果 |
|---|---|
| 训练 | 从 e30 续训到 **100 epoch**（约 70 分钟，约 46–55 s/epoch） |
| 最终模型 | `assets/weights/speaker_a.pth`（57.5 MB）+ `speaker_a_e100_s5300.pth` |
| 检索 index | `added_IVF934_Flat_nprobe_1_speaker_a_v2.index`（faiss，36453 个特征向量，已链接到 `assets/indices/`） |
| 训练 loss 趋势 | mel 34 → 22–26，disc 稳定 ~3，明显收敛 |

### “纯音色”声纹校验（WavLM-XVector，与 6 段speaker_a参考音频的余弦相似度）

| 音频 | 相似度 |
|---|---|
| speaker_a自身一致性（参考） | **0.978** |
| e30 自转换 | 0.964 |
| **e100 自转换（+index）** | **0.968** |
| e30 跨人声转换（台湾腔/男声/女声） | 0.876 / 0.818 / 0.848 |
| e100 中性合成源（无 index） | 0.408 |
| **e100 中性合成源（+index）** | **0.586** |
| 中性合成源本身（无真人身份） | 0.293 |

结论：

1. **模型确实学到了speaker_a的音色**（自转换 0.968 ≈ 她本人 0.978）。
2. **跨人声转换会残留源说话人的特征**（0.82–0.88），这就是“混合感”的来源——RVC 只换音色，
   源音频的咬字、气息、节奏会保留。要“纯她”，就用**她自己的声音**做源（自转换）。
3. **检索 index 明显提升相似度**（0.408 → 0.586），推理时 `--index-rate 0.75`。
4. 中性合成源本身不自然（WavLM 对合成音频打分偏低），输出会带电子味；
   若要“她唱新歌”，更合适的路线是用她的声音训练 **GPT-SoVITS**（TTS/歌唱合成，不需要别人的声音当源）。

### 最终交付音频（本地 `rvc-demo/`）

- `self_e100_idx075.wav` —— 她本人 → 模型（纯她的音色，0.968）
- `neutral_e100_idx075.wav` —— 中性合成旋律 → 她的音色（不含任何真人源）
- `neutral_e30.wav` / `neutral_e100_idx075.wav` —— 30 vs 100 epoch 对比

---

## 8. GPT-SoVITS 零样本 TTS（2026-11-08 晚，CPU 跑通）

目标：不依赖任何别人的声音，用speaker_a的参考音频直接合成她说话/唱歌。

### 环境

| 项 | 值 |
|---|---|
| 仓库 | gitee 镜像 `Snake-Konginchrist/GPT-SoVITS`（main，commit b7a904a）→ `/mnt/workspace/GPT-SoVITS` |
| venv | `/mnt/workspace/gptsovits-venv`（system-site-packages 复用 torch 2.7.1 + torch_npu） |
| 关键依赖 | torchaudio **2.7.1**（替换镜像自带的 2.11，版本不匹配）、transformers 4.43.1、gradio 4.24.0、librosa 0.11、numba 0.68、pytorch-lightning 2.6、onnxruntime、opencc-python-reimplemented |
| 预训练 | HF `lj1995/GPT-SoVITS` v2 文件（hubert 189MB / bert 651MB / s1 155MB / s2G 106MB），经 hf-mirror 下载 |
| G2PWModel | paddlespeech 官方链接已 404，从 HF 镜像 `alextomcat/G2PWModel` 下载到 `GPT_SoVITS/text/G2PWModel/` |
| 推理设备 | CPU（`tts_infer.yaml` default_v2 默认 cpu；NPU 推理待后续 patch） |

### 踩坑与修复（Python 3.11 / aarch64 / torch 2.7）

1. `torchaudio` 镜像自带 2.11 与 torch 2.7.1 ABI 不匹配 → 固定装 `torchaudio==2.7.1`（aarch64 wheel）。
2. **libgomp 静态 TLS**：torchaudio、sentencepiece 各带一份 libgomp 会冲突 →
   预加载 `sentencepiece.libs/libgomp-*.so*` 或 `/lib/aarch64-linux-gnu/libgomp.so.1`。
3. `LangSegment` PyPI 版本缺 `setLangfilters/getLangfilters` → 写一个中文版兼容 shim
   （`gptsovits_shim.sh`，全部文本按中文处理）。
4. `jieba_fast`（C 扩展）在 aarch64 无轮子 → 写 `jieba_fast` 兼容包转发到 `jieba`；
   并把 `tone_sandhi.py` 的导入改为 try/except。
5. torch 2.7 下 `patched_mha_with_cache.py` 缺 `typing.Tuple` 导入 → 补丁。
6. `gradio`、`pytorch-lightning`、`onnxruntime`、`opencc-python-reimplemented` 按需补装。
7. `tools/my_utils.load_audio` 依赖 `ffmpeg` CLI → 把 imageio 的静态 ffmpeg 软链到 venv/bin。
8. G2PWModel 官方下载链接 404 → 用 HF 镜像仓库替代。

### 推理命令

```bash
source /home/developer/Ascend/cann-9.0.0/set_env.sh
source /mnt/workspace/gptsovits-venv/bin/activate
export LD_PRELOAD=$(ls /mnt/workspace/gptsovits-venv/lib/python3.11/site-packages/sentencepiece.libs/libgomp-*.so*)
cd /mnt/workspace/GPT-SoVITS
python GPT_SoVITS/inference_cli.py \
  --gpt_model "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt" \
  --sovits_model "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth" \
  --ref_audio /mnt/workspace/demo_src/ref_speaker_a8.wav \
  --ref_text /mnt/workspace/demo_src/ref_speaker_a8.txt \
  --ref_language 中文 \
  --target_text /mnt/workspace/demo_src/target_speaker_a.txt \
  --target_language 中文 \
  --output_path /mnt/workspace/gptsovits_demo
# 输出：/mnt/workspace/gptsovits_demo/output.wav
```

### 实测结果

| 项 | 结果 |
|---|---|
| 参考音频 | speaker_a清唱 8 秒（Whisper 转写参考文本） |
| 合成文本 | "你好呀，我是speaker_a。今天想用我自己的声音，和你说一声晚安。" |
| 输出 | `gptsovits_demo/output.wav`（约 10 秒） |
| 声纹相似度 | 输出 0.754 vs speaker_a参考（该 8 秒参考片段本身 0.795） |
| 耗时 | CPU：模型加载 ~40s + 合成 ~30s（约 8 it/s） |

> 零样本（未微调）只能做到"方向像"；要更像需要在该数据集上微调 s1/s2，
> 或换 v3/v4 权重。这一步跑通说明**不需要任何别人的声音**就能合成她的音色。

---

## 9. GPT-SoVITS 微调（2026-11-08 深夜，NPU + CPU）

### 9.1 数据预处理（GPT-SoVITS 格式）

复用 RVC 的 207 段 3.7s 切片（`rvc-src/logs/speaker_a/0_gt_wavs`）：

| 步骤 | 结果 |
|---|---|
| ASR | transformers Whisper-small 逐条转写 → `asr.list`（`wav|歌手|ZH|文本`）207 条 |
| 1a 文本+BERT | 204 条 → `2-name2text.txt`（wav/phones/word2ph/text） |
| 1b HuBERT+wav32k | 207 个 `4-cnhubert/*.pt` + `5-wav32k/*.wav` |
| 1c 语义 token | 207 条 → `6-name2semantic.tsv` |

**关键点**：`GPT_SoVITS/prepare_datasets/*.py` 靠**环境变量**传参（webui 用 os.environ 注入），
并且必须 `export PYTHONPATH="$PWD/GPT_SoVITS:$PWD"`，否则 `text`/`feature_extractor`/`utils` 导入失败。

### 9.2 s2（SoVITS）微调 —— NPU

`s2_train.py` 非 CUDA 分支本来就是 `.to(device)`，只需把全局 `device="cpu"` 改成 `npu:0`。

| 补丁 | 内容 |
|---|---|
| 设备 | `device = "npu:0"`（torch_npu 可用时） |
| DataLoader | `num_workers=0`、`pin_memory=False`、`persistent_workers=False`（64MB `/dev/shm`） |
| 权重加载 | `torch.load(..., weights_only=False)`（torch 2.7 默认拦截） |
| matplotlib | `tostring_rgb()` → `buffer_rgba()`（matplotlib ≥3.10 移除） |

配置：`fp16_run=false`、`batch_size=4`、`epochs=8`、`save_every_epoch=2`、预训练 s2G/s2D。
实测：51 步/epoch、约 1.2s/步，~65s/epoch；权重 `SoVITS_weights/speaker_a_e2/e4/e6/e8_s*.pth`。

### 9.3 s1（GPT）微调 —— CPU（PyTorch Lightning）

s1_train 走 `Trainer(accelerator="cpu")`（NPU 无 lightning 插件），配置 `s1longer-v2.yaml`：
`precision=32`、`batch_size=4`、`epochs=4`。补丁：`torch.load(..., weights_only=False)`。
实测：204 条中过滤后 121 条有效（phoneme/sec 3–25），31 步/epoch；e2/e4 权重 `GPT_weights/speaker_a-e2/e4.ckpt`。

### 9.4 推理对比（WavLM 声纹相似度，越接近 1 越像）

| 版本 | 相似度 |
|---|---|
| 零样本（未微调） | 0.754 |
| 微调 s2（e8） | 0.855 |
| **微调 s1(e2) + s2(e8)** | **0.920** |
| 参考片段本身（8s 清唱） | 0.795 |
| 微调 s1(e4) + s2(e8) | 0.733（e4 提前 EOS、输出截断） |

结论：
1. 两个模块微调都有效，s1+s2 组合最佳（0.920，高于参考片段本身）；
2. s1 小数据下**训练 2 epoch 足够，4 epoch 开始过拟合**（提前 EOS/截断），数据更多时再增加轮数；
3. 最终交付：`gptsovits_demo_best/output.wav`（e2 GPT + e8 SoVITS）。

### 9.5 复现命令（云端）

```bash
bash training/gptsovits_prep2.sh          # 数据预处理
bash training/gptsovits_launch_s2.sh      # s2 训练（NPU，8 epoch）
bash training/gptsovits_launch_s1.sh      # s1 训练（CPU，4 epoch）
bash training/gptsovits_infer_best.sh     # 用 e2 GPT + e8 SoVITS 合成
bash training/run_check_gptsovits.sh      # 声纹相似度对比
```

---

## 10. 更多数据重训（2026-11-13，32 段 / 17.7 分钟）

新下载 `speaker_a` 目录：32 个视频 / 137 MB / **17.7 分钟**（旧数据 21 个 / 11.7 分钟），
干声指标一致（低频能量 0.031、立体声宽度 0.037）。

| 步骤 | 结果 |
|---|---|
| 去重抽取 | 32 条 48 kHz 单声道 WAV（同名文件内容不同，按不同录音保留） |
| 切片 | 312 段（RVC preprocess 3.7s） |
| 预处理 | ASR 312 条 / BERT 308 / HuBERT+wav32k 312 / 语义 token 312 |
| s2 训练 | **从上一版 e8 继续**，NPU 10 epoch（77 步/epoch，~95s/epoch）→ `speaker_a_v2_e2..e10` |
| s1 训练 | **从上一版 e2 继续**，CPU 2 epoch → `speaker_a_v2-e1/e2.ckpt` |

### A/B 对比（同一句、同一参考）

| 组合 | 声纹相似度 | 输出时长 |
|---|---|---|
| 旧 s1(e2)+旧 s2(e8) | **0.911** | 10.0 s（完整） |
| 新 s1(e1)+新 s2(e10) | **0.904** | 6.5 s（完整） |
| 新 s1(e2)+新 s2(e10) | 0.885 | 4.4 s（偏短） |
| 旧 s1(e2)+新 s2(e10) | 0.869 | 6.0 s |
| 参考片段本身 | 0.795 | — |

结论：
1. 新数据模型与旧模型**同一水平（≈0.90）**，差异在单句方差范围内，但更多数据对泛化更有利；
2. s1 继续训练 2 epoch 已接近过拟合边界（e2 输出变短），e1 更稳；
3. 日常使用建议：新 s1(e1) + 新 s2(e10)（泛化更好），或旧 s1(e2) + 旧 s2(e8)（当前单句最高分）。

---

## 11. 大数据集（71 分钟）重训（2026-11-14）

`speaker_a` 扩充到 199 个文件后，清洗出 **128 个干净媒体文件 / 71.1 分钟**，
重跑全流程（脚本 `training/big_pipeline.sh`）：

| 步骤 | 结果 |
|---|---|
| 切片 | **1245 段**（48 kHz / 3.7 s） |
| ASR | faster-whisper small（int8，~1.9x 实时），45 分钟 |
| 特征 | BERT 995 / HuBERT 998 / wav32k 998 / 语义 token 995 |
| s1（GPT） | CPU 1 epoch → `GPT_weights/speaker_a_v3-e1.ckpt` |
| s2（SoVITS） | NPU 6 epoch → `SoVITS_weights/speaker_a_v3_e2..e6_s*.pth` |

### A/B 结果（同一句、20 段参考，相似度）

| 组合 | 相似度 | 输出时长 |
|---|---|---|
| 旧 17.7min 最佳（旧 GPT e2 + 旧 s2 e8） | **0.938** | 10.0 s |
| 旧 GPT(e2) + **大数据 s2(e6)** | **0.896** | 6.6 s |
| speaker_a_v2 GPT(e1) + 大数据 s2(e6) | 0.867 | 7.2 s |
| 大数据 GPT(e1) + 大数据 s2(e6) | 0.774 | 2.9 s（严重截断） |
| 大数据 GPT(e1) + 旧 s2(e10) | 0.765 | 5.4 s |
| 参考片段本身 | 0.811 | — |

结论：
1. **大数据 s2 可用**（0.896），但与旧 s2 的差异仍在单句方差内；
2. **大数据 GPT 只训 1 epoch 明显欠拟合**（输出截断、相似度降到 0.77），需要 2–3 epoch；
3. 当前最佳仍是旧 17.7min 组合（0.938）。

### 本次踩坑

- s2 配置漏写 `train.gpu_numbers` → `HParams has no attribute 'gpu_numbers'`，训练直接失败；
- 流水线里 `python ... | tail` 会把失败吞成 0 退出码，导致后续步骤继续跑（应加 `set -o pipefail`）；
- 修复后 s2 重跑正常（125 步/epoch，~3 分钟/epoch）。

---

## 12. 大数据 GPT 续训尝试（2026-11-14 下午）

在 `speaker_a_v3-e1` 基础上**再训 2 epoch**（batch 8、precision 32、沿用 s1longer-v2 的 lr=0.01）：

| 组合 | 相似度 | 输出时长 |
|---|---|---|
| 旧 17.7min 最佳 | **0.938** | 10.0 s |
| 旧 GPT(e2) + 大数据 s2(e6) | **0.896** | 6.6 s |
| speaker_a_v2 GPT(e1) + 大数据 s2(e6) | 0.867 | 7.2 s |
| speaker_a_v3b GPT(e2) + 大数据 s2(e6) | 0.789 | 3.9 s |
| speaker_a_v3b GPT(e1) + 大数据 s2(e6) | 0.756 | 3.8 s |
| speaker_a_v3b GPT(e2) + speaker_a_v2 s2(e10) | 0.632 | 2.6 s |
| 参考片段本身 | 0.811 | — |

结论：

1. **大数据 GPT 续训不稳定**：输出明显偏短（提前 EOS/截断），相似度反而下降；
2. 可能原因：ASR 文本质量参差（faster-whisper 对长切片）、lr=0.01 对续训偏高、有效样本 995 条但仍不足；
3. 目前排名：**旧 17.7min 组合（0.938）> 旧 GPT + 大数据 s2（0.896）> speaker_a_v2 GPT + 大数据 s2（0.867）**；
4. 若继续优化：建议 **lr 降到 0.002 左右、训 3–4 epoch**，并优先清洗 ASR 文本（按歌词人工校对），而不是继续加原始数据。







