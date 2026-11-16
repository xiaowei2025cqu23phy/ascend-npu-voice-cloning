# 脚本清单

所有脚本中的路径均为占位符/示例（`<PROJECT_ROOT>`、`<DATA_DIR>`、`/workspace`、`GPT-SoVITS/`、`rvc-src/`），
运行前请按自己的环境调整。脚本依赖对应上游仓库的目录结构。

## 1. NPU 兼容补丁

| 脚本 | 作用 |
|---|---|
| `rvc_npu_patch_infer.py` | RVC 推理：parselmouth 惰性导入；RMVPE 在 NPU 上退回 CPU |
| `rvc_npu_patch_dataloader.py` | RVC 训练 DataLoader：小 `/dev/shm` 下改 `num_workers=0` |
| `rvc_npu_patch_ckpt.py` | RVC 训练 checkpoint：`torch.load(weights_only=False)` |
| `gptsovits_patch_s2.py` | GPT-SoVITS s2：`device=npu:0`、DataLoader、checkpoint、matplotlib 兼容 |
| `gptsovits_patch_s1.py` | GPT-SoVITS s1：`torch.load(weights_only=False)` |
| `gptsovits_patch_infer.py` | GPT-SoVITS 推理：`torch.load(weights_only=False)` |
| `rvc_infer_launcher.py` | RVC CLI 启动器（放宽可信本地权重加载） |

> 说明：RVC 的 `configs/config.py`、`train/train.py`、`extract_f0.py`、`extract_hubert_feature.py`
> 的设备适配补丁未随仓库提供，行为见 `docs/npu-engineering-notes.md`。

## 2. 数据质量与审计

| 脚本 | 作用 |
|---|---|
| `audit_local_dataset.py` | 单文件夹音频体检（时长/响度/静音/低频/立体声宽度） |
| `audit_large_dataset.py` | 大批量媒体质量审计（SHA-1 去重 + 指标 + 汇总） |
| `quality_report.py` | WavLM 声纹一致性 + 指标 + 审计图 |
| `filter_quality.py` | 按阈值过滤并输出干净文件列表 |
| `plot_quality.py` | 生成质量图表 |
| `dedupe_extract.py` | 去重并抽取 48 kHz 单声道 WAV |
| `extract_clean.py` | 按干净文件列表批量抽取 32 kHz WAV |

## 3. ASR

| 脚本 | 作用 |
|---|---|
| `gptsovits_build_asr_fw.py` | faster-whisper 快速转写（GPT-SoVITS 列表格式） |
| `re_asr_quality.py` | 质量模式 ASR（beam=5 + VAD，保存 avg_logprob） |
| `test_fw_quality.sh` | 测量质量模式 ASR 速度 |

## 4. 训练与推理流水线

| 脚本 | 作用 |
|---|---|
| `gptsovits_prep3.sh` | GPT-SoVITS 数据预处理（ASR → BERT → HuBERT → 语义 token） |
| `big_pipeline.sh` | 端到端示例：解包 → 切片 → ASR → 特征 → s2(NPU) → s1(CPU) → 推理 |
| `gptsovits_make_s1_config2.py` | 生成 s1 训练配置（低学习率/续训） |
| `gptsovits_make_s2_config2.py` | 生成 s2 训练配置（NPU） |
| `make_filelist.py` | 生成 RVC 训练 filelist |

## 5. 评估

| 脚本 | 作用 |
|---|---|
| `check_gptsovits.py` | WavLM-XVector 声纹相似度（多参考质心） |
| `check_intelligibility.sh` | 用 ASR 回读检验合成语音可懂度 |

## 6. 未随仓库提供的早期工具

- 环境自检脚本（昇腾算子/训练冒烟探测）；
- 数据集切片工具（静音感知切片、缝隙拼接）。
两者行为与参数已记录在 `docs/npu-engineering-notes.md`，可据此重建。
