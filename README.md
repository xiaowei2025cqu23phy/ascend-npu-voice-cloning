# Ascend NPU 个性化语音克隆：研究代码与报告

> Research code and report: personalized voice cloning (RVC and GPT-SoVITS) on
> Huawei Ascend NPU (910B3 / CANN 9.0.0 / PyTorch 2.7.1 + torch_npu).

本仓库整理了一套在单卡昇腾 910B3 上完成「数据清洗 → 特征提取 → 微调 → 推理 → 评估」的
语音克隆实验材料，包含去隐私化的技术报告、数据质量审计工具、ASR 与训练/推理流水线脚本，
以及昇腾环境适配补丁。

**数据集、音频、模型权重、训练日志均不在本仓库中**（隐私与授权原因，见下）。

---

## 1. 主要内容

| 路径 | 说明 |
|---|---|
| [`docs/report.md`](docs/report.md) | 技术报告（Markdown，去隐私化） |
| [`docs/report.docx`](docs/report.docx) | 技术报告（Word 版） |
| [`docs/npu-engineering-notes.md`](docs/npu-engineering-notes.md) | 昇腾适配与踩坑记录（RVC / GPT-SoVITS） |
| [`docs/quality-audit.png`](docs/quality-audit.png) | 数据质量审计图（响度/伴奏风险/时长分布） |
| [`scripts/`](scripts/) | 数据审计、ASR、训练/推理、评估、兼容补丁脚本 |
| [`data/README.md`](data/README.md) | 数据集要求与清洗流程说明（数据不随仓库提供） |

## 2. 主要实验结果

| 任务 | 声纹相似度（WavLM-XVector） | 备注 |
|---|---|---|
| RVC 自转换（100 epoch） | **0.968** | 目标说话人声音 → 模型 |
| RVC 跨人声转换 | 0.818–0.876 | 保留源咬字/气息，有“混合感” |
| GPT-SoVITS 零样本 | 0.754 | 未微调，可懂 |
| GPT-SoVITS 仅微调 s2 | 0.855 | 可懂 |
| GPT-SoVITS 微调 s1+s2（11.7 min） | **0.920**（扩增参考集 0.938） | 相似度最高，但 ASR 回读不可懂 |
| GPT-SoVITS 大数据 s2 + 稳定 GPT（71 min） | **0.896** | 推荐：相似度与可懂度平衡 |

关键结论（详见报告）：

1. **音色模型（s2）受益于更多数据；韵律模型（s1）更吃标注质量与训练稳定性**；
2. s1 在“唱歌为主、ASR 标注不完美”的数据上继续微调，会导致咬字含糊、提前 EOS/截断；
3. **声纹相似度与可懂度可能严重背离**，评估应同时看两项指标；
4. 推荐配置：用大数据微调 s2，保持稳定的 GPT（不继续微调）。

## 3. 环境

- 硬件：Ascend 910B3 × 1，HBM 64 GB
- 软件：aarch64 Linux 容器，CANN 9.0.0，PyTorch 2.7.1（CPU 构建）+ torch_npu 2.7.1.post4
- 上游框架：
  - RVC（去 fairseq 版本）
  - GPT-SoVITS（main 分支）

## 4. 复现概要

1. 克隆上游 RVC / GPT-SoVITS 仓库；
2. 应用 [`scripts/`](scripts/) 中的昇腾兼容补丁（`rvc_npu_patch_*.py`、`gptsovits_patch_*.py`）；
3. 准备**你有权使用**的单人干声数据（格式与清洗见 [`data/README.md`](data/README.md)）；
4. 数据质量审计：`audit_large_dataset.py` → `quality_report.py` → `filter_quality.py`；
5. 切片、ASR（`re_asr_quality.py` / `gptsovits_build_asr_fw.py`）、特征提取；
6. 训练 s2（NPU）与 s1（CPU），参考 `scripts/big_pipeline.sh`；
7. 推理与评估：`check_gptsovits.py`（声纹相似度）、`check_intelligibility.sh`（ASR 可懂度）。

> 脚本中的路径为占位符（`<PROJECT_ROOT>`、`<DATA_DIR>`、`/workspace` 等），运行前请按你的环境替换。

## 5. 隐私与伦理

- 本仓库**不包含**任何说话人音频、视频、数据集、模型权重或训练日志；
- 报告中的说话人已匿名（“说话人 A”），数据集以编号表示，不含姓名、账号、歌名、文件名、URL、本机路径或云实例标识；
- 语音克隆属于敏感技术，训练与使用需取得说话人明确授权，并遵守当地法律法规；禁止用于冒充、诈骗、侵权等用途；
- 若你基于本项目训练模型，请自行承担数据授权与合规责任。

## 6. 许可

代码与文档以 MIT License 发布；上游 RVC / GPT-SoVITS / HuBERT / WavLM / Whisper 等遵循其各自许可证。

## 7. 参考

- RVC: https://github.com/RVC-Boss/Retrieval-based-Voice-Conversion-WebUI
- GPT-SoVITS: https://github.com/RVC-Boss/GPT-SoVITS
- Huawei Ascend CANN: https://www.hiascend.com/
- torch_npu: https://gitee.com/ascend/pytorch
