# 面向昇腾 NPU 的个性化歌声与语音克隆系统：实现、数据质量与实证研究

> 技术报告（去隐私化版本）
> 日期：2026-11-14
> 说明：本报告不包含说话人姓名、账号、视频标题、文件名、链接、本机路径或云实例标识。文中说话人以 **说话人 A（Subject A）** 代称，数据集以 **D1 / D2 / D3** 编号。

---

## 摘要

本文报告一套在华为昇腾（Ascend）910B3 NPU 上完成端到端训练与推理的个性化歌声/语音克隆系统。系统包含两条技术路线：（1）基于 RVC 的音色转换（voice conversion）；（2）基于 GPT-SoVITS 的两阶段语音合成（AR 语言模型 s1 + VITS 声学模型 s2）。工作重点解决三类问题：

1. **异构算力适配**：PyTorch 2.7.1 + torch_npu 2.7.1.post4 在 aarch64 容器中的算子、依赖与权重加载兼容性；
2. **小样本数据工程**：对短视频来源的干声素材进行去重、质量审计与切片，构建可训练数据集；
3. **数据规模、标注质量与合成相似度之间的关系**：通过多组对照实验，量化“更多数据”和“更好标注/更稳训练”对音色与韵律的差异化影响。

实验以 WavLM-XVector 声纹余弦相似度作为主客观混合指标。在 D1（11.7 分钟）上，GPT-SoVITS 微调后相似度由零样本的 **0.754** 提升至 **0.920**（扩增参考集口径下 0.938）；RVC 模型在自转换任务上达到 **0.968**。将数据扩至 D3（71.1 分钟）后，声学模型（s2）保持可用（0.896），但语言模型（s1）出现提前 EOS/截断，相似度回落至 0.756–0.789，说明**对韵律建模而言，标注质量与训练稳定性比数据总量更关键**。进一步用 ASR 回读发现**声纹相似度与可懂度可能严重背离**：相似度最高的模型反而咬字含糊，而“稳定 GPT + 微调 s2”的组合在保持可懂的同时达到 0.896–0.855。据此推荐分模块训练与双指标评估。本文进一步给出可复现的适配补丁、超参数建议与隐私保护流程。

**关键词**：语音克隆；歌声合成；昇腾 NPU；RVC；GPT-SoVITS；数据质量；隐私保护

---

## Abstract

We present an end-to-end system for personalized singing/speech voice cloning trained and evaluated on a Huawei Ascend 910B3 NPU. Two routes are implemented: RVC-based timbre conversion and GPT-SoVITS two-stage synthesis (an autoregressive language model s1 plus a VITS acoustic model s2). The work addresses heterogeneous-accelerator adaptation (PyTorch 2.7.1 with torch_npu on aarch64), small-data engineering (deduplication, quality auditing and slicing of dry vocal clips), and the relationship between data scale, label quality and speaker similarity.

Using WavLM-XVector cosine similarity as the primary metric, GPT-SoVITS fine-tuning raises similarity from **0.754** (zero-shot) to **0.920** on an 11.7-minute dataset (0.938 under an expanded reference set). The RVC model reaches **0.968** on self-conversion. Scaling the dataset to 71.1 minutes keeps the acoustic model usable (0.896) but destabilizes the language model (early EOS / truncated outputs), dropping similarity to 0.756–0.789. This indicates that for prosody modeling, **label quality and training stability matter more than raw data volume**. ASR read-back further reveals a **divergence between speaker similarity and intelligibility**: the highest-similarity model is the least intelligible, whereas "stable GPT + fine-tuned s2" stays intelligible at 0.896–0.855 similarity. We therefore recommend modular training and dual-metric evaluation, and release the adaptation notes, hyper-parameter guidance and a de-identification protocol.

**Keywords**: voice cloning; singing synthesis; Ascend NPU; RVC; GPT-SoVITS; data quality; privacy

---

## 1 引言

个性化音色合成在内容创作、无障碍播报与虚拟人交互中具有广泛应用，但通常依赖大显存 GPU。本文研究在单卡昇腾 910B3（64 GB HBM）上完成“数据准备 → 微调 → 推理”全流程的可行性，并回答两个实践问题：

- **RQ1**：在昇腾 NPU 上，主流开源语音克隆框架（RVC、GPT-SoVITS）需要哪些最小改动才能稳定训练与推理？
- **RQ2**：在单人、小样本、来源为消费级短视频的场景下，数据规模、ASR 标注质量与训练策略分别如何影响合成相似度？

本文贡献：

1. 给出昇腾 NPU 上两套框架的可复现适配方案与踩坑清单；
2. 提出一套面向“干声短视频”的质量审计—清洗—切片流程；
3. 通过 10 余组对照实验，区分“音色模型（s2）”与“韵律模型（s1）”对数据的不同敏感度，指出**数据更多不一定更好**的具体机制（标注噪声放大与续训漂移）。

---

## 2 相关工作

- **RVC（Retrieval-based Voice Conversion）**：以 HuBERT 特征 + 检索索引 + VITS 声码器实现音色转换；训练目标为生成对抗 + 梅尔重建 + 对比损失，适合“保留源音频内容、替换音色”的场景。
- **GPT-SoVITS**：两阶段 TTS/克隆框架。s1 为自回归语义语言模型（文本 → 语义 token），s2 为 VITS 类声学模型（语义 + 文本 → 音频）。零样本推理仅需数秒参考音频；微调可显著提升相似度与稳定性。
- **声纹相似度评估**：采用 WavLM-XVector（说话人确认模型）提取嵌入，计算与参考语音质心的余弦相似度，作为可复现的客观代理指标。
- **昇腾生态**：CANN 9.0.0 + torch_npu 提供 PyTorch 算子后端；部分 CUDA 专属或低精度算子（如 fp32 GRU）在 NPU 上不可用，需要框架级绕行。

---

## 3 系统实现

### 3.1 硬件与软件栈

| 组件 | 配置 |
|---|---|
| 加速卡 | Ascend 910B3 × 1，HBM 64 GB |
| 操作系统 | aarch64 Linux 容器 |
| 深度学习 | PyTorch 2.7.1 (CPU 构建) + torch_npu 2.7.1.post4 |
| 加速库 | CANN 9.0.0 |
| 关键依赖 | transformers 4.43–4.49、torchaudio 2.7.1、librosa、faiss-cpu、onnxruntime、faster-whisper |
| 训练框架 | RVC（去 fairseq 版本）、GPT-SoVITS（main 分支） |

### 3.2 昇腾适配与兼容性修复

| 问题 | 现象 | 解决 |
|---|---|---|
| fp32 GRU 不支持 | `DynamicGRUV2 ... DT_FLOAT not supported` | RMVPE 音高提取改 CPU；或使用 fp16 |
| torchaudio ABI 不匹配 | `_torchaudio... undefined symbol` | 固定 `torchaudio==2.7.1`（aarch64 wheel） |
| libgomp 静态 TLS | `cannot allocate memory in static TLS block` | 预加载 sentencepiece/sklearn 自带 libgomp，或系统 libgomp |
| 权重加载拦截 | torch 2.6+ 默认 `weights_only=True` | 训练/推理入口显式 `weights_only=False`（仅本地可信权重） |
| 容器共享内存小 | DataLoader `Bus error`（`/dev/shm` 64 MB） | `num_workers=0`、关闭 `prefetch_factor` |
| matplotlib 3.10 移除 API | `FigureCanvasAgg.tostring_rgb` | 改为 `buffer_rgba()` |
| 预处理脚本依赖环境变量 | `ModuleNotFoundError: text/feature_extractor` | 注入 `PYTHONPATH` 与 webui 同款环境变量 |
| 旧版下载链接失效 | G2PW 模型 404 | 使用 HF 镜像仓库替代 |

### 3.3 数据流水线

1. **抽取**：视频 → 48 kHz 单声道 WAV；
2. **去重**：SHA-1 内容哈希；
3. **质量审计**：时长、RMS、峰值/削波率、静音比例、低频能量比（伴奏代理）、频谱质心/平坦度、SNR 估计；
4. **清洗**：剔除 vlog/非歌唱、过轻、静音、疑似伴奏片段；
5. **切片**：3.7 s 定长切片（RVC 预处理）；
6. **标注**：Whisper/faster-whisper 转写（含质量模式与置信度过滤）；
7. **特征**：BERT 文本特征、HuBERT 语义特征、32 kHz 波形、s2 语义 token；
8. **训练**：s2（NPU）/ s1（CPU，PyTorch Lightning）。

---

## 4 数据与隐私保护

### 4.1 数据来源与伦理

数据在说话人知情同意的前提下采集，仅用于本研究的个性化音色实验；模型与音频不对外发布。语音克隆属于高敏感技术，任何对外分享（模型、翻唱、合成音频）均需再次取得说话人书面授权，并遵守当地关于深度合成与人格权的法规。

### 4.2 去标识化流程

| 处理 | 措施 |
|---|---|
| 身份 | 说话人统一记作“说话人 A / Subject A” |
| 内容 | 不出现歌名、视频标题、账号名、话题标签 |
| 文件 | 用序号（`0001.wav`…）与数据集编号（D1–D3）代替原始文件名 |
| 元数据 | 删除下载时间、平台 ID、URL、地理位置 |
| 路径 | 报告中不出现本机路径、用户名、云实例 ID |
| 示例 | 合成目标语句使用通用礼貌用语，不引用原始歌词 |
| 可复原性 | 不提供哈希、时长指纹等可用于反查的细粒度组合信息 |

### 4.3 质量审计结果（D3）

| 指标 | 数值 |
|---|---|
| 媒体文件 | 142 个（另有图片/图文，已忽略） |
| 重复文件 | 0 |
| 总时长 | 74.6 分钟 |
| 时长分布 | 最短 1.8 s / 中位 32.1 s / 最长 84.8 s |
| 音量范围 | −120 ~ −7.8 dBFS（均值 −16.1） |
| 削波 > 0.1% | 15 个（均 < 0.5%） |
| 静音 > 30% | 5 个 |
| 低频能量比 > 0.10 | 8 个（伴奏风险） |
| 清洗后 | **128 个文件 / 71.1 分钟** |

被剔除的 14 个文件（3.5 分钟）包括：vlog/非歌唱 4 个、音量过低 6 个、疑似伴奏 6 个（存在重叠）、空文件 3 个。

---

## 5 实验设置

- **数据集**：D1 = 11.7 分钟（207 切片）；D2 = 17.7 分钟（312 切片）；D3 = 71.1 分钟（1245 切片，质量过滤后约 900 条用于文本建模）。
- **评估**：WavLM-XVector 说话人嵌入；参考集为多段参考音频的归一化质心；对同一句目标文本合成后计算余弦相似度。指标越接近 1 越像。
- **训练**：
  - RVC：批量 4，fp32，NPU，30→100 epoch；
  - GPT-SoVITS s2：批量 4/8，fp32，NPU，6–10 epoch；
  - GPT-SoVITS s1：PyTorch Lightning CPU，批量 4/8，precision 32，1–4 epoch。
- **推理**：参考音频 3–10 秒（8 秒），目标文本为固定通用句；s2 推理 CPU，端到端约 1–2 分钟/句。

---

## 6 实验结果

### 6.1 RVC 音色转换

| 设置 | 相似度 | 说明 |
|---|---|---|
| 30 epoch 自转换 | 0.964 | 说话人 A 的声音 → 模型 |
| 100 epoch 自转换 | 0.968 | 增加训练轮数 |
| 30 epoch 跨人声（3 种源） | 0.818 / 0.848 / 0.876 | 保留源咬字/气息，存在“混合感” |
| 中性合成源（e30） | 0.372 | 源本身非自然人声，指标偏低 |
| 中性合成源（e100 + 检索 index） | 0.586 | index 提升相似度 |

RVC 训练速度：100 epoch 约 45–50 s/epoch（D1）。结论：RVC 对音色建模有效，但跨人声转换会保留源特征；适合“同人不同曲”的转换，不适合从零合成新内容。

### 6.2 GPT-SoVITS 零样本与微调

| 数据 | 方法 | 相似度 | 输出时长 |
|---|---|---|---|
| D1 | 零样本（预训练） | 0.754 | 完整 |
| D1 | 仅微调 s2 | 0.855 | 完整 |
| D1 | 微调 s1(e2) + s2(e8) | **0.920** | 完整 |
| D1 | 同组合（扩增参考集） | **0.938** | 10.0 s |
| D2 | s1(e1) + s2(e10) | 0.904 / 0.893* | 6.5 s |
| D3 | 旧 s1(e2) + D3 s2(e6) | **0.896** | 6.6 s |
| D3 | D3 s1(e1) + D3 s2(e6) | 0.774 | 2.9 s（截断） |
| D3 | D3 s1 续训 2 epoch（e1/e2） | 0.756 / 0.789 | 3.8–3.9 s（截断） |
| D3 | 稳定版 s1(e1) + D3 s2 | 0.867 | 7.2 s |
| D3 | 低学习率 s1(e1/e2/e3) + D3 s2 | 0.826 / 0.758 / **0.906** | 4.4 / 1.6 / 3.5 s |

\* 括号内为扩增参考集口径。低学习率组：lr=0.002、warmup=100、batch=8、3 epoch，文本来自质量模式 ASR（952 条过滤后 948 条，最终有效训练样本 640 条）。

### 6.3 相似度 ≠ 可懂度：ASR 回读分析

仅用声纹相似度无法反映合成语音是否“听得懂”。我们用语音识别回读各版本输出（目标语句为固定通用句），结果如下：

| 版本 | 时长 | ASR 回读 | 可懂度 |
|---|---|---|---|
| 零样本（未微调） | 7.90 s | “今天想用我自己的声音和你说一声晚安” | **可懂**（基本完整） |
| 仅微调 s2（D1） | 7.82 s | “今天想用我自己的生意和你说一声晚安” | **可懂**（个别同音字） |
| D1 s1+s2（相似度 0.938） | 6.22 s | “晴天下阳我四角一射一射一射” | **乱码** |
| D2 s1+s2（0.904） | 6.58 s | “今天教育我亲戚的生意上” | **部分乱码** |
| D3 s2 + 稳定 GPT（0.896） | 6.66 s | “今天想要我自己和你说一说晚安” | **可懂**（缺前半句） |
| D3 低 LR s1 e3（0.906） | 3.46 s | “才能救自己的睡着” | 截断/乱码 |

结论：

1. **s1（GPT/韵律）微调是风险来源**：在“以唱歌为主、文本标注不完美”的数据上微调 s1，会破坏文本—韵律对齐，导致咬字含糊甚至输出截断；相似度可能不降反升，但可懂度显著下降。
2. **仅微调 s2（音色）最安全**：零样本与 s2-only 输出均可懂，适合作为默认配置。
3. **推荐组合**：D3 的 s2（音色更稳）+ 稳定代际的 GPT（s1 不再继续微调）——相似度 0.896，且回读基本可懂。
4. 评估协议应同时报告**声纹相似度**与**ASR 可懂度**（或人工 MOS），单看相似度会选出“像但听不清”的模型。

### 6.4 关键发现：数据更多反而更差？

D3 的 s2（音色）在旧 s1 配合下达到 0.896，证明**声学模型受益于更大的数据量**。但 D3 的 s1（GPT）无论只训 1 epoch 还是续训 2 epoch，都出现**提前 EOS/输出截断**（2.6–3.9 s，正常 6–10 s），相似度降至 0.756–0.789。原因分析：

1. **ASR 标注质量下降**：D3 使用 int8 + beam=1 的快速转写，995 条文本中错字、半句比例升高；GPT 学习“文本 → 韵律”，标注噪声被放大。
2. **数据异构**：D3 包含不同录音环境、歌曲与演唱方式，音色模型可受益，但韵律模型会学到更宽且不一致的分布，倾向提前结束。
3. **续训漂移**：D3 的 GPT 是从已多次微调的模型继续训练（第 4 代），在高学习率（0.01）下误差累积。
4. **学习率调度**：`warmup_steps=2000` 远大于实际训练步数（125–375 步），有效学习率长期处于预热区，训练不充分。

### 6.5 训练成本

| 阶段 | 数据 | 设备 | 成本 |
|---|---|---|---|
| RVC 100 epoch | D1 207 切片 | NPU | ~80 分钟 |
| s2 微调 10 epoch | D2 312 切片 | NPU | ~55 分钟 |
| s2 微调 6 epoch | D3 998 切片 | NPU | ~18 分钟 |
| s1 微调 1–2 epoch | D2/D3 | CPU | ~12 分钟/epoch |
| 零样本/微调推理 | 单句 | CPU | ~1–2 分钟 |

---

## 7 讨论

1. **“多”与“好”的取舍**：对音色（s2），数据量总体有益；对韵律（s1），标注一致性与训练稳定性优先。若数据量扩大但标注质量下降，应优先修复标注，而非继续堆数据。
2. **相似度与可懂度的分离**：实验显示两者可以严重背离——D1 的 s1+s2 模型声纹相似度最高（0.938），但 ASR 回读为乱码；仅微调 s2 的版本相似度较低（0.855），却可懂。**评估协议必须同时报告两者**，否则会选出“像但听不清”的模型。
3. **训练策略**：小样本微调应使用更低学习率（0.002 量级）、缩短 warmup，并按 epoch 试听；出现截断即停止。
4. **续训链条**：避免“多代连续微调”，每轮尽量从原始预训练权重出发或保留上一代稳定快照。
5. **模块分工**：音色（s2）可以持续吸收更多数据；韵律/咬字（s1）应保持在少量高质量、经校对的样本上，或干脆不微调（零样本 GPT + 微调 s2 即可获得可懂输出）。
6. **隐私**：去标识化应贯穿数据、日志与报告；训练数据与模型均应限制访问。

---

## 8 局限性

- 单一说话人、单一语言（普通话）、以清唱/翻唱为主，泛化性有限；
- ASR 标注未经逐句人工校对，存在残余错误；
- 声纹相似度只反映说话人身份，不反映韵律自然度与音质；
- 未评估长文本、跨风格与实时推理；
- 数据规模仍远小于商业级语音克隆系统。

---

## 9 结论

本文验证了在单卡昇腾 910B3 上完成 RVC 与 GPT-SoVITS 训练/推理的可行性，并给出完整的兼容性适配与数据清洗方案。实验表明：GPT-SoVITS 微调可将声纹相似度从 0.754（零样本）提升到 0.920（D1），RVC 自转换达到 0.968；将数据扩展到 71 分钟后，声学模型（s2）保持可用（0.896），但语言模型（s1）的连续微调出现提前结束与咬字退化。

更重要的是，ASR 回读揭示了**相似度与可懂度的背离**：相似度最高的模型反而不可懂，而“零样本 GPT + 微调 s2”的配置在相似度 0.896–0.855 的同时保持可懂。因此对个性化语音克隆而言，**“高质量标注 + 稳定训练 + 分模块控制”比“更多原始数据”更能决定最终可用性**；评估应同时报告声纹相似度与可懂度，避免单一指标误导。

---

## 参考文献

1. RVC: Retrieval-based Voice Conversion WebUI. https://github.com/RVC-Boss/Retrieval-based-Voice-Conversion-WebUI
2. GPT-SoVITS. https://github.com/RVC-Boss/GPT-SoVITS
3. Hsu W.-N., et al. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. IEEE/ACM TASLP, 2021.
4. Chen S., et al. WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing. IEEE JSTSP, 2022.
5. Radford A., et al. Robust Speech Recognition via Large-Scale Weak Supervision (Whisper). ICML, 2023.
6. FAISS: A Library for Efficient Similarity Search. https://github.com/facebookresearch/faiss
7. Huawei Ascend CANN Documentation. https://www.hiascend.com/
8. torch_npu. https://gitee.com/ascend/pytorch
9. RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music. 2023.

---

> 附：本报告所有数值来自本地/云端实验记录；说话人、文件与路径信息均已去标识化。
