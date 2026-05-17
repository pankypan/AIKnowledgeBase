# 应用型AI算法工程师修炼路线

> **适用人群**：Python 高级开发工程师，熟悉 Agent 开发，希望向算法方向延伸  
> **目标岗位**：应用型算法工程师 / 上下文工程算法工程师 / AI 算法开发工程师  
> **学习时长**：8-10 周（可业余投入）

---

## 一、为什么需要这份路线图

### 工业界"算法工程师"的真实画像

在大多数人的认知中，算法工程师 = 读论文 + 写论文 + 设计新算法。但在工业界，AI 相关岗位实际是一个从"研究"到"工程"的连续谱：

```
                        偏研究 ◄──────────────────────► 偏工程
                        人数少                           人数多

┌──────────────────────────────────────────────────────────────────────────┐
│  研究科学家 / Research Scientist                                          │
│  PhD，发顶会论文，设计新模型架构                                            │
│  占比：≈5%（大厂研究院、AI 创业公司核心团队、高校）                            │
├──────────────────────────────────────────────────────────────────────────┤
│  研究型算法工程师                                                          │
│  硕博，复现 + 改进论文，偶尔发论文                                           │
│  占比：≈10%（大厂核心算法团队）                                             │
├──────────────────────────────────────────────────────────────────────────┤
│  ⭐ 应用型算法工程师 ← 本路线图的目标                                       │
│  本硕均可，读论文 → 选型适配 → 落地优化 → 实验验证 → 业务交付                   │
│  占比：≈30%（所有 AI 公司都需要，岗位需求量大）                               │
├──────────────────────────────────────────────────────────────────────────┤
│  AI 开发工程师 / MLOps / AI Infra                                         │
│  工程为主，框架调用 + 系统搭建 + 模型部署 + 数据管线 + 业务落地                 │
│  占比：≈55%（需求量最大，与上层存在大量交叉）                                  │
└──────────────────────────────────────────────────────────────────────────┘

注：比例为估算，实际因公司类型和阶段不同而差异较大。
    大模型时代，第三层与第四层的边界正在快速模糊——
    越来越多岗位同时要求"懂算法原理"和"能工程落地"。
```

### 应用型算法工程师的真实日常

很多人以为算法工程师整天在读论文、推公式。实际上，基于招聘平台 JD 和一线从业者反馈，应用型算法工程师的时间分配更接近这样：

| 工作内容 | 占比 | 典型活动 |
|---------|------|---------|
| 方案选型与落地优化 | 35-40% | 读论文/开源方案 → 评估可行性 → 适配业务 → 调参调优 |
| 数据与特征工程 | 20-25% | 数据清洗、标注质量把控、构造训练/评测数据集 |
| 实验设计与效果验证 | 15-20% | 对比实验、消融分析、A/B 测试、撰写实验报告 |
| 工程化与部署交付 | 10-15% | 模型服务化、性能优化、与开发团队联调上线 |
| 跨团队沟通与汇报 | 5-10% | 向业务方解释算法方案、对齐需求、评审方案可行性 |
| 原创性研究与论文 | 0-5% | 锦上添花，多数团队不做硬性要求 |

> **关键认知**：工业界不缺写论文的人，缺的是能把论文变成生产力的人。
> 注意"数据工程"和"跨团队沟通"占据了近 1/3 的时间——这是很多转型者容易低估的部分。

### 真实 JD 长什么样

以下是典型的应用型算法岗位职责（综合多个真实招聘信息提炼）：

**核心职责（几乎所有 JD 都会提到）：**
- 负责特定方向的算法设计与开发（NLP / CV / 推荐 / 多模态，因业务而异）
- 跟踪前沿技术，评估并落地到业务场景
- 设计实验方案，通过数据验证算法效果

**高频要求（大部分 JD 会提到）：**
- 推动大模型（LLaMA、Qwen 等）在业务中的落地，包括微调、量化及部署
- 构建和优化数据管线，保障训练数据和评测数据的质量
- 主导项目全生命周期开发，从需求分析到量产交付

**加分项（部分 JD 会提到）：**
- 有开源项目或技术博客
- 有 Agent / RAG / 多模态等方向的实战经验
- 了解模型推理优化（vLLM、TensorRT 等）

**所有 JD 的共同模式：**

```
跟踪前沿 → 选型评估 → 适配落地 → 实验验证 → 数据驱动迭代
```

---

## 二、你的优势与需要补的短板

### 已有的优势（直接可用）

| 能力 | 对算法岗的价值 | 面试/工作中的体现 |
|------|-------------|----------------|
| Python 高级开发 | 代码质量远超多数算法研究者，生产级代码能力 | 手撕代码环节碾压、Code Review 能力强 |
| Agent 开发经验 | 理解 Agent 架构，能快速做算法验证和落地 | 当前最热门的算法落地方向之一 |
| 工程化能力 | 高并发、部署、监控——算法团队最稀缺的能力 | 能独立把模型从实验跑到上线 |
| 系统设计能力 | 能把算法模块集成到完整的生产系统中 | 系统设计面试环节加分项 |

> **你的差异化竞争力**：多数算法岗候选人的短板恰好是你的强项（工程能力）。
> 反过来，他们的强项（数学基础、论文经验）恰好是你需要补的。这是互补关系，不是劣势。

### 需要补的短板（本路线图重点）

| 短板 | 目标水位（够用即可，不必到科研级） | 优先级 |
|------|-------------------------------|-------|
| LLM 底层原理 | 能从矩阵运算层面解释 Attention、能画出 Transformer 数据流 | 🔴 高 |
| 算法优化方法论 | 掌握微调（LoRA/全量）、量化、检索优化等，能独立完成选型和调优 | 🔴 高 |
| 实验设计能力 | 会设计对比实验、消融实验、选择评估指标，能写出令人信服的实验报告 | 🔴 高 |
| 论文阅读能力 | 能快速提取核心思想并判断落地可行性——不是写论文，而是用论文 | 🟡 中 |
| 数据工程意识 | 理解数据质量对模型效果的决定性影响，能主导数据清洗和评测集构建 | 🟡 中 |

---


## 三、简历定位建议

### 你的标签

不要简单地写"算法工程师"或"开发工程师"，而是突出复合能力：

```
核心定位：AI 应用算法工程师（工程 + 算法双背景）

关键词：
  - LLM 应用算法优化（RAG / Agent / Memory）
  - 模型微调与部署（LoRA / 量化 / vLLM）
  - 生产级 AI 系统架构（高并发 / 高可用 / 可观测）
  - 开源贡献（GitHub 项目 + 技术博客）
```

### 你可以投的岗位类型

| 岗位名称 | 匹配度 | 说明 |
|---------|--------|------|
| AI/LLM 应用算法工程师 | ⭐⭐⭐⭐⭐ | 最精准匹配，路线图完成后直接投 |
| RAG/Agent 算法工程师 | ⭐⭐⭐⭐⭐ | 你的 Agent 经验是核心竞争力 |
| 上下文工程算法工程师 | ⭐⭐⭐⭐ | 新兴方向，岗位数量还在增长中 |
| 智能体算法开发 | ⭐⭐⭐⭐ | 偏落地的算法岗，工程能力加分 |
| 模型部署/推理优化工程师 | ⭐⭐⭐⭐ | 补完量化/推理优化知识后可投 |
| AI 开发工程师（算法方向） | ⭐⭐⭐ | 你的舒适区，但天花板和薪资可能偏低 |
| 模型算法工程师（研究型） | ⭐⭐ | 通常要求论文发表经历，短期不建议 |

---


## 四、详细学习计划

> **总体节奏**：10 周，每天 2-4 小时业余投入即可。每周 7 个学习单元，可根据实际情况灵活调整节奏。  
> **核心理念**：不追求从零推导，而是"理解原理 → 读懂论文 → 复现优化 → 实验验证 → 开源产出"。

### **第 1 周：Transformer 原理速通 + 论文阅读方法论**

> **学习内容:**
> - **Transformer 架构**: Encoder/Decoder、Self-Attention、Multi-Head Attention、Positional Encoding
> - **矩阵运算层面**: Q/K/V 的计算流程、Attention Score 的维度变化
> - **论文阅读方法论**: 三遍阅读法（摘要扫读 → 结构精读 → 复现验证）
> - **模型家族**: GPT、BERT、T5 的架构差异与适用场景
>
> **手撕系列:**
> - [ ] 用 PyTorch 手撕 Multi-Head Attention（不借助 nn.MultiheadAttention）
> - [ ] 用 Excel 或手算推导一次完整的 Attention 矩阵运算
> - [ ] 精读 Attention Is All You Need，输出 1 页核心要点笔记
>
> **面试预热:**
> - Q: 请从矩阵运算层面解释 Self-Attention 的计算过程。
> - Q: 为什么要除以 √d_k？不除会怎样？
>
> **解锁技能:**
> - 能在白板上画出 Transformer 架构并解释每个组件
> - 能从矩阵运算层面解释 Attention 机制（面试核心考点）
> - 掌握高效的论文阅读方法论

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
| ------ | ---------- | ---------- | ------- |
| 1 | Transformer 宏观理解 | 博客: [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)<br>可视化: [Interactive Transformer](https://poloclub.github.io/transformer-explainer/)<br>图解: [Transformer 算法原理图](https://github.com/changyeyu/LLM-RL-Visualized) | 理解 Encoder/Decoder 结构，画出完整架构图 |
| 2 | Self-Attention 矩阵运算 | 论文: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)<br>教程: [Transformer from scratch in Excel](https://www.youtube.com/watch?v=k_P-tprA6-Q)<br>详解: [Transformer 数学原理](https://kexue.fm/) | 逐步推导 Q/K/V 计算流程，理解维度变化 |
| 3 | 手撕 Multi-Head Attention | 教程: [Let's build GPT: from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY)<br>代码: [nanoGPT](https://github.com/karpathy/nanoGPT)<br>参考: [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | 纯 PyTorch 实现 Multi-Head Attention |
| 4 | GPT/BERT/T5 架构对比 | 书籍: [《大语言模型》](https://llmbook-zh.github.io/) 第1-3章<br>教程: [HuggingFace NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1) | 理解 Encoder-only / Decoder-only / Encoder-Decoder 的区别与适用场景 |
| 5 | 论文阅读方法论 | 指南: [How to Read a Paper (三遍阅读法)](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf)<br>工具: [Connected Papers](https://www.connectedpapers.com/)、[Semantic Scholar](https://www.semanticscholar.org/) | 掌握三遍阅读法，建立自己的论文笔记模板 |
| 6 | 精读 Attention Is All You Need | 论文: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)<br>解读: [论文逐段解读](https://nlp.seas.harvard.edu/annotated-transformer/) | 用三遍阅读法完成精读，输出 1 页核心要点笔记 |
| 7 | 周度总结与查漏补缺 | 视频: [State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A)<br>教程: [《动手学大模型 Dive into LLMs》](https://github.com/Lordog/dive-into-llms) | 回顾本周内容，确保能在白板上完整解释 Transformer |

---

### **第 2 周：LLM 训练全流程 + 核心论文精读**

> **学习内容:**
> - **预训练流程**: 数据处理、Tokenization、训练目标（CLM / MLM）
> - **SFT 流程**: 指令微调的数据格式、训练策略
> - **对齐技术概览**: RLHF、DPO、GRPO 的核心思想（理解即可，不需推导）
> - **核心论文精读**: DPR、LoRA（为后续微调实战打基础）
>
> **手撕系列:**
> - [ ] 精读 DPR 论文，画出双编码器架构图
> - [ ] 精读 LoRA 论文，理解低秩分解的核心思想
> - [ ] 梳理 "预训练 → SFT → RLHF" 全流程，画出完整 Pipeline 图
>
> **面试预热:**
> - Q: 请解释 LLM 从预训练到可用的完整流程。
> - Q: LoRA 的核心思想是什么？为什么能用更少的参数达到接近全量微调的效果？
>
> **解锁技能:**
> - 完整理解 LLM 训练 Pipeline（面试必考）
> - 掌握 LoRA 和 DPR 的核心原理
> - 建立对齐技术的全景认知

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
| ------ | ---------- | ---------- | ------- |
| 8 | 预训练流程概览 | 书籍: [《大语言模型》](https://llmbook-zh.github.io/) 第4-5章<br>视频: [清华大模型公开课第二季](https://www.bilibili.com/video/BV1pf421z757)<br>教程: [《从零开始的大语言模型原理与实践》](https://github.com/datawhalechina/happy-llm) | 理解预训练的数据处理、训练目标和 Scaling Laws |
| 9 | SFT 与指令微调 | 论文: [InstructGPT](https://arxiv.org/abs/2203.02155)<br>教程: [《面向开发者的 LLM 入门教程》](https://github.com/datawhalechina/llm-cookbook)<br>工具: [Easy-Dataset](https://github.com/ConardLi/easy-dataset) | 理解 SFT 数据格式、训练策略，了解数据质量的重要性 |
| 10 | 对齐技术全景 (RLHF/DPO/GRPO) | 博客: [RLHF 详解](https://huggingface.co/blog/rlhf)<br>图解: [DPO/PPO/GRPO 算法原理图](https://github.com/changyeyu/LLM-RL-Visualized)<br>论文: [DPO](https://arxiv.org/abs/2305.18290) | 理解三种对齐方法的核心思想和差异（无需推导） |
| 11 | LoRA 论文精读 | 论文: [LoRA](https://arxiv.org/abs/2106.09685)<br>博客: [LoRA 详解](https://huggingface.co/blog/lora)<br>扩展: [QLoRA](https://arxiv.org/abs/2305.14314) | 理解低秩分解原理，画出 LoRA 插入位置示意图 |
| 12 | DPR 论文精读 | 论文: [DPR](https://arxiv.org/abs/2004.04906)<br>教程: [Sentence Transformers](https://www.sbert.net/)<br>扩展: [ColBERT](https://arxiv.org/abs/2004.12832) | 理解双编码器架构、对比学习训练方式 |
| 13 | Self-RAG 论文精读 | 论文: [Self-RAG](https://arxiv.org/abs/2310.11511)<br>相关: [CRAG](https://arxiv.org/abs/2401.15884), [Adaptive-RAG](https://arxiv.org/abs/2403.14403) | 理解 Reflection Tokens 机制，评估其落地可行性 |
| 14 | 周度总结与 Pipeline 梳理 | 参考: [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)<br>笔记: [LLM Interview Note](https://github.com/wdndev/llm_interview_note) | 画出完整的 "预训练→SFT→对齐" Pipeline 图，整理论文笔记 |

---

### **第 3 周：RAG 算法优化实战**

> **学习内容:**
> - **检索算法**: BM25 vs Dense Retrieval vs Hybrid Search 的原理与权衡
> - **重排算法**: Cross-Encoder Reranker 的工作原理
> - **查询改写**: HyDE, Multi-Query, Step-Back Prompting
> - **RAG 评估**: RAGAs 框架的核心指标（Faithfulness, Answer Relevancy, Context Precision）
> - **实验设计入门**: 如何设计 RAG 优化的对比实验
>
> **手撕系列:**
> - [ ] 手撕 BM25 算法（理解 TF-IDF 的加权逻辑）
> - [ ] 实现 BM25 + Dense 的混合检索，并引入 Reranker
> - [ ] 使用 RAGAs 评估优化前后的 RAG 系统，输出对比报告
> - [ ] 设计一组 A/B 实验，证明混合检索优于单一检索
>
> **面试预热:**
> - Q: 密集检索和稀疏检索的优缺点？为什么 Hybrid Search 通常更好？
> - Q: 你的 RAG 优化提升了多少？怎么验证的？
>
> **解锁技能:**
> - 掌握 RAG 全链路优化方法（检索 → 重排 → 生成）
> - 能够设计 RAG 优化实验并用数据证明效果
> - 建立"改一处 → 量化 → 对比"的实验思维

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
| ------ | ---------- | ---------- | ------- |
| 15 | BM25 算法与手撕 | 教程: [BM25 from scratch](https://www.pinecone.io/learn/series/bm25/bm25-pragmatic-guide/)<br>参考: [All-in-RAG](https://github.com/datawhalechina/all-in-rag) | 理解 TF-IDF 和 BM25 原理，手动实现 BM25 |
| 16 | Dense Retrieval 与混合检索 | 教程: [Sentence Transformers](https://www.sbert.net/)<br>论文: [Modular RAG](https://arxiv.org/pdf/2407.21059)<br>技术: [RAG Techniques](https://github.com/NirDiamant/RAG_Techniques) | 实现 BM25 + Embedding 的混合检索 |
| 17 | Reranker 集成 | 教程: [LlamaIndex Reranking](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/CohereRerank.html)<br>工具: [Cohere Rerank](https://docs.cohere.com/docs/reranking) | 集成 Reranker，对比加入前后的检索精度 |
| 18 | Query Transformation | 教程: [LlamaIndex Query Transforms](https://docs.llamaindex.ai/en/stable/module_guides/querying/query_transforms/root.html)<br>教程: [RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch) | 实现 HyDE, Multi-Query 等查询改写策略 |
| 19 | RAGAs 评估框架 | 文档: [RAGAs](https://docs.ragas.io/)<br>工具: [DeepEval](https://github.com/confident-ai/deepeval), [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) | 学习核心评估指标，搭建自动化评估流水线 |
| 20 | 实验设计：RAG A/B 对比 | 参考: [All-in-RAG](https://github.com/datawhalechina/all-in-rag)<br>工具: [LangSmith](https://docs.smith.langchain.com/) | 设计对比实验，量化混合检索 + Reranker 的提升效果 |
| 21 | 周度总结与实验报告 | | 输出一份完整的 RAG 优化实验报告（含对比数据、图表） |

---

### **第 4 周：模型微调技术（LoRA / SFT / 数据工程）**

> **学习内容:**
> - **LoRA/QLoRA 实战**: 使用 LLaMA-Factory 或 Unsloth 完成一次完整微调
> - **数据工程**: 微调数据的构造、清洗、质量评估（数据决定效果上限）
> - **SFT 最佳实践**: 学习率调度、训练轮次、过拟合判断
> - **DPO 实战**: 使用偏好数据进行对齐微调
>
> **手撕系列:**
> - [ ] 使用 LLaMA-Factory 对 Qwen 进行一次 LoRA SFT 微调
> - [ ] 构造一份高质量微调数据集（至少 500 条）
> - [ ] 使用 Unsloth 体验高效微调，对比训练速度
> - [ ] 设计微调效果的评估方案（自动评估 + 人工评估）
>
> **面试预热:**
> - Q: LoRA 的 rank 怎么选？太大或太小会怎样？
> - Q: 微调数据质量不高时，你会怎么处理？
>
> **解锁技能:**
> - 能独立完成从数据准备到模型微调的全流程
> - 掌握微调超参调优的实战经验
> - 理解"数据质量 > 数据数量 > 模型大小"的核心原则

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
| ------ | ---------- | ---------- | ------- |
| 22 | LoRA/QLoRA 原理复习与实战准备 | 论文: [LoRA](https://arxiv.org/abs/2106.09685), [QLoRA](https://arxiv.org/abs/2305.14314)<br>图解: [LoRA/QLoRA 算法原理图](https://github.com/changyeyu/LLM-RL-Visualized) | 巩固 LoRA 原理，理解 rank、alpha、target_modules 等关键参数 |
| 23 | 微调数据工程 | 工具: [Easy-Dataset](https://github.com/ConardLi/easy-dataset)<br>教程: [《开源大模型食用指南》](https://github.com/datawhalechina/self-llm)<br>参考: [大模型微调系列](https://mp.weixin.qq.com/s/aQCY8873d09zFIhMhrx7Pg) | 学习微调数据格式 (Alpaca/ShareGPT)，构造一份高质量数据集 |
| 24 | LLaMA-Factory SFT 实战 | 文档: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)<br>教程: [LLaMA-Factory 快速入门](https://llamafactory.readthedocs.io/) | 使用 LLaMA-Factory Web UI 完成一次 LoRA SFT 训练 |
| 25 | Unsloth 高效微调 | 文档: [Unsloth](https://github.com/unslothai/unsloth)<br>教程: [Unsloth 微调教程](https://docs.unsloth.ai/) | 使用 Unsloth 训练同样任务，对比 LLaMA-Factory 的速度和显存 |
| 26 | DPO 对齐微调实战 | 教程: [TRL DPO Trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)<br>框架: [LLaMA-Factory DPO](https://github.com/hiyouga/LLaMA-Factory) | 用 LLaMA-Factory 完成一次 DPO 训练，理解偏好数据格式 |
| 27 | 微调效果评估 | 工具: [OpenCompass](https://github.com/open-compass/opencompass)<br>参考: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 设计评估方案，对比微调前后的模型表现（自动指标 + 人工抽检） |
| 28 | 周度总结与最佳实践整理 | 博客: [Hugging Face PEFT Blog](https://huggingface.co/blog/peft)<br>教程: [《AI-Guide-and-Demos》](https://github.com/Hoper-J/AI-Guide-and-Demos-zh_CN) | 整理微调全流程的最佳实践文档，包括踩坑记录 |

---

### **第 5 周：模型量化与推理部署**

> **学习内容:**
> - **量化技术**: INT8、INT4、GPTQ、AWQ、GGUF 的原理与适用场景
> - **推理引擎**: vLLM 的 PagedAttention、连续批处理等核心技术
> - **性能基准**: 如何科学地进行推理性能评测（吞吐量、延迟、显存）
> - **端侧部署**: llama.cpp / Ollama 的使用场景
>
> **手撕系列:**
> - [ ] 使用 AutoGPTQ 或 AWQ 对一个模型进行量化
> - [ ] 部署 vLLM 并通过 OpenAI 兼容 API 进行推理
> - [ ] 设计性能基准测试，对比量化前后的推理速度、显存占用和效果损失
> - [ ] 使用 Ollama 在本地部署一个量化模型
>
> **面试预热:**
> - Q: GPTQ 和 AWQ 的区别是什么？各自适用什么场景？
> - Q: 量化会带来多少精度损失？如何评估？
>
> **解锁技能:**
> - 掌握主流量化技术的原理和实操
> - 能独立完成模型量化 → 部署 → 性能评测的全流程
> - 理解推理优化的核心技术（KV Cache、PagedAttention）

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
| ------ | ---------- | ---------- | ------- |
| 29 | 量化技术原理 | 博客: [A Gentle Introduction to Quantization](https://huggingface.co/blog/merve/quantization)<br>图解: [量化算法原理图](https://github.com/changyeyu/LLM-RL-Visualized)<br>书籍: [《大语言模型》](https://llmbook-zh.github.io/) 相关章节 | 理解 INT8/INT4 量化原理、GPTQ vs AWQ vs GGUF 的区别 |
| 30 | GPTQ/AWQ 量化实战 | 工具: [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ), [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)<br>教程: [HF Quantization Guide](https://huggingface.co/docs/transformers/quantization) | 对一个 7B 模型进行量化，对比不同量化方法的效果 |
| 31 | vLLM 部署实战 | 文档: [vLLM Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)<br>替代: [SGLang](https://github.com/sgl-project/sglang), [LMDeploy](https://github.com/InternLM/lmdeploy) | 部署 vLLM，通过 OpenAI 兼容 API 调用推理 |
| 32 | KV Cache 与 PagedAttention | 论文: [Efficient Memory Management for LLM Serving (vLLM)](https://arxiv.org/abs/2309.06180)<br>博客: [PagedAttention 详解](https://blog.vllm.ai/2023/06/20/vllm.html) | 理解 vLLM 的核心技术，能在面试中解释 PagedAttention |
| 33 | 推理性能基准测试 | 工具: [LLMPerf](https://github.com/ray-project/llmperf)<br>概览: [Awesome Inference](https://github.com/WangRongsheng/awesome-LLM-resources#%E6%8E%A8%E7%90%86-inference) | 设计基准测试方案，输出量化前后的性能对比报告（吞吐量、延迟、显存） |
| 34 | 端侧部署 (Ollama / llama.cpp) | 工具: [Ollama](https://github.com/ollama/ollama), [llama.cpp](https://github.com/ggml-org/llama.cpp)<br>格式: [GGUF 说明](https://huggingface.co/docs/hub/gguf) | 使用 Ollama 在本地跑通一个量化模型，理解 GGUF 格式 |
| 35 | 周度总结与部署方案对比 | 概览: [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | 整理"量化方案选型决策树"：场景 → 量化方法 → 部署引擎 |

---

### **第 6 周：Agent Memory 与上下文工程算法**

> **学习内容:**
> - **Memory 架构**: 短期记忆 vs 长期记忆、分层记忆设计
> - **记忆算法**: 重要性评分（Recency + Importance + Relevance）、压缩与总结
> - **上下文工程**: 上下文选择策略、压缩算法、动态上下文构建
> - **核心论文**: MemGPT 的分层记忆机制
>
> **手撕系列:**
> - [ ] 精读 MemGPT 论文，画出分层记忆架构图
> - [ ] 实现一个记忆重要性评分算法（语义相似度 + 时间衰减 + 任务相关性）
> - [ ] 设计一个上下文压缩方案，对比压缩前后的效果
>
> **面试预热:**
> - Q: 如何设计 Agent 的长期记忆机制？写入、更新、读取流程是怎样的？
> - Q: MemGPT 和传统 RAG 在处理长上下文时的本质区别是什么？
>
> **解锁技能:**
> - 掌握 Agent 记忆系统的算法设计方法
> - 能够设计上下文选择与压缩策略
> - 理解上下文工程在 Agent 系统中的核心地位

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
| ------ | ---------- | ---------- | ------- |
| 36 | Agent Memory 全景 | 博客: [LLM Powered Agents - Memory](https://lilianweng.github.io/posts/2023-06-23-agent/#memory)<br>工具: [Mem0](https://github.com/mem0ai/mem0), [MemoryScope](https://github.com/modelscope/MemoryScope) | 梳理 Agent 记忆的分类（感知/短期/长期）和设计挑战 |
| 37 | Generative Agents 记忆机制 | 论文: [Generative Agents](https://arxiv.org/abs/2304.03442)<br>解读: [Generative Agents 解读](https://blog.ml.cmu.edu/2023/09/08/generative-agents/) | 学习 Recency + Importance + Relevance 的记忆评分机制 |
| 38 | MemGPT 论文精读 | 论文: [MemGPT](https://arxiv.org/abs/2310.08560)<br>代码: [MemGPT 开源库](https://github.com/cpacker/MemGPT) | 理解分层记忆和函数调用管理虚拟上下文的方法 |
| 39 | 记忆评分算法实现 | 参考: [LangMem](https://github.com/langchain-ai/langmem)<br>工具: [LlamaIndex Memory](https://docs.llamaindex.ai/en/stable/) | 用 Python 实现一个记忆评分算法（语义相似度 + 时间衰减） |
| 40 | 上下文压缩技术 | 论文: [LongLLMLingua](https://arxiv.org/abs/2310.06839)<br>论文: [Lost in the Middle](https://arxiv.org/abs/2307.03172) | 学习上下文压缩策略，理解信息保真度与长度限制的权衡 |
| 41 | 上下文工程最佳实践 | 教程: [LlamaIndex Node Postprocessors](https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors.html)<br>博客: [Anthropic Context Engineering](https://www.anthropic.com/) | 实现一个自定义上下文过滤器（按时间、相关性、重要性） |
| 42 | 周度总结与方案设计 | | 设计一个完整的分层记忆方案（含评分、压缩、检索），画出架构图 |

---

### **第 7 周：实验设计方法论 + 评估体系搭建**

> **学习内容:**
> - **实验设计**: 对比实验、消融实验、超参搜索的规范流程
> - **评估指标**: 不同任务的指标选择（分类/生成/检索/对话）
> - **统计分析**: 结果的统计显著性、置信区间
> - **评估流水线**: 使用 OpenCompass / RAGAs 搭建自动化评估
> - **技术报告**: 如何用图表和数据讲清楚"为什么你的方案更好"
>
> **手撕系列:**
> - [ ] 为之前的 RAG 优化设计一组完整的消融实验
> - [ ] 搭建一个自动化评估流水线（一键跑 Baseline + 改进方案 + 输出对比表格）
> - [ ] 撰写一份完整的技术实验报告（含动机、方法、实验、结论）
>
> **面试预热:**
> - Q: 你的算法优化提升了 X%，怎么确保这个提升是显著的？
> - Q: 消融实验怎么设计？为什么需要消融实验？
>
> **解锁技能:**
> - 掌握算法实验设计的标准方法论
> - 能够用数据和图表证明方案有效（区别于纯工程开发的核心能力）
> - 具备撰写专业技术报告的能力

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
| ------ | ---------- | ---------- | ------- |
| 43 | 实验设计基础 | 指南: [ML Experiment Design](https://neptune.ai/blog/ml-experiment-tracking)<br>参考: [How to Design ML Experiments](https://www.deeplearning.ai/) | 理解对比实验、消融实验、控制变量的基本原则 |
| 44 | 评估指标选择 | 文档: [RAGAs Metrics](https://docs.ragas.io/)<br>工具: [OpenCompass](https://github.com/open-compass/opencompass)<br>参考: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | 梳理不同任务类型（检索/生成/对话）的评估指标体系 |
| 45 | 消融实验设计实战 | 参考本路线图第 3 周的 RAG 优化项目 | 为 RAG 优化设计消融实验：分别去掉 Reranker、混合检索、Query 改写 |
| 46 | 自动化评估流水线搭建 | 工具: [RAGAs](https://docs.ragas.io/), [DeepEval](https://github.com/confident-ai/deepeval)<br>工具: [LangSmith](https://docs.smith.langchain.com/) | 搭建 "一键评估" 脚本，输入数据集 → 输出对比表格 |
| 47 | 结果分析与数据可视化 | 工具: [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/)<br>参考: [科研画图教程](https://github.com/garrettj403/SciencePlots) | 学习用图表清晰呈现实验结果（柱状图、雷达图、热力图） |
| 48 | 技术报告撰写 | 模板: [技术报告模板](https://www.overleaf.com/)<br>参考: 顶会论文的 Experiments 章节格式 | 撰写一份完整的实验报告（动机 → 方法 → 实验设计 → 结果分析 → 结论） |
| 49 | 周度总结与方法论沉淀 | | 整理"实验设计 Checklist"，作为未来项目的标准流程 |

---

### **第 8-9 周：项目实战与开源产出**

> **核心目标**：完成 1 个可写进简历的完整算法优化项目，产出开源代码 + 技术博客。
>
> 从以下两个方向中选择一个（根据你的兴趣和目标岗位选择）：

#### **项目方向 A：Self-RAG 优化 —— 让 RAG 学会"自我纠错"**

> **问题定义**: 传统 RAG "检索一次定成败"，无法处理检索结果不准或信息不足的情况。
>
> **你要做的事**:
> 1. **复现 Baseline**: 搭建一个标准的 Naive RAG 系统
> 2. **实现 Self-RAG 机制**: 基于 Self-RAG 论文，实现 Reflection Tokens 驱动的自适应检索
> 3. **工程优化**: 加入混合检索 + Reranker + 查询改写，提升端到端效果
> 4. **完整实验**: 设计对比实验和消融实验，用数据证明每个模块的贡献
>
> **实验设计**:
> - **数据集**: [HotpotQA](https://hotpotqa.github.io/)（多跳推理）, [Natural Questions](https://ai.google.com/research/NaturalQuestions)
> - **Baseline**: Naive RAG, RAG + Reranker
> - **评估指标**: F1, Faithfulness (RAGAs), Answer Relevancy, 检索轮次（效率）
> - **消融实验**: 分别去掉 Self-Correction / Reranker / Query Rewrite
>
> **简历亮点**: Self-RAG 算法优化 + 完整消融实验 + 开源代码

#### **项目方向 B：Agent Memory 优化 —— 分层记忆让 Agent "记得更好"**

> **问题定义**: 现有 Agent 记忆机制多为扁平向量存储，长对话场景下检索效率低、信息丢失严重。
>
> **你要做的事**:
> 1. **复现 Baseline**: 基于 LangChain 搭建一个 Sliding Window Memory Agent
> 2. **设计分层记忆**: 实现 Working Memory（原始对话）+ Event Memory（摘要）双层结构
> 3. **实现记忆管理算法**: 自动摘要、重要性评分、分层检索
> 4. **完整实验**: 对比扁平记忆 vs 分层记忆在长对话场景的表现
>
> **实验设计**:
> - **数据集**: 构造多轮长对话数据（模拟客服/助手场景，20+ 轮）
> - **Baseline**: Sliding Window, Naive Vector Memory
> - **评估指标**: Information Recall（信息保留率）, Compression Ratio, 检索延迟
> - **消融实验**: 验证分层结构和自动摘要模块各自的贡献
>
> **简历亮点**: 分层记忆算法设计 + 对比实验验证 + 开源代码

**🌟 学习计划 (2 周)**

| **天数** | **学习主题** | **资源链接** | **目标** |
| ------ | ---------- | ---------- | ------- |
| 50-51 | 项目选题与方案设计 | 论文: [Self-RAG](https://arxiv.org/abs/2310.11511) 或 [MemGPT](https://arxiv.org/abs/2310.08560)<br>参考: [All-in-RAG](https://github.com/datawhalechina/all-in-rag) | 选定方向，完成技术方案文档（问题定义、算法设计、实验方案） |
| 52-53 | 搭建实验框架与 Baseline | 框架: [LangChain](https://github.com/langchain-ai/langchain), [LlamaIndex](https://github.com/run-llama/llama_index)<br>评估: [RAGAs](https://docs.ragas.io/) | 搭建实验框架（数据加载、评估脚本），实现 Baseline |
| 54-57 | 核心算法实现与调优 | 参考选定方向的相关论文和开源实现 | 实现核心优化模块，进行初步调试和效果验证 |
| 58-59 | 完整实验与消融分析 | 工具: [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/) | 运行所有对比实验和消融实验，生成结果图表 |
| 60-61 | 技术博客撰写 | 平台: [掘金](https://juejin.cn/), [知乎](https://www.zhihu.com/) | 撰写一篇 3000+ 字的技术博客，讲清楚动机、方法和实验结果 |
| 62-63 | 代码开源与文档 | 指南: [如何写好 README](https://www.makeareadme.com/)<br>平台: [GitHub](https://github.com/), [Hugging Face](https://huggingface.co/) | 整理代码，撰写 README（含架构图、快速开始、实验结果） |

---

### **第 10 周：面试冲刺与简历打磨**

> **学习内容:**
>
> **简历打磨:**
> - [ ] 用"算法 + 工程"复合背景重写简历
> - [ ] 每个项目经历用 STAR 法则量化描述
> - [ ] 突出实验数据和算法优化成果
>
> **面试准备:**
> - [ ] 算法原理题：Transformer、Attention、LoRA、RAG 检索算法
> - [ ] 实验设计题：消融实验、评估指标选择、结果分析
> - [ ] 系统设计题：如何设计一个生产级 RAG/Agent 系统
> - [ ] 论文讨论题：讲一篇你读过的论文，说说创新点和局限性
>
> **面试话术准备 (STAR - 应用算法版):**
> - **Situation**: 业务场景和技术痛点（如"RAG 检索准确率不足"）
> - **Task**: 你的优化目标（如"将 Faithfulness 从 0.72 提升到 0.85"）
> - **Action**: 你的算法方案 + 为什么选这个方案 + 做了哪些实验
> - **Result**: 量化结果 + 消融实验证明 + 产出（开源/博客）
>
> **解锁技能:**
> - 一份突出"算法 + 工程"复合能力的简历
> - 能在面试中流畅地阐述算法选型、实验设计和优化成果
> - 从容应对算法原理、论文讨论和系统设计类面试题

**🌟 每日学习计划**

| **天数** | **学习主题** | **资源链接** | **目标** |
| ------ | ---------- | ---------- | ------- |
| 64 | 简历撰写 | 指南: [Tech Resume Guide](https://www.techinterviewhandbook.org/resume/)<br>参考: [AI 面试指南](https://github.com/WangRongsheng/awesome-LLM-resources/tree/main/docs/04-interview) | 按"应用型算法工程师"定位重写简历，突出算法优化 + 实验验证 |
| 65 | 项目经历打磨 (STAR 法则) | 模板: [STAR 方法](https://www.indeed.com/career-advice/interviewing/how-to-use-the-star-interview-response-technique) | 准备 3-5 分钟的项目介绍逐字稿，覆盖 S/T/A/R |
| 66 | 算法原理高频题 | 题库: [LLM Interview Note](https://github.com/wdndev/llm_interview_note)<br>图解: [100+ 算法原理图](https://github.com/changyeyu/LLM-RL-Visualized) | 刷 Transformer、LoRA、RAG、量化等核心面试题 |
| 67 | 实验设计 & 论文讨论题 | 准备: 讲解 Self-RAG / MemGPT 的创新点和局限性<br>题库: [ML Papers Explained](https://github.com/dair-ai/ML-Papers-Explained) | 准备"讲一篇论文"和"消融实验怎么设计"类问题 |
| 68 | 系统设计题 | 资源: [OpenAI Cookbook](https://github.com/openai/openai-cookbook)<br>题库: [RAG Interview Questions](https://www.analyticsvidhya.com/blog/2024/04/rag-interview-questions/) | 准备"设计一个生产级 RAG 系统"等系统设计题 |
| 69 | 模拟面试 | 课程: [LLM Evaluation: A Complete Course](https://www.comet.com/site/llm-course/) | 进行 1v1 模拟面试，重点练习算法追问和项目深挖 |
| 70 | 总结复盘与查漏补缺 | | 复盘面试薄弱环节，针对性补强 |








---

## 五、学习资源精选

> 只列出对应用型算法工程师最有价值的资源，避免信息过载。

### 必读论文（按优先级排序）

| 优先级 | 论文 | 读法 |
|--------|------|------|
| P0 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 精读 + 手撕 |
| P0 | [DPR](https://arxiv.org/abs/2004.04906) | 精读，理解双编码器 |
| P0 | [Self-RAG](https://arxiv.org/abs/2310.11511) | 精读，可作为项目基础 |
| P1 | [LoRA](https://arxiv.org/abs/2106.09685) | 精读，理解低秩分解 |
| P1 | [DPO](https://arxiv.org/abs/2305.18290) | 理解损失函数推导 |
| P1 | [MemGPT](https://arxiv.org/abs/2310.08560) | 理解分层记忆设计 |
| P2 | [ReAct](https://arxiv.org/abs/2210.03629) | 快速过（你已有 Agent 基础） |
| P2 | [GRPO](https://arxiv.org/pdf/2402.03300) | 理解核心思想即可 |
| P2 | [GraphRAG](https://www.microsoft.com/en-us/research/project/graphrag/) | 理解创新点和适用场景 |

### 核心工具框架

| 类别 | 工具 | 用途 |
|------|------|------|
| 微调 | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | 一站式微调平台 |
| 微调 | [Unsloth](https://github.com/unslothai/unsloth) | 高效微调（速度快、显存省） |
| 部署 | [vLLM](https://github.com/vllm-project/vllm) | 高性能推理服务 |
| 评估 | [RAGAs](https://docs.ragas.io/) | RAG 系统评估 |
| 评估 | [OpenCompass](https://github.com/open-compass/opencompass) | 模型能力评估 |
| 可视化 | [100+ 算法原理图](https://github.com/changyeyu/LLM-RL-Visualized) | 算法理解利器 |

### 系统学习资源

| 资源 | 用途 |
|------|------|
| [《大语言模型》](https://llmbook-zh.github.io/) | 大模型最佳中文书籍 |
| [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | 从零构建大模型 |
| [All-in-RAG](https://github.com/datawhalechina/all-in-rag) | RAG 全流程算法优化 |
| [LLM Interview Note](https://github.com/wdndev/llm_interview_note) | 面试题库 |

---


