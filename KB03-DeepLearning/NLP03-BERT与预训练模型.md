# BERT 与预训练模型

本笔记基于《动手学深度学习》第 14 章后半部分（`bert.ipynb` / `bert-dataset.ipynb` / `bert-pretraining.ipynb`）整理，聚焦 **context-sensitive** 的语义表示与基于 Transformer encoder 的自监督预训练范式。

> 前置阅读：[RNN02-自然语言基础](./RNN02-自然语言基础.md) —— 介绍 word2vec / GloVe / fastText 等 **context-independent** 词嵌入方案。
>
> 代码具体实现见 `chapter_natural-language-processing-pretraining/` 下对应 `.ipynb`，本笔记仅记录关键函数名。



## 1. 从 context-independent 到 context-sensitive

### 1.1 静态词向量的局限

word2vec、GloVe、fastText 等模型属于 **context-independent representation**：对任意 token $x$，其表示是一个仅依赖 $x$ 本身的函数 $f(x)$。

**问题**：自然语言充满 polysemy（一词多义）。

- *"a **crane** is flying"*（鹤）
- *"a **crane** driver came"*（吊车）

同一个 "crane" 在不同语境中含义完全不同，但 word2vec / GloVe 会给出 **完全相同的向量**。

### 1.2 上下文敏感表示

**Context-sensitive representation**：token $x$ 的表示是 $f(x, c(x))$，依赖 $x$ 及其上下文 $c(x)$。

代表性模型演进：

| 模型 | 年份 | 上下文方向 | 架构通用性 | 关键思路 |
| ---- | ---- | ---------- | ---------- | -------- |
| **TagLM** | 2017 | 双向 | 任务相关 | 语言模型增强的序列标注 |
| **CoVe** | 2017 | 双向 | 任务相关 | Context Vectors，借助机器翻译编码器 |
| **ELMo** | 2018 | **双向**（独立训练正反向 LSTM 拼接） | task-specific | 冻结预训练 BiLSTM，作为附加特征拼到下游模型 |
| **GPT** | 2018 | 从左到右（Transformer decoder） | **task-agnostic** | 微调全部参数；自回归 LM；只能看左侧 |
| **BERT** | 2018 | **双向**（Transformer encoder） | **task-agnostic** | 微调全部参数；最少架构改动适配多种任务 |

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/elmo-gpt-bert.svg" width="700px">
</div>

> - **ELMo**：双向但 task-specific，每个下游任务都要定制架构；
> - **GPT**：task-agnostic 但单向，"i went to the **bank** to deposit cash" 与 "i went to the **bank** to sit down" 会得到相同的 bank 表示；
> - **BERT**：兼具双向编码与 task-agnostic，11 项 NLP 任务上达到 SOTA，将 NLP 预训练范式推向新阶段。



## 2. BERT 输入表示

### 2.1 输入序列格式

BERT input sequence 同时支持单文本与文本对：

- **单文本**：`<cls>` + tokens + `<sep>`
- **文本对**：`<cls>` + tokensA + `<sep>` + tokensB + `<sep>`

特殊词元说明：

- `<cls>`：分类标记，其编码输出作为整条输入的句向量；
- `<sep>`：分隔标记，区分两段文本。

### 2.2 三种嵌入之和

每个位置的最终输入嵌入由三个嵌入相加得到：

$$
\text{Input} = \underbrace{\text{Token Embedding}}_{\text{词元身份}} + \underbrace{\text{Segment Embedding}}_{e_A/e_B,\ 区分文本对} + \underbrace{\text{Position Embedding}}_{\text{可学习}}.
$$

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/bert-input.svg" width="600px">
</div>

> 与原始 Transformer 不同，BERT 使用 **learnable positional embedding**，而非 sinusoidal 位置编码。

**关键函数 / 类**：

| 名称 | 作用 |
| ---- | ---- |
| `get_tokens_and_segments(tokens_a, tokens_b=None)` | 拼接特殊词元并返回 tokens 与 segments |
| `BERTEncoder` | 由 `token_embedding` + `segment_embedding` + 可学习 `pos_embedding` + 多层 `EncoderBlock` 组成 |



## 3. 预训练任务

BERT 使用两个 self-supervised 任务联合预训练，总损失 = **MLM loss + NSP loss**。

### 3.1 Masked Language Modeling（MLM）

**核心思想**：随机遮蔽部分 token，用其双向上下文预测被遮蔽的原始 token。

**做法**：

- 在输入中随机选 **15%** 的 token 作为预测目标；
- 为减轻 pretrain / finetune 阶段 mismatch（`<mask>` 仅在 pretrain 阶段出现），被选中的 token：
  - **80%** 替换为 `<mask>`；
  - **10%** 替换为词表中的 **随机 token**；
  - **10%** 保持 **不变**。

> 后两种"噪声"鼓励模型在所有位置都保持上下文编码能力，而非仅在 `<mask>` 处推断。

**预测**：在被选位置的 encoder 输出上，经一个单隐层 MLP 输出 vocab 上的分布，使用 cross-entropy 训练。

**关键类**：

| 名称 | 作用 |
| ---- | ---- |
| `MaskLM(vocab_size, num_hiddens, num_inputs)` | MLM 任务头：`Linear → ReLU → LayerNorm → Linear(vocab_size)` |



### 3.2 Next Sentence Prediction（NSP）

**核心思想**：让模型学习两个文本片段之间的 **逻辑关系**。

**做法**：

- 训练样本由句对 $(A,B)$ 组成；
- **50%** 为真实连续句对（IsNext）；
- **50%** 第二句从语料中随机抽取（NotNext）；
- 利用 `<cls>` 位置的 encoder 输出经一个隐藏层 + 线性层进行二分类。

> NSP 弥补了 MLM 只能建模 token 级关系、无法显式建模句子关系的局限，对 NLI、QA 等下游任务有帮助。

**关键类**：

| 名称 | 作用 |
| ---- | ---- |
| `NextSentencePred(num_inputs)` | NSP 任务头：单线性层输出 2 维 logits |



## 4. BERT 整体模型

`BERTModel` 由三部分组合：

- `encoder = BERTEncoder(...)`：双向 Transformer encoder；
- `hidden`：取 `<cls>` 表示后接 `Linear → Tanh`，供 NSP 使用；
- `mlm = MaskLM(...)` 与 `nsp = NextSentencePred(...)`：两个任务头。

前向输出：

```text
encoded_X, mlm_Y_hat, nsp_Y_hat = net(tokens, segments, valid_lens, pred_positions)
```

### 模型规模

| 配置 | layers | hidden | heads | 参数量 |
| ---- | ------ | ------ | ----- | ------ |
| **BERT-BASE** | 12 | 768 | 12 | 110M |
| **BERT-LARGE** | 24 | 1024 | 16 | 340M |



## 5. 预训练数据集（WikiText-2）

BERT 原版在 **BookCorpus（8 亿词）+ English Wikipedia（25 亿词）** 上预训练；演示版改用更小的 **WikiText-2**。

### 5.1 相较 PTB 的优势

| 维度 | PTB | WikiText-2 |
| ---- | --- | ---------- |
| 大小写 | 全部小写 | **保留** |
| 标点 | 删除 | **保留**（可用于按 `.` 切句） |
| 数字 | 替换为 N | **保留** |
| 规模 | 较小 | **约 2 倍以上** |

标点和句号的保留使其天然适合构造 **NSP 样本**。

### 5.2 数据生成流程

1. 读取段落，每段至少含两句话，用 ` . ` 切分句子；
2. **NSP 样本生成**：
   - 50% 取真实连续句对；
   - 50% 随机从其它段落抽一句替换 B；
   - 调用 `get_tokens_and_segments` 拼装 BERT input；
3. **MLM 样本生成**：
   - 排除 `<cls>` 与 `<sep>` 后，随机选 15% 位置；
   - 按 80% / 10% / 10% 规则替换；
4. **填充对齐**：将所有变长字段填充至 `max_len`，并构造 `valid_lens`、`mlm_weights`（用于在损失中屏蔽 padding 预测位）。

### 5.3 关键函数 / 类

| 名称 | 作用 |
| ---- | ---- |
| `_read_wiki(data_dir)` | 读取并按句切分 WikiText-2 段落 |
| `_get_next_sentence(...)` | 生成单条 NSP 样本（IsNext 或 NotNext） |
| `_get_nsp_data_from_paragraph(...)` | 从段落构造一批 NSP 样本 |
| `_replace_mlm_tokens(...)` | 按 80/10/10 规则替换 token，返回新输入与预测标签 |
| `_get_mlm_data_from_tokens(...)` | 从 tokens 构造 MLM 训练数据 |
| `_pad_bert_inputs(...)` | 统一填充所有变长字段 |
| `_WikiTextDataset` | 封装为 `torch.utils.data.Dataset` |
| `load_data_wiki(batch_size, max_len)` | 返回 `train_iter` 与 `vocab` |

> 一条 batch 输出 7 个张量：`tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y`。
>
> 原始 BERT 使用 **WordPiece** 分词（vocab size = 30000），其本质是 BPE 的变体；演示中用 `d2l.tokenize` 简化处理，过滤出现次数 < 5 的低频 token。



## 6. 预训练 BERT 的训练流程

### 6.1 训练目标

总损失：

$$
\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}},
$$

均为 cross entropy。MLM 部分按 `mlm_weights` 屏蔽 padding 位（乘 0 权重）。

### 6.2 关键函数

| 名称 | 作用 |
| ---- | ---- |
| `_get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)` | 计算单 batch 的 MLM、NSP 与总损失 |
| `train_bert(train_iter, net, loss, vocab_size, devices, num_steps)` | 多 GPU 训练 BERT，按 step 而非 epoch 控制 |
| `get_bert_encoding(net, tokens_a, tokens_b=None)` | 用预训练好的 BERT 编码单文本 / 文本对，返回 token 级表示 |

### 6.3 实验观察：context-sensitive 验证

对一词多义 "crane"：

| 输入 | 含义 | "crane" 位置编码（前 3 维） |
| ---- | ---- | --------------------------- |
| `a crane is flying` | 鹤 | `[-0.5007, -1.0034,  0.8718]` |
| `a crane driver came / he just left` | 吊车 | `[ 0.5101, -0.4041, -1.2749]` |

同一个 token 在不同语境下的 BERT 表示 **不同**，直接验证了 BERT 输出是 **context-sensitive** 的。

> 通常使用 `encoded_X[:, 0, :]`（即 `<cls>` 位置）作为整段输入的句向量；token 级表示则取相应位置的输出向量。



## 7. 小结

1. **静态词向量** (word2vec / GloVe / fastText) 无法表达 polysemy；
2. ELMo 用 BiLSTM 提供双向上下文但依赖 task-specific 架构；GPT 用 Transformer decoder 提供 task-agnostic 微调框架但只能单向编码；
3. **BERT** 用 **Transformer encoder + 双向上下文 + task-agnostic 微调** 统一了二者优点；
4. BERT 输入嵌入 = **Token + Segment + 可学习 Position**；
5. 预训练任务为 **MLM**（80/10/10 替换策略） + **NSP**（50/50 IsNext / NotNext），无需人工标注；
6. 下游任务微调时只需在 BERT 输出上加少量任务头，并对全部参数进行 fine-tuning；
7. WikiText-2 相比 PTB 保留了标点 / 大小写 / 数字，适合构造 NSP 样本。



## References

- 《动手学深度学习》第 14 章 [自然语言处理：预训练](https://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/index.html)
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2018
- Peters et al., *Deep Contextualized Word Representations*, 2018（ELMo）
- Radford et al., *Improving Language Understanding by Generative Pre-Training*, 2018（GPT）
- Vaswani et al., *Attention Is All You Need*, 2017（Transformer）
- Wu et al., *Google's Neural Machine Translation System*, 2016（WordPiece）
