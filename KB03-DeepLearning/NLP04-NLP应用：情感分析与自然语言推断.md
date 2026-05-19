# 自然语言处理应用：情感分析与自然语言推断

本笔记基于《动手学深度学习》第 15 章 *Natural Language Processing: Applications* 整理，聚焦如何将预训练的文本表示（GloVe、BERT 等）送入下游 NLP 模型解决两个代表性任务：

- **Sentiment Analysis**（情感分析）：单文本分类；
- **Natural Language Inference, NLI**（自然语言推断）：文本对分类。

> 前置阅读：
>
> - [NLP02-自然语言基础](./NLP02-自然语言基础.md)：word2vec / GloVe / fastText 等 context-independent embedding；
> - [NLP03-BERT与预训练模型](./NLP03-BERT与预训练模型.md)：context-sensitive 预训练模型 BERT。
>
> 代码具体实现见 `chapter_natural-language-processing-applications/` 下对应 `.ipynb`，本笔记仅记录关键函数名。



## 1. 章节概览

预训练的文本表示可以与不同下游架构搭配使用。本章聚焦的两条主线如下：

| 任务 | 输入 | 输出 | 数据集 | 本章实现的下游架构 |
| ---- | ---- | ---- | ------ | ------------------ |
| Sentiment Analysis | 单段文本 | 极性二分类（pos / neg） | IMDb (large movie review) | GloVe + **BiRNN** / GloVe + **textCNN** |
| Natural Language Inference | (premise, hypothesis) 文本对 | 三分类（entailment / contradiction / neutral） | SNLI | GloVe + **Decomposable Attention (MLP)** / **Fine-tuned BERT** |

> 资源充足时直接 fine-tune BERT 即可获得最少架构改动下的强基线；资源受限时，基于 RNN / CNN / Attention 精心构造的小模型仍是更可行的方案。



## 2. 情感分析数据集（IMDb）

### 2.1 任务定义

将可变长度的影评文本映射到固定的极性标签：

$$
f: \text{tokens} \mapsto \{\text{positive}, \text{negative}\}.
$$

属于 **text classification**：可变长输入 → 固定类别。

### 2.2 数据集说明

**Stanford Large Movie Review Dataset (IMDb)**：

- 25 000 训练 + 25 000 测试，均为 IMDb 影评；
- 标签平衡：pos / neg 各 12 500；
- 每条样本是一段较长的英文 review。

### 2.3 预处理流程

1. **读取原始文本与标签**：按 `pos` / `neg` 子目录分别打标签；
2. **Tokenization**：单词级 `d2l.tokenize`，构造 `Vocab` 并过滤 `min_freq < 5`，加入 `<pad>` 等保留 token；
3. **Truncate & Pad**：把每条 review 长度统一为 `num_steps = 500`（不足填 `<pad>`，超出截断）；
4. **Mini-batch**：使用 `d2l.load_array` 包装 `(features, labels)` 为 iterator。

### 2.4 关键函数

| 名称 | 作用 |
| ---- | ---- |
| `read_imdb(data_dir, is_train)` | 读取 IMDb 训练 / 测试集，返回 `(reviews, labels)` |
| `load_data_imdb(batch_size, num_steps=500)` | 一站式接口，返回 `train_iter, test_iter, vocab` |



## 3. 情感分析：BiRNN + GloVe

### 3.1 思路

- Token 级用预训练 **GloVe** 初始化的 embedding 表示；
- 整个序列用 **multi-layer Bidirectional LSTM** 编码；
- 取 BiLSTM 顶层 **首末时间步** 隐状态拼接，作为整段文本的固定长度表示；
- 接一个全连接层做二分类。

### 3.2 模型结构（BiRNN）

输入形状 `(batch, num_steps)`，主要层：

- `embedding`：`nn.Embedding(vocab_size, embed_size)`，权重由 GloVe 拷贝并 **frozen**（`requires_grad = False`）；
- `encoder`：`nn.LSTM(embed_size, num_hiddens, num_layers, bidirectional=True)`；
- `decoder`：`nn.Linear(4 * num_hiddens, 2)`。

forward 关键步骤：

1. 将 `inputs` 转置为 `(num_steps, batch)` 以匹配 LSTM 的 time-major 输入；
2. 取 `outputs` 中第 0 步与最后一步隐状态（已含双向）做 `torch.cat(..., dim=1)`，得到 `(batch, 4 * num_hiddens)`；
3. 经 `decoder` 输出 logits。

### 3.3 训练配置

- 超参：`embed_size=100, num_hiddens=100, num_layers=2`；
- 优化：`Adam, lr=0.01`，`num_epochs=5`；
- 损失：`CrossEntropyLoss(reduction='none')` + `d2l.train_ch13`；
- 典型结果：train acc ≈ 0.89，test acc ≈ 0.86。

### 3.4 关键函数 / 类

| 名称 | 作用 |
| ---- | ---- |
| `BiRNN(vocab_size, embed_size, num_hiddens, num_layers)` | BiLSTM 情感分类模型 |
| `init_weights(m)` | 对 `Linear` / `LSTM` 做 Xavier 初始化 |
| `predict_sentiment(net, vocab, sequence)` | 给定字符串预测 `'positive' / 'negative'` |



## 4. 情感分析：textCNN

### 4.1 一维卷积（1D Cross-correlation）

把文本看作一维 "图像"：

- 输入：`(channel, length)`；
- 1-D kernel 沿序列方向滑动逐元素相乘求和；
- 多输入通道：每通道独立做 1-D 卷积后逐元素相加。

> 多输入通道的 1-D 互相关 ≡ 单输入通道的 2-D 互相关（kernel 高度等于通道数）。

### 4.2 Max-over-Time Pooling

对每个通道沿时间步取 **最大值**，类似 1-D global max pooling：

- 把变长序列在每个通道上压缩成一个 scalar；
- 自然支持不同 kernel 产出的不同通道宽度；
- 抽取 "整段中最显著的 n-gram 特征"。

### 4.3 textCNN 架构

输入：`n` 个 token，每个 `d` 维 → 视为宽度 `n`、高度 `1`、通道 `d` 的 1-D 多通道输入。

模型流水线（Kim, 2014）：

1. **Dual embedding**：`embedding`（可训练）+ `constant_embedding`（GloVe 冻结），沿 channel 维拼接；
2. **多尺寸 1-D conv**：多组 `nn.Conv1d` 使用不同 `kernel_sizes`（如 3 / 4 / 5）和各自的 `num_channels`，捕获不同 n-gram；
3. **Max-over-Time pooling**：对每条卷积输出做 `AdaptiveMaxPool1d(1)`；
4. **Concat + MLP head**：所有标量输出拼接 → `Dropout` → `Linear` → 二分类。

> 输入形状变换：`embedding` 输出 `(batch, n, 2d)` → `permute(0, 2, 1)` 得 `(batch, 2d, n)`，再喂入 `Conv1d`。

### 4.4 训练配置

- 超参：`embed_size=100, kernel_sizes=[3,4,5], num_channels=[100,100,100], dropout=0.5`；
- 优化：`Adam, lr=0.001`，`num_epochs=5`；
- 典型结果：train acc ≈ 0.98，test acc ≈ 0.86（与 BiRNN 接近，但训练更快）。

### 4.5 关键函数 / 类

| 名称 | 作用 |
| ---- | ---- |
| `corr1d(X, K)` | 单通道 1-D cross-correlation |
| `corr1d_multi_in(X, K)` | 多输入通道 1-D cross-correlation |
| `TextCNN(vocab_size, embed_size, kernel_sizes, num_channels)` | textCNN 模型，含 dual embedding + 多 kernel + max-over-time pooling |



## 5. 自然语言推断与 SNLI 数据集

### 5.1 任务定义

给定 **premise** 与 **hypothesis** 两段文本，判定二者的逻辑关系：

| 标签 | 说明 |
| ---- | ---- |
| **entailment**（蕴涵） | hypothesis 可由 premise 推出 |
| **contradiction**（矛盾） | hypothesis 的否定可由 premise 推出 |
| **neutral**（中性） | 既不蕴涵也不矛盾 |

NLI 又称 *recognizing textual entailment*，被广泛用作衡量句对级语义理解能力的基准任务。

### 5.2 SNLI 数据集

**Stanford Natural Language Inference (SNLI)**：

- 约 550 000 训练对、10 000 测试对；
- 三种标签数量基本平衡（实际略偏 `entailment`）；
- 文件中每行：`label \t premise \t hypothesis \t ...`，标签需过滤掉 `-`（标注不一致样本）。

### 5.3 预处理

1. **`read_snli`**：用正则去掉括号、合并多余空格，过滤无效标签；
2. **Tokenization & Vocab**：在前提与假设的合并 token 上构造 vocab，`min_freq=5`，加入 `<pad>`；
3. **Padding**：设序列长度 `num_steps=50`，不足补 `<pad>`，超出截断；
4. **Dataset**：`SNLIDataset.__getitem__` 返回 `((premise, hypothesis), label)`。

### 5.4 关键函数 / 类

| 名称 | 作用 |
| ---- | ---- |
| `read_snli(data_dir, is_train)` | 解析 SNLI，返回 `(premises, hypotheses, labels)` |
| `SNLIDataset(dataset, num_steps, vocab=None)` | `torch.utils.data.Dataset` 封装，支持复用训练集 vocab |
| `load_data_snli(batch_size, num_steps=50)` | 返回 `train_iter, test_iter, vocab` |

> 测试集 **必须** 复用训练集的 vocab，避免引入训练时未见的 token。



## 6. 自然语言推断：Decomposable Attention

### 6.1 整体思想

Parikh et al. (2016) 提出 **Decomposable Attention Model**：用 attention + MLP 替代 RNN / CNN，参数更少且当时取得 SNLI SOTA。共三步：

1. **Attend**（软对齐）；
2. **Compare**（逐 token 比较）；
3. **Aggregate**（汇总分类）。

整体输入仅由 GloVe 词向量提供，不显式建模 token 顺序。

### 6.2 Attend：跨序列软对齐

记 premise 与 hypothesis 为 $\mathbf{A}=(\mathbf{a}_1,\dots,\mathbf{a}_m)$、$\mathbf{B}=(\mathbf{b}_1,\dots,\mathbf{b}_n)$，$\mathbf{a}_i,\mathbf{b}_j\in\mathbb{R}^d$。

注意力打分：

$$
e_{ij} = f(\mathbf{a}_i)^\top f(\mathbf{b}_j),
$$

其中 $f$ 是共享的 MLP。**Decomposition trick**：$f$ 分别作用于单侧 token 而不是对偶组合，使得复杂度从 $O(mn)$ 降至 $O(m+n)$。

软对齐表示：

$$
\boldsymbol{\beta}_i = \sum_{j=1}^{n}\frac{\exp(e_{ij})}{\sum_{k=1}^{n}\exp(e_{ik})}\mathbf{b}_j,
\qquad
\boldsymbol{\alpha}_j = \sum_{i=1}^{m}\frac{\exp(e_{ij})}{\sum_{k=1}^{m}\exp(e_{kj})}\mathbf{a}_i.
$$

- $\boldsymbol{\beta}_i$：与 $\mathbf{a}_i$ 软对齐的 hypothesis 信息；
- $\boldsymbol{\alpha}_j$：与 $\mathbf{b}_j$ 软对齐的 premise 信息。

### 6.3 Compare：拼接后逐位比较

将每个 token 与其软对齐表示拼接，喂入共享 MLP $g$：

$$
\mathbf{v}_{A,i} = g([\mathbf{a}_i, \boldsymbol{\beta}_i]), \quad
\mathbf{v}_{B,j} = g([\mathbf{b}_j, \boldsymbol{\alpha}_j]).
$$

### 6.4 Aggregate：求和后分类

两侧分别按 token 求和，再拼接送入 MLP $h$：

$$
\mathbf{v}_A = \sum_{i=1}^{m}\mathbf{v}_{A,i},\quad
\mathbf{v}_B = \sum_{j=1}^{n}\mathbf{v}_{B,j},\quad
\hat{\mathbf{y}} = h([\mathbf{v}_A, \mathbf{v}_B]).
$$

输出 3 维 logits 对应 entailment / contradiction / neutral。

### 6.5 模型结构与训练

- `embed_size=100`（GloVe `glove.6b.100d`）；
- `num_hiddens=200`，所有 MLP（`f, g, h`）共享同一形状结构（含 Dropout）；
- 优化：`Adam, lr=0.001`，`num_epochs=4`，batch size 256；
- 典型结果：train acc ≈ 0.81，test acc ≈ 0.82。

### 6.6 关键函数 / 类

| 名称 | 作用 |
| ---- | ---- |
| `mlp(num_inputs, num_hiddens, flatten)` | 通用两层 MLP（含 Dropout / ReLU），`flatten` 控制是否在 batch 维之外展平 |
| `Attend(num_inputs, num_hiddens)` | 实现 attend 步骤，返回 `beta, alpha` |
| `Compare(num_inputs, num_hiddens)` | 实现 compare 步骤，返回 `V_A, V_B` |
| `Aggregate(num_inputs, num_hiddens, num_outputs)` | 实现 aggregate + 分类，返回 `Y_hat` |
| `DecomposableAttention(vocab, embed_size, num_hiddens, ...)` | 组合 embedding + attend + compare + aggregate |
| `predict_snli(net, vocab, premise, hypothesis)` | 推断三分类标签 |



## 7. BERT 微调适配下游任务

BERT 通过 task-agnostic 的 `<cls>` / `<sep>` 输入格式，使下游应用只需 **额外一层全连接** 即可。下游训练时 BERT 全部参数会被一起微调，仅 MLM / NSP 头不再更新。

可以将 NLP 应用归纳为两类：

### 7.1 序列级任务（Sequence-level）

输入：单段文本或文本对；输出：单一标签或回归值。BERT 取 **`<cls>` 位置的输出向量** 经 MLP 得到结果。

| 子任务 | 输入 | 例子 |
| ------ | ---- | ---- |
| **Single-text classification** | 单文本 | sentiment analysis、CoLA（grammatical acceptability） |
| **Text-pair classification** | 文本对 | natural language inference |
| **Text-pair regression** | 文本对 | semantic textual similarity（STS-B，0–5 连续分） |

> 回归任务仅需把分类头改成输出 1 维并将损失换为 MSE。

### 7.2 词元级任务（Token-level）

对输入序列中的 **每一个 token** 输出标签。BERT 取每个位置的输出向量，送入 **共享** 的 MLP head。

| 子任务 | 输入 | 例子 |
| ------ | ---- | ---- |
| **Text tagging** | 单文本 | POS tagging（如 Penn Treebank tagset） |
| **Question answering** | 段落 + 问题 | SQuAD v1.1（预测段落中片段的 start / end） |

**SQuAD 输出方式**：

- 用两组独立的全连接层把每个 token 的表示分别投到 start score $s_i$ 与 end score $e_i$；
- softmax 后得到 start / end 位置概率分布；
- 训练目标：最大化真实 start / end 的对数似然；
- 推理：在 $i \le j$ 上选择 $s_i + e_j$ 最大的片段。



## 8. 自然语言推断：微调 BERT

### 8.1 任务设置

把 SNLI 视为 **text-pair classification**：

- 输入序列：`<cls>` + premise + `<sep>` + hypothesis + `<sep>`；
- segment id 用 0 / 1 区分两段；
- 在 `<cls>` 位置的 BERT 输出上接 MLP 输出 3 类 logits。

### 8.2 加载预训练 BERT

提供两个预训练版本（在 WikiText-2 上预训练，仅作演示，不代表原始 BERT 规模）：

| 版本 | num_hiddens | ffn_num_hiddens | num_heads | num_layers | 备注 |
| ---- | ----------- | --------------- | --------- | ---------- | ---- |
| `bert.base` | 768 | 3072 | 12 | 12 | 与 BERT-BASE 同规模 |
| `bert.small` | 256 | 512 | 4 | 2 | 演示用小模型 |

每个 zip 内包含：

- `vocab.json`：词表；
- `pretrained.params`：BERTModel 参数。

### 8.3 数据集封装：SNLIBERTDataset

为加速生成，使用 `multiprocessing.Pool(4)` 并行处理：

1. 对 premise / hypothesis 各自做 lowercased tokenization；
2. **截断长文本**：保留 3 个特殊 token 位置（`<cls>` + 两个 `<sep>`），超出 `max_len-3` 时反复弹出较长那侧的最后一个 token（`_truncate_pair_of_tokens`）；
3. 调用 `d2l.get_tokens_and_segments` 拼接特殊词元，生成 `tokens` 与 `segments`；
4. 用 vocab 转 id 并 pad 到 `max_len`，同时记录 `valid_len`；
5. `__getitem__` 返回 `((token_ids, segments, valid_len), label)`。

> 实验配置：`batch_size=512, max_len=128`；显存不足时减 `batch_size`；原始 BERT 用 `max_len=512`。

### 8.4 微调用分类器

`BERTClassifier`：

- `self.encoder = bert.encoder`（共享预训练 encoder）；
- `self.hidden = bert.hidden`（即 NSP 中 `<cls>` 后的 `Linear → Tanh`）；
- `self.output = nn.Linear(256, 3)`（**新增**，3 类 logits）。

forward：取 `encoded_X[:, 0, :]`（`<cls>` 位置）→ `hidden` → `output`。

### 8.5 训练细节与 stale gradient

- 优化：`Adam, lr=1e-4`，`num_epochs=5`，损失 `CrossEntropyLoss(reduction='none')`；
- `MaskLM` 与 `NextSentencePred` 的参数仍在 `net.parameters()` 内，但下游只用到 `<cls>` 表示，这些 head 的参数 **没有梯度** → 设置 `ignore_stale_grad=True` 跳过它们；
- 典型结果：train acc ≈ 0.79，test acc ≈ 0.78（受 `bert.small` 与 WikiText-2 预训练规模限制；换用 `bert.base` 通常能突破 0.86）。

### 8.6 关键函数 / 类

| 名称 | 作用 |
| ---- | ---- |
| `load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens, num_heads, num_layers, dropout, max_len, devices)` | 实例化 `BERTModel` 并加载预训练参数与 `vocab` |
| `SNLIBERTDataset(dataset, max_len, vocab)` | 把 SNLI 转成 BERT 输入：`(token_ids, segments, valid_len)` + label |
| `BERTClassifier(bert)` | 在 BERT encoder + hidden 之上加分类头，输出 3 类 logits |



## 9. 章节小结

1. NLP 下游任务可按 **输入 / 输出粒度** 划分为序列级（单文本 / 文本对分类、回归）与词元级（tagging、QA）；
2. **情感分析** 是单文本极性分类的代表，IMDb 是常用基准；
3. 在 IMDb 上，**BiLSTM + GloVe** 与 **textCNN + 双 embedding** 都能达到 ≈ 0.86 的测试精度，textCNN 通过 1-D 卷积 + max-over-time pooling 提取 n-gram 特征，训练更快；
4. **NLI** 通过判定 entailment / contradiction / neutral 衡量句对级语义理解，**SNLI** 是常用数据集；
5. **Decomposable Attention** 用三段（attend / compare / aggregate）和 decomposition trick 把对齐复杂度从 $O(mn)$ 降至 $O(m+n)$，无 RNN / CNN 即可在 SNLI 取得不俗表现；
6. **BERT 微调** 用统一的 `<cls>` / `<sep>` 输入格式适配多种任务，仅需新增一层全连接；微调时全部参数都会更新，但只与预训练相关的 head（MLM / NSP）会出现 stale gradient，需要在训练循环中显式忽略；
7. 计算受限时，精心构造的小模型 + 预训练词向量仍是可行的工程选项；资源充足时，fine-tune BERT 通常是最简单且最强的基线。



## References

- 《动手学深度学习》第 15 章 [自然语言处理：应用](https://zh-v2.d2l.ai/chapter_natural-language-processing-applications/index.html)
- Maas et al., *Learning Word Vectors for Sentiment Analysis*, 2011（IMDb）
- Kim, *Convolutional Neural Networks for Sentence Classification*, 2014（textCNN）
- Collobert et al., *Natural Language Processing (Almost) from Scratch*, 2011（max-over-time pooling）
- Bowman et al., *A Large Annotated Corpus for Learning Natural Language Inference*, 2015（SNLI）
- Parikh et al., *A Decomposable Attention Model for Natural Language Inference*, 2016
- Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, 2018
- Warstadt, Singh, Bowman, *Neural Network Acceptability Judgments*, 2019（CoLA）
- Cer et al., *SemEval-2017 Task 1: Semantic Textual Similarity*, 2017（STS-B）
- Rajpurkar et al., *SQuAD: 100,000+ Questions for Machine Comprehension of Text*, 2016
