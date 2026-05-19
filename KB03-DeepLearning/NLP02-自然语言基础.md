# 自然语言基础：词嵌入与预训练

本笔记基于《动手学深度学习》第 14 章 *Natural Language Processing: Pretraining* 整理，聚焦如何从大规模语料中以 **自监督**（self-supervised）方式预训练 token 的向量表示（word embedding），覆盖 word2vec、GloVe、fastText、相似性 & 类比任务等内容。

> 📌 **后续阅读**：上下文敏感的预训练模型（ELMo / GPT / BERT）独立成文，见 [NLP01-BERT与预训练模型](./NLP01-BERT与预训练模型.md)。
>
> 代码具体实现见 `chapter_natural-language-processing-pretraining/` 下的对应 `.ipynb` 文件，本笔记仅记录关键函数名。



## 1. 词嵌入概览（Word Embedding）

### 1.1 为何 one-hot 不是好的选择

将词典大小记为 $N$，one-hot vector 为长度 $N$ 的稀疏向量。其问题：

- 任意两个不同词的 one-hot vector 间 **cosine similarity 恒为 0**，无法表达词之间的相似度；
- 维度随词表线性增长，计算与存储开销大。

cosine similarity 定义：

$$
\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|} \in [-1,1].
$$



### 1.2 词嵌入的核心思想

将每个 token 映射到一个固定长度的稠密实向量（dense vector），让语义相似的词在向量空间中相近。常见预训练手段：

| 方法 | 训练目标 | 输入粒度 | 上下文敏感 |
| ---- | -------- | -------- | ---------- |
| word2vec (Skip-Gram / CBOW) | 局部 context 预测 | word | 否 |
| GloVe | 全局共现矩阵拟合 | word | 否 |
| fastText | subword n-gram 求和 | subword | 否 |
| BERT *(见后续笔记)* | MLM + NSP，Transformer encoder | subword (WordPiece) | 是 |

> **上下文无关 vs 上下文敏感**：word2vec / GloVe / fastText 是 **context-independent**，同一个词无论上下文都用同一个向量；ELMo / GPT / BERT 是 **context-sensitive**，向量随上下文变化。本笔记只覆盖前者，后者见 [NLP01-BERT与预训练模型](./NLP01-BERT与预训练模型.md)。



## 2. word2vec

word2vec 包含两个 self-supervised 模型：**Skip-Gram** 与 **CBOW**。

对词典中索引为 $i$ 的任意词，分别用 $\mathbf{v}_i\in\mathbb{R}^d$ 与 $\mathbf{u}_i\in\mathbb{R}^d$ 表示其作为 *center word* 与 *context word* 的向量。



### 2.1 Skip-Gram

**核心思想**：给定 center word 预测窗口内的 context words。

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/skip-gram.svg" width="380px">
</div>

给定 center word $w_c$，生成 context word $w_o$ 的条件概率通过 softmax 建模：

$$
P(w_o \mid w_c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{\sum_{i\in\mathcal V}\exp(\mathbf{u}_i^\top \mathbf{v}_c)}.
$$

长度为 $T$ 的文本序列、窗口大小为 $m$ 时，Skip-Gram 的似然函数：

$$
\prod_{t=1}^{T}\prod_{-m\le j\le m,\ j\ne 0} P(w^{(t+j)} \mid w^{(t)}).
$$

对应的负对数损失：

$$
-\sum_{t=1}^{T}\sum_{-m\le j\le m,\ j\ne 0}\log P(w^{(t+j)} \mid w^{(t)}).
$$

对中心词向量 $\mathbf{v}_c$ 的梯度为：

$$
\frac{\partial \log P(w_o\mid w_c)}{\partial \mathbf{v}_c}
= \mathbf{u}_o - \sum_{j\in\mathcal V} P(w_j\mid w_c)\mathbf{u}_j.
$$

> 训练完成后，Skip-Gram 通常使用 **中心词向量** $\mathbf{v}_i$ 作为该词的最终表示。



### 2.2 CBOW（Continuous Bag of Words）

**核心思想**：给定窗口内若干 context words 预测 center word（与 Skip-Gram 输入输出方向相反）。

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/cbow.svg" width="380px">
</div>

由于存在多个 context words，CBOW 对其向量取平均：

$$
\bar{\mathbf{v}}_o = \frac{1}{2m}\sum_{k=1}^{2m}\mathbf{v}_{o_k}.
$$

条件概率：

$$
P(w_c \mid \mathcal{W}_o) = \frac{\exp(\mathbf{u}_c^\top \bar{\mathbf{v}}_o)}{\sum_{i\in\mathcal V}\exp(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)}.
$$

> 训练完成后，CBOW 通常使用 **上下文词向量** $\mathbf{v}_i$ 作为该词的最终表示（符号约定与 Skip-Gram 相反）。



### 2.3 计算瓶颈

Skip-Gram 和 CBOW 的梯度中含 $\sum_{i\in\mathcal V}\exp(\mathbf{u}_i^\top \mathbf{v}_c)$，每步更新需要遍历整个 vocabulary，开销 $\mathcal O(|\mathcal V|)$。当词表规模数十万到百万级时不可接受。

为此需要 **近似训练**：Negative Sampling、Hierarchical Softmax。



## 3. 近似训练（Approximate Training）

### 3.1 Negative Sampling

将原 softmax 多分类问题改造为 **多个二分类问题**。事件 $D=1$ 表示 $w_o$ 来自 $w_c$ 的 context window：

$$
P(D=1 \mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c), \quad
\sigma(x)=\frac{1}{1+\exp(-x)}.
$$

为防止平凡解，再从噪声分布 $P(w)$ 采样 $K$ 个 **noise word** 作为负例：

$$
P(w^{(t+j)}\mid w^{(t)}) = P(D=1\mid w^{(t)},w^{(t+j)})\prod_{k=1}^{K} P(D=0\mid w^{(t)},w_k).
$$

对应的损失：

$$
-\log\sigma(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}) - \sum_{k=1}^{K}\log\sigma(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}).
$$

- 每步梯度计算复杂度由 $\mathcal O(|\mathcal V|)$ 降至 $\mathcal O(K)$；
- 通常使用 unigram 分布的 3/4 次幂作为采样分布：$P(w)\propto f(w)^{0.75}$。



### 3.2 Hierarchical Softmax

将词表组织成一棵二叉树（如 Huffman tree），每个叶节点对应一个词。设词 $w$ 在树中从根到叶的路径长度为 $L(w)$，节点向量为 $\mathbf{u}_{n(w,j)}$，则：

$$
P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1}\sigma\big([\![n(w_o,j+1)=\text{leftChild}(n(w_o,j))]\!]\cdot \mathbf{u}_{n(w_o,j)}^\top \mathbf{v}_c\big).
$$

- $[\![x]\!]=1$ 若 $x$ 为真，否则 $-1$；
- 每步计算复杂度由 $\mathcal O(|\mathcal V|)$ 降至 $\mathcal O(\log_2|\mathcal V|)$；
- 由 $\sigma(x)+\sigma(-x)=1$ 可证 $\sum_{w\in\mathcal V}P(w\mid w_c)=1$。

| 方法 | 复杂度 | 思路 |
| ---- | ------ | ---- |
| Negative Sampling | $\mathcal O(K)$ | 二分类 + 噪声词 |
| Hierarchical Softmax | $\mathcal O(\log\|\mathcal V\|)$ | 二叉树路径乘积 |



## 4. 用于预训练词嵌入的数据集

以 PTB（Penn Tree Bank）数据集 + Skip-Gram + Negative Sampling 为例。

### 4.1 关键步骤

1. **读取语料并构建 Vocab**：低频词以 `<unk>` 替换；
2. **下采样高频词（subsampling）**：高频词 (如 the、a) 提供信息有限，按概率丢弃：

   $$
   P(w_i) = \max\left(1-\sqrt{\frac{t}{f(w_i)}},\ 0\right),\quad t=10^{-4}.
   $$

3. **提取 center 与 context**：对每个 center word，随机采样一个不超过 `max_window_size` 的窗口大小，窗口内其余 token 为 context；
4. **Negative Sampling**：按 $f(w)^{0.75}$ 加权采样噪声词，且噪声词不能出现在当前 context 中；
5. **小批量整合**：将不等长 (context + negative) 用 0 填充至 `max_len`，并构造：
   - `centers`：(batch, 1)
   - `contexts_negatives`：(batch, max_len)
   - `masks`：标识有效位置（用于损失计算时屏蔽 padding）
   - `labels`：标识正例 (context) vs 负例 (negative)。

### 4.2 关键函数

| 函数 | 作用 |
| ---- | ---- |
| `read_ptb` | 加载 PTB 语料 |
| `subsample` | 下采样高频词 |
| `get_centers_and_contexts` | 提取 center & context |
| `RandomGenerator` | 按权重缓存式抽样 |
| `get_negatives` | 生成 negative samples |
| `batchify` | 小批量整合（含 mask、label） |
| `load_data_ptb` | 一次性获取 `data_iter` 与 `vocab` |



## 5. 预训练 word2vec

### 5.1 模型实现

- 使用两个 `nn.Embedding` 分别承担 center vector $\mathbf v$ 与 context vector $\mathbf u$；
- 前向通过 batch matrix multiplication 计算 $\mathbf v_c^\top \mathbf u$，输出形状 `(batch, 1, max_len)`；
- 损失使用带掩码的 binary cross entropy with logits（`SigmoidBCELoss`）。

### 5.2 关键函数 / 类

| 名称 | 作用 |
| ---- | ---- |
| `skip_gram(center, contexts_and_negatives, embed_v, embed_u)` | Skip-Gram 前向 |
| `SigmoidBCELoss` | 带 mask 的 BCEWithLogitsLoss |
| `train(net, data_iter, lr, num_epochs)` | 训练循环 |
| `get_similar_tokens(query_token, k, embed)` | 用 cosine similarity 查最相似的 $k$ 个词 |

### 5.3 应用

训练后的 embedding 可用于：

- **词相似度**：cosine similarity top-k；
- **下游 NLP 任务**的初始化向量。



## 6. GloVe（Global Vectors）

### 6.1 与 Skip-Gram 的关系

记 $x_{ij}$ 为词 $w_j$ 出现在 $w_i$ 上下文窗口的全局共现计数，$x_i=\sum_j x_{ij}$，$p_{ij}=x_{ij}/x_i$。

Skip-Gram 的损失可改写为加权交叉熵：

$$
-\sum_{i\in\mathcal V} x_i \sum_{j\in\mathcal V} p_{ij}\log q_{ij}.
$$

其问题：

- $q_{ij}$ 归一化代价高昂；
- 大语料下罕见事件被交叉熵不成比例地放大。

### 6.2 GloVe 的三点修改

1. 用 $p'_{ij}=x_{ij}$ 与 $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ 取 log 后做 **squared loss**；
2. 为每个词加入两个标量偏置：center 偏置 $b_i$，context 偏置 $c_j$；
3. 用单调递增的权重函数 $h(x_{ij})$ 替代固定权重：

   $$
   h(x) = \begin{cases} (x/c)^\alpha, & x<c \\ 1, & x\ge c\end{cases},\quad c=100,\ \alpha=0.75.
   $$

最终损失：

$$
\sum_{i\in\mathcal V}\sum_{j\in\mathcal V} h(x_{ij})\big(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log x_{ij}\big)^2.
$$

### 6.3 特点

- 由 $x_{ij}=x_{ji}$，GloVe 拟合的是 **对称** 概率 $\log x_{ij}$，中心词向量与上下文词向量在数学上等价，最终输出常取二者之和；
- 从 **共现概率比值** $p_{ij}/p_{ik}$ 角度，可推导出 $\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log x_{ij}$ 的拟合目标。



## 7. 子词嵌入（Subword Embedding）

word2vec 和 GloVe 把每个词视为原子单元，无法利用 morphology（如 helps / helped / helping 共享词根）。

### 7.1 fastText

将每个 word 分解为 character n-gram（通常 $n\in[3,6]$）外加一个表示整词的特殊 token：

- 例：`where` 在 $n=3$ 时切分为 `<wh`、`whe`、`her`、`ere`、`re>`、`<where>`；
- 词向量 = 其所有 subword 向量之和：

  $$
  \mathbf v_w = \sum_{g\in\mathcal G_w} \mathbf z_g.
  $$

- 其余结构与 Skip-Gram 相同；
- 优势：罕见词、OOV (out-of-vocabulary) 词可通过 subword 复用参数获得合理表示；
- 代价：subword 数量大于 word 数量，模型更大、计算更慢。



### 7.2 Byte Pair Encoding（BPE）

fastText 要求 n-gram 长度固定，词表大小不可控。**BPE** 通过统计压缩允许 **可变长度 subword**：

**算法步骤**：

1. 初始 symbol 词表：所有单字符（如英文 26 字母）+ 特殊词尾 `_` + `[UNK]`；
2. 将语料中每个 word 拆为字符序列并附加 `_` 作为词尾；
3. 重复执行：统计所有相邻符号对的频次 → 合并出现次数最多的一对，形成新 symbol；
4. 重复 $K$ 次后得到最终 symbol 词表。

**应用切分**：贪心地从左向右匹配尽可能长的 subword（若失败则回退至更短，最终为 `[UNK]`）。

**关键函数**：

| 函数 | 作用 |
| ---- | ---- |
| `get_max_freq_pair(token_freqs)` | 统计并返回最高频的相邻符号对 |
| `merge_symbols(max_freq_pair, token_freqs, symbols)` | 合并选定对、更新词表 |
| `segment_BPE(tokens, symbols)` | 用学到的 symbol 切分新 token |

> BPE 与其变体（WordPiece、SentencePiece）被 GPT-2、RoBERTa、BERT 等主流预训练模型用作输入子词切分方案。



## 8. 词的相似性与类比任务（Similarity & Analogy）

预训练好的 GloVe / fastText 向量可直接复用，用于：

### 8.1 词相似度

计算 query 与全词表中所有词向量的 cosine similarity，取 top-$k$。

### 8.2 词类比（Analogy）

形式：$a:b :: c:d$，已知 $a,b,c$ 求 $d$。利用向量算术：

$$
\text{vec}(d) \approx \text{vec}(b) - \text{vec}(a) + \text{vec}(c).
$$

示例：

- 性别：man : woman :: son : **daughter**
- 首都：beijing : china :: tokyo : **japan**
- 比较级：bad : worst :: big : **biggest**
- 时态：do : did :: go : **went**

### 8.3 关键类 / 函数

| 名称 | 作用 |
| ---- | ---- |
| `TokenEmbedding(embedding_name)` | 加载 GloVe / fastText 预训练向量 |
| `knn(W, x, k)` | 在向量矩阵 $W$ 中找与 $x$ cosine 相似度 top-$k$ |
| `get_similar_tokens(query_token, k, embed)` | 相似度任务 |
| `get_analogy(token_a, token_b, token_c, embed)` | 类比任务 |



## 9. 章节小结

1. **词嵌入 (Word Embedding)** 用稠密向量解决 one-hot 不能表达相似度的问题；
2. **word2vec** 提供 Skip-Gram / CBOW 两个自监督模型；当词表巨大时，借助 **Negative Sampling** 或 **Hierarchical Softmax** 把每步开销由 $\mathcal O(|\mathcal V|)$ 降低；
3. **GloVe** 用 squared loss 拟合全局共现统计 $\log x_{ij}$，避免 softmax 归一化代价；
4. **fastText / BPE** 引入 subword 表示，提升罕见词与 OOV 表现，词表大小可控；
5. word2vec、GloVe、fastText 均 **context-independent**，无法解决一词多义问题——这正是后续 **ELMo / GPT / BERT** 等 context-sensitive 预训练模型的切入点，详见 [NLP01-BERT与预训练模型](./NLP01-BERT与预训练模型.md)。



## References

- 《动手学深度学习》第 14 章 [自然语言处理：预训练](https://zh-v2.d2l.ai/chapter_natural-language-processing-pretraining/index.html)
- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality*, 2013（word2vec / Negative Sampling）
- Pennington, Socher, Manning, *GloVe: Global Vectors for Word Representation*, 2014
- Bojanowski et al., *Enriching Word Vectors with Subword Information*, 2017（fastText）
- Sennrich, Haddow, Birch, *Neural Machine Translation of Rare Words with Subword Units*, 2015（BPE）
