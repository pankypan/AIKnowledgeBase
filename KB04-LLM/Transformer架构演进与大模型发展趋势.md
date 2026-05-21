# Transformer 架构演进与大模型发展趋势

> 本文整理自一期关于"大语言模型架构演进"的视频讲解，梳理了从 GPT-2（2019）到 DeepSeek V4（2026）共 67 个核心开源模型的架构变化，从位置编码、归一化、残差连接、FFN 层到注意力机制，系统性地展示了 Transformer 架构各模块的创新脉络与发展瓶颈。

---

## 一、Transformer 架构简化回顾

> 📄 原始论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)

自 ChatGPT 出现至今已三年多，所有主流大模型都只是 Transformer 架构的一个分支。其核心结构可以简化为：

```
输入 Token → 编码器 → [注意力层 + FFN层] × N → 解码器 → 输出 Token
```

| 模块 | 功能 |
|------|------|
| **编码器（Embedding）** | 将文字（Token）转换为计算用的向量 |
| **注意力层（Attention）** | 向量间的加权求和，让每个词包含上下文信息 |
| **前馈神经网络层（FFN）** | 对每个 Token 向量做非线性变换，是预测下一个词的关键步骤 |
| **解码器（Output Head）** | 将向量转换回文字，完成"预测下一个词"的任务 |

---

## 二、位置编码（Positional Encoding）的演进

位置编码的作用是让模型感知 Token 在序列中的位置信息。

### 2.1 演进路线

| 方法 | 说明 | 论文 |
|------|------|------|
| **固定位置编码（Sinusoidal PE）** | Transformer 原始论文方案，通过正弦/余弦函数算出固定值加到原向量上 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| **旋转位置编码（RoPE）** | 通过旋转变换使向量在计算点积时自然获得位置相关性，已成为主流方案 | [RoFormer](https://arxiv.org/abs/2104.09864) |
| **YaRN** | 对 RoPE 的扩展方法，支持更长的上下文窗口 | [YaRN](https://arxiv.org/abs/2309.00071) |
| **NoPE（No Position Encoding）** | 不使用显式位置编码，在某些场景下被证明有效 | [NoPE](https://arxiv.org/abs/2305.19466) |

### 2.2 现状

位置编码的创新空间已基本被挖掘殆尽，实际应用已经**世界线收束**——绝大多数模型采用 RoPE 或其变体。

---

## 三、归一化（Normalization）的演进

归一化的目的是将向量控制在稳定范围内（均值→0，方差→1），保证训练稳定性。

### 3.1 归一化算法

| 方法 | 说明 | 论文 |
|------|------|------|
| **LayerNorm** | Transformer 原始论文采用的层归一化方法 | [Layer Normalization](https://arxiv.org/abs/1607.06450) |
| **RMSNorm** | 当代大模型更常用，计算更高效，省去了均值计算步骤 | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) |

### 3.2 归一化位置

归一化层可以安插在模型中的不同位置，产生多种组合：

> 📄 参考：[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)

| 位置策略 | 说明 |
|----------|------|
| **Pre-Norm** | 归一化在注意力层/FFN层之前 |
| **Post-Norm** | 归一化在注意力层/FFN层之后 |
| **Sandwich Norm** | 前后各一个归一化层 |
| **残差连接前/后** | Post-Norm 还可以放在残差连接的不同位置 |

### 3.3 归一化作用对象

归一化不仅作用于主干向量，还可以作用于注意力层内部的 Q/K/V 向量：

- **QNorm**：对 Query 向量归一化
- **KNorm**：对 Key 向量归一化
- **VNorm**：对 Value 向量归一化
- **QKNorm**：同时对 Q 和 K 归一化
- **KVNorm**：同时对 K 和 V 归一化

### 3.4 现状

所有位置组合与作用对象的搭配均已被充分探索。

---

## 四、残差连接（Residual Connection）的演进

### 4.1 原始形式

Transformer 中的残差连接就是一个简单的加法操作：

$$\text{output} = x + \text{SubLayer}(x)$$

多年来几乎无人触碰这一"默认正确"的部分。

### 4.2 近期创新

| 方法 | 提出者 | 说明 | 论文 |
|------|--------|------|------|
| **HC（Hyper-Connections）** | 字节跳动 | 动态调整不同深度特征间的连接强度 | [Hyper-Connections](https://arxiv.org/abs/2409.19606) |
| **MHC（Multi-Head Connection）** | DeepSeek | 多头残差连接 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) |
| **Attention Residual** | Kimi | 基于注意力的残差连接 | [Kimi-Linear](https://arxiv.org/abs/2510.26692) |

### 4.3 现状

这块还比较新，目前仍处于探索阶段，花样有限但已开始动刀。

---

## 五、前馈神经网络层（FFN）→ 混合专家模型（MoE）

### 5.1 问题

FFN 层本质是全连接神经网络，在大模型中占据了绝大部分参数量（通常占总参数的 2/3）。

### 5.2 MoE 解决思路

```
原始 FFN（一个大网络）
    ↓ 拆分
多个小 FFN（Expert 专家） + 可训练路由层（Router）
    ↓ 效果
总参数量大，但每次推理只激活少量专家 → 效率与能力的平衡
```

### 5.3 演进路线

| 方法 | 说明 | 论文 |
|------|------|------|
| **标准 MoE** | 将大 FFN 拆为多个小专家网络，通过路由层分配 Token | [Switch Transformers](https://arxiv.org/abs/2101.03961) |
| **DeepSeek MoE** | 专家拆分更细粒度，引入共享专家（Shared Expert）机制 | [DeepSeekMoE](https://arxiv.org/abs/2401.06066) |

### 5.4 参数标注方式

现在模型名称中常见的 `AxxB` 表示法：
- 例如 **DeepSeek-V3 671B/37B**：总参数量 671B，每次推理只激活 37B

### 5.5 术语

| 术语 | 含义 |
|------|------|
| **Dense（稠密模型）** | 所有参数都参与计算（传统方式） |
| **Sparse（稀疏模型）** | 只激活部分参数（即 MoE 架构） |

### 5.6 现状

当代大模型基本全面 MoE 化，只是专家数量和激活比例各有配置。

---

## 六、注意力机制（Attention）的演进

注意力层是 Transformer 的精髓，也是创新最密集的领域。其核心问题是：**每个 Token 都需要和所有其他 Token 计算一次注意力，计算复杂度为 O(n²)**。

### 6.1 三大优化方向

#### 方向一：降低单次计算量（数量不变，难度降低）

仍然是所有 Token 之间都计算注意力，但通过压缩向量维度等方式减少计算量。

| 方法 | 说明 | 论文 |
|------|------|------|
| **MHA（Multi-Head Attention）** | Transformer 原始方案，多头注意力 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| **MQA（Multi-Query Attention）** | 多个 Query 头共享一组 K/V，大幅减少 KV Cache | [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150) |
| **GQA（Grouped-Query Attention）** | MHA 与 MQA 的折中，将 Query 头分组共享 K/V | [GQA](https://arxiv.org/abs/2305.13245) |
| **MLA（Multi-head Latent Attention）** | DeepSeek 提出，通过低秩压缩 KV 到潜空间，推理时无需传统 KV Cache | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) |

#### 方向二：减少计算数量（稀疏注意力）

不再对所有 Token 两两计算，而是选择性地计算部分 Token 对。

| 方法 | 说明 | 论文 |
|------|------|------|
| **SWA（Sliding Window Attention）** | 只计算固定窗口内的 Token | [Longformer](https://arxiv.org/abs/2004.05150) / [Mistral 7B](https://arxiv.org/abs/2310.06825) |
| **DSA / CSA / HCA** | DeepSeek 的 Native Sparse Attention 系列方案 | [NSA](https://arxiv.org/abs/2502.11089) |

#### 方向三：线性化注意力（复杂度从 O(n²) → O(n)）

彻底改变注意力的计算方式，使其随序列长度线性增长。

| 方法 | 说明 | 论文 |
|------|------|------|
| **KDA（Kimi Delta Attention）** | Kimi 提出的带衰减的线性注意力 | [Kimi-Linear](https://arxiv.org/abs/2510.26692) |
| **GateDeltaNet** | 门控增量网络，结合 gating 与 delta rule | [Gated Delta Networks](https://arxiv.org/abs/2412.06464) |
| **Lightning Attention** | IO-aware 的线性注意力高效实现 | [Lightning Attention-2](https://arxiv.org/abs/2401.04658) |
| **Mamba** | 基于选择性状态空间模型（SSM）的序列建模 | [Mamba](https://arxiv.org/abs/2312.00752) |

### 6.2 混合注意力（Hybrid Attention）

由于线性注意力与传统注意力各有优劣，当前趋势是将两者**混合使用**——部分层用传统注意力，部分层用线性注意力。这一领域目前百花齐放，尚无定论。

### 6.3 现状

注意力机制的创新极度密集，能改的点已基本被穷举探索。

---

## 七、整体趋势与三大定律

### 7.1 架构创新的瓶颈

Transformer 架构的五大核心模块——位置编码、归一化、残差连接、FFN、注意力——能动刀子的地方已所剩无几。单纯依靠优化 Transformer 架构来突破模型能力上限，这条路的空间已经不大。

来自斯坦福年度 AI 报告的证据：
- 开源模型与闭源模型的差距在不断缩小
- 各厂商之间的差距也在不断缩小

> 如果发展是线性的，差距应保持不变；如果是指数型的，差距应越拉越大；**只有发展逐渐放缓时，差距才会逐渐缩小**。

### 7.2 AI 发展的三大定律

| 定律 | 核心含义 | 当前困境 | 参考 |
|------|----------|----------|------|
| **Scaling Law（规模定律）** | 模型越大、数据越多、算力越强，性能越好 | 互联网数据已基本挖尽，算力与模型规模难以短时间内大幅提升 | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) |
| **Bitter Lesson（苦涩的教训）** | 长期来看，人为设计的精巧架构 trick 在更大规模面前反而是阻碍 | 人类的"巧思"可能阻碍了模型的学习效率，尤其在架构已被过度雕琢的当下 | [The Bitter Lesson (Rich Sutton, 2019)](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) |
| **Moravec's Paradox（莫拉维克悖论）** | 对人类简单的对机器很难，反之亦然 | 机器擅长算术但不擅长物理世界交互，Transformer 架构本身不一定是"正确答案" | Moravec, *Mind Children*, 1988 |

### 7.3 未来可能的方向

- **多模态（Multimodal）**：融合视觉、语音等多种输入模态，但当前部分方案仅做了模态对齐，底层仍依赖语言模型
- **世界模型（World Model）**：试图让 AI 理解物理世界的运作规律，但边界和定义尚不清晰

---

## 八、总结

```
┌─────────────────────────────────────────────────────┐
│              Transformer 架构创新全景                  │
├─────────────────────────────────────────────────────┤
│  位置编码：Fixed PE → RoPE → YaRN → NoPE    [收束]  │
│  归一化层：LayerNorm → RMSNorm + 多种位置    [收束]  │
│  残差连接：简单加法 → HC/MHC/Attn Residual   [探索]  │
│  FFN 层 ：Dense FFN → MoE → DeepSeek MoE    [收束]  │
│  注意力层：MHA → GQA/MLA → 稀疏/线性/混合    [激烈]  │
├─────────────────────────────────────────────────────┤
│  结论：架构层面的低垂果实已基本摘完，                   │
│       单靠 Transformer 优化难以带来质的飞跃            │
└─────────────────────────────────────────────────────┘
```

**核心观点**：
1. 不应高估单个技术创新的贡献
2. 不应低估技术量变产生质变的过程
3. 未来的突破可能来自架构之外——数据范式、训练方式、多模态融合或全新的计算范式

---

## 附录：论文快速索引

| 类别 | 技术 | 论文链接 |
|------|------|----------|
| **基础架构** | Transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| **位置编码** | RoPE | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) |
| | YaRN | [YaRN: Efficient Context Window Extension of LLMs](https://arxiv.org/abs/2309.00071) |
| | NoPE | [The Impact of Positional Encoding on Length Generalization](https://arxiv.org/abs/2305.19466) |
| **归一化** | LayerNorm | [Layer Normalization](https://arxiv.org/abs/1607.06450) |
| | RMSNorm | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) |
| | Pre/Post-Norm 分析 | [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) |
| **残差连接** | Hyper-Connections | [Hyper-Connections](https://arxiv.org/abs/2409.19606) (ByteDance, ICLR 2025) |
| | Frac-Connections | [Frac-Connections](https://arxiv.org/abs/2503.14125) (ByteDance, 2025) |
| **MoE** | Switch Transformer | [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) |
| | DeepSeekMoE | [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/abs/2401.06066) |
| **注意力（压缩型）** | MQA | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) |
| | GQA | [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) |
| | MLA | [DeepSeek-V2: A Strong, Economical, and Efficient MoE Model](https://arxiv.org/abs/2405.04434) |
| **注意力（稀疏型）** | Longformer (SWA) | [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) |
| | NSA (DSA/CSA/HCA) | [Native Sparse Attention](https://arxiv.org/abs/2502.11089) (DeepSeek, ACL 2025) |
| **注意力（线性型）** | KDA | [Kimi-Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) |
| | GateDeltaNet | [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) (ICLR 2025) |
| | Lightning Attention | [Lightning Attention-2: Unlimited Sequence Lengths](https://arxiv.org/abs/2401.04658) |
| | Mamba | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) |
| **综合技术报告** | DeepSeek-V3 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) |
| **理论/定律** | Scaling Law | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020) |
| | Bitter Lesson | [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) (Rich Sutton, 2019) |
