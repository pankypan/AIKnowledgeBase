# Quantization for Large Language Models (LLMs)

## 1. 为什么需要量化

### 1.1 神经网络中的数据类型

<div align="center">
<img src="https://i-blog.csdnimg.cn/blog_migrate/8c75c00012e91c6ae6c799c4fe864b38.png" width="600px">
</div>

- **FP32**：在深度学习中，单精度浮点数格式FP32是一种广泛使用的数据格式，其可以表示很大的实数范围，足够深度学习训练和推理中使用。这种格式使用4个bytes（32bits）表示。
- **Tensor Float 32**: Tensor Float 32是Tensor Core支持新的数值类型，从NVIDIA A100中开始支持。
  - 在深度学习中，其实我们对浮点数的表示范围比较看重，而有效数字不是那么重要。
  - A100的普通FP32的峰值计算速度为19.5TOPs，而TF32的峰值计算速度为156TOPs，提升了非常多。
- **FP16**: FP16是一种半精度浮点格式，深度学习有使用FP16而不是FP32的趋势，因为较低精度的计算对于神经网络来说似乎并不重要。
  - 额外的精度没有任何作用，同时速度较慢，需要更多内存并降低通信速度。
  - 在实际测试中，FP16的精度水平已经足够应对深度学习负载，只是表示的范围不够广而已。
- **BFLOAT16**: 由Google开发的16位浮点格式称为“Brain Floating Point Format”，简称“bfloat16”。这个名字来源于“Google Brain”，这是谷歌的一个人工智能研究小组。



### 1.2 量化技术的目的

LLM大模型的量化技术主要是通过对模型参数进行压缩和量化，从而降低模型的存储和计算复杂度。具体来说如下：
- **参数压缩**：通过将模型中的**浮点数参数转换为低精度的整数参数**，量化技术可以实现参数的压缩。这不仅可以**减少模型所需的存储空间**，还可以**降低模型加载的时间**。
- **计算加速**：由于低精度整数运算的速度远快于浮点数运算，量化技术还可以通过降低计算复杂度来实现计算加速。这可以在保证模型性能的同时，**提高模型的推理速度**。


综上：量化就是把Float类型(FP32，FP16)的模型参数和激活值，用整数(Int8，Int4)来代替。同时尽可能减少量化后模型推理的误差。
1. 节省显存：减少模型的存储空间和显存的占用。
2. 降低通讯量：减少显存和Tensorcore之间的数据传输量，从而降低通讯延迟。
3. 加速计算：显卡对整数运算速度快于浮点型数据。从而加快模型推理时间。



### 1.3 量化是如何缩小模型的？

实际上，对于大模型最常见的就是8bits量化(FP8/INT8)和4bits量化(FP4/NF4/INT4)。量化通过减少每个模型权重所需的位数，显著降低了模型的大小。模型一个典型的场景是将权重从FP16（16位浮点）减少到INT4（4位整数）。


研究表明，这种影响因所使用的技术而异，较大的模型受到精度变化的影响较小。更大的型号（超过70B）即使转换为4bits也能保持其性能。
- 较大的模型（如超过70B）使用4bit量化其性能没有影响;
- 较小的模型使用8bit量化可能更好;


下面以Qwen-7B-Chat为例展示INT8和INT4量化的效果：
<div align="center">
<img src="https://i-blog.csdnimg.cn/blog_migrate/4da191b03132c33a8d447f4f8b37c8b9.png" width="600px">
</div>

- **MMLU**: 是一个新的基准，用于衡量在零样本（zero-shot）和少样本（few-shot）情形下，大模型在预训练期间获得的世界知识。它的难度从初级到高级，既考验世界知识，又考验解决问题的能力。 学科的粒度和广度使该基准成为识别模型盲点的理想选择。
- **C-Eval**: 是一个全面的中文基础模型评估套件。它包含了13948个多项选择题，涵盖了52个不同的学科和四个难度级别。
- **GSM8K**: 该数据集是由 OpenAI 发布的小学数学题数据集。
- **HumanEval**: Hand-Written Evaluation Set，是《Evaluating Large Language Models Trained on Code》中提到的一个代码评测基准。




## 2. 量化技术原理

量化就是使 $x_{f} \xrightarrow{\text { 量化 }} x_{q}, x_{q} \xrightarrow{\text { 反量化 }} x_{f}^{\prime}$，同时让 $x_f$ 和 $x_{f}^{\prime}$ 尽可能接近。

通用的量化与反量化公式可以写成：
$$
x_{q}=\operatorname{clip}\left(\operatorname{round}\left(\frac{x_{f}}{s}+z\right), q_{\min }, q_{\max }\right)
$$
$$
x_{f}^{\prime}=\left(x_{q}-z\right) \times s
$$
其中：
- $s$（**scale**，浮点数）：缩放因子（量化步长），决定每一个整数刻度代表多大的浮点数；
- $z$（**zero_point**，整数）：零点，表示浮点数 $0$ 在量化后整型空间中对应的位置；
- $q_{min}, q_{max}$：量化整型的取值范围，例如 INT8 有符号量化时 $q_{min}=-128$，$q_{max}=127$；
- $\operatorname{clip}(\cdot)$：把超出范围的值截断到边界；$\operatorname{round}(\cdot)$：四舍五入取整。

按 $s$、$z$ 的不同取法，可以划分出**对称量化** / **非对称量化**两种基本方案。



### 2.1 对称量化与非对称量化

#### 对称量化（Symmetric Quantization）

**对称量化**：如下图，对称量化的原理就是找到 $x_{f}$ 中绝对值的最大值，然后对其进行缩放得到量化后的值，然后进行反量化得到原来的值，可以看出量化是存在一定误差的，具体原理如下图：
<div align="center">
<img src="https://pic1.zhimg.com/v2-45fd7d5ede2b518c31e40c24a5085f32_1440w.jpg" width="600px">
</div>

对称量化的核心是 **zero_point = 0**，量化后的整型值域以 0 为中心对称分布，即 $q_{max} = -q_{min}$。其计算公式为：
$$
s=\frac{\max \left(\left|x_{f}\right|\right)}{2^{b-1}-1}, \quad z=0
$$
$$
x_{q}=\operatorname{clip}\left(\operatorname{round}\left(\frac{x_{f}}{s}\right),-2^{b-1}+1, 2^{b-1}-1\right)
$$
其中 $b$ 为量化位宽。以 INT8 为例（$b=8$）：缩放因子 $s = \max(|x_f|)/127$，量化后的取值范围被限制在 $[-127, 127]$（注意为了对称性，通常舍弃 $-128$ 这一个值）。

这样量化的缺点就是，坐标轴上有一段数值空间被浪费了，对应图中 `-127` 那一部分。例如 ReLU 之后的激活值全部为正数，使用对称量化时整个负半轴 $[-127, 0)$ 都不会被使用，量化精度只能利用一半的整型空间。基于这个缺点，为了让量化后的坐标轴上的数值被充分利用，引入非对称量化。


#### 非对称量化（Asymmetric Quantization）

非对称量化使用两个统计量 $x_{min}$、$x_{max}$，把浮点数据真实的取值范围**整体平移**到 $[q_{min}, q_{max}]$，因此能更充分地利用量化空间。其计算公式为：
$$
s=\frac{x_{\max }-x_{\min }}{q_{\max }-q_{\min }}, \quad z=q_{\min }-\operatorname{round}\left(\frac{x_{\min }}{s}\right)
$$
$$
x_{q}=\operatorname{clip}\left(\operatorname{round}\left(\frac{x_{f}}{s}\right)+z, q_{\min }, q_{\max }\right)
$$

INT8 无符号非对称量化时 $q_{min}=0, q_{max}=255$；INT8 有符号非对称量化时 $q_{min}=-128, q_{max}=127$。零点 $z$ 是一个整数，保证浮点 $0$ 能被精确表示（这对 padding、ReLU 等运算非常重要）。


#### 一个具体的对比示例

以浮点向量 $x_f = [-0.5, -3.1, 0.8, 5.4]$ 进行 INT8 量化为例：

| 方法 | scale $s$ | zero_point $z$ | 量化结果 $x_q$ | 反量化 $x_f'$ |
|---|---|---|---|---|
| 对称量化 | $5.4/127 \approx 0.0425$ | $0$ | $[-12, -73, 19, 127]$ | $[-0.51, -3.10, 0.81, 5.40]$ |
| 非对称量化（无符号） | $(5.4-(-3.1))/255 \approx 0.0333$ | $0 - \operatorname{round}(-3.1/0.0333)=93$ | $[78, 0, 117, 255]$ | $[-0.50, -3.10, 0.80, 5.40]$ |

可以看出**非对称量化把 $[-3.1, 5.4]$ 这段不对称的浮点区间整体映射到了 $[0, 255]$**，量化分辨率更高；而对称量化牺牲了一部分整型空间换取了简化的计算。


#### 工程实践中的选择

实际部署中通常采用 **权重对称量化、激活非对称量化** 的混合策略。原因可以从 `Linear` 层 $y = Wx$ 的矩阵乘法展开看出：
$$
W x \approx s_{W}\left(W_{q}-z_{W}\right) \cdot s_{x}\left(x_{q}-z_{x}\right)=s_{W} s_{x}\left(W_{q} x_{q}-W_{q} z_{x}-z_{W} x_{q}+z_{W} z_{x}\right)
$$
- 当 $z_W = 0$（权重对称量化）时，公式简化为 $s_W s_x(W_q x_q - W_q z_x)$，其中第二项 $W_q z_x$ 可以提前算好作为偏置；
- 若权重也做非对称量化，则推理时多出 $z_W x_q$ 这一项需要在线计算，开销更大。

因此：**权重对称、激活非对称** 是兼顾精度与推理效率的常见做法。



### 2.2 如何对神经网络进行量化？

#### 量化对象：权重 vs 激活

神经网络中可被量化的张量主要有两类：

| 量化对象 | 特点 | 量化时机 |
|---|---|---|
| **权重 W** | 训练完成后即固定，分布可提前统计 | 离线（部署前完成） |
| **激活 A** | 与输入数据相关，每次推理都不同 | 静态（用校准集统计）或动态（运行时统计） |

按照同时量化 W、A 的位宽，常见组合有：
- **W8A8**：经典的 INT8 量化方案，权重和激活都使用 INT8；
- **W4A16**：LLM 中常用，权重 INT4，激活保持 FP16（典型代表 GPTQ / AWQ）；
- **W8A16**：权重 INT8、激活 FP16，兼顾精度和速度。


#### 量化粒度（Granularity）

量化粒度决定 **scale / zero_point 在张量内的共享范围**，粒度越细精度越高，但元数据开销也越大：

- **Per-tensor（逐张量）**：整个张量共享一个 $s$ 和 $z$，最简单但容易被离群值"拖累"；
- **Per-channel（逐通道）**：每个输出通道（卷积的输出通道、Linear 的列）一组 $s, z$，对权重最常用；
- **Per-token（逐 token）**：Transformer 中对激活的每个 token 维度独立量化，能有效隔离不同 token 之间的离群值；
- **Per-group（分组量化）**：把一个张量按 128 / 64 / 32 等大小切分成若干组，每组独立量化，是 GPTQ、AWQ 等 W4 量化的标配。


#### 静态量化 vs 动态量化

| 类型 | 激活的 scale 来源 | 优点 | 缺点 |
|---|---|---|---|
| **静态量化** | 部署前用**校准集**前向跑一遍，统计每层激活分布得到 $s, z$ | 推理无额外开销 | 校准集分布要与真实分布一致 |
| **动态量化** | 推理时**实时**根据当前输入计算 $s, z$ | 鲁棒性好，无需校准集 | 每次推理多一步统计开销 |


#### 训练后量化 PTQ vs 量化感知训练 QAT

按是否需要重新训练，量化方法分为两大类：

- **PTQ（Post-Training Quantization，训练后量化）**：模型已经训好，**不再重新训练**，只用少量校准数据统计分布即可完成量化。优点是流程简单、几乎不需要算力；缺点是**精度损失相对较大**，对低位宽（INT4）尤其敏感。代表方法：GPTQ、AWQ、SmoothQuant 等。
- **QAT（Quantization-Aware Training，量化感知训练）**：在训练阶段就**让模型感知到量化误差**，通过反向传播让模型自己适应量化噪声，从而获得比 PTQ 更高的量化精度。详见 2.3 节。


#### Linear 层的 INT8 矩阵乘法流程

以 $Y_{fp16} = X_{fp16} \cdot W_{fp16}$ 这一典型的 Linear 层为例，INT8 量化推理的过程为：

```bash
       ┌───────────────────┐
X_fp16 │ 1. 在线量化 X     │  → X_int8, s_x
       └───────────────────┘
                │
       ┌───────────────────┐
       │ 2. 加载离线量化   │  → W_int8, s_w  （部署前预先完成）
       │    好的权重 W     │
       └───────────────────┘
                │
       ┌───────────────────┐
       │ 3. INT8 矩阵乘    │  Y_int32 = X_int8 · W_int8
       └───────────────────┘
                │
       ┌───────────────────┐
       │ 4. 反量化         │  Y_fp16 = Y_int32 × (s_x ⊗ s_w)
       └───────────────────┘
                │
              Y_fp16
```

关键点：**矩阵乘法在 INT8 域完成**（享受硬件 INT8 Tensor Core 的加速），结果先累加到 INT32（防止溢出），最后乘上 scale 反量化回 FP16 输出。



### 2.3 量化感知训练

#### 核心思想

量化感知训练（**Quantization-Aware Training, QAT**）的基本思想是：**在训练的前向传播中插入"伪量化"（FakeQuant）节点**，模拟出量化-反量化引入的舍入误差，让模型在训练过程中提前"见识"到量化带来的扰动，并通过梯度下降自适应地调整参数来抵消这些误差，最终得到一个对量化噪声鲁棒的模型。


#### 伪量化节点（FakeQuant）

伪量化节点的本质是 **先量化再反量化**，输入输出都是浮点数，但取值被限制在量化网格上：
$$
Q(x)=\operatorname{FakeQuant}(x)=\operatorname{DeQuant}(\operatorname{Quant}(x))=s \cdot\left(\operatorname{clip}\left(\operatorname{round}\left(\frac{x}{s}\right)+z, q_{\min }, q_{\max }\right)-z\right)
$$

FakeQuant 通常插入到两个位置：
- **权重前**：模拟权重量化误差；
- **激活后**：模拟激活量化误差。

训练过程中权重 $W$ 始终以 **FP32** 存储和更新，只是在前向计算时被 FakeQuant 处理过。


#### 正向传播

前向传播时，FakeQuant 将 FP32 的权重和激活 **量化到 INT8 再反量化回 FP32**，在计算图中显式地引入量化误差，使损失函数包含量化扰动。这样，训练时模型的 loss 就反映了"量化模型"的真实表现，优化器自然会朝着对量化更友好的方向更新参数。


#### 反向传播与直通估计器（STE）

QAT 的关键技术难点是：**$\operatorname{round}(\cdot)$ 在几乎所有点处导数都为 0**（在整数刻度处不可导），如果直接求导梯度会全是 0，反向传播无法进行。

解决方案是 **直通估计器（Straight-Through Estimator，STE）**：**在反向传播时，假装 $\operatorname{round}(\cdot)$ 是恒等函数，让梯度"直接穿过"FakeQuant 节点**。形式化地写：
$$
\frac{\partial Q(x)}{\partial x} \approx \begin{cases} 1, & \text { if } x \in[x_{\min }, x_{\max }] \\ 0, & \text { otherwise }\end{cases}
$$

也就是说，对落在量化范围内的值，梯度无损穿过；对被 `clip` 截断的值，梯度置 0（因为这些值实际上已经饱和）。STE 是一个数学上不严格但工程上极为有效的近似。


#### QAT vs PTQ 对比

| 维度 | PTQ | QAT |
|---|---|---|
| 是否需要训练数据 | 仅需少量校准集 | 需要完整训练集 |
| 是否需要重新训练 | 否 | 是（通常是微调） |
| 算力开销 | 极小 | 较大 |
| 量化精度损失 | 较大（尤其低位宽） | 较小（典型 < 1%） |
| 适用场景 | 大模型、快速部署 | 对精度要求严格的边缘部署 |

实践中通常**先做 PTQ 校准得到一个 baseline，再以此为初始化做 QAT 微调**，可以兼顾收敛速度和最终精度。



## 3. 大模型的Int8量化

前面介绍的 PTQ / QAT 量化方法在 BERT、ResNet 这一类中小规模模型上工作得很好。但当 Transformer 参数规模超过 **6.7B** 时，**朴素的 INT8 量化会出现灾难性的精度崩塌**：困惑度（perplexity）急剧上升，零样本任务上的准确率退化到接近随机。这一现象由 Dettmers 等人在 2022 年发表的 [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) 中系统性地揭示并解决。


### 3.1 现象：大模型中的离群特征（Emergent Outlier Features）

研究者在对 OPT、BLOOM 等模型逐层观察隐藏状态时发现，当模型规模超过约 **6.7B 参数** 时，会出现一种**系统性的离群特征**：

- **幅度极大**：离群值的幅度比普通特征大 **20 倍**以上；
- **维度极少**：在 13B 模型中，整个 Transformer 中只集中在 **最多 7 个** 特征/隐藏维度（约占总维度的 0.1%）；
- **覆盖广泛**：这些少数维度会在 **所有层** 的约 **75% 序列位置** 上同时出现；
- **影响巨大**：将这些维度强制置零后，验证集困惑度暴涨 **600~1000%**，top-1 softmax 概率下降 20% 以上；而置零同样数量的随机维度只会让困惑度上升 0.1%。

**为什么离群值会让 INT8 量化失效？**
INT8 量化的 scale 是由 $\max(|x|)$ 决定的：一旦张量中混入了一两个 20 倍大的离群值，scale 会被它们"撑得"非常大，导致正常值在量化网格中被压缩到极小的区间——大多数普通值四舍五入后都变成 0，整个张量的有效精度被一两个离群值彻底破坏。


### 3.2 LLM.int8() 算法

针对上述问题，LLM.int8() 提出了**两个相互配合**的关键技术：

#### 技术一：向量级量化（Vector-wise Quantization）

不再对整个张量使用一个 scale，而是把矩阵乘法 $Y = X \cdot W$ 拆解为一系列**行向量与列向量的内积**：
- 对 $X \in \mathbb{R}^{s \times h}$ 的**每一行**分配一个 scale $c_x \in \mathbb{R}^{s}$；
- 对 $W \in \mathbb{R}^{h \times o}$ 的**每一列**分配一个 scale $c_w \in \mathbb{R}^{o}$；
- 整型矩阵乘的结果通过 $c_x \otimes c_w$（外积）反量化回 FP16：
$$
Y_{f 16} \approx \frac{1}{c_{x} \otimes c_{w}} \cdot\left(Q\left(X_{f 16}\right) \cdot Q\left(W_{f 16}\right)\right)
$$

向量级量化把离群值的影响**限制在它所在的那一行/列内**，不再污染整个张量。该方法可以使 2.7B 以下的模型量化无损，但**对 6.7B 以上的离群特征仍然不够**。


#### 技术二：混合精度分解（Mixed-precision Decomposition）

这是 LLM.int8() 的**核心创新**。既然离群值只集中在极少数（≤7 个）固定的特征维度，那就**把这些维度单独拎出来用 FP16 算；其余 99.9% 的维度照常用 INT8 算**，最后再相加。

具体步骤：
1. **离群检测**：扫描输入 $X$ 中每一列（特征维度），把存在幅值 $\geq \alpha = 6.0$ 的列索引集合记为 $O$；
2. **分解矩阵乘**：
   $$
   Y \approx \sum_{h \in O} X_{f 16}^{h} W_{f 16}^{h}+S_{f 16} \cdot \sum_{h \notin O} X_{i 8}^{h} W_{i 8}^{h}
   $$
   - 前一项：离群维度（约 0.1%）用 **FP16 矩阵乘**，保留全精度；
   - 后一项：剩余维度（99.9%）用 **INT8 向量级量化**矩阵乘，最后乘上反量化项 $S_{f16}$ 还原为 FP16；
3. **结果累加**：将 FP16 部分和反量化后的 INT8 部分**在 FP16 域相加**得到最终输出。

整体流程示意（对应论文 Figure 2）：

```bash
                    ┌─────────────── X_fp16 ───────────────┐
                    │                                       │
       ┌────────────┴───────────┐         ┌────────────────┴──────────┐
       │  离群列（0.1%, FP16）  │         │  普通列（99.9%, INT8 量化）│
       └────────────┬───────────┘         └────────────────┬──────────┘
                    │                                       │
       FP16 × FP16 矩阵乘                INT8 × INT8 矩阵乘 → 反量化回 FP16
                    │                                       │
                    └───────────────┬───────────────────────┘
                                    ▼
                                 Y_fp16
```


### 3.3 效果

LLM.int8() 在多种规模的 LLM 上做到了**几乎零精度损失**的 INT8 推理：

| 模型 | FP16 精度 | LLM.int8() 精度 | 差异 |
|---|---|---|---|
| OPT-175B (hellaswag acc_norm) | 0.7849 | 0.7849 | 0.000 |
| OPT-175B (piqa acc) | 0.7959 | 0.7965 | +0.0006 |
| BLOOM-176B (lambada acc) | 0.6718 | 0.6808 | +0.009 |

工程价值：
- **显存减半**：BLOOM-176B 从 ~352 GB（BF16）压缩到 ~180 GB（INT8），原本需要 8 张 80GB A100 才能装下的模型，可以在 4 张 80GB A100 上运行；
- **推理速度**：176B 这种巨型矩阵乘上可获得约 **2× 加速**；但中等模型（如 T5-3B/11B）由于离群分解的额外开销，反而会有 15%~23% 的降速；
- **生态集成**：通过 `bitsandbytes` 库已经无缝集成进 Hugging Face `transformers`，只需 `load_in_8bit=True` 即可启用：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1",
    device_map="auto",
    load_in_8bit=True,
)
```


### 3.4 小结

LLM.int8() 之所以能成为大模型 INT8 推理的"通用解"，关键在于它把**对大模型的分析洞察**和**工程化的混合精度方案**结合到了一起：

1. **理论洞察**：揭示了 6.7B 以上 Transformer 中"少数离群特征主导精度"的现象；
2. **工程方案**：用 0.1% 的 FP16 精度换回 99.9% 维度的 INT8 加速与压缩；
3. **零退化**：在 175B 规模上首次做到 INT8 推理无精度损失。

它也启发了后续一系列 LLM 量化工作：例如 **SmoothQuant** 通过把激活的离群幅度迁移到权重上来避免分解；**GPTQ / AWQ** 则进一步把权重压到 INT4 实现更激进的压缩。这些方法将在后续章节单独展开。


**References:**
- [大模型（LLM）的量化技术Quantization原理学习](https://blog.csdn.net/penriver/article/details/136411485)
- [大模型(LLM)量化(Quantization)原理学习](https://zhuanlan.zhihu.com/p/29140505773)
- [【深度学习】模型量化-笔记/实验](https://blog.csdn.net/qq_40035462/article/details/123745290)

----




# 预备知识-深入理解Float

## 什么是 Float Point

Float Point 是用于表示浮点数的一种数据类型，用**符号、尾数、指数和基数**这四部分来表示的小数。
<div align="center">
<img src="https://polarisxu.studygolang.com/posts/basic/imgs/float-point02.png" width="600px">
</div>




## 二进制与十进制的转换

**二进制转十进制：**
$$
[11.101]_{2}=[1 \times 2^{1} + 1 \times 2^{0} + 1 \times 2^{-1}+ 0 \times 2^{-2}+0 \times 2^{-3}]_{10}=[3.625]_{10}
$$


**十进制转二进制：** $[11.625]_{10}=[1011.101]_{2}$
1. 整数部分：`11`，十进制整数转二进制采用 “除 2 取余，逆序排列”法。
   ```bash
   11 / 2 = 5 余 1
   5 / 2 = 2 余 1
   2 / 2 = 1 余 0
   1 / 2 = 0 余 1
   ```
2. 小数部分：`0.625`，十进制小数转二进制采用 “乘 2 取整，顺序排列”法。
   ```bash
   0.625 * 2 = 1.25 取整 1
   0.25 * 2 = 0.5 取整 0
   0.5 * 2 = 1.0 取整 1
   ```




## 浮点数的 IEEE754 表示

一般地，IEEE754 浮点数有两种类型：单精度浮点数（float32）和双精度浮点数（float64）。
<div align="center">
<img src="https://polarisxu.studygolang.com/posts/basic/imgs/float-point03.png" width="500px">
</div>

- **符号位(Sign)**：0 表示正数，1 表示负数。
- **指数位(Exponent)**：指数部分采用了 The Biased exponent（有偏指数）。IEEE754 规定，$2^{e-1}-1$ 的值是 $0$，其中 $e$ 表示指数部分的位数，小于这个值表示负数，大于这个值表示正数。
  - 对于单精度浮点数而言，$e=8$，$2^{8-1}-1=127$ 是 0；
  - 对于双精度浮点数而言，$e=11$，$2^{11-1}-1=1023$ 是 0；
- **尾数位(Fraction)**：浮点数名称的由来在于小数点是浮动的。但具体存储时，需要固定一种形式，这叫做尾数的标准化。
  - IEEE754 规定，在二进制数中，通过移位，将小数点前面的值固定为 1。IEEE754 称这种形式的浮点数为规范化浮点数（normal number）。
  - 尾数决定了精度，对于单精度浮点数，因为只有 `23` 位，而 `1<<23` 对应十进制是 `8388608`，因此不能完整表示全部的 `7` 个十进制位，所以说，单精度浮点数**有效小数位最多 `7` 位**；




## IEEE754 转换案例

`0.15625` 的 IEEE754 表示
<div align="center">
<img src="https://polarisxu.studygolang.com/posts/basic/imgs/float-point04.png" width="600px">
</div>

**Float32 转换为十进制：**
1. 计算公式:
   $$
   Sign \times (1 + Fraction) \times 2^{(Exponent - bias)} 
   $$
2. 代入公式:
   $$
   [0 \quad 01111100 \quad 01000000000000000000000]_{2} \\ 
   = [(-1)^0 \times 2^{(124 - 127)} \times (1 + 0 * 2^{-1} + 1 * 2^{-2}) ]_{10} \\
   = [1.25 \times 0.125]_{10} = [0.15625]_{10}
   $$



**十进制转换为 Float32 进行存储和表示：**
1. 将十进制小数转换为二进制小数：使用 "乘 2 取整，顺序排列" 法
   ```bash
   0.15625 * 2 = 0.3125 取整 0
   0.3125 * 2 = 0.625 取整 0
   0.625 * 2 = 1.25 取整 1
   0.25 * 2 = 0.5 取整 0
   0.5 * 2 = 1.0 取整 1
   ```
   因此，$[0.15625]_{10}=[0.00101]_{2}$
2. 规范化二进制表示：IEEE754 要求将浮点数规范化为 $1.M \times 2^n$ 的形式
  小数点需要向右移动3位，得到 $[0.00101]_{2}=[1.01 \times 2^{-3}]_{2}$
3. 计算 IEEE754 各部分
   - 符号位(Sign)：$0$ （正数）
   - 指数位(Exponent)：$n=-3$，$e=n+127=124$，二进制表示为 `01111100`（有偏指数）
   - 尾数位(Fraction)：规范化后的尾数：`01`（只存储小数点后的部分）,因为尾数位最多存储 `23` 位，补齐到`23`位，得到`01000000000000000000000`
4. 最终表示为 `0 01111100 01000000000000000000000`


[Base Convert: IEEE 754 Floating Point](https://baseconvert.com/ieee-754-floating-point)




## 经典Float相加问题

经典Float相加问题：
```bash
0.1 + 0.2 = 0.30000000000000004
```

示例代码：
```go
package main

import (
	"fmt"
)

func main() {
	var a, b float64 = 0.1, 0.2
	fmt.Println(a + b)
}
```

根本原因：
1. 出现这种情况的根本原因是，有些十进制小数无法转换为二进制数。
   <div align="center">
   <img src="https://polarisxu.studygolang.com/posts/basic/imgs/float-point06.png" width="300px">
   </div>
   
   在小数点后 4 位时，连续的二进制数，对应的十进制数却是不连续的，因此只能增加位数来尽可能近似的表示。
2. $0.1$ 和 $0.2$ 在计算机中存储时的 float32 表示：
   <div align="center">
   <img src="https://polarisxu.studygolang.com/posts/basic/imgs/float-point07.png" width="500px">
   </div>
   
   同样的方法，$0.2$ 用单精度浮点数表示是：`0.20000000298023223876953125`。
3. 相加后的结果：`0.300000004470348358154296875`
   <div align="center">
   <img src="https://polarisxu.studygolang.com/posts/basic/imgs/float-point08.png" width="400px">
   </div>





## Float32的表示范围


IEEE754 浮点数，指数是关键，根据**指数**，将其分为：**特殊值、非规范化浮点数**和 **规范化浮点数**。
<div align="center">
<img src="https://polarisxu.studygolang.com/posts/basic/imgs/float-point14.png" width="700px">
</div>

从上图规范化和非规范化浮点数的表示范围可以看出，两种类型的表示是具有连续性的。这也就是为什么非规范化浮点数指数规定为比规范形式的偏移值小 1（即单精度为 -126，双精度为 -2046）。

在数轴上，浮点数的分布：
<div align="center">
<img src="https://polarisxu.studygolang.com/posts/basic/imgs/float-point15.png" width="500px">
</div>

----

**Float32表示：**

- **特殊值**：infinity（无穷）和 NaN（Not a Number）
    <div align="center">
    <img src="https://polarisxu.studygolang.com/posts/basic/imgs/float-point11.png" width="500px">
    </div>

- **最大值（正数）**： $[0 \quad 11111110 \quad 11111111111111111111111]_{2}$
    <div align="center">
    <img src="https://polarisxu.studygolang.com/posts/basic/imgs/float-point12.png" width="500px">
    </div>

- **最小值（正数）**： $[0 \quad 00000000 \quad 00000000000000000000001]_{2}$，也是 **最小步进（ULP）**
  <div align="center">
  <img src="https://polarisxu.studygolang.com/posts/basic/imgs/float-point13.png" width="500px">
  </div>




## 乘法的计算过程(Float32)

浮点数的乘法运算遵循IEEE754标准，主要包括符号位、指数位和尾数位三个部分的独立计算。

乘法运算规则，对于两个浮点数 $A$ 和 $B$：
$$
A = (-1)^{S_A} \times 2^{E_A - 127} \times (1 + M_A)
$$
$$
B = (-1)^{S_B} \times 2^{E_B - 127} \times (1 + M_B)
$$

它们的乘积为：
$$
A \times B = (-1)^{S_A \oplus S_B} \times 2^{(E_A + E_B - 127) - 127} \times [(1 + M_A) \times (1 + M_B)]
$$

**计算步骤：**
1. **符号位(Sign)**：$S_{result} = S_A \oplus S_B$（异或运算）
   - 两个数同号结果为正（0）
   - 两个数异号结果为负（1）

2. **指数位(Exponent)**：$E_{result} = E_A + E_B - 127$
   - 将两个指数相加
   - 减去一个偏置值127（因为两个数的指数都已经加过偏置）

3. **尾数位(Fraction)**：$M_{result} = (1 + M_A) \times (1 + M_B)$
   - 将两个尾数（包含隐含的1）相乘
   - 如果结果 $\geq 2.0$，需要规范化：右移一位，指数加1
   - 如果结果 $< 1.0$，需要规范化：左移直到最高位为1

4. **结果规范化**：确保尾数在 $[1.0, 2.0)$ 范围内

5. **舍入处理**：如果尾数超过23位，需要按照IEEE754舍入规则进行舍入



**特殊情况处理**：
1. **零乘以任何数**：结果为零（保持符号）
2. **无穷大乘以非零数**：结果为无穷大
3. **无穷大乘以零**：结果为NaN
4. **指数溢出**（$E > 254$）：结果为无穷大
5. **指数下溢**（$E < 0$）：结果为0或非规范化数




**References:**
- [15 张图带你深入理解浮点数](https://polarisxu.studygolang.com/posts/basic/diagram-float-point/)
