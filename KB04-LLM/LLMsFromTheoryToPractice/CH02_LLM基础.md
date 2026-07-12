# 2. 大语言模型基础


## 2.1 Transformer 结构

Transformer 结构是由Google 在2017 年提出并首先应用于机器翻译的神经网络模型架构。机器翻译的目标是从源语言（Source Language）转换到目标语言（Target Language）。Transformer结构完全通过注意力机制完成对源语言序列和目标语言序列全局依赖的建模。如今，几乎全部大语言模型都是基于Transformer 结构的。


基于Transformer 的编码器和解码器结构如图2.1 所示，左侧和右侧分别对应着编码器（Encoder）和解码器（Decoder）结构，它们均由若干个基本的Transformer 块（Block）组成（对应图中的灰色框）。这里 $N \times$ 表示进行了 $N$ 次堆叠。每个Transformer 块都接收一个向量序列 $\{x_i\}_{i=1}^{t}$作为输入，并输出一个等长的向量序列作为输出 $\{y_i\}_{i=1}^{t}$。这里的 $x_i$ 和 $y_i$ 分别对应文本序列中的一个词元（Token）的表示。$y_i$ 是当前Transformer 块对输入 $x_i$ 进一步整合其上下文语义后对应的输出。

<div align="center">
    <img src="./assets/ch02/pic-2.1.png" alt="2.1" />
</div>

在从输入 $\{x_i\}_{i=1}^{t}$ 到输出 $\{y_i\}_{i=1}^{t}$ 的语义抽象过程中，主要涉及如下几个模块:

- **注意力层**：使用 **多头注意力（Multi-Head Attention）机制** 整合上下文语义。多头注意力并行运行多个独立注意力机制，进而从多维度捕捉输入序列信息。它使得序列中任意两个单词之间的依赖关系可以直接被建模而不基于传统的循环结构，从而更好地解决文本的长程依赖问题。
- **位置感知前馈网络层（Position-wise Feed-Forward Network）**：通过全连接层对输入文本序列中的每个单词表示进行更复杂的变换。
- **残差连接**：对应图中的 Add 部分。它是一条分别作用在上述两个子层中的直连通路，被用于连接两个子层的输入与输出，使信息流动更高效，有利于模型的优化。
- **层归一化**：对应图中的Norm 部分。它作用于上述两个子层的输出表示序列，对表示序列进行层归一化操作，同样起到稳定优化的作用。





### 2.1.1 嵌入表示层

对于输入文本序列，先通过输入嵌入层（Input Embedding）将每个单词转换为其相对应的向量表示。通常，直接对每个单词创建一个向量表示。在送入编码器端建模其上下文语义之前，一个非常重要的操作是在词嵌入中加入位置编码（Positional Encoding）这一特征。具体来说，序列中每一个单词所在的位置都对应一个向量。这一向量会与单词表示对应相加并送入后续模块中做进一步处理。在训练过程中，模型会自动地学习到如何利用这部分位置信息。


为了得到不同位置所对应的编码，Transformer 结构使用不同频率的正余弦函数，如下所示。

$$
\operatorname{PE}(\operatorname{pos}, 2 i)=\sin \left(\frac{\operatorname{pos}}{10000^{2 i / d}}\right) \tag{2.1}
$$

$$
\operatorname{PE}(\operatorname{pos}, 2 i+1)=\cos \left(\frac{\operatorname{pos}}{10000^{2 i / d}}\right) \tag{2.2}
$$

其中，$pos$ 表示单词所在的位置，$2i$ 和 $2i+1$ 表示位置编码向量中的对应维度，$d$ 则对应位置编码的总维度。通过上面这种方式计算位置编码有以下两个好处：

1. 正余弦函数的范围是 $[-1, +1]$，导出的位置编码与原词嵌入相加不会使得结果偏离过远而破坏原有单词的语义信息；
2. 依据三角函数的基本性质，可以得知第 $pos+k$ 个位置编码是第 $pos$ 个位置编码的线性组合，这就意味着位置编码中蕴含着单词之间的距离信息。






### 2.1.2 注意力层

**自注意力（Self-Attention）** 操作是基于Transformer 的机器翻译模型的基本操作，在源语言的编码和目标语言的生成中频繁地被使用，以建模源语言、目标语言任意两个单词之间的依赖关系。将由单词语义嵌入及其位置编码叠加得到的输入表示为 $\{x_i \in \mathbb{R}^d\}_{i=1}^{L}$。

如图2.2 所示，通过三个线性变换 $W^Q \in \mathbb{R}^{d \times d_q}$ , $W^K \in \mathbb{R}^{d \times d_k}$ , $W^V \in \mathbb{R}^{d \times d_v}$ 将输入序列中的每一个单词表示 $x_i$ 转换为其对应的 $q_i \in \mathbb{R}^{d_q}$ , $k_i \in \mathbb{R}^{d_k}$ , $v_i \in \mathbb{R}^{d_v}$ 向量。

<div align="center">
    <img src="./assets/ch02/pic-2.2.png" alt="2.2" />
</div>

对于输入 $\{x_i \in \mathbb{R}^d\}_{i=1}^{L}$，$Q$、$K$ 和 $V$ 矩阵可以通过如下公式所示：

$$
Q =X W^{Q} \tag{2.3}
$$

$$
K =X W^{K} \tag{2.4}
$$

$$
V =X W^{V} \tag{2.5}
$$


注意力计算公式如下所示：

$$
Z = \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \tag{2.6}
$$

其中 $Q ∈ \mathbb{R}^{L \times d_q}$ , $K ∈ \mathbb{R}^{L \times d_k}$ , $V ∈ \mathbb{R}^{L \times d_v}$ 分别表示输入序列中的不同单词的 $q$ , $k$ , $v$ 向量拼接组成的矩阵，$L$ 表示序列长度，$Z ∈ \mathbb{R}^{L \times d_v}$ 表示自注意力操作的输出。

> 注意，由于计算 $QK^T$ 要求矩阵维度匹配，因此必须满足 $d_q = d_k$，在标准 Transformer 中二者始终相等。

---

为了进一步增强自注意力机制聚合上下文信息的能力，提出了多头注意力机制，以关注上下文的不同侧面。具体来说，上下文中每一个单词的表示 $x_i$ 经过多组线性 $\left\{W_{j}^{Q}, W_{j}^{K}, W_{j}^{V}\right\}_{j=1}^{N}$ 映射到不同的表示子空间中。公式(2.6) 会在不同的子空间中分别计算并得到不同的上下文相关的单词序列表示 $\left\{Z_j\right\}_{j=1}^{N}$：

$$
Z_{i}=\operatorname{Attention}\left(Q_{i}, K_{i}, V_{i}\right)=\operatorname{Softmax}\left(\frac{Q_{i} K_{i}^{\top}}{\sqrt{d}}\right) V_{i} \tag{2.7}
$$


在此基础上，经过线性变换 $W^O \in \mathbb{R}^{(N d_v) \times d}$ 用于综合不同子空间中的上下文表示并形成注意力层最终的输出 $\{x_i \in \mathbb{R}^d\}_{i=1}^{L}$，可得到 **多头自注意力（Multi-Head Self-Attention）** 表示：

$$
Z=\operatorname{Concat}\left(Z_{1}, Z_{2}, \ldots, Z_{N}\right) W^{O} \tag{2.8}
$$





### 2.1.3 前馈层

前馈层接收自注意力子层的输出作为输入，并通过一个带有 **ReLU** 激活函数的两层全连接网络对输入进行更复杂的非线性变换。

$$
\operatorname{FFN}(\boldsymbol{x})=\operatorname{ReLU}\left(\boldsymbol{x} \boldsymbol{W}_{1}+\boldsymbol{b}_{1}\right) \boldsymbol{W}_{2}+\boldsymbol{b}_{2} \tag{2.9}
$$

其中 $\boldsymbol{W}_1, \boldsymbol{b}_1, \boldsymbol{W}_2, \boldsymbol{b}_2$ 表示前馈子层的参数。





### 2.1.4 残差连接与层归一化

残差连接主要是指使用一条直连通道直接将对应子层的输入连接到输出，避免在优化过程中因网络过深而产生潜在的梯度消失问题：

$$
x^{l+1}=f\left(x^{l}\right)+x^{l} \tag{2.10}
$$

其中 $x^l$ 表示第 $l$ 层的输入，$f(·)$ 表示一个映射函数。


为了使每一层的输入/输出稳定在一个合理的范围内，**层归一化** 技术被进一步引入每个Transformer 块中：

$$
\operatorname{LN}(x)=\alpha \cdot \frac{x-\mu}{\sigma}+b \tag{2.11}
$$

其中 $\mu$ 和 $\sigma$ 分别是 $x$ 的均值和标准差，用于将数据平移缩放到均值为0、方差为1 的标准分布，$\alpha$ 和 $b$ 是可学习的参数。层归一化技术可以有效地缓解优化过程中潜在的不稳定、收敛速度慢等问题。







### 2.1.5 编码器和解码器结构

基于上述模块，根据图2.1 给出的网络架构，编码器端较容易实现。相比于编码器端，解码器端更复杂。具体来说，解码器的每个Transformer 块的第一个自注意力子层额外增加了注意力掩码，对应图中的掩码多头注意力（Masked Multi-Head Attention）部分。


编码器仅需要考虑如何融合上下文语义信息。解码器端则负责生成目标语言序列，这一生成过程是自回归的，即对于每一个单词的生成过程，仅有当前单词之前的目标语言序列是可以被观测的，因此这一额外增加的掩码是用来掩盖后续的文本信息的，以防模型在训练阶段直接看到后续的文本序列，进而无法得到有效的训练。


此外，解码器端额外增加了一个 **多头交叉注意力（Multi-Head Cross-Attention）** 模块，使用交叉注意力（Cross-Attention）方法，同时接收来自编码器端的输出和当前Transformer 块的前一个掩码注意力层的输出。查询是通过解码器前一层的输出进行投影的，而键和值是使用编码器的输出进行投影的


解码器端以自回归的方式生成目标语言文本，即在每个时间步 $t$，根据编码器端输出的源语言文本表示，以及前 $t - 1$ 个时刻生成的目标语言文本，生成当前时刻的目标语言单词。











## 2.2 生成式预训练语言模型GPT

受到计算机视觉领域采用ImageNet 对模型进行一次预训练，使得模型可以通过海量图像充分学习如何提取特征，再根据任务目标进行模型微调的范式影响，自然语言处理领域基于预训练语言模型的方法也逐渐成为主流。

以GPT 和BERT 为代表的基于Transformer 的大规模预训练语言模型的出现，使得自然语言处理全面进入了预训练微调范式新时代。

将预训练模型应用于下游任务时，不需要了解太多的任务细节，不需要设计特定的神经网络结构，只需要“微调”预训练模型，即使用具体任务的标注数据在预训练语言模型上进行监督训练，就可以取得显著的性能提升。


OpenAI 公司在2018 年提出的生成式预训练语言模型（Generative Pre-Training，GPT）是典型的生成式预训练语言模型之一。GPT 的模型结构如图2.3 所示，它是由多层Transformer 组成的单向语言模型，主要分为输入层、编码层和输出层三部分。

<div align="center">
    <img src="./assets/ch02/pic-2.3.png" alt="2.3" />
</div>






### 2.2.1 自监督预训练

GPT 采用生成式预训练方法，单向意味着模型只能从左到右或从右到左对文本序列建模，所采用的Transformer 结构和解码策略保证了输入文本每个位置只能依赖过去时刻的信息。


给定文本序列 $w = w_1, w_2, \cdots, w_n$，GPT 首先在输入层中将其映射为稠密的向量：

$$
v_{i}=v_{i}^{\mathrm{t}} + v_{i}^{\mathrm{P}} \tag{2.12}
$$

其中，$v_i^t$ 是词 $w_i$ 的词向量，$v_i^P$ 是词 $w_i$ 的位置向量，$v_i$ 为第 $i$ 个位置的单词经过模型输入层（第0层）后的输出。


GPT 模型的输入层与前文中介绍的神经网络语言模型的不同之处在于其需要添加位置向量，这是Transformer 结构自身无法感知位置导致的，因此需要来自输入层的额外位置信息。


经过输入层编码，模型得到表示向量序列 $v = v_1, v_2, \cdots, v_n$，随后将 $v$ 送入模型编码层。编码层由 $L$ 个Transformer 模块组成，在自注意力机制的作用下，每一层的每个表示向量都会包含之前位置表示向量的信息，使每个表示向量都具备丰富的上下文信息，而且，经过多层编码，GPT能得到每个单词层次化的组合式表示，其计算过程表示为：

$$
\boldsymbol{h}^{(l)}=\text { Transformer-Block }^{(l)}\left(\boldsymbol{h}^{(l-1)}\right), \quad l = 1, 2, \cdots, L \tag{2.13}
$$

其中 $\boldsymbol{h}^{(0)}$ 为输入层编码得到的初始表示，$\boldsymbol{h}^{(l)} \in \mathbb{R}^{n \times d}$ 表示第 $l$ 层的表示向量序列，$n$ 为序列长度，$d$ 为模型隐藏层维度，$L$ 为模型总层数。


GPT 模型的输出层基于最后一层的表示 $\boldsymbol{h}^{(L)}$，预测每个位置上的条件概率，其计算过程可以表示为：

$$
P\left(w_{i} \mid w_{1}, w_{2}, \cdots, w_{i-1}\right)=\operatorname{Softmax}\left(\boldsymbol{h}_{i}^{(L)} \boldsymbol{W}^{e}+\boldsymbol{b}^{\text {out }}\right) \tag{2.14}
$$

其中 $\boldsymbol{W}^e \in \mathbb{R}^{d \times V}$ 是输出层的权重矩阵，$V$ 是词表大小。


**单向语言模型** 按照阅读顺序输入文本序列 $w = w_1, w_2, \cdots, w_n$，用常规语言模型目标优化 $w$ 的最大似然估计，使之能根据输入历史序列对当前词做出准确的预测：

$$
\mathcal{L}^{\mathrm{PT}}(w)=-\sum_{i=1}^{n} \log P\left(w_{i} \mid w_{0}, w_{1}, \cdots, w_{i-1} ; \boldsymbol{\theta}\right) \tag{2.15}
$$

其中 $\boldsymbol{\theta}$ 代表模型参数。也可以基于马尔可夫假设，只使用部分过去词进行训练。预训练时通常使用随机梯度下降法进行反向传播，优化该负对数似然函数。


> **负对数似然与交叉熵的等价性：** 公式（2.15）的最大似然估计目标与工程中常用的交叉熵损失在数学上完全等价。在位置 $i$，真实标签为 one-hot 向量 $y_i$（仅在正确词 $w_i$ 处为 1），模型输出 Softmax 概率分布 $\hat{y}_i$，交叉熵为：
>
> $$H(y_i, \hat{y}_i) = -\sum_{w \in V} y_i(w) \log \hat{y}_i(w) = -\log P(w_i \mid w_0, \cdots, w_{i-1}; \boldsymbol{\theta})$$
>
> 对所有位置求和后，交叉熵损失与负对数似然完全一致：
>
> $$\sum_{i=1}^n \text{CrossEntropy}(y_i, \hat{y}_i) = -\sum_{i=1}^n \log P(w_i \mid \cdots) = \mathcal{L}^{\mathrm{PT}}(w)$$
>
> 两者只是视角不同：最大似然从统计学出发（"让训练数据的概率最大"），交叉熵从信息论出发（"让预测分布与真实分布的差距最小"），殊途同归。







### 2.2.2 有监督下游任务微调

下游任务微调（Downstream Task Fine-tuning）的目的是在通用语义表示的基础上，根据下游任务的特性进行适配。下游任务通常需要利用有标注数据集进行训练，数据集使用 $D$ 进行表示，每个样例由输入长度为 $n$ 的文本序列 $x = x_1, x_2, \cdots, x_n$ 和对应的标签 $y$ 构成。


先将文本序列 $x$ 输入GPT 模型，获得最后一层的最后一个词所对应的隐藏层输出 $\boldsymbol{h}^{(L)}_n$，在此基础上，通过全连接层变换结合 Softmax 函数，得到标签预测结果

$$
P\left(y \mid x_{1}, x_{2}, \cdots, x_{n}\right)=\operatorname{Softmax}\left(\boldsymbol{h}_{n}^{(L)} \boldsymbol{W}^{y}\right) \tag{2.16}
$$

其中 $\boldsymbol{W}^y \in \mathbb{R}^{d \times k}$ 是输出层的权重矩阵，$k$ 是标签类别数。


通过对整个标注数据集 $D$ 优化如下目标函数精调下游任务：

$$
\mathcal{L}^{\mathrm{FT}}(\mathbb{D})=-\sum_{(x, y)} \log P\left(y \mid x_{1}, x_{2}, \cdots, x_{n}\right) \tag{2.17}
$$


在微调过程中，下游任务针对任务目标进行优化，很容易使得模型遗忘预训练阶段所学习的通用语义知识表示，从而损失模型的通用性和泛化能力，导致出现 **灾难性遗忘（Catastrophic Forgetting）**问题。因此，通常采用混合预训练任务损失和下游微调损失的方法来缓解上述问题。在实际应用中，通常采用式（2.18）进行下游任务微调：

$$
\mathcal{L}=\mathcal{L}^{\mathrm{FT}}(\mathbb{D})+\lambda \mathcal{L}^{\mathrm{PT}}(\mathbb{D}) \tag{2.18}
$$

其中 $\lambda$ 的取值为 $[0, 1]$，用于调节预训练任务的损失占比。





### 2.2.3 预训练语言模型实践

HuggingFace 是一个开源自然语言处理软件库，其目标是通过提供一套全面的工具、库和模型，使自然语言处理技术对开发人员和研究人员更易于使用。

HuggingFace 最著名的贡献之一是transformers 库，基于此，研究人员可以快速部署训练好的模型，以及实现新的网络结构。除此之外，HuggingFace 提供了Dataset 库，可以非常方便地下载自然语言处理研究中经常使用的基准数据集。


以构建 BERT 模型为例，介绍基于 HuggingFace 的 BERT 模型的构建和使用方法

1. **数据集准备**：常见的用于预训练语言模型的大规模数据集都可以在Dataset 库中直接下载并加载。
2. **训练词元分析器**：BERT 采用 WordPiece 分词算法，根据训练数据中的词频决定是否将一个完整的词切分为多个词元。因此，需要先训练词元分析器（Tokenizer）。可以使用transformers 库中的BertWordPiece-Tokenizer 类来完成任务
3. **预处理数据集**：在启动整个模型训练之前，还需要将预训练数据根据训练好的词元分析器进行处理。如果文档长度超过512 个词元，就直接截断。
4. **模型训练**：在构建处理好的预训练数据之后，就可以开始模型训练。
5. **模型使用**：可以针对不同应用需求使用训练好的模型，比如句子补全、文本分类等。

> 具体代码请见代码仓









## 2.3 大语言模型的结构

当前，绝大多数大语言模型都采用类似GPT 的架构，使用基于Transformer 结构构建的仅由解码器组成的网络结构，采用自回归的方式构建语言模型，但是在位置编码、层归一化位置、激活函数等细节上各有不同。


本节将以 LLaMA 模型为例，介绍大语言模型架构在 Transformer 原始结构上的改进，并介绍Transformer 结构中空间和时间占比最大的注意力机制的优化方法。





### 2.3.1 LLaMA 的模型结构

LLaMA 采用的Transformer 结构和细节，与2.1 节介绍的Transformer 结构的不同之处为采用了前置层归一化（Pre-normalization）方法并使用RMSNorm 归一化函数（Root Mean Square Normalizing Function），激活函数更换为SwiGLU，使用了旋转位置嵌入（Rotary Positional Embeddings，RoPE），如图2.4 所示。

<div align="center">
    <img src="./assets/ch02/pic-2.4.png" alt="2.4" />
</div>





#### RMSNorm 归一化函数

为了使模型训练过程更加稳定，GPT-2 相较于GPT 引入了前置层归一化方法，将第一个层归一化移动到多头自注意力层之前，将第二个层归一化移动到全连接层之前。同时，残差连接的位置调整到多头自注意力层与全连接层之后。层归一化中也采用了RMSNorm 归一化函数。针对输入向量 $\boldsymbol{a}$，RMSNorm 函数的计算公式如下：

$$
\operatorname{RMS}(a)=\sqrt{\frac{1}{n} \sum_{i=1}^{n} a_{i}^{2}} \tag{2.19}
$$

$$
\bar{a}_{i}=\frac{a_{i}}{\operatorname{RMS}(a)} \tag{2.20}
$$

此外，RMSNorm 还可以引入可学习的缩放因子 $g_i$ 和偏移参数 $b_i$，从而得到 $\bar{a}_{i}=\frac{a_{i}}{\operatorname{RMS}(\boldsymbol{a})} g_{i}+b_{i}$。






#### SwiGLU 激活函数

在 LLaMA 中，全连接层使用带有 SwiGLU 激活函数的位置感知前馈网络的计算公式如下：

$$
\operatorname{FFN}_{\operatorname{SwiGLU}}\left(x, W, V, W_{2}\right)=\operatorname{SwiGLU}(x, W, V) W_{2} \tag{2.21}
$$

$$
\operatorname{SwiGLU}(x, W, V)=\operatorname{Swish}_{\beta}(x W) \otimes x V \tag{2.22}
$$

$$
\operatorname{Swish}_{\beta}(x)=x \sigma(\beta x) \tag{2.23}
$$

其中，$\sigma(x)$ 是Sigmoid 函数。


图2.5 给出了 Swish 激活函数在参数 $\beta$ 取不同值时的形状。可以看出：

- 当 $\beta$ 趋近于0 时，Swish 函数趋近于线性函数 $y = x$；
- 当 $\beta$ 趋近于无穷大时，Swish 函数趋近于ReLU 函数 $y = \max(0, x)$；
- 当 $\beta$ 取值为1 时，Swish 函数是光滑且非单调的。

<div align="center">
    <img src="./assets/ch02/pic-2.5.png" alt="2.5" />
</div>





#### RoPE

在位置编码上，使用旋转位置嵌入代替原有的绝对位置编码。RoPE 借助复数的思想，出发点是通过绝对位置编码的方式实现相对位置编码。其目标是通过下述运算给 $q, k$ 添加绝对位置信息：

$$
\tilde{q}_{m}=f(q, m), \tilde{k}_{n}=f(k, n) \tag{2.24}
$$


详细的证明和求解过程可以参考文献，最终可以得到二维情况下用复数表示的RoPE：

$$
f(\boldsymbol{q}, m)=R_{f}(\boldsymbol{q}, m) \mathrm{e}^{\mathrm{i} \Theta_{f}(\boldsymbol{q}, m)}=\|\boldsymbol{q}\| \mathrm{e}^{\mathrm{i}(\Theta(\boldsymbol{q})+m \theta)}=\boldsymbol{q} \mathrm{e}^{\mathrm{i} m \theta} \tag{2.25}
$$


根据复数乘法的几何意义，上述变换实际上是对应向量旋转，所以位置向量称为“旋转式位置编码”。还可以使用矩阵形式表示：

$$
f(\boldsymbol{q}, m)=\left(\begin{array}{cc}
\cos m \theta & -\sin m \theta \\
\sin m \theta & \cos m \theta
\end{array}\right)\binom{\boldsymbol{q}_{0}}{\boldsymbol{q}_{1}} \tag{2.26}
$$


根据内积满足线性叠加的性质，任意偶数维的RoPE 都可以表示为二维情形的拼接，即

$$
f(\boldsymbol{q}, m)=\underbrace{\left(\begin{array}{ccccccc}
\cos m \theta_{0} & -\sin m \theta_{0} & 0 & 0 & \cdots & 0 & 0 \\
\sin m \theta_{0} & \cos m \theta_{0} & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m \theta_{1} & -\sin m \theta_{1} & \cdots & 0 & 0 \\
0 & 0 & \sin m \theta_{1} & \cos m \theta_{1} & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m \theta_{d / 2-1} & -\sin m \theta_{d / 2-1} \\
0 & 0 & 0 & 0 & \cdots & \sin m \theta_{d / 2-1} & \cos m \theta_{d / 2-1}
\end{array}\right)}_{\boldsymbol{R}_{d}}\left(\begin{array}{c}
\boldsymbol{q}_{0} \\
\boldsymbol{q}_{1} \\
\boldsymbol{q}_{2} \\
\boldsymbol{q}_{3} \\
\vdots \\
\boldsymbol{q}_{d-2} \\
\boldsymbol{q}_{d-1}
\end{array}\right)
$$

由于上述矩阵 $\boldsymbol{R}_d$ 具有稀疏性，因此可以使用逐位相乘 $\otimes$ 操作进一步提高计算速度。





#### LLaMA 模型整体框架

基于上述模型和网络结构可以实现解码器层，根据自回归方式利用训练数据进行模型训练的过程与2.2.3 节介绍的过程基本一致。不同规模的LLaMA 模型使用的超参数如表2.1 所示。由于大语言模型的参数量非常大，并且需要大量的数据进行训练，因此仅利用单个GPU 很难完成训练，需要依赖分布式模型训练框架。


<center>表 2.1 不同规模的 LLaMA 模型使用的超参数</center>

| 参数规模 | 层数 | 自注意力头数 | 嵌入表示维度 | 学习率   | 全局批次大小 | 训练词元数量（个） |
| :------- | :--: | :---------: | :---------: | :-----: | :---------: | :---------------: |
| 6.7B     |  32  |     32      |    4096     | 3.0e-4  |   400 万    |     1.0 万亿      |
| 13.0B    |  40  |     40      |    5120     | 3.0e-4  |   400 万    |     1.0 万亿      |
| 32.5B    |  60  |     52      |    6656     | 1.5e-4  |   400 万    |     1.4 万亿      |
| 65.2B    |  80  |     64      |    8192     | 1.5e-4  |   400 万    |     1.4 万亿      |






### 2.3.2 注意力机制优化

在Transformer 结构中，自注意力机制的时间和存储复杂度与序列的长度呈平方的关系，因此占用了大量的计算设备内存并消耗了大量的计算资源。如何优化自注意力机制的时空复杂度、增强计算效率是大语言模型面临的重要问题。




#### 稀疏注意力机制

对一些训练好的Transformer 结构中的注意力矩阵进行分析时发现，其中很多是稀疏的，因此可以通过限制Query-Key 对的数量来降低计算复杂度。这类方法称为稀疏注意力（Sparse Attention）机制。可以将稀疏化方法进一步分成 **基于位置** 的和 **基于内容** 的两类。


**基于位置的稀疏注意力机制** 的基本类型如图2.6 所示

1. **全局注意力（Global Attention）**：为了增强模型建模长距离依赖关系的能力，可以加入一些全局节点。
2. **带状注意力（Band Attention）**：大部分数据都带有局部性，限制Query 只与相邻的几个节点进行交互。
3. **膨胀注意力（Dilated Attention）**：与CNN 中的Dilated Conv 类似，通过增加空隙获取更大的感受野。
4. **随机注意力（Random Attention）**：通过随机采样，提升非局部的交互能力。
5. **局部块注意力（Block Local Attention）**：使用多个不重叠的块（Block）来限制信息交互。

<div align="center">
    <img src="./assets/ch02/pic-2.6.png" alt="2.6" />
</div>


现有的稀疏注意力机制，通常是基于上述五种基于位置的稀疏注意力机制的复合模式，图2.7给出了一些典型的稀疏注意力模型。

<div align="center">
    <img src="./assets/ch02/pic-2.7.png" alt="2.7" />
</div>

---

**基于内容的稀疏注意力机制** 根据输入数据创建稀疏注意力，其中一种很简单的方法是选择和给定查询（Query）有很高相似度的键（Key）。


Routing Transformer 采用 K-means 聚类方法，针对 $Query = \{q_i\}_{i=1}^{T}$ 和 $Key = \{k_i\}_{i=1}^{T}$ 进行聚类，类中心向量集合为 $\{\mu_i\}_{i=1}^{k}$，其中 $k$ 是类中心的个数。每个Query 只与其处在相同簇（Cluster）下的Key 进行交互。中心向量采用滑动平均的方法进行更新：

$$
\widetilde{\mu} \leftarrow \lambda \widetilde{\mu}+(1-\lambda)\left(\sum_{i: \mu\left(\boldsymbol{q}_{i}\right)=\mu} q_{i}+\sum_{j: \mu\left(\boldsymbol{k}_{j}\right)=\mu} k_{j}\right) \tag{2.28}
$$

$$
c_{\mu} \leftarrow \lambda c_{\mu}+(1-\lambda)|\mu| \tag{2.29}
$$

$$
\mu \leftarrow \frac{\widetilde{\mu}}{c_{\mu}} \tag{2.30}
$$

其中 $|\mu|$ 表示在簇 $\mu$ 中向量的数量。


Reformer 则采用 **局部敏感哈希（Local-Sensitive Hashing，LSH）** 的方法为每个Query 选择Key-Value 对。其主要思想是使用LSH 函数对Query 和Key 进行哈希计算，将它们划分到多个桶内，以提升在同一个桶内的Query 和Key 参与交互的概率。假设 $b$ 是桶的个数，给定一个大小为 $[D_k, b/2]$ 的随机矩阵 $R$，LSH 函数的定义为：

$$
h(x)=\arg \max ([x R ;-x R]) \tag{2.31}
$$

当 $hq_i = hk_j$ 时，$q_i$ 才可以与相应的 $k_j$ 进行交互。






#### FlashAttention

GPU 显存分为全局内存（Global Memory）、本地内存（Local Memory）、共享存储（Shared Memory，SRAM）、寄存器（Register）、常量内存（Constant Memory）、纹理内存（Texture Memory）六大类。图2.8 为NVIDIA GPU 的整体内存结构示意图。全局内存和本地内存使用的高带宽显存（High Bandwidth Memory，HBM）位于板卡RAM 存储芯片上，该部分内存容量很大。所有线程都可以访问全局内存，而本地内存只能由当前线程访问。

<div align="center">
    <img src="./assets/ch02/pic-2.8.png" alt="2.8" />
</div>

NVIDIA H100中全局内存有80GB 空间，其访问速度虽然可以达到3.35TB/s，但当全部线程同时访问全局内存时，其平均带宽仍然很低。共享存储和寄存器位于GPU 芯片上，因此容量很小，并且只有在同一个GPU 线程块（Thread Block）内的线程才可以并行访问共享存储，而寄存器仅限于同一个线程内部访问。虽然NVIDIA H100 中每个GPU 线程块在流式多处理器（Stream Multi-processor，SM）上可以使用的共享存储容量仅有228KB，但是其速度比全局内存的访问速度快很多。

---


前文介绍了自注意力机制的原理，在GPU 中进行计算时，传统的方法还需要引入两个中间矩阵 $S$ 和 $P$ 并存储到全局内存中。具体计算过程如下：

$$
S=Q K, \quad P=\operatorname{Softmax}(S), \quad O=P V \tag{2.32}
$$

按照上述计算过程，需要先从全局内存中读取矩阵 $Q$ 和 $K$，并将计算好的矩阵 $S$ 写入全局内存，然后从全局内存中获取矩阵 $S$，计算 $\text{Softmax}$ 得到矩阵 $P$，再将其写入全局内存，最后读取矩阵$P$ 和矩阵 $V$ ，计算得到矩阵 $O$。

这样的过程会极大地占用显存的带宽。在自注意力机制中，GPU的计算速度比内存速度快得多，因此**计算效率越来越受全局内存访问的制约**。


FlashAttention 利用GPU 硬件中的特殊设计，针对全局内存和共享存储的 I/O 速度的不同，尽可能地避免从HBM 中读取或写入注意力矩阵。FlashAttention 的目标是尽可能高效地使用SRAM来加快计算速度，避免从全局内存中读取和写入注意力矩阵。达成该目标需要做到在不访问整个输入的情况下计算 $\text{Softmax}$ 函数，并且后向传播中不能存储中间注意力矩阵。


在标准Attention 算法中，$\text{Softmax}$ 计算按行进行，即在与 $V$ 做矩阵乘法之前，需要完成 $Q$、$K$ 每个分块中的一整行的计算。在得到Softmax 的结果后，再与矩阵V 分块做矩阵乘。而在FlashAttention 中，将输入分割成块，并在输入块上进行多次传递，以增量的方式执行 $\text{Softmax}$ 计算。


自注意力算法的标准实现将计算过程中的矩阵 $S$、$P$ 写入全局内存，而这些中间矩阵的大小与输入的序列长度有关且为二次型。因此，FlashAttention 就提出了不使用中间注意力矩阵，通过存储归一化因子来减少全局内存消耗的方法。FlashAttention 算法并没有将 $S$、$P$ 整体写入全局内存，而是通过分块写入，存储前向传播的 $\text{Softmax}$ 归一化因子，在后向传播中快速重新计算片上注意力，这比从全局内存中读取中间注意力矩阵的标准方法更快。虽然大幅减少了全局内存的访问量，重新计算也导致 FLOPS 增加，但总体来看运行的速度更快且使用的显存更少。具体算法如代码2.1 所示，其中内层循环和外层循环所对应的计算可以参考图2.9。

> **代码 2.1: FlashAttention 算法**
>
> **输入:** $Q, K, V \in \mathbb{R}^{N \times d}$ 位于 HBM 中，GPU 芯片中的 SRAM 大小为 $M$
>
> **输出:** $O$
>
> 1. $B_c = \lceil \frac{M}{4d} \rceil$，$B_r = \min(\lceil \frac{M}{4d} \rceil, d)$ // 设置块大小（block size）
> 2. 在 HBM 中初始化 $O = (0)_{N \times d} \in \mathbb{R}^{N \times d}$，$l = (0)_N \in \mathbb{R}^N$，$m = (-\infty)_N \in \mathbb{R}^N$
> 3. 将矩阵 $Q$ 切分成 $T_r = \lceil \frac{N}{B_r} \rceil$ 块 $Q_1, Q_2, \cdots, Q_{T_r}$，$Q_i \in \mathbb{R}^{B_r \times d}$
> 4. 将矩阵 $K$ 切分成 $T_c = \lceil \frac{N}{B_c} \rceil$ 块 $K_1, K_2, \cdots, K_{T_c}$，$K_i \in \mathbb{R}^{B_c \times d}$
> 5. 将矩阵 $V$ 切分成 $T_c$ 块 $V_1, V_2, \cdots, V_{T_c}$，$V_i \in \mathbb{R}^{B_c \times d}$
> 6. 将矩阵 $O$ 切分成 $T_r$ 块 $O_1, O_2, \cdots, O_{T_r}$，$O_i \in \mathbb{R}^{B_r \times d}$
> 7. 将 $l$ 切分成 $T_r$ 块 $l_1, l_2, \cdots, l_{T_r}$，$l_i \in \mathbb{R}^{B_r}$
> 8. 将 $m$ 切分成 $T_r$ 块 $m_1, m_2, \cdots, m_{T_r}$，$m_i \in \mathbb{R}^{B_r}$
> 9. **for** $j = 1$ to $T_c$ **do**
> 10. &emsp; 将 $K_j$ 和 $V_j$ 从芯片外部的 HBM 中读入芯片内部存储 SRAM
> 11. &emsp; **for** $i = 1$ to $T_r$ **do**
> 12. &emsp;&emsp; 计算 $S_{ij} = Q_i K_j^T \in \mathbb{R}^{B_r \times B_c}$
> 13. &emsp;&emsp; 计算 $\tilde{m}_{ij} = \text{rowmax}(S_{ij}) \in \mathbb{R}^{B_r}$，$\tilde{P}_{ij} = \exp(S_{ij} - \tilde{m}_{ij}) \in \mathbb{R}^{B_r \times B_c}$
> 14. &emsp;&emsp; 计算 $\tilde{l}_{ij} = \text{rowsum}(\tilde{P}_{ij}) \in \mathbb{R}^{B_r}$
> 15. &emsp;&emsp; 计算 $m_i^{\text{new}} = \max(m_i, \tilde{m}_{ij}) \in \mathbb{R}^{B_r}$，$l_i^{\text{new}} = e^{m_i - m_i^{\text{new}}} l_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{l}_{ij} \in \mathbb{R}^{B_r}$
> 16. &emsp;&emsp; 将 $O \leftarrow \text{diag}(l_i^{\text{new}})^{-1} \big( \text{diag}(l_i) e^{m_i - m_i^{\text{new}}} O_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}} \tilde{P}_{ij} V_j \big)$ 写回 HBM 中
> 17. &emsp;&emsp; 将 $l_i \leftarrow l_i^{\text{new}}$ 和 $m_i \leftarrow m_i^{\text{new}}$ 写回 HBM 中
> 18. &emsp; **end**
> 19. **end**
> 20. **return** $O$


<div align="center">
    <img src="./assets/ch02/pic-2.9.png" alt="2.9" />
</div>






#### 多查询注意力

多查询注意力（Multi Query Attention） 是多头注意力的一种变体。它的特点是，在多查询注意力中不同的注意力头共享一个键和值的集合，每个头只单独保留了一份查询参数，因此键和值的矩阵仅有一份，这大幅减少了显存占用，使其更高效。由于多查询注意力改变了注意力机制的结构，因此模型通常需要从训练开始就支持多查询注意力。文献的研究结果表明，可以通过对已经训练好的模型进行微调来添加多查询注意力支持，仅需要约5% 的原始训练数据量就可以达到不错的效果。







#### 多头潜在注意力

多头潜在注意力（Multi-Head Latent Attention，MLA） 是在DeepSeek-V2 中引入的注意力优化模型。多头潜在注意力通过在键值层利用低秩矩阵，实现对压缩潜在键值状态的缓存，从而大幅减少了KV 缓存大小，有效缓解了通信瓶颈。


具体来说，MLA 方法的核心是是将传统多头注意力中的键（Key）和值（Vale）进行低秩联合压缩，得到一个低秩表示形式，以减少键值（KV）缓存。设 $d$ 为嵌入维度，$n_h$ 为注意力头的数量，$d_h$ 为每个头的维度，$h_t \in \mathbb{R}^d$ 是注意力层中第 $t$ 个词元的输入。标准的多头注意力机制（MHA）首先通过三个矩阵 $W^Q, W^K, W^V \in \mathbb{R}^{d_h n_h \times d}$ 生成 $q_t, k_t, v_t \in \mathbb{R}^{d_h n_h}$。MLA 方法则通过如下公式对 KV 缓存进行压缩：

$$
c_{t}^{K V} =W^{D K V} h^{t} \tag{2.33}
$$

$$
k_{t}^{C} =W^{U K} c_{t}^{K V} \tag{2.34}
$$

$$
v_{t}^{C} =W^{U V} c_{t}^{K V} \tag{2.35}
$$

其中

- $c_{t}^{K V} \in \mathbb{R}^{d_c}$ 是键和值的压缩潜在向量（Comressed Latent Vector）;
- $d_c \ll d_h n_h$ 表示键值压缩维度；
- $W^{D K V} \in \mathbb{R}^{d_c \times d}$ 是下投影矩阵；
- $W^{U K} \in \mathbb{R}^{d_h n_h \times d_c}$ 和 $W^{U V} \in \mathbb{R}^{d_h n_h \times d_c}$ 分别是键和值的上投影矩阵。

在推理过程中，MLA 方法只需要缓存 $c_{t}^{K V}$ ，因此其键值缓存仅有 $d_c l$ 个元素，其中 $l$ 表示层数。


此外，在推理过程中，由于 $W^{U K}$ 可以合并到 $W^Q$ 中，$W^{U V}$ 可以合并到 $W^O$ 中，甚至无需在注意力计算中真正获得键和值。为了在训练过程中减少激活内存，还可以进一步对查询（Query）进行低秩压缩：

$$
c_{t}^{Q} =W_{D Q} h_{t} \tag{2.36}
$$

$$
q_{t}^{C} =W^{U Q} c_{t}^{Q} \tag{2.37}
$$

其中，
- $c_{t}^{Q} \in \mathbb{R}^{d_c'}$ 是查询的压缩潜在向量；
- $d_c' \ll d_h n_h$ 表示查询压缩维度，$W^{D Q} \in \mathbb{R}^{d_c' \times d}$ 和 $W^{U Q} \in \mathbb{R}^{d_h n_h \times d_c'}$ 分别是查询的下投影矩阵和上投影矩阵。








## 2.4 混合专家模型

随着GPT-4、Mixtral-8x7B、DeepSeek-V3 等模型的相继推出，混合专家模型(Mixed Expert Models，MoEs) 日益受到关注。依据大模型缩放法则，模型规模是提升性能的关键，然而规模扩大必然使计算资源大幅增加。因此，在有限计算资源预算下，如何用更少训练步数训练更大模型成为关键问题。

为解决该问题，混合专家模型基于一个简洁的思想：模型不同部分（即“专家”）专注不同任务或数据层面。混合专家架构的引入使得训练具有数千亿甚至万亿参数的模型成为可能。

---

在采用混合专家架构的大语言模型中，MoE 层通常由门控网络（Gating Network） $G$ 和 $N$ 个专家网络（Experts Network）$\{f_1, f_2, \cdots, f_N\}$ 组成。

- 门控网络充当着选择器的角色，也称为路由，它负责决定将哪些输入数据发送给哪些专家；
- 专家网络则分别处理特定的不同子任务。

在这一过程中，并非所有专家都同时运作，而是由门控网络依据数据特性，精准地将数据路由到与之最为相关的专家那里，最终再根据一个或者多个专家输出的结果综合得到整体的预测结果。

---

在模型架构的设计中，MoE 层通常安置于每个Transformer 模块中前馈层（FFN）。当模型不断扩大时，FFN层在计算方面的需求也越来越高。


混合专家架构中，每个专家网络 $f_i$ 通常由一个前馈层组成，其参数使用 $W_i$ 表示。对于给入的输入 $X$，其输出使用 $f_i(X;W_i)$ 表示。门控网络 $G$ 通常使用线性Softmax（Linear-Softmax）网络构成，使用 $\Theta$ 表示其参数，其输出使用 $G(x;\Theta)$ 表示。

如图2.10 所示，混合专家模型按照门控网络（Gate）类型，可以从广义上讲可以分为三个大类：
- **稀疏混合专家模型（Sparse MoE）**
- **稠密混合专家模型（Dense MoE）**
- **软混合专家模型（Soft MoE）**

<div align="center">
    <img src="./assets/ch02/pic-2.10.png" alt="2.10" />
</div>






### 2.4.1 稀疏混合专家模型

稀疏混合专家模型，如图2.10(a) 所示，对于每个输入词元，在前向计算中仅激活专家集合中的一个子集。门控网络对专家子集进行选择，通过计算排名前K 位专家的输出加权和来实现稀疏性。这个过程可以形式化的表示为：

$$
\mathcal{F}_{\text {Sparse }}^{M o E}\left(\mathbf{x} ; \Theta ;\left\{\mathbf{W}_{i}\right\}_{i=1}^{N}\right)=\sum_{i=1}^{N} \mathcal{G}(\mathbf{x} ; \Theta)_{i} f_{i}\left(\mathbf{x} ; \mathbf{W}_{i}\right) \tag{2.38}
$$

$$
\mathcal{G}(\mathrm{x} ; \Theta)_{i}=\operatorname{softmax}\left(\operatorname{TopK}\left(g(\mathrm{x} ; \Theta)+\mathcal{R}_{\text {noise }}, K\right)\right)_{i} \tag{2.39}
$$

$$
\operatorname{Top}-\mathrm{K}(g(\mathrm{x} ; \Theta), K)_{i}=\left\{\begin{array}{l}
g(\mathrm{x} ; \Theta)_{i}, g(\mathrm{x} ; \Theta)_{i} \text { 的值属于前 } \mathrm{K} \text { 项 } \\
-\infty, \text { 其他 }
\end{array}\right.
\tag{2.40}
$$

其中，
- $g(x;\Theta)$ 表示在进行softmax 操作之前的门控值，
- $G(x;\Theta)_i$ 表示门控网络针对第i 个专家的输出，
- $\text{TopK}(g(x;\Theta), K)_i$ 函数的目标是保持向量的前 K 项不变，其它维度设置为 $-\infty$。

鉴于 softmax 函数自身所具有的独特性质，当把其中某些项设置为 $-\infty$ 时，这些项所对应的值会近似等同于 $0$。超参数 $K$ 是根据具体应用来选取的，常见的取值选择为 $K = 1^{[67, 69]}$ 或者 $K = 2^{[66, 70-72]}$。


添加噪声项 $R_{\text{noise}}$ 是训练稀疏混合专家层的一种常用策略：

- 一方面，它能够为模型创造更多的探索空间，促使不同专家模块之间展开多样化的尝试与协作，挖掘出潜在的优化路径；
- 另一方面，通过打破可能出现的局部最优情况，提高了整个混合专家训练过程的稳定性。

---

稀疏混合专家模型中采用常规的门控策略时，分配给不同专家的词元可能需要一些共有知识或信息才能处理。因此，多个专家可能会在各自的参数中获取同样的知识，进而导致专家参数出现冗余。如果构建专门用于捕捉并整合不同情境下共有知识的共享专家，那么其他专家之间的参数冗余情况将可能得到缓解。这种冗余情况的缓解，有助于构建一个参数利用更高效且专家专业性更强的模型。因此，DeepSeekMoE 提出了分离 $K_s$ 个专家作为共享专家的思路。无论门控网络所给出的结果如何，每个词元都将被确定性地分配给这些共享专家，如图2.11 所示，深色块Shared FFN 为共享专家，所有输入都会分配给共享专家。为保持计算成本恒定，其他经门控网络分配的专家中被激活专家的数量将减少 $K_s$ 个。

<div align="center">
    <img src="./assets/ch02/pic-2.11.png" alt="2.11" />
</div>



稀疏混合专家模型中的MoE 层对于并行计算也十分友好，能更便捷地在单个GPU 上实现高效计算。常规稠密模型中，全部参数都会参与对所有输入数据的处理流程。与之不同，稀疏混合专家模型具备的稀疏特性，使得计算仅在系统的特定局部展开。也就是说，并非所有参数在处理各个输入时都会被触发或启用，而是依据输入的具体特性与需求，仅有特定的部分参数集被唤起并运行。因此，在并行计算中可以有效利用上述特性。此外，MoE 层可以通过标准的模型并行技术分布到多个GPU 上，还可以借助专家并行（Expert Parallelism，EP） 实现特殊的分区策略。





### 2.4.2 稠密混合专家模型

稠密混合专家模型，如图2.10(b) 所示，对于每个输入词元，在前向计算中激活所有专家网络$\{f_1, \cdots, f_N\}$。门控网络根据输入赋予专家不同的权重。这个过程可以形式化的表示为：

$$
\mathcal{F}_{\text {Dense }}^{M o E}\left(\mathbf{x} ; \Theta ;\left\{\mathbf{W}_{i}\right\}_{i=1}^{N}\right)=\sum_{i=1}^{N} \mathcal{G}(\mathbf{x} ; \Theta)_{i} f_{i}\left(\mathbf{x} ; \mathbf{W}_{i}\right) \tag{2.41}
$$

$$
\mathcal{G}(\mathrm{x} ; \Theta)_{i}=\operatorname{softmax}(g(\mathrm{x} ; \Theta))_{i}=\frac{\exp \left(g(\mathrm{x} ; \Theta)_{i}\right)}{\sum_{j}^{N} \exp \left(g(\mathrm{x} ; \Theta)_{j}\right)} \tag{2.42}
$$

由于稠密混合专家模型在前向计算过程中会激活所有参数，不能降低模型计算量。因此，大语言模型采用稠密混合专家结构的并不多。

---


虽然稠密混合专家模型需要使用全部参数进行计算，并不能减少模型计算时间，但是研究人员却发现，如果能够将LoRA 方法和MoE 相结合，可以在占用很少GPU 显存的同时，减少微调数据的大规模扩增与模型世界知识维持之间存在的冲突。有监督微调是大语言模型应用的一个关键步骤，当模型需要与更广泛的下游任务保持一致，或者希望显著提高在特定任务上的表现时，大规模增加微调数据通常成为解决方案。然而当指令数据的大规模扩增可能会破坏大语言模型中之前储存的世界知识，即世界知识遗忘。


LoRAMoE 采用融合混合专家和LoRA 插件的思想，插件形式确保了在训练阶段冻结主模型，保证了主模型世界知识的完整性。


LoRAMoE 模型架构如图2.12所示。基于插件的微调能够将参数的改动集中在额外引入的插件中，从而保证了模型知识的完整性，有机会引入其他插件来通过与主模型的交互来缓解知识遗忘。LoRAMoE 引入了多个与前反馈神经网络并列的专家，并通过路由相连，如图2.12中标注了“火焰”符号的部分，这些部分也是需要在后续学习中进行参数学习的结构。LoRAMoE 在训练阶段使用局部平衡约束损失（Localized Balancing Constraint），这种约束能够让专家自动划分为两个组：使一部分专家在专注于做下游任务的同时，另一部分专家专注于将指令与主模型的世界知识对齐，以缓解世界知识遗忘。同时局部平衡约束还能防止单个专家组内的专家退化现象，使路由平衡地关注于单个专家组的所有专家，防止个别专家长期占据优势，而其他专家未被充分训练或使用。这有助于专家之间相互配合以提高下游任务能力。微调后的LoRAMoE 中的路由能够根据数据类型灵活地关注相应的专家，并使专家们相互配合，在保证下游任务表现的同时，也几乎不丧失世界知识。

<div align="center">
    <img src="./assets/ch02/pic-2.12.png" alt="2.12" />
</div>






### 2.4.3 软混合专家模型

软混合专家模型，如图2.10(c) 所示，门控网络依然根据输入为各个专家分配不同的权重，但与稠密混合专家模型在前向计算中激活所有专家网络不同，软混合专家模型引入了融合前馈层（Merged FFN）。该方法通过门控网络分配的权重对不同专家的参数进行融合，仅对融合后的前馈层参数进行计算。这种设计既能在几乎不增加计算成本的情况下完成计算，又保留了稠密混合专家模型中可使用基于梯度的训练方法的优势。这个过程可以形式化的表示为：

$$
\mathcal{F}_{\text {Soft }}^{M o E}\left(\mathbf{x} ; \Theta ;\left\{\mathbf{W}_{i}\right\}_{i=1}^{N}\right)=f_{\text {merged }}\left(\mathbf{x} ; \sum_{i=1}^{N} \mathcal{G}(\mathbf{x} ; \Theta)_{i} \mathbf{W}_{i}\right) \tag{2.43}
$$

$$
\mathcal{G}(\mathbf{x} ; \boldsymbol{\Theta})_{i}=\operatorname{softmax}(g(\mathbf{x} ; \boldsymbol{\Theta}))_{i}=\frac{\exp \left(g(\mathbf{x} ; \boldsymbol{\Theta})_{i}\right)}{\sum_{j}^{N} \exp \left(g(\mathbf{x} ; \boldsymbol{\Theta})_{j}\right)} \tag{2.44}
$$

其中，$f_{\text{merged}}$ 表示融合前向层，其结构与其余专家网络 $f_i$ 的结构相同。SMEAR 算法就采用了这种软混合专家结构。


软混合专家模型始终只计算单个专家的输出，其计算成本可能与单专家稀疏混合模型相当，明显低于稠密混合专家模型。但是，软混合专家模型的平均操作仍然会产生不可忽视的计算成本。为了量化这一成本，文献[85] 分析了SMEAR 算法的计算复杂度。假设专家网络架构是一个从 $d$ 维激活值投射到 $m$ 维向量的稠密计算，随后经过非线性变换，再附加一个从 $m$ 维投射回 $d$ 维的稠密计算。为简便起见，这里忽略成本相对较小的非线性变换成本。假定输入是一个长度为 $L$ 的激活值序列，其大小为 $L \times d$。在这种情况下，计算合并专家的输出会产生大约 $L \times 4 \times d \times m$ 次浮点运算（FLOPs）的计算成本，而采用 $N$ 个专家的稠密混合专家模型则需要 $N \times L \times 4 \times d \times m$ 次浮点运算。此外，软混合专家模型还必须对 $N$ 个专家的参数进行平均，这又会额外产生 $N \times 2 \times d \times m$次浮点运算的成本。整体上SMEAR 算法的计算复杂度是 $(L \times 4 + N \times 2) \times d \times m$。综合整体计算成本，软混合专家模型计算复杂度仍然远低于稠密混合专家模型。








