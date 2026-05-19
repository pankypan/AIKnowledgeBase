# RNN相关的神经网络模型总结

## Full Connected Neural Network

### 基本结构

将许多全连接层堆叠在一起。 每一层都输出到上面的层，直到生成最后的输出。 我们可以把前 $L-1$ 层看作表示，把最后一层看作线性预测器。 这种架构通常称为多层感知机（multilayer perceptron），通常缩写为MLP。

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/mlp.svg" width="500px">
</div>

这个多层感知机有4个输入，3个输出，其隐藏层包含5个隐藏单元。



### 数学模型

对于一个只有单隐藏层的多层感知机。 设隐藏层的激活函数为 $\phi$， 给定一个小批量样本 $\mathbf{X} \in \mathbb{R}^{n \times d}$， 其中批量大小为 $n$ ，输入维度为 $d$ ， 则隐藏层的输出 $\mathbf{H} \in \mathbb{R}^{n \times h}$ 通过下式计算：

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h).$$

其中：
- 隐藏层权重参数为 $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$
- 偏置参数为 $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$
- 隐藏单元的数目为 $h$

接下来，将隐藏变量 $\mathbf{H}$ 用作输出层的输入。 输出层由下式给出：

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q,$$

其中：
- $O$ 表示输出，$\mathbf{O} \in \mathbb{R}^{n \times q}$
- 输出层权重参数为 $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$
- 偏置参数为 $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$

如果是分类问题，我们可以用$softmax(O)$来计算输出类别的概率分布。



### 普通神经网络的局限性

从传统的神经网络结构我们可以看出，信号流从输入层到输出层依次流过，同一层级的神经元之间，信号是不会相互传递的。
- 这样就会导致一个问题，输出信号只与输入信号有关，而与输入信号的先后顺序无关。
- 并且神经元本身也不具有存储信息的能力，整个网络也就没有“记忆”能力，当输入信号是一个跟时间相关的信号时，如果我们想要通过这段信号的“上下文”信息来理解一段时间序列的意思，传统的神经网络结构就显得无力了。

因此，我们需要构建具有“记忆”能力的神经网络模型，用来处理需要理解上下文意思的信号，也就是时间序列数据。



## RNN

### 基本结构

RNN是一种特殊的神经网路结构，其本身是包含循环的网络，允许信息在神经元之间传递，如下图所示：

<div align="center"><img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png" width="220px"></div>

上图是一个RNN结构示意图：

- $A$ 表示一个神经网模型
- $x_t$ 表示在第 $t$ 个时间步的输入，$h_t$ 表示在第 $t$ 个时间步的输出。

关键在于输入信号是一个时间序列，跟时间 $t$ 有关。也就是说，在 $t$ 时刻，输入信号 $x_t$ 作为神经网络 $A$ 的输入，$A$ 的输出分流为两部分，一部分输出给 $h_t$，一部分作为一个隐藏的信号流被输入到 $A$ 中，在下一次时刻输入信号 $x_{t+1}$ 时，这部分隐藏的信号流也作为输入信号输入到了 $A$ 中。

如果我们把上面那个图根据时间 $t$ 展开来看，就是：

<div align="center">
<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" width="600px">
</div>


这样链式的结构揭示了RNN本质上是与序列相关的，是对于时间序列数据最自然的神经网络架构。并且理论上，RNN可以保留以前任意时刻的信息。RNN在语音识别、自然语言处理、图片描述、视频图像处理等领域已经取得了一定的成果，而且还将更加大放异彩。在实际使用的时候，用得最多的一种RNN结构是LSTM




### 数学模型

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/rnn.svg" width="700px">
</div>

假设我们在时间步 $t$ 有小批量输入 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$，其中批量大小为 $n$ ，输入维度为 $d$ 。接下来，用 $\mathbf{H}_t \in \mathbb{R}^{n \times h}$ 表示时间步 $t$ 的隐藏变量:

1. 隐状态 $\mathbf{H}_t \in \mathbb{R}^{n \times h}$：

    $$
    \mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh} + \mathbf{b}_h),
    $$
2. 输出 $\mathbf{Y}_t \in \mathbb{R}^{n \times q}$：

    $$
    \mathbf{Y}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.
    $$




### RNN的补充说明

**循环神经网络在三个相邻时间步的计算逻辑**。在任意时间步$t$，隐状态的计算可以被视为：

1. 拼接当前时间步$t$的输入 $\mathbf{X}_t$ 和前一时间步$t-1$的隐状态 $\mathbf{H}_{t-1}$；
2. 将拼接的结果送入带有激活函数 $\phi$ 的全连接层。 全连接层的输出是当前时间步$t$的隐状态 $\mathbf{H}_t$。
3. 将当前时间步$t$的隐状态 $\mathbf{H}_t$ 作为输出层的输入，输出层计算得到当前时间步$t$的输出 $\mathbf{O}_t$。


**补充说明：**

1. 与多层感知机不同的是， 我们在这里保存了前一个时间步的隐藏变量 $\mathbf{H}_{t-1}$， 并引入了一个新的权重参数 $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$， 来描述如何在当前时间步中使用前一个时间步的隐藏变量。
2. 从相邻时间步的隐藏变量 $\mathbf{H}_{t-1}$ 和 $\mathbf{H}_t$ 之间的关系可知， 这些变量捕获并保留了序列直到其当前时间步的历史信息， 就如当前时间步下神经网络的状态或记忆， 因此这样的隐藏变量被称为**隐状态（hidden state）**。 由于在当前时间步中， 隐状态使用的定义与前一个时间步中使用的定义相同， 因此 $\mathbf{H}_t$ 的计算是循环的（recurrent）。 于是基于循环计算的隐状态神经网络被命名为 **循环神经网络（recurrent neural network）。**
3. 值得一提的是，即使在不同的时间步，循环神经网络也总是使用这些模型参数。 因此，循环神经网络的参数开销不会随着时间步的增加而增加。




### RNN的代码实现

**基类 `WithHiddenStateModel`：**

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


class WithHiddenStateModel(object):
    """带有隐藏状态的模型"""
    def __init__(self, vocab_size, num_hiddens, device):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = self.get_init_params(vocab_size, num_hiddens, device)
    
    def get_init_params(self, vocab_size, num_hiddens, device):
        raise NotImplementedError

    def begin_state(self, batch_size, device):
        raise NotImplementedError

    def forward(self, inputs, state, params):
        raise NotImplementedError

    def __call__(self, X, state):
        # 转置inputs的形状，并转换为one-hot编码 -> (时间步数量，批量大小，词表大小)
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward(X, state, self.params)
```


**RNNModelScratch类是RNN模型的实现：**

```python
class RNNModelScratch(WithHiddenStateModel):
    """从零开始实现的循环神经网络模型"""
    def get_init_params(self, vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size

        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        # 隐藏层参数
        W_xh = normal((num_inputs, num_hiddens))
        W_hh = normal((num_hiddens, num_hiddens))
        b_h = torch.zeros(num_hiddens, device=device)
        # 输出层参数
        W_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        # 附加梯度
        params = [W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def begin_state(self, batch_size, device):
        return (torch.zeros((batch_size, self.num_hiddens), device=device), )
    
    def forward(self, inputs, state, params):
        # inputs的形状：(时间步数量，批量大小，词表大小)
        W_xh, W_hh, b_h, W_hq, b_q = params
        H, = state
        outputs = []
        # X的形状：(批量大小，词表大小)
        for X in inputs:
            H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
            Y = torch.mm(H, W_hq) + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,)
```


**测试RNN模型的输出和状态**：

```python
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

# ============================= Test the RNN Model =============================
# test the RNN model calculate the output and the state
num_hiddens = 512
rnn_model = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu())

X, Y = next(iter(train_iter))  # X = torch.arange(10).reshape((2, 5))
state = rnn_model.begin_state(X.shape[0], d2l.try_gpu())
Y_hat, new_state = rnn_model(X.to(d2l.try_gpu()), state)
print("RNN Model Output and State: ", Y_hat.shape, len(new_state), new_state[0].shape)
```



### RNN的局限性

RNN利用了神经网络的“内部循环”来保留时间序列的上下文信息，可以使用过去的信号数据来推测对当前信号的理解，这是非常重要的进步，并且理论上RNN可以保留过去任意时刻的信息。但实际使用RNN时往往遇到问题

有时候我们只需要用到最近的时刻的信息。例如预测“我喜欢妈妈做的菜”最后这个词“菜”，此时信息传递是这样的：

<div align="center">
<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-shorttermdepdencies.png" width="600px">
</div>

“菜”这个词与“我”、“喜欢”、“妈妈”、“做”、“的”这几个词关联性比较大，距离也比较近，所以可以直接利用这几个词进行最后那个词语的推测。



而有时候我们又需要用到很早以前时刻的信息，例如预测“我最常说汉语”最后的这个词“汉语”。此时信息传递是这样的：

<div align="center">
<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-longtermdependencies.png" width="600px">
</div>

此时，我们要预测“汉语”这个词，仅仅依靠“我”、“最”、“常”、“说”这几个词还不能得出我说的是汉语，必须要追溯到更早的句子“我是一个中国人”，由“中国人”这个词语来推测我最常说的是汉语。


RNN虽然在理论上可以保留所有历史时刻的信息，但在实际使用时， **信息的传递往往会因为时间间隔太长而逐渐衰减，传递一段时刻以后其信息的作用效果就大大降低了。** 因此，普通RNN对于信息的长期依赖问题没有很好的处理办法。




## LSTM

为了克服RNN的长期依赖问题，Hochreiter等人在1997年改进了RNN，提出了一种特殊的RNN模型——LSTM网络，可以学习长期依赖信息，在后面的20多年被改良和得到了广泛的应用，并且取得了极大的成功。



### 基本结构

长短期记忆（Long Short Term Memory，LSTM）网络是一种特殊的RNN模型，其特殊的结构设计使得它可以避免长期依赖问题，记住很早时刻的信息是LSTM的默认行为，而不需要专门为此付出很大代价。


普通的RNN模型中，其重复神经网络模块的链式模型如下图所示，这个重复的模块只有一个非常简单的结构，一个单一的神经网络层（例如tanh层），这样就会导致信息的处理能力比较低。

<div align="center">
<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png" width="600px">
</div>


而LSTM在此基础上将这个结构改进了，不再是单一的神经网络层，而是4个神经网络层，并且这4个神经网络层之间有连接，这样就大大提高了信息的处理能力。

<div align="center">
<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png" width="600px">
</div>

其中:

<div align="center">
<img src="https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM2-notation.png" width="500px">
</div>




### 数学模型

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/lstm-3.svg" width="700px">
</div>

对于给定时间步 $t$，假设有 $h$ 个隐藏单元，批量大小为 $n$ ，输入数（维度）为 $d$ 。输入为 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$， 前一时间步的隐状态为 $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$ ：

1. 输入门 $\mathbf{I}_t \in \mathbb{R}^{n \times h}$, 遗忘门 $\mathbf{F}_t \in \mathbb{R}^{n \times h}$, 输出门 $\mathbf{O}_t \in \mathbb{R}^{n \times h}$ ：

    $$
    \begin{split}\begin{aligned}
    \mathbf{I}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xi} + \mathbf{H}_{t-1} \mathbf{W}_{hi} + \mathbf{b}_i),\\
    \mathbf{F}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xf} + \mathbf{H}_{t-1} \mathbf{W}_{hf} + \mathbf{b}_f),\\
    \mathbf{O}_t &= \sigma(\mathbf{X}_t \mathbf{W}_{xo} + \mathbf{H}_{t-1} \mathbf{W}_{ho} + \mathbf{b}_o),
    \end{aligned}\end{split}
    $$
2. 候选记忆元（candidate memory cell） $\tilde{\mathbf{C}}_t \in \mathbb{R}^{n \times h}$ ：

    $$
    \tilde{\mathbf{C}}_t = \text{tanh}(\mathbf{X}_t \mathbf{W}_{xc} + \mathbf{H}_{t-1} \mathbf{W}_{hc} + \mathbf{b}_c),
    $$
3. 记忆元 $\mathbf{C}_t \in \mathbb{R}^{n \times h}$：

    $$
    \mathbf{C}_t = \mathbf{F}_t \odot \mathbf{C}_{t-1} + \mathbf{I}_t \odot \tilde{\mathbf{C}}_t.
    $$
4. 隐状态 $\mathbf{H}_t \in \mathbb{R}^{n \times h}$

    $$
    \mathbf{H}_t = \mathbf{O}_t \odot \tanh(\mathbf{C}_t).
    $$
5. 输出 $\mathbf{Y}_t \in \mathbb{R}^{n \times q}$：

    $$
    \mathbf{Y}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.
    $$


**补充说明：**

1. 在长短期记忆网络中，有一种机制来控制输入和遗忘（或跳过）。它包含两个部分：
    - 输入门 $\mathbf{I}_t$ 控制采用多少来自 $\tilde{\mathbf{C}}_t$ 的新数据
    - 遗忘门 $\mathbf{F}_t$ 控制保留多少过去的 记忆元 $\mathbf{C}_{t-1} \in \mathbb{R}^{n \times h}$ 的内容。
2. 只要输出门接近 $1$ ，我们就能够有效地将所有记忆信息传递给预测部分， 而对于输出门接近 $0$ ，我们只保留记忆元内的所有信息，而不需要更新隐状态。




### LSTM的代码实现

**LSTMModelScratch类是LSTM模型的实现：**

```python
class LSTMModelScratch(WithHiddenStateModel):
    """从零开始实现的LSTM模型"""
    def begin_state(self, batch_size, device):
        return (torch.zeros((batch_size, self.num_hiddens), device=device),
                torch.zeros((batch_size, self.num_hiddens), device=device))

    def get_init_params(self, vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size

        def normal(shape):
            return torch.randn(size=shape, device=device)*0.01

        def three():
            return (normal((num_inputs, num_hiddens)),
                    normal((num_hiddens, num_hiddens)),
                    torch.zeros(num_hiddens, device=device))

        W_xi, W_hi, b_i = three()  # 输入门参数
        W_xf, W_hf, b_f = three()  # 遗忘门参数
        W_xo, W_ho, b_o = three()  # 输出门参数
        W_xc, W_hc, b_c = three()  # 候选记忆元参数
        # 输出层参数
        W_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        # 附加梯度
        params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
                b_c, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params
    
    def forward(self, inputs, state, params):
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, 
        W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
        (H, C) = state
        
        outputs = []
        for X in inputs:
            I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
            F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
            O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
            C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
            Y = (H @ W_hq) + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H, C)
```


**测试LSTM模型的输出和状态**：

```python
# ============================= Test the LSTM Model =============================
# test the LSTM model calculate the output and the state
num_hiddens = 256
lstm_model = LSTMModelScratch(len(vocab), num_hiddens, d2l.try_gpu())

X, Y = next(iter(train_iter))  # X = torch.arange(10).reshape((2, 5))
state = lstm_model.begin_state(X.shape[0], d2l.try_gpu())
Y_hat, new_state = lstm_model(X.to(d2l.try_gpu()), state)
print("LSTM Model Output and State: ", Y_hat.shape, len(new_state), new_state[0].shape, new_state[1].shape)
```




## GRU

GRU（Gated Recurrent Unit）是另一种常用的循环神经网络模型，与LSTM相比，GRU的结构更简单，计算速度更快，但性能稍逊于LSTM。

门控循环单元(GRU)与普通的循环神经网络(RNN)之间的关键区别在于：前者支持隐状态(hidden state)的门控。这意味着模型有专门的机制来确定应该何时更新隐状态，以及应该何时重置隐状态。这些机制是可学习的，并且能够解决了上面列出的问题。



### 基本结构

**GRU的主要结构：**

1. **重置门(reset gate)**：$\mathbf{R}_t \in \mathbb{R}^{n \times h}$
   - 重置门 允许我们控制“可能还想记住”的过去状态的数量；
2. **更新门(update gate)**：$\mathbf{Z}_t \in \mathbb{R}^{n \times h}$
   - 重置门和更新门被设计成$(0, 1)$区间中的向量，这样我们就可以进行凸组合；
   - 更新门将允许我们控制新状态中有多少个是旧状态的副本；
3. **候选隐状态(candidate hidden state)**：$\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$
4. **隐状态(hidden state)**：$\mathbf{H}_t \in \mathbb{R}^{n \times h}$


**门控循环单元具有以下两个显著特征：**

- 重置门有助于捕获序列中的短期依赖关系；
- 更新门有助于捕获序列中的长期依赖关系；




### 数学模型

<div align="center">
<img src="https://zh-v2.d2l.ai/_images/gru-3.svg" width="700px">
</div>

对于给定的时间步 $t$，假设输入是一个小批量 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$（样本个数$n$，输入维度$d$），上一个时间步的隐状态是 $\mathbf{H}_{t-1} \in \mathbb{R}^{n \times h}$（隐藏单元个数$h$）:

1. 重置门 $\mathbf{R}_t \in \mathbb{R}^{n \times h}$ 和更新门 $\mathbf{Z}_t \in \mathbb{R}^{n \times h}$：

    $$
    \begin{aligned}
    \mathbf{R}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xr} + \mathbf{H}_{t-1} \mathbf{W}_{hr} + \mathbf{b}_r),\\
    \mathbf{Z}_t = \sigma(\mathbf{X}_t \mathbf{W}_{xz} + \mathbf{H}_{t-1} \mathbf{W}_{hz} + \mathbf{b}_z),
    \end{aligned}
    $$
2. 候选隐状态 $\tilde{\mathbf{H}}_t \in \mathbb{R}^{n \times h}$：

    $$
    \tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t \mathbf{W}_{xh} + \left(\mathbf{R}_t \odot \mathbf{H}_{t-1}\right) \mathbf{W}_{hh} + \mathbf{b}_h),
    $$
3. 隐状态 $\mathbf{H}_t \in \mathbb{R}^{n \times h}$：
    $$
    \mathbf{H}_t = \mathbf{Z}_t \odot \mathbf{H}_{t-1}  + (1 - \mathbf{Z}_t) \odot \tilde{\mathbf{H}}_t,
    $$
4. 输出 $\mathbf{Y}_t \in \mathbb{R}^{n \times q}$：
    $$
    \mathbf{Y}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.
    $$


**补充说明：**

1. 重置门有助于捕获序列中的短期依赖关系；
2. 更新门有助于捕获序列中的长期依赖关系；




### GRU的代码实现

**GRUModelScratch类是GRU模型的实现：**

```python
class GRUModelScratch(WithHiddenStateModel):
    """从零开始实现的GRU模型"""

    def begin_state(self, batch_size, device):
        return (torch.zeros((batch_size, self.num_hiddens), device=device), )

    def get_init_params(self, vocab_size, num_hiddens, device):
        num_inputs = num_outputs = vocab_size

        def normal(shape):
            return torch.randn(size=shape, device=device)*0.01

        def three():
            return (normal((num_inputs, num_hiddens)),
                    normal((num_hiddens, num_hiddens)),
                    torch.zeros(num_hiddens, device=device))

        W_xz, W_hz, b_z = three()  # 更新门参数
        W_xr, W_hr, b_r = three()  # 重置门参数
        W_xh, W_hh, b_h = three()  # 候选隐状态参数
        # 输出层参数
        W_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        # 附加梯度
        params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def forward(self, inputs, state, params):
        W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
        H, = state
        outputs = []
        
        for X in inputs:
            Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
            R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
            H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
            H = Z * H + (1 - Z) * H_tilda
            Y = H @ W_hq + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H,)
```


**测试GRU模型的输出和状态**：

```python
# ============================= Test the GRU Model =============================
# test the GRU model calculate the output and the state
num_hiddens = 256
gru_model = GRUModelScratch(len(vocab), num_hiddens, d2l.try_gpu())

X, Y = next(iter(train_iter))  # X = torch.arange(10).reshape((2, 5))
state = gru_model.begin_state(X.shape[0], d2l.try_gpu())
Y_hat, new_state = gru_model(X.to(d2l.try_gpu()), state)
print("GRU Model Output and State: ", Y_hat.shape, len(new_state), new_state[0].shape)
```




## Deep RNNs

Deep RNNs(深度循环神经网络，Deep Recurrent Neural Networks)是RNN的扩展，它将多个RNN层堆叠在一起，可以捕捉更复杂的模式。


我们可以将多层循环神经网络堆叠在一起，通过对几个简单层的组合，产生了一个灵活的机制。特别是，数据可能与不同层的堆叠有关。



### 基本结构

下图描述了一个具有 $L$ 个隐藏层的深度循环神经网络，每个隐状态都连续地传递到当前层的下一个时间步和下一层的当前时间步。

<div align="center">
    <img src="https://zh-v2.d2l.ai/_images/deep-rnn.svg" width="500px">
</div>




### 数学模型

假设在时间步$t$有一个小批量的输入数据 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$（样本数：$n$，每个样本中的输入维度：$d$）。同时，将$l^\mathrm{th}$隐藏层（$l=1,\ldots,L$）的隐状态设为$\mathbf{H}_t^{(l)}  \in \mathbb{R}^{n \times h}$（隐藏单元数：$h$），输出层变量设为$\mathbf{Y}_t \in \mathbb{R}^{n \times q}$（输出数：$q$）。第$l$个隐藏层的隐状态使用激活函数$\phi_l$，则：

1. 设置：
    $$
    \mathbf{H}_t^{(0)} = \mathbf{X}_t
    $$
2. 第$l$个隐藏层的隐状态：
    $$
    \mathbf{H}_t^{(l)} = \phi_l(\mathbf{H}_t^{(l-1)} \mathbf{W}_{xh}^{(l)} + \mathbf{H}_{t-1}^{(l)} \mathbf{W}_{hh}^{(l)}  + \mathbf{b}_h^{(l)}),
    $$
3. 输出层的计算仅基于第$l$个隐藏层最终的隐状态：

    $$
    \mathbf{Y}_t = \mathbf{H}_t^{(L)} \mathbf{W}_{hq} + \mathbf{b}_q.
    $$




## Bi-RNNs

Bi-RNNs(双向循环神经网络，Bidirectional Recurrent Neural Networks)是RNN的扩展，它将两个RNN层堆叠在一起，可以捕捉更复杂的模式。




### 基本结构

*双向循环神经网络*（bidirectional RNNs）添加了反向传递信息的隐藏层，以便更灵活地处理此类信息。下图是一个具有单个隐藏层的双向循环神经网络的架构示意图：

<div align="center">
    <img src="https://zh-v2.d2l.ai/_images/birnn.svg" width="500px">
</div>




### 数学模型

对于任意时间步 $t$，给定一个小批量的输入数据 $\mathbf{X}_t \in \mathbb{R}^{n \times d}$（样本数$n$，每个示例中的输入数$d$），并且令隐藏层激活函数为$\phi$。在双向架构中，我们设该时间步的前向和反向隐状态分别为 $\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$ 和 $\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$，其中$h$是隐藏单元的数目。

1. 前向和反向隐状态的更新：

    $$
    \begin{aligned}
    \overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)}),\\
    \overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)}),
    \end{aligned}
    $$
2. 连接 $\overrightarrow{\mathbf{H}}_t$ 和 $\overleftarrow{\mathbf{H}}_t$，获得需要送入输出层的隐状态 $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$：

    $$\mathbf{H}_t = [\overrightarrow{\mathbf{H}}_t, \overleftarrow{\mathbf{H}}_t],$$

3. 输出层计算得到的输出为 $\mathbf{Y}_t \in \mathbb{R}^{n \times q}$（$q$是输出单元的数目）：

    $$\mathbf{Y}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$




### Bi-RNN的计算代价及应用

双向循环神经网络的一个关键特性是：使用来自序列两端的信息来估计输出。也就是说，我们使用来自过去和未来的观测信息来预测当前的观测。但是在对下一个词元进行预测的情况中，这样的模型并不是我们所需的。因为在预测下一个词元时，我们终究无法知道下一个词元的下文是什么，所以将不会得到很好的精度。

具体地说，在训练期间，我们能够利用过去和未来的数据来估计现在空缺的词；而在测试期间，我们只有过去的数据，因此精度将会很差。

另一个严重问题是，双向循环神经网络的计算速度非常慢。其主要原因是网络的前向传播需要在双向层中进行前向和后向递归，并且网络的反向传播还依赖于前向传播的结果。因此，梯度求解将有一个非常长的链。

双向层的使用在实践中非常少，并且仅仅应用于部分场合。例如，填充缺失的单词、词元注释（例如，用于命名实体识别）以及作为序列处理流水线中的一个步骤对序列进行编码（例如，用于机器翻译）。




## Encoder-Decoder Architecture

为了能处理输入和输出都是变长序列的场景，我们构建两个称为编码器和解码器的基本框架， 它们可以用来训练神经网络模型来学习将一个序列转换为另一个序列。




### 基本结构

一个包含两个主要组件的 *编码器-解码器* 架构：

1. 第一个组件是一个 *编码器*（encoder）：它接受一个长度可变的序列作为输入，并将其转换为具有固定形状的编码状态。
2. 第二个组件是 *解码器*（decoder）：它将固定形状的编码状态映射到长度可变的序列。

这被称为 *编码器-解码器* 架构，如下图所示。

<div align="center">
    <img src="https://zh-v2.d2l.ai/_images/encoder-decoder.svg" width="500px">
</div>




### 代码架构

**编码器：**

```python
from torch import nn


class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```


**解码器：**

```python
class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```


**编码器-解码器架构的实现：**

```python
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```




## Seq2Seq Learning

本节，我们将使用两个循环神经网络的编码器和解码器，并将其应用于*序列到序列*（sequence to sequence，seq2seq）类的学习任务。

**编码器－解码器架构的设计原则：**

1. 循环神经网络编码器使用长度可变的序列作为输入，将其转换为固定形状的隐状态。换言之，输入序列的信息被*编码*到循环神经网络编码器的隐状态中。
2. 为了连续生成输出序列的词元，独立的循环神经网络解码器是基于输入序列的编码信息和输出序列已经看见的或者生成的词元来预测下一个词元。




### 基本结构

下图演示了如何在机器翻译中使用两个循环神经网络进行序列到序列学习。

<div align="center">
    <img src="https://zh-v2.d2l.ai/_images/seq2seq.svg" width="500px">
</div>

特定的 `<eos>` 表示序列结束词元。一旦输出序列生成此词元，模型就会停止预测。在循环神经网络解码器的初始化时间步，有两个特定的设计决定：

1. 首先，特定的 `<bos>` 表示序列开始词元，它是解码器的输入序列的第一个词元。
2. 其次，使用循环神经网络编码器最终的隐状态来初始化解码器的隐状态。

编码器最终的隐状态在每一个时间步都作为解码器的输入序列的一部分。




### 数学模型

使用循环神经网络实现的“编码器－解码器”模型中的各层如下图所示：

<div align="center">
    <img src="https://zh-v2.d2l.ai/_images/seq2seq-details.svg" width="500px">
</div>




#### 编码器

从技术上讲，编码器将长度可变的输入序列转换成形状固定的上下文变量$\mathbf{c}$，并且将输入序列的信息在该上下文变量中进行编码。

考虑由一个序列组成的样本（批量大小是$1$）。假设输入序列是$x_1, \ldots, x_T$，其中$x_t$是输入文本序列中的第$t$个词元。使用一个函数$f$来描述循环神经网络的循环层所做的变换：

1. 在时间步$t$，循环神经网络将词元$x_t$的输入特征向量$\mathbf{x}_t$和$\mathbf{h} _{t-1}$（即上一时间步的隐状态）转换为$\mathbf{h}_t$（即当前步的隐状态）：

    $$
    \mathbf{h}_t = f(\mathbf{x}_t, \mathbf{h}_{t-1}),
    $$
2. 编码器通过选定的函数$q$，将所有时间步的隐状态转换为上下文变量$\mathbf{c}$：

    $$
    \mathbf{c} =  q(\mathbf{h}_1, \ldots, \mathbf{h}_T),
    $$
    例如 $q$ 函数为取最后一个隐状态的操作，则有 $\mathbf{c} = \mathbf{h}_T$




#### 解码器

编码器输出的上下文变量 $\mathbf{c}$ 对整个输入序列$x_1, \ldots, x_T$进行编码。来自训练数据集的输出序列$y_1, y_2, \ldots, y_{T'}$，对于每个时间步$t'$（与输入序列或编码器的时间步$t$不同），解码器输出$y_{t'}$的概率取决于先前的输出子序列 $y_1, \ldots, y_{t'-1}$和上下文变量$\mathbf{c}$，即$P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$。

为了在序列上模型化这种条件概率，我们可以使用另一个循环神经网络作为解码器。在输出序列上的任意时间步 $t^\prime$，循环神经网络将来自上一时间步的输出 $y_{t^\prime-1}$ 和上下文变量$\mathbf{c}$作为其输入，然后在当前时间步将它们和上一隐状态 $\mathbf{s}_{t^\prime-1}$ 转换为隐状态 $\mathbf{s}_{t^\prime}$。因此，可以使用函数 $g$ 来表示解码器的隐藏层的变换：

$$\mathbf{s}_{t^\prime} = g(y_{t^\prime-1}, \mathbf{c}, \mathbf{s}_{t^\prime-1}),$$

在获得解码器的隐状态之后，我们可以使用输出层和softmax操作来计算在时间步 $t^\prime$时输出$y_{t^\prime}$ 的条件概率分布：

$$P(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \mathbf{c}) = \text{softmax}(\mathbf{s}_{t^\prime} \mathbf{W}_{hq} + \mathbf{b}_q),$$

当实现解码器时，我们直接使用编码器最后一个时间步的隐状态来初始化解码器的隐状态。这就要求使用循环神经网络实现的编码器和解码器具有相同数量的层和隐藏单元。为了进一步包含经过编码的输入序列的信息，上下文变量在所有的时间步与解码器的输入进行拼接（concatenate）。为了预测输出词元的概率分布，在循环神经网络解码器的最后一层使用全连接层来变换隐状态。





















References:

- [[干货]深入浅出LSTM及其Python代码实现](https://zhuanlan.zhihu.com/p/104475016)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [《动手学深度学习》-8.4 循环神经网络](https://zh-v2.d2l.ai/chapter_recurrent-neural-networks/rnn.html)
- [《动手学深度学习》-9.2. 长短期记忆网络（LSTM）](https://zh-v2.d2l.ai/chapter_recurrent-modern/lstm.html)


