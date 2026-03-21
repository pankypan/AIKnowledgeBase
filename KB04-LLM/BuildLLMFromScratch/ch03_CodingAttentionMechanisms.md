<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/02.webp" width="800px">
</div>



## 3.1 The problem with modeling long sequences

As shown in Figure 3.3, we can't simply translate a text word by word due to the grammatical structures in the source and target language.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/03.webp" width="800px">
</div>

---

In an encoder-decoder RNN, the input text is fed into the encoder, which processes it sequentially. The encoder updates its hidden state (the internal values at the hidden layers) at each step, trying to capture the entire meaning of the input sentence in the final hidden state, as illustrated in Figure 3.4. The decoder then takes this final hidden state to start generating the translated sentence, one word at a time. It also updates its hidden state at each step, which is supposed to carry the context necessary for the next-word prediction.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/04.webp" width="800px">
</div>

> Figure 3.4: Before the advent of transformer models, encoder-decoder RNNs were a popular choice for machine translation. The encoder takes a sequence of tokens from the source language as input, where a hidden state (an intermediate neural network layer) of the encoder encodes a compressed representation of the entire input sequence. Then, the decoder uses its current hidden state to begin the translation, token by token.

While we don't need to know the inner worki ngs of these encoder-decoder RNNs, the key idea here is that the encoder part processes the entire input text into a hidden state (memory cell). The decoder then takes in this hidden state to produce the output.



## 3.2 Capturing data dependencies with attention mechanisms

Hence, researchers developed the so-called Bahdanau attention mechanism for RNNs in 2014 (named after the first author of the respective paper), which modifies the encoder-decoder RNN such that the decoder can selectively access different parts of the input sequence at each decoding step as illustrated in Figure 3.5.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/05.webp" width="800px">
</div>

> Figure 3.5 Using an attention mechanism, the text-generating decoder part of the network can access all input tokens selectively. This means that some input tokens are more important than others for generating a given output token. The importance is determined by the so-called attention weights, which we will compute later.
> 
> Note that this figure shows the general idea behind attention and does not depict the exact implementation of the Bahdanau mechanism, which is an RNN method outside this book's scope.

---

Self-attention is a mechanism that allows each position in the input sequence to attend to all positions in the same sequence when computing the representation of a sequence. Self-attention is a key component of contemporary LLMs based on the transformer architecture, such as the GPT series.

This chapter focuses on coding and understanding this self-attention mechanism used in GPT-like models, as illustrated in Figure 3.6.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/06.webp" width="700px">
</div>

> Figure 3.6 Self-attention is a mechanism in transformers that is used to compute more efficient input representations by allowing each position in a sequence to interact with and weigh the importance of all other positions within the same sequence.



## 3.3 Attending to different parts of the input with self-attention

### 3.3.1 A simple self-attention mechanism without trainable weights

In this section, we implement a simplified variant of self-attention, free from any trainable weights, which is summarized in Figure 3.7

<div align="center">
<img src="./assests/fig3.7.png" width="700px">
</div>

---

Consider the following input sentence, which has already been embedded into 3-dimensional vectors as discussed in chapter 2. We choose a small embedding dimension for illustration purposes to ensure it fits on the page without line breaks:

```python
import torch


inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
)
```


**Compute step-by-step:**

1. The first step of implementing self-attention is to compute the intermediate values $\omega$, referred to as attention scores, as illustrated in Figure 3.8.
    $$\omega_{ij} = x^{(i)} \cdot (x^{(j)})^T, \quad \text{x.shape = (T, d)}$$
    
    <div align="center">
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/08.webp" width="700px">
    </div>

2. In the next step, as shown in Figure 3.9, we normalize each of the attention scores that we computed previously.
    $$\alpha_{ij} = \frac{\exp(\omega_{ij})}{\sum_{k=1}^T \exp(\omega_{ik})}$$

    <div align="center">
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/09.webp" width="700px">
    </div>

3. Now that we computed the normalized attention weights, we are ready for the final step illustrated in Figure 3.10: calculating the context vector $z^{(2)}$ by multiplying the embedded input tokens, $x^{i}$, with the corresponding attention weights and then summing the resulting vectors.
    $$z^{(i)} = \sum_{j=1}^T \alpha_{ij} x^{(j)}$$
    
    <div align="center">
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/10.webp" width="700px">
    </div>

    例如 $z^{(2)}$ 的计算(矩阵化形式)：
    $$
    z^{(2)} = W^{(2)}X = 
    \begin{bmatrix}
    \alpha_{21} &  \alpha_{22} &..,  &\alpha_{2T}
    \end{bmatrix}
    \begin{bmatrix}
    x^{(1)}\\
    x^{(2)}\\
    ...,\\
    x^{(T)}
    \end{bmatrix}
    $$

The context vector $z^{(2)}$ depicted in Figure 3.10 is calculated as a weighted sum of all input vectors.



### 3.3.2 Computing attention weights for all input tokens

In the previous section, we computed attention weights and the context vector for input 2, as shown in the highlighted row in Figure 3.11. Now, we are extending this computation to calculate attention weights and context vectors for all inputs.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/11.webp" width="700px">
</div>

---


We follow the same three steps as before, as summarized in Figure 3.12:

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/12.webp" width="700px">
</div>

设输入矩阵 $X\in\mathbb{R}^{n\times d}$：

1. First, in step 1 as illustrated in Figure 3.12, we add an additional for-loop to compute the dot products for all pairs of inputs;
   $$
   S = XX^T
   $$
   $S \in \mathbb{R}^{n\times n} \text{ is the attention scores matrix}$
2. In step 2, as illustrated in Figure 3.12, we now normalize each row so that the values in each row sum to 1;
   $$
   W = \mathrm{softmax}_{\text{row}}(S),\quad
   W_{ij}=\frac{\exp(S_{ij})}{\sum_{k=1}^{n}\exp(S_{ik})}
   $$
   $W \in \mathbb{R}^{n\times n} \text{ is the attention weights matrix}$
3. In the third and last step, we now use these attention weights to compute all context vectors via matrix multiplication.
   $$C = WX$$
   $C \in \mathbb{R}^{n\times d} \text{ is the context vectors matrix}$



## 3.4 Implementing self-attention with trainable weights

These trainable weight matrices are crucial so that the model (specifically, the attention module inside the model) can learn to produce "good" context vectors.



### 3.4.1 Computing the attention weights step by step

We will implement the self-attention mechanism step by step by introducing the three trainable weight matrices $W_q, W_k$, and $W_v$. These three matrices are used to project the embedded input tokens, $x^{(i)}$, into query, key, and value vectors as illustrated in Figure 3.14.

- $W_q$、$W_k$ 和 $W_v$, 分别是Query、Key、Value,专有名词
- 这三个矩阵用于通过矩阵乘法将嵌入的输入标记 $x^{(i)}$ 映射到查询向量、键向量和值向量：
  - 查询向量：$q^{(i)} = W_q \,x^{(i)}$  
  - 键向量：$k^{(i)} = W_k \,x^{(i)}$  
  - 值向量：$v^{(i)} = W_v \,x^{(i)}$  

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/14.webp" width="800px">
</div>

1. Let's begin by defining a few variables; Next, we initialize the three weight matrices $W_q$, $W_k$, and $W_v$ that are shown in Figure 3.14;
   $$
   x^{(2)} = \text{inputs[1]}, \quad d_{in} = \text{inputs.shape[1]}, \quad d_{out} = 2
   $$
   
   $$
   q^{(2)} = x^{(2)} W_q, \quad k^{(2)} = x^{(2)} W_k, \quad v^{(2)} = x^{(2)} W_v
   $$

   $$Q=X W_q, \quad K=X W_k, \quad V=X W_v$$
2. The second step is now to compute the attention scores, as shown in Figure 3.15;
    $$
    \omega_{i} = q^{(i)} \cdot K^T, \quad \omega_{ij} = q^{(i)} \cdot (k^{(j)})^T
    $$
    <div align="center">
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/15.webp" width="800px">
    </div>
3. The third step is now going from the attention scores to the attention weights, as illustrated in Figure 3.16;
    $$
    \alpha_{ij} = \frac{\exp(\omega_{ij})}{\sum_{k=1}^{n}\exp(\omega_{ik})}
    $$
    <div align="center">
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/16.webp" width="800px">
    </div>
4. Now, the final step is to compute the context vectors, as illustrated in Figure 3.17.
    $$
    z^{(i)} = \sum_{j=1}^T \alpha_{ij} v^{(j)}
    $$
    <div align="center">
    <img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/17.webp" width="800px">
    </div>



### 3.4.2 Implementing a compact self-attention Python class

Figure 3.18 summarizes the self-attention mechanism we just implemented.

1. Compute the queries, keys, and values:
    $$
    Q=X W_q, \quad K=X W_k, \quad V=X W_v
    $$
2. Compute the attention weight matrix:
    $$
    W = QK^T
    $$
3. Normalize the attention weight matrix:
    $$
    A = \mathrm{softmax}_{row}(W), \quad A_{ij} = \frac{\exp(W_{ij})}{\sum_{k=1}^{n}\exp(W_{ik})}
    $$
4. Compute the context vectors:
    $$
    Z = A V
    $$

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/18.webp" width="700px">
</div>

> Figure 3.18 In self-attention, we transform the input vectors in the input matrix X with the three weight matrices, $W_q$, $W_k$, and $W_v$. 
>
> Then, we compute the attention weight matrix based on the resulting queries (Q) and keys (K). Using the attention weights and values (V), we then compute the context vectors (Z). 



## 3.5 Hiding future words with causal attention

Causal attention, also known as masked attention, is a specialized form of self-attention. It restricts a model to only consider previous and current inputs in a sequence when processing any given token. This is in contrast to the standard self-attention mechanism, which allows access to the entire input sequence at once.

Consequently, when computing attention scores, the causal attention mechanism ensures that the model only factors in tokens that occur at or before the current token in the sequence.

To achieve this in GPT-like LLMs, for each token processed, we mask out the future tokens, which come after the current token in the input text, as illustrated in Figure 3.19.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/19.webp" width="700px">
</div>



### 3.5.1 Applying a causal attention mask

In this section, we implement the causal attention mask in code. We start with the procedure summarized in Figure 3.20.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/20.webp" width="700px">
</div>

---

While we could be technically done with implementing causal attention at this point, we can take advantage of a mathematical property of the softmax function and implement the computation of the masked attention weights more efficiently in fewer steps, as shown in Figure 3.21.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/21.webp" width="700px">
</div>

```python
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print(attn_weights)
```



### 3.5.2 Masking additional attention weights with dropout

Here, we will apply the dropout mask after computing the attention weights, as illustrated in Figure 3.22, because it's the more common variant in practice.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/22.webp" width="600px">
</div>



## 3.6 Extending single-head attention to multi-head attention

The term "multi-head" refers to dividing the attention mechanism into multiple "heads," each operating independently. In this context, a single causal attention module can be considered single-head attention, where there is only one set of attention weights processing the input sequentially.



### 3.6.1 Stacking multiple single-head attention layers

Figure 3.24 illustrates the structure of a multi-head attention module, which consists of multiple single-head attention modules, as previously depicted in Figure 3.18, stacked on top of each other.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/24.webp" width="700px">
</div>

---

For example, if we use this `MultiHeadAttentionWrapper` class with two attention heads (via `num_heads=2`) and `CausalAttention` output dimension `d_out=2`, this results in a 4-dimensional context vectors (`d_out * num_heads=4`), as illustrated in Figure 3.25.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/25.webp" width="700px">
</div>

> 多头注意力的核心思想是使用不同的学习到的线性投影，并行地多次运行注意力机制。这使得模型能够在不同位置同时关注来自不同表示子空间的信息。



### 3.6.2 Implementing multi-head attention with weight splits

The `MultiHeadAttention` class takes an integrated approach. It starts with a multi-head layer and then internally splits this layer into individual attention heads, as illustrated in Figure 3.26.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch03_compressed/26.webp" width="700px">
</div>


