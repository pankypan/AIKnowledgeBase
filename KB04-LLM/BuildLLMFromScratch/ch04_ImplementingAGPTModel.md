## 4.1 Coding an LLM architecture

LLMs, such as GPT (which stands for Generative Pretrained Transformer), are large deep neural network architectures designed to generate new text one word (or token) at a time.

Figure 4.2 provides a top-down view of a GPT-like LLM, with its main components highlighted.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/02.webp" width="700px">
</div>

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}
```

---

Using the configuration above, we will start this chapter by implementing a GPT placeholder architecture (DummyGPTModel) in this section, as shown in Figure 4.3. This will provide us with a big-picture view of how everything fits together and what other components we need to code in the upcoming sections to assemble the full GPT model architecture.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/03.webp" width="700px">
</div>

```python
# A placeholder GPT model architecture class
import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"]) 
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5): 
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
```

---

Next, we will prepare the input data and initialize a new GPT model to illustrate its usage. Building on the figures we have seen in chapter 2, where we coded the tokenizer, Figure 4.4 provides a high-level overview of how data flows in and out of a GPT model.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/04.webp?123" width="700px">
</div>

---

```python
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)  # torch.Size([2, 4])

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)  # torch.Size([2, 4, 50257])
```

The output tensor has two rows corresponding to the two text samples. Each text sample consists of 4 tokens; each token is a 50,257-dimensional vector, which matches the size of the tokenizer's vocabulary.

The embedding has 50,257 dimensions because each of these dimensions refers to a unique token in the vocabulary. At the end of this chapter, when we implement the postprocessing code, we will convert these 50,257-dimensional vectors back into token IDs, which we can then decode into words.



## 4.2 Normalizing activations with layer normalization

Training deep neural networks with many layers can sometimes prove challenging due to issues like **vanishing** or **exploding gradients**. 

We will implement layer normalization to improve the stability and efficiency of neural network training.

The main idea behind layer normalization is to adjust the activations (outputs) of a neural network layer to have a mean of 0 and a variance of 1, also known as **unit variance**.

Figure 4.5 provides a visual overview of how layer normalization functions.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/05.webp" width="600px">
</div>

---

$x$ is a vector of shape $(1, d)$, and $\mu$ and $\sigma^2$ are the mean and variance of $x$, respectively.

1. Calculate the mean and variance of $x$:
    $$
    \mu = \frac{1}{d} \sum_{i=1}^{d} x_i, \quad
    \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
    $$

2. Normalize $x$ using the mean and variance:
    $$
    \hat{x_{i}} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
    $$

**Tips:**

- $x_i$ is the $i$-th element of $x$;
- $\hat{x_{i}}$ is the normalized element;
- $\epsilon$ is a small constant to avoid division by zero.


The `dim` parameter specifies the dimension along which the calculation of the statistic (here, mean or variance) should be performed in a tensor, as shown in Figure 4.6.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/06.webp" width="600px">
</div>



## 4.3 Implementing a feed forward network with GELU activations

Historically, the ReLU activation function has been commonly used in deep learning due to its simplicity and effectiveness across various neural network architectures. However, in LLMs, several other activation functions are employed beyond the traditional ReLU. Two notable examples are GELU (Gaussian Error Linear Unit) and SwiGLU (Swish-Gated Linear Unit).

GELU and SwiGLU are more complex and smooth activation functions incorporating Gaussian and sigmoid-gated linear units, respectively.

The GELU activation function can be implemented in several ways; the exact version is defined as $\text{GELU}(x)=x⋅\Phi(x)$, where $\Phi(x)$ is the cumulative distribution function of the standard Gaussian distribution. In practice, however, it's common to implement a computationally cheaper approximation (the original GPT-2 model was also trained with this approximation):

$$
\operatorname{GELU}(x) \approx 0.5 \cdot x \cdot\left(1+\tanh \left[\sqrt{(2 / \pi)} \cdot\left(x+0.044715 \cdot x^{3}\right]\right)\right.
$$

```python
import torch.nn as nn


class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(
            torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
```

---

Draw the GELU and ReLU activation functions using the code below, as shown in Figure 4.8.

```python
import matplotlib.pyplot as plt

gelu, relu = GELU(), nn.ReLU()

x = torch.linspace(-5, 5, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()
```

As we can see in the resulting plot in Figure 4.8, ReLU is a piecewise linear function that outputs the input directly if it is positive; otherwise, it outputs zero. GELU is a smooth, nonlinear function that approximates ReLU but with a non-zero gradient for negative values.

The smoothness of GELU, as shown in Figure 4 .8, can lead to better optimization properties during training, as it allows for more nuanced adjustments to the model's parameters.

---

Figure 4.9 shows how the embedding size is manipulated inside this small feed forward neural network when we pass it some inputs.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/09.webp?12" width="600px">
</div>


Although the input and output dimensions of this module are the same, it internally expands the embedding dimension into a higher-dimensional space through the first linear layer as illustrated in Figure 4.10. This expansion is followed by a non-linear GELU activation, and then a contraction back to the original dimension with the second linear transformation. Such a design allows for the exploration of a richer representation space.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/10.webp" width="600px">
</div>



## 4.4 Adding shortcut connections

Originally, shortcut connections were proposed for deep networks in computer vision (specifically, in residual networks) to mitigate the challenge of vanishing gradients. The vanishing gradient problem refers to the issue where gradients (which guide weight updates during training) become progressively smaller as they propagate backward through the layers, making it difficult to effectively train earlier layers, as illustrated in Figure 4.12.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/12.webp?123" width="600px">
</div>



## 4.5 Connecting attention and linear layers in a transformer block

In this section, we are implementing the transformer block, a fundamental building block of GPT and other LLM architectures. This block combines several concepts we have previously covered: multi-head attention, layer normalization, dropout, feed forward layers, and GELU activations, as illustrated in Figure 4.13. 

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/13.webp?1" width="700px">
</div>

The idea is that the self-attention mechanism in the multi-head attention block identifies and analyzes relationships between elements in the input sequence. In contrast, the feed forward network modifies the data individually at each position. This combination not only enables a more nuanced understanding and processing of the input but also enhances the model's overall capacity for handling complex data patterns.



## 4.6 Coding the GPT model

Before we assemble the GPT-2 model in code, let's look at its overall structure in Figure 4.15, which combines all the concepts we covered so far in this chapter

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/15.webp" width="700px">
</div>



## 4.7 Generating text

How a generative model like an LLM generates text one word (or token) at a time, as shown in Figure 4.16.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/16.webp" width="700px">
</div>

---

The process by which a GPT model goes from output tensors to generated text involves several steps, as illustrated in Figure 4.17. These steps include decoding the output tensors, selecting tokens based on a probability distribution, and converting these tokens into human-readable text.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/17.webp" width="800px">
</div>

```python
def generate_text_simple(model: nn.Module, idx: torch.Tensor, 
                         max_new_tokens: int, context_size: int) -> torch.Tensor:
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        
        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probs, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

---

This process of generating one token ID at a time and appending it to the context using the `generate_text_simple` function is further illustrated in Figure 4.18.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch04_compressed/18.webp" width="800px">
</div>

```python
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension (torch.Size([1, 4]))

model.eval()
out = generate_text_simple(
    model=model, idx=encoded_tensor, max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"])
print("Output:", out)  # torch.Size([1, 4 + 6 = 10])

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
```


