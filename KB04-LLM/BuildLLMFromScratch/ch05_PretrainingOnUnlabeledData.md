## 5.1 Evaluating generative text models

### 5.1.1 Using GPT to generate text

图 5.3 展示了使用 GPT 模型生成文本的三步流程。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/gpt-process.webp" width="800px">
</div>

```python
import torch

from st04_implement_gpt_model import GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256, #A
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1, #B
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()
```



### 5.1.2 Calculating the text generation loss

本节探讨通过计算所谓的 text generation loss 来数值化评估训练过程中生成文本质量的技术。

图 5.4 展示了从输入文本到 LLM 生成文本的整体流程，包含五个步骤。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/proba-to-text.webp" width="800px">
</div>

```python
inputs = torch.tensor([
    [16833, 3626, 6100],  # ["every effort moves",
    [40, 1107, 588]  # "I really like"]
])

# Matching these inputs, the `targets` contain the token IDs we aim for the model to produce:
targets = torch.tensor([
    [3626, 6100, 345 ],  # [" effort moves you",
    [107, 588, 11311]  # " really like ai"]
])

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)  # Probability of each token in vocabulary
print(probas.shape)  # (batch_size, context_size, vocab_size) torch.Size([2, 3, 50257])

# token_ids = torch.argmax(probas, dim=-1, keepdim=True)
```

---

模型生成的是与目标文本不同的随机文本，因为它尚未经过训练。

我们在本节剩余部分实现的文本评估过程的一部分，是衡量生成的 token 与正确预测（targets）之间的"距离"。

模型训练的目标是提高与正确目标 token ID 对应的索引位置上的 softmax 概率，如图 5.6 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/proba-index.webp" width="800px">
</div>

---

我们计算两个示例 batch 的概率分数的 loss，即 `target_probas_1` 和 `target_probas_2`。主要步骤如图 5.7 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/cross-entropy.webp?123" width="800px">
</div>

```python
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
avg_log_probas = torch.mean(log_probas)  # scalar value
neg_avg_log_probas = avg_log_probas * -1  # scalar value
# neg_avg_log_probas == loss
```

---

使用 **Cross Entropy** 计算 loss：

$$
H(y, \hat{y}) = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
$$

```python
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)  # scalar value
# loss == neg_avg_log_probas
```



### 5.1.3 Calculating the training and validation set losses

首先，我们准备本章后续用于训练 LLM 的 training set 和 validation set。

接下来，我们将数据集划分为 training set 和 validation set，并使用第 2 章中的 data loader 为 LLM 训练准备 batch。该过程如图 5.9 所示。

然后，我们计算 training set 和 validation set 的 cross entropy。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/batching.webp" width="800px">
</div>

首先，我们定义两个函数来计算单个 batch 和整个 loader 的 loss：

```python
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()
    return total_loss / num_batches
```

然后，我们加载数据集并创建 data loader：

```python
import os
from st02_working_with_text_data import create_dataloader_v1


file_path = os.path.join(os.path.dirname(os.getcwd()), "the-verdict.txt")
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

train_loader = create_dataloader_v1(
    train_data, 
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
with torch.no_grad(): 
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=10)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=10)
print(f"Train loss: {train_loss:.4f}")
print(f"Validation loss: {val_loss:.4f}")
```



## 5.2 Training an LLM

为此，我们聚焦于一个简洁的训练循环，如图 5.11 所示，以保持代码简洁易读。不过，感兴趣的读者可以进一步学习更高级的技术，包括 learning rate warmup、cosine annealing 和 gradient clipping。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/train-steps.webp" width="800px">
</div>

让我们通过使用 AdamW optimizer 和之前定义的 `train_model_simple` 函数，对一个 GPTModel 实例进行 10 个 epoch 的训练来看看实际效果。

```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1) #A
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=1,
    start_context="Every effort moves you", tokenizer=tokenizer
)
```

---

训练和验证的 loss 曲线如图 5.12 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/loss-plot.webp" width="800px">
</div>

```python
import matplotlib.pyplot as plt


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax2 = ax1.twiny() #A
    ax2.plot(tokens_seen, train_losses, alpha=0) #B
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()


epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
```



## 5.3 Decoding strategies to control randomness

本节我们将介绍文本生成策略（也称为 decoding strategies），以生成更具原创性的文本。

然后，我们将介绍两种改进该函数的技术：**temperature scaling** 和 **top-k sampling**。



### 5.3.1 Temperature scaling

本节介绍 **temperature scaling**，这是一种为 next-token 生成任务添加概率选择过程的技术。

之前，在 `generate_text_simple` 函数中，我们总是使用 `torch.argmax` 选择概率最高的 token 作为下一个 token，这也被称为 **greedy decoding**。

为了生成更多样化的文本，我们可以将 `argmax` 替换为从概率分布中采样的函数。


概率分布：


```python
vocab = {
    "closer": 0, "every": 1, "effort": 2, "forward": 3, "inches": 4, 
    "moves": 5, "pizza": 6, "toward": 7, "you": 8,
}
inverse_vocab = {v: k for k, v in vocab.items()}

next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)
probas = torch.softmax(next_token_logits, dim=0)


# use argmax to sample the next token
next_token_id = torch.argmax(probas).item()


# use multinomial distribution to sample the next token
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"token: freq -> {inverse_vocab[i]}: {freq}")

print_sampled_tokens(probas)
```

我们可以通过一个叫做 **temperature scaling** 的概念进一步控制分布和选择过程，其中 temperature scaling 本质上就是将 logits 除以一个大于 0 的数：

```python
"""
Temperatures greater than 1 result in more uniformly distributed token probabilities

Temperatures smaller than 1 will result in more confident (sharper or more peaky) distributions. """


def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)


temperatures = [1, 0.1, 5]
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15
fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(x + i * bar_width, scaled_probas[i],
                   bar_width, label=f'Temperature = {T}')
ax.set_ylabel('Probability')
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()
```



### 5.3.2 Top-k sampling

在 top-k sampling 中，我们可以将采样的 token 限制为概率最高的前 k 个 token，并通过将其他所有 token 的概率分数遮蔽来排除它们，如图 5.15 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/topk.webp" width="800px">
</div>

```python
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)
print("Top positions:", top_pos)

new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input=torch.tensor(float('-inf')),
    other=next_token_logits
)
print(new_logits)

topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)
```



### 5.3.3 Modifying the text generation function

本节中，我们将修改 `generate_text_simple` 函数，添加 temperature scaling 和 top-k sampling 功能。

```python
def generate(model, idx, max_new_tokens: int, context_size: int, 
             temperature: float=1.0, top_k: int=None, eos_id: int=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # top k
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Stop generating early if end-of-sequence token is encountered and eos_id is specified
        if idx_next == eos_id:
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx
```

使用 `generate` 函数生成文本：

```python
torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.5
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```



## 5.4 Loading and saving model weights in PyTorch

幸运的是，保存 PyTorch 模型相对简单。推荐的方式是使用 `torch.save` 函数保存模型的 `state_dict`（一个将每一层映射到其参数的字典），方法如下：

```python
torch.save(model.state_dict(), "model.pth")
```

然后，在通过 `state_dict` 保存模型权重之后，我们可以将模型权重加载到一个新的 `GPTModel` 模型实例中，方法如下：

```python
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

使用 `torch.save`，我们可以同时保存模型和 optimizer 的 state_dict 内容，方法如下：

```python
torch.save(
    {"model_state_dict": model.state_dict(),
     "optimizer_state_dict": optimizer.state_dict()},
    "model_and_optimizer.pth"
)
```



## 5.5 Loading pretrained weights from OpenAI

关于从 OpenAI 加载 pretrained weights 的详细信息，请阅读项目中的参考章节。

