## 5.1 Evaluating generative text models

### 5.1.1 Using GPT to generate text

Figure 5.3 illustrates a three-step text generation process using a GPT model. 

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

This section explores techniques for numerically assessing text quality generated during training by calculating a so-called text generation loss. 

Figure 5.4 illustrates the overall flow from input text to LLM-generated text using a five step procedure.

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

The model produces random text that is different from the target text because it has not been trained yet.

Part of the text evaluation process that we implement in the remainder of this section, is to measure "how far" the generated tokens are from the correct predictions (targets).

The model training aims to increase the softmax probability in the index positions corresponding to the correct target token IDs, as illustrated in Figure 5.6.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/proba-index.webp" width="800px">
</div>

---

We calculate the loss for the probability scores of the two example batches, `target_probas_1` and `target_probas_2`. The main steps are illustrated in Figure 5.7.

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

Use **Cross Entropy** to calculate the loss:

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

First, we prepare the training and validation datasets that we will use to train the LLM later in this chapter.

Next, we divide the dataset into a training and a validation set and use the data loaders from chapter 2 to prepare the batches for LLM training. This process is visualized in Figure 5.9.

Then, we calculate the cross entropy for the training and validation sets.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/batching.webp" width="800px">
</div>

First, we define two functions to calculate the loss for a batch and a loader:

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

Then, we load the dataset and create the data loaders:

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

For this, we focus on a straightforward training loop, as illustrated in Figure 5.11, to keep the code concise and readable. However, interested readers can learn about more advanced techniques, including learning rate warmup, cosine annealing, and gradient clipping.

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch05_compressed/train-steps.webp" width="800px">
</div>

Let's see this all in action by training a GPTModel instance for 10 epochs using an AdamW optimizer and the `train_model_simple` function we defined earlier.

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

The resulting training and validation loss plot is shown in Figure 5.12.

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

In this section, we will cover text generation strategies (also called decoding strategies) to generate more original text. 

Then, we will cover two techniques, **temperature scaling**, and **top-k sampling**, to improve this function.



### 5.3.1 Temperature scaling

This section introduces **temperature scaling**, a technique that adds a probabilistic selection process to the next-token generation task.

Previously, inside the `generate_text_simple` function, we always sampled the token with the highest probability as the next token using `torch.argmax`, also known as **greedy decoding**. 

To generate text with more variety, we can replace the `argmax` with a function that samples from a probability distribution 


Probability distribution:


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

We can further control the distribution and selection process via a concept called **temperature scaling**, where temperature scaling is just a fancy description for dividing the logits by a number greater than 0:

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

In top-k sampling, we can restrict the sampled tokens to the top-k most likely tokens and exclude all other tokens from the selection process by masking their probability scores, as illustrated in Figure 5.15.

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

In this section, we will modify the `generate_text_simple` function to add temperature scaling and top-k sampling.

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

Use the `generate` function to generate text:

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

Fortunately, saving a PyTorch model is relatively straightforward. The recommended way is to save a model's so-called `state_dict`, a dictionary mapping each layer to its parameters, using the `torch.save` function as follows:

```python
torch.save(model.state_dict(), "model.pth")
```

Then, after saving the model weights via the `state_dict`, we can load the model weights into a new `GPTModel` model instance as follows:

```python
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth"))
model.eval()
```

Using `torch.save`, we can save both the model and optimizer state_dict contents as follows:

```python
torch.save(
    {"model_state_dict": model.state_dict(),
     "optimizer_state_dict": optimizer.state_dict()},
    "model_and_optimizer.pth"
)
```



## 5.5 Loading pretrained weights from OpenAI

For detail about loading pretrained weights from OpenAI, please read the reference section from the Project.


