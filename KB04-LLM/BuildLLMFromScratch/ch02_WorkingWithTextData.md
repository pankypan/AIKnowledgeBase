## 2.1 Understanding word embeddings

将数据转换为向量格式的概念通常被称为 embedding。通过使用特定的神经网络层或其他预训练的神经网络模型，我们可以对不同的数据类型进行 embedding，例如视频、音频和文本，如 Figure 2.2 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp" width="700px">
</div>

Figure 2.2: 深度学习模型无法直接处理视频、音频和文本等原始数据格式。因此，我们使用 embedding model 将原始数据转换为密集的向量表示，使深度学习架构能够轻松理解和处理。具体来说，该图展示了将原始数据转换为三维数值向量的过程。

---

Word2Vec 通过预测目标词的上下文（或反过来）来训练神经网络架构以生成 word embeddings。Word2Vec 背后的核心思想是：出现在相似上下文中的词往往具有相似的含义。因此，当将其投影到二维 word embeddings 进行可视化时，可以看到相似的词汇聚集在一起，如 Figure 2.3 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/03.webp" width="700px">
</div>



## 2.2 Tokenizing text

本节介绍如何将输入文本拆分为单独的 token，这是为 LLM 创建 embeddings 所必需的预处理步骤。这些 token 可以是单独的词或特殊字符，包括标点符号，如 Figure 2.4 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/04.webp" width="700px">
</div>

---

如 Figure 2.5 所示，我们目前实现的 tokenization 方案将文本拆分为单独的词和标点字符。在该图所示的具体示例中，示例文本被拆分为 10 个单独的 token。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/05.webp" width="700px">
</div>



## 2.3 Converting tokens into token IDs

要将之前生成的 token 映射为 token ID，我们首先需要构建一个所谓的 vocabulary。这个 vocabulary 定义了如何将每个唯一的词和特殊字符映射到一个唯一的整数，如 Figure 2.6 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/06.webp" width="800px">
</div>

---

让我们用 Python 实现一个完整的 tokenizer 类，其中包含一个 encode 方法，用于将文本拆分为 token，并通过 vocabulary 执行字符串到整数的映射以生成 token ID。此外，我们还实现一个 decode 方法，执行反向的整数到字符串映射，将 token ID 转换回文本。

使用 SimpleTokenizerV1 Python 类，我们现在可以通过现有的 vocabulary 实例化新的 tokenizer 对象，然后用它来编码和解码文本，如 Figure 2.8 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/08.webp?123" width="800px">
</div>



## 2.4 Adding special context tokens

具体来说，我们将修改上一节中实现的 vocabulary 和 tokenizer（即 SimpleTokenizerV2），使其支持两个新的 token：`<|unk|>` 和 `<|endoftext|>`，如 Figure 2.9 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/09.webp?123" width="800px">
</div>

如 Figure 2.9 所示，我们可以修改 tokenizer，使其在遇到不在 vocabulary 中的词时使用 `<|unk|>` token。

---

此外，我们在不相关的文本之间添加一个 token。例如，当在多个独立的文档或书籍上训练类 GPT 的 LLM 时，通常会在前一个文本源之后的每个文档或书籍之前插入一个 token，如 Figure 2.10 所示。这有助于 LLM 理解，虽然这些文本源被拼接在一起用于训练，但它们实际上是不相关的。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/10.webp" width="800px">
</div>



## 2.5 Byte-pair encoding (BPE)

本节介绍一种基于 byte pair encoding (BPE) 概念的更复杂的 tokenization 方案。本节涵盖的 BPE tokenizer 被用于训练 GPT-2、GPT-3 以及 ChatGPT 最初使用的模型。

由于实现 BPE 可能相当复杂，我们将使用一个名为 [tiktoken](https://github.com/openai/tiktoken) 的现有 Python 开源库，它基于 Rust 源代码非常高效地实现了 BPE 算法。

```python
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))
```

安装完成后，我们可以按如下方式从 tiktoken 实例化 BPE tokenizer：

```python
tokenizer = tiktoken.get_encoding("gpt2")
text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

tokenizer.decode(integers)
```


如 Figure 2.11 所示，将未知词拆分为单个字符的能力确保了 tokenizer 以及使用它训练的 LLM 可以处理任何文本，即使其中包含训练数据中未出现过的词。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/11.webp" width="700px">
</div>


对 BPE 的详细讨论和实现超出了本书的范围，但简单来说，它通过迭代地将高频字符合并为子词、将高频子词合并为词来构建 vocabulary。合并操作由频率阈值决定。



## 2.6 Data sampling with a sliding window

在最终为 LLM 创建 embeddings 之前，下一步是生成训练 LLM 所需的 input-target pairs。

这些 input-target pairs 是什么样的？正如我们在第 1 章中学到的，LLM 通过预测文本中的下一个词来进行预训练，如 Figure 2.12 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/12.webp" width="700px">
</div>

---

具体来说，我们需要返回两个 tensor：一个包含 LLM 看到的文本的 input tensor，以及一个包含 LLM 需要预测的目标的 target tensor，如 Figure 2.13 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/13.webp?123" width="700px">
</div>

---

如 Figure 2.14 所示，当从输入数据集创建多个 batch 时，我们在文本上滑动一个输入窗口。如果 stride 设置为 1，我们在创建下一个 batch 时将输入窗口移动 1 个位置。如果将 stride 设置为等于输入窗口大小，则可以防止 batch 之间的重叠。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/14.webp" width="700px">
</div>



## 2.7 Creating token embeddings

为 LLM 训练准备输入文本的最后一步是将 token ID 转换为 embedding 向量，如 Figure 2.15 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/15.webp" width="700px">
</div>

除了 Figure 2.15 中概述的过程外，需要注意的是，我们使用随机值初始化这些 embedding 权重作为初始步骤。

---

换句话说，embedding layer 本质上是一个查找操作，通过 token ID 从 embedding layer 的权重矩阵中检索对应的行。

该输出矩阵中的每一行都是通过从 embedding 权重矩阵中进行查找操作获得的，如 Figure 2.16 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/16.webp?123" width="700px">
</div>

Figure 2.16: Embedding layer 执行查找操作，从 embedding layer 的权重矩阵中检索与 token ID 对应的 embedding 向量。例如，token ID 5 的 embedding 向量是 embedding layer 权重矩阵的第六行（之所以是第六行而不是第五行，是因为 Python 从 0 开始计数）。为了便于说明，我们假设 token ID 是由我们在 2.3 节中使用的小型 vocabulary 生成的。



## 2.8 Encoding word positions

前面介绍的 embedding layer 的工作方式是：相同的 token ID 始终映射到相同的向量表示，无论该 token ID 在输入序列中的位置如何，如 Figure 2.17 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/17.webp" width="700px">
</div>

---

Absolute positional embeddings 与序列中的特定位置直接关联。对于输入序列中的每个位置，都会在 token 的 embedding 上添加一个唯一的 embedding 来表示其确切位置。例如，第一个 token 将有一个特定的 positional embedding，第二个 token 有另一个不同的 embedding，依此类推，如 Figure 2.18 所示。

<div align="center">
<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/18.webp" width="700px">
</div>

---

现在我们考虑更实际和有用的 embedding 尺寸，将输入 token 编码为 256 维的向量表示。这比原始 GPT-3 模型使用的维度要小（GPT-3 的 embedding 大小为 12,288 维），但对于实验来说仍然是合理的。此外，我们假设 token ID 是由我们之前实现的 BPE tokenizer 创建的，其 vocabulary size 为 50,257：

```python
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)  # torch.Size([8, 4, 256])
```


对于 GPT 模型的 absolute embedding 方法，我们只需要创建另一个与 `token_embedding_layer` 维度相同的 embedding layer：

```python
context_length = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)  # torch.Size([4, 256])
```


如我们所见，positional embedding tensor 由四个 256 维向量组成。我们现在可以将它们直接加到 token embeddings 上，其中 PyTorch 会将 4x256 维的 `pos_embeddings` tensor 加到 8 个 batch 中每个 4x256 维的 token embedding tensor 上：

```python
input_embeddings = token_embeddings + pos_embeddings
input_embeddings.shape  # torch.Size([8, 4, 256])
```
