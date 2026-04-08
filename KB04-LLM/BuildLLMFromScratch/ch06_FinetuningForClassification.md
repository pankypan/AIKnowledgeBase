## 6.1 Different categories of finetuning

微调语言模型最常见的方式是 **instruction-finetuning** 和 **classification-finetuning**。


Instruction-finetuning 涉及使用特定指令在一组任务上训练语言模型，以提高其理解和执行自然语言 prompt 中描述的任务的能力，如图 6.2 所示。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch6/2.png" width=700px>
</div>



在 classification-finetuning 中，模型被训练来识别一组特定的类别标签，例如 "spam" 和 "not spam"。

关键点在于，经过 classification-finetuning 的模型只能预测其在训练过程中遇到的类别——例如，它可以判断某内容是 "spam" 还是 "not spam"，如图 6.3 所示，但它无法对输入文本做出其他任何表述。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch6/3.png" width=700px>
</div>





## 6.2 Preparing the dataset

我们将修改并对前几章中实现和预训练的 GPT 模型进行 classification-finetuning。首先从下载和准备数据集开始，如图 6.4 所示。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch6/4.png" width=700px>
</div>


使用由垃圾邮件和非垃圾邮件组成的数据集来对 LLM 进行分类微调:
1. 下载并解压缩数据集; (下载代码详见项目内)
2. 加载数据集并进行平衡处理; (为了简化处理，并且出于教学目的考虑，我们选择使用小规模数据集，这有助于更快地对大语言模型进行微调。因此，我们对数据集进行了下采样，使每个类别包含 747 个实例。)
3. 将标签 "ham" 和 "spam" 转换为整数类标签"0"和"1";
4. 自定义一个函数,用于把数据集随机划分为训练集、验证集、测试集;





## 6.3 Creating data loaders

数据处理关键点：
- 由于文本消息长度随机, 因此在批量化组合训练数据之前要做数据归一化, 我们有两种操作可供选择
  1. 将所有消息截断到数据集中最短消息的长度或批次长度
  2. 将所有消息填充到数据集中最长消息的长度或批次长度

- 这里我们选择操作2, 填充数据
- 并且, 我们使用 `<|endoftext|>` 作为填充标识符, 即在数据集中最长的文本消息后面添加 `<|endoftext|>` 对应的 token ID


<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch6/5.png" width=700px>
</div>

---


使用数据集作为输入，我们可以像第 2 章中那样实例化 data loader。但在这种情况下，目标表示的是类别标签而不是文本中的下一个 token。例如，选择 batch size 为 8 时，每个 batch 将由 8 个长度为 120 的训练样本及每个样本对应的类别标签组成，如图 6.7 所示。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch6/6.png" width=700px>
</div>








## 6.4 Initializing a model with pretrained weights

步骤：（代码详见项目内代码）
1. 定义模型超参数；
2. 下载并加载预训练模型权重；
3. 使用预训练模型生成文本(正常文本)；
4. 使用预训练模型生成文本(垃圾邮件分类)；






## 6.5 Adding a classification head

在本节中，我们修改预训练的大语言模型，为 classification-finetuning 做准备。为此，我们将原始输出层（将隐藏表示映射到 50,257 大小的词表）替换为一个更小的输出层，该层映射到两个类别：0（"not spam"）和 1（"spam"），如图 6.9 所示。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch6/8.png" width=700px>
</div>


接下来，我们用一个新的输出层替换 `out_head`，如图 6.9 所示，我们将对这个新的输出层进行微调。

为了使模型准备好进行 classification-finetuning，我们首先冻结模型，即将所有层设为不可训练。

---

这个新的 `model.out_head` 输出层的 `requires_grad` 属性默认设置为 `True`，这意味着它是模型中唯一会在训练过程中被更新的层。

从技术上讲，仅训练我们刚添加的输出层就已经足够了。然而，通过实验我发现，微调额外的层可以显著提升微调模型的预测性能。

此外，我们将最后一个 transformer block 和连接该 block 与输出层的最终 LayerNorm 模块配置为可训练的，如图 6.10 所示。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch6/9.png" width=700px>
</div>

---

请记住，我们的目标是微调该模型，使其返回一个类别标签，指示模型输入是 spam 还是 not spam。为实现这一目标，我们不需要微调所有 4 个输出行，而是可以专注于单个输出 token。具体来说，我们将专注于对应最后一个输出 token 的最后一行，如图 6.11 所示。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch6/10.png" width=700px>
</div>






## 6.6 Calculating the classification loss and accuracy

在本章中，我们采用相同的方法来计算模型对给定输入的 "spam" 或 "not spam" 预测，如图 6.14 所示，唯一的区别在于我们处理的是 2 维而非 50,257 维的输出。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch6/13.png" width=700px>
</div>


为了确定分类准确率，我们将基于 argmax 的预测代码应用于数据集中的所有样本，并通过定义一个 `calc_accuracy_loader` 函数来计算正确预测的比例。






## 6.7 Finetuning the model on supervised data

我们定义并使用训练函数来微调预训练的 LLM，以提高其垃圾邮件分类的准确率。如图 6.15 所示，训练循环与我们在第 5 章中使用的整体训练循环相同，唯一的区别在于我们计算分类准确率而非生成样本文本来评估模型。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch6/14.png" width=700px>
</div>






## 6.8 Using the LLM as a spam classifier

使用步骤：（代码详见项目内代码）
1. 将微调后的 GPT 模型投入实际应用;
2. `classify_review` 函数实现了类似于我们之前实现的 `SpamDataset` 的数据预处理步骤;
3. 函数返回模型预测的整数类别标签，并返回对应的类别名称;
4. 最后，让我们保存模型，以便以后如果需要重用模型时，无需重新训练。




