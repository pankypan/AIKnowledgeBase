## 7.1 Introduction to instruction finetuning

在本章中，我们专注于提升 LLM 遵循指令并生成期望响应的能力，如图 7.2 所示。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch7/2.png" width=600px>
</div>



## 7.2 Preparing a dataset for supervised instruction finetuning


Instruction finetuning，也称为 supervised instruction finetuning，是指在一个数据集上训练模型，其中输入-输出对（如我们从 JSON 文件中提取的那些）是明确提供的。有多种方法可以为 LLM 格式化这些条目。图 7.4 展示了两种不同的格式示例，通常称为 prompt styles，用于训练知名的 LLM，如 Alpaca 和 Phi-3。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch7/4.png" width=600px>
</div>

我们默认使用Alpaca风格的提示格式，这是一种较早公开并被广泛使用的指令微调提示模板。





## 7.3 Organizing data into training batches


在本节中，我们将分几个步骤处理批处理过程，包括编写自定义 collate function，如图 7.6 所示。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch7/6.png" width=600px>
</div>

---


首先，为了实现图 7.6 中所示的步骤 2.1 和 2.2，我们编写了一个 `InstructionDataset` 类，该类应用了上一节中的 `format_input` 并对数据集中的所有输入进行预 tokenize，类似于第 6 章中的 `SpamDataset`。这两个步骤在图 7.7 中有更详细的说明。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch7/7.png" width=600px>
</div>

---


这个自定义 collate function 将每个 batch 中的训练样本填充到相同长度，同时允许不同 batch 具有不同的长度，如图 7.8 所示。这种方法通过仅将序列扩展到每个 batch 中最长序列的长度（而非整个数据集的最长长度）来最小化不必要的 padding。

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch7/12.png" width=600px>
</div>

---


但请注意，我们在目标列表中保留了一个 end-of-text token（ID 50256），如图 7.12 所示。这使得 LLM 能够学习在响应指令时何时生成 end-of-text token，我们将其用作生成响应完成的指示标志。

<img src="assets/figure-7.12.png" width="700px">

---


关键小结：
- 如上述所见，这3个训练样本计算得到的损失与我们从2个样本计算得到的损失相同，可以看出交叉熵损失函数忽略了带有 -100 标签的训练样本。
- 默认情况下，PyTorch 的 `cross_entropy(..., ignore_index=-100)` 设置会忽略对应于标签 -100 的样本。
- 使用这个 -100 的 ignore_index，我们可以忽略在批次中填充训练样本到相同长度时使用的额外结束 token（填充 token）。
- 然而，我们忽略第一个结束 token（50256）也不是个好选择，因为这个 token 有助于向LLM发出响应已完成的信号。
- 除了屏蔽填充词元，实践中我们通常还会屏蔽与指令相关的目标token ID

<div align="center">
<img src="https://raw.githubusercontent.com/MLNLP-World/LLMs-from-scratch-CN/main/imgs/ch7/13.png" width=600px>
</div>






## 7.4 Creating data loaders for an instruction dataset

- 在本节中，我们使用 `InstructionDataset` 类和 `custom_collate_fn` 函数来实例化训练集、验证集和测试集数据加载器。
- 之前的 `custom_collate_fn` 函数的另一个改进之处在于，我们现在直接将数据移动到目标设备（例如GPU），而不是在主训练循环中执行。这提高了效率，因为当我们将 `custom_collate_fn` 作为数据加载器的一部分使用时，数据的移动可以在后台进行。
- 我们使用 Python 标准库中的 `functools` 模块的 `partial` 函数，创建了一个新函数，其中原始函数的 `device` 参数已预先填充。


步骤：（代码详见项目内代码）
1. 接下来，我们像之前的章节一样实例化数据加载器，唯一不同的是，我们现在为批处理过程提供了自定义的collate函数；
2. 看看输入和输出批次的维度是怎样的；
3. 观察上面的输出，所有批次的批次大小为8，但长度各不相同，正如预期的那样；
4. 通过输出 `inputs` 批次中第一个训练样本的内容，再次确认输入中包含了与 token ID 50256 对应的 `<|endoftext|>` 填充 token。
5. 通过输出，直观地检查目标中是否包含 -100 占位符 token。






## 7.5 Loading a pretrained LLM

我们没有加载1.24亿参数的最小模型，而是选择了3.55亿参数的中型版本，因为1.24亿参数的模型对于通过指令微调获得合理的结果来说过于简单。

- 在下一节开始微调模型之前，可以先看一下它在一个验证集数据上的表现。




## 7.6 Finetuning the LLM on instruction data

1. 在开始训练之前，让我们计算初始的训练集和验证集损失
2. 因为模型更大了, 我们的计算成本就比之前高了不少，下面列出了不同设备运行该模型的时间；
3. 使用Adam训练, 并定义了学习率、权重衰减等参数
4. 从模型的训练输出可以看出，模型训练得很好，训练损失和验证损失值不断下降。
5. 此外，从每个epoch结束后输出的响应文本来看，我们可以看到模型正确地执行了指令，将输入句子 `'The chef cooks the meal every day.'` 转换为被动语态 `'The meal is cooked every day by the chef.'`。
6. 最后，让我们看看训练损失和验证损失曲线。



<div style="text-align: left;">
    
| Model              | Device                | Runtime for 2 Epochs |
|--------------------|-----------------------|----------------------|
| gpt2-medium (355M) | CPU (M3 MacBook Air)  | 15.78 minutes        |
| gpt2-medium (355M) | GPU (M3 MacBook Air)  | 10.77 minutes        |
| gpt2-medium (355M) | GPU (L4)              | 1.83 minutes         |
| gpt2-medium (355M) | GPU (A100)            | 0.86 minutes         |
| gpt2-small (124M)  | CPU (M3 MacBook Air)  | 5.74 minutes         |
| gpt2-small (124M)  | GPU (M3 MacBook Air)  | 3.73 minutes         |
| gpt2-small (124M)  | GPU (L4)              | 0.69 minutes         |
| gpt2-small (124M)  | GPU (A100)            | 0.39 minutes         |

</div>




## 7.7 Extracting and saving responses

- 在本节中，我们保存测试集的响应，以便在下一节进行评估。
- 我们还保存了模型的副本，以备将来使用。
- 但首先，让我们粗略的查看一下微调后的模型生成的响应。


结论：
- 从测试集中的指令、给定的响应以及模型的响应来看，模型的表现相对较好。
- 第一个和最后一个指令的回答显然是正确的。
- 最重要的是，我们可以看到，模型评估结果不像第六章那样直接，因为在第六章中我们只需要计算正确的垃圾邮件/非垃圾邮件类别标签的百分比来获得分类准确率。
- 实际上，指令微调后的LLM（如聊天机器人）通常通过多种方法进行评估：
  - 短答案和多选基准，如MMLU（“大规模多任务语言理解测量”，https://arxiv.org/abs/2009.03300），测试模型的知识。
  - 与其他LLM的人工偏好比较，如LMSYS聊天机器人竞技场（https://arena.lmsys.org）。
  - 自动化对话基准，其中使用另一个LLM，如GPT-4，来评估响应，例如AlpacaEval（https://tatsu-lab.github.io/alpaca_eval/）。
- 在下一节中，我们将使用类似AlpacaEval的方法，使用另一个LLM来评估我们模型的响应；不过，我们将使用自己的测试集，而不是公开可用的基准数据集。
- 为此，我们将模型的响应添加到 `test_data` 字典中，并将其保存为 "instruction-data-with-response.json" 文件，以便记录，这样我们可以在需要时在单独的Python会话中加载并分析它。
- 最后保存这个模型以便日后复用。






## 7.8 Evaluating the finetuned LLM

大步骤1：
- 在本节中，我们通过使用另一个更强大的LLM来自动化微调后的LLM的响应评估。
- 具体来说，我们使用了Meta AI发布的8B参数指令微调Llama 3模型，该模型可以通过ollama本地运行（https://ollama.com）。
- 请注意，Ollama 是用于生成文本（进行模型推理）的工具，而不是用于训练或微调LLM的工具。
- 在运行以下代码之前，请访问 https://ollama.com 安装Ollama，并按照指示操作
- 通常，在我们通过命令行使用ollama之前，需要先启动ollama应用程序或在单独的终端中运行 `ollama serve`。
- 在另一个终端运行ollama应用程序或 ollama serve 后，在命令行中执行以下命令，尝试使用8B参数的Llama 3模型（该模型占用4.7GB的存储空间，第一次执行此命令时会自动下载）。

```bash
# 8B 模型
ollama run llama3
```


大步骤2：
- 现在，与我们之前使用的 ollama run 命令互动模型的另一种方式是通过其REST API，在Python中通过以下函数进行操作。
- 在运行notebook中的下一个单元格之前，请确保ollama仍在运行（前面的代码单元格应显示 "Ollama running: True"）。
- 接下来，运行以下代码单元格来查询模型。



大步骤3：
- 现在，使用我们上面定义的 query_model 函数，我们可以评估微调后的模型的响应；让我们在前面一节中查看的前三个测试集响应上试试。


大步骤4：
- 如我们所见，Llama 3 模型给出了合理的评估
- 如果模型的回答不完全正确，它会根据部分正确的内容给予相应的分数，例如“积云”这个回答。
- 请注意，之前的提示会返回详细的评估结果；我们可以调整提示，使其生成介于0到100之间的整数分数（其中100为最佳），以便计算模型的平均得分。










