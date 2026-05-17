# Chapter 1: Introduction, Foundations, and Career Roadmap for LLMs

## Overview

**Chapter overview:**

- 本章作为全书的导航地图，在深入讲解之前先为读者建立整体框架。它从系统层面定义了什么是 large language model，展示了现代 LLM 技术栈的构成，梳理了实用的学习路线，并总结了影响 GenAI 岗位招聘的行业趋势。本章的目标是让读者更容易理解后续章节——因为你已经明白 tokenization、embeddings、attention、retrieval、adaptation、evaluation 和 serving 为何按这个顺序出现。
- 优秀的面试候选人很少一上来就背诵架构术语。他们会先梳理工作负载，找出模型创造价值的环节，再解释将原始模型能力转化为生产产品的周边系统。这种"先建框架"的思维方式，正是本章要培养的。


**Interview Anchor**:

- **面试官真正在考察什么**：你能否将 LLM 解释为一个工程系统，而不是孤立的研究热词。
- **强答案模式**：将 LLM 定义为嵌入在更大应用栈中的预训练 next-token 模型，然后将该技术栈与 retrieval、prompting、evaluation、serving、governance 以及可衡量的产品结果联系起来。
- **常见失误**：候选人往往直接跳到模型名称或热点话题。优秀候选人会解释要完成的任务、数据路径、可靠性控制，以及灵活性、成本与风险之间的权衡。



**INTERVIEW CHEATSHEET:**

- **要传递的信号**：LLM 不是完整的产品，它是更大 retrieval、tool-use、evaluation 和交付工作流中的推理与生成引擎。
- **最佳示例**：解释为什么一个客服助手除了需要强大的 base model，还需要 prompt 设计、retrieval 质量、监控、升级规则和输出控制。
- **追问角度**：提及从 tokenization 和 attention 到 RAG、PEFT、serving、evaluation 和 governance 的路线图。
- **高级候选人的加分点**：将多模态、更小的专用模型、推理优化等当前趋势与真实的产品决策联系起来。
- **红旗警示**：将 LLM 工程视为纯粹的 prompting，而不讨论数据、质量度量或运营约束。


## Foundations: What an LLM really is

large language model 是一个在超大规模下训练以预测序列中下一个 token 的神经网络。这个简单的目标之所以强大，是因为模型在海量语料库中内化了关于语法、事实、结构、风格和任务行为的统计规律。然而在实践中，一个达到面试水准的解释应该更上一层楼：LLM 之所以有价值，不仅仅是因为它能生成文本，更是因为它可以被嵌入到工作流中，完成分类、检索、摘要、工具推理和结构化输出草稿等任务。

这就是为什么本手册的内容从 tokenization 和 embeddings 延伸到 retrieval、adaptation、prompting、evaluation 和 serving。这些层次并非为了学习方便而拼凑在一起的独立话题，它们是决定 GenAI 系统是否实用、有据可查、快速、安全且经济可持续的真实运营层次。

下方的路线图将本书转化为一系列学习层次，帮助读者理解为何早期的机制章节要先讲，以及为何后续章节聚焦于 retrieval、adaptation 和部署，而不是止步于 base model 概念。

路线图之所以重要，是因为许多候选人在还无法解释底层机制之前就急于涉猎时髦话题。更好的顺序是先建立机制层面的理解，再向上延伸到产品模式、evaluation 和部署。

<div align="center">
<img src="./assets/figure-1.1.png">
</div>


## Roadmap for LLM and GenAI roles

对大多数工程师来说，最有效的路线图是分层的，而非按时间顺序排列的。从文本基础和模型机制开始，然后学习 retrieval 如何改变 context 质量，再学习 adaptation 和 serving 如何使系统具备生产就绪能力。之后，再专攻 evaluation、agents、多模态系统、安全或特定领域 copilot 等方向。

同样的分层视角也有助于简历和面试中的自我定位。它让你能够清晰地介绍自己：你可以说你在 retrieval 和 evaluation 方面最强，或者在 serving 和优化方面，或者在 agent 工作流的产品化方面。这比泛泛声称拥有广泛专业知识而没有清晰的技术栈叙事要可信得多。

下方的趋势表旨在将本手册锚定在当前的行业方向上。它不是一份热词清单，每一行都指向一个改变团队招聘方式、项目范围界定和技术深度评估的技能领域。

将趋势表作为优先级过滤器来阅读。差异化竞争力往往不在于又一个通用模型教程，而在于你对 retrieval 质量、evaluation、推理约束以及人类在何处仍需介入的推理能力。

Table 1.1: LLM trends that most affect engineering roadmaps and interview expectations

| Trend | Why it matters | What strong candidates should be ready to discuss |
|-------|----------------|-----------------------------------------------------|
| Longer context windows | More information can fit into a prompt, but irrelevant context still hurts answers and cost. | Why retrieval, ranking, and context compression still matter even when the model supports very large windows. |
| Multimodal systems | Text-only products are no longer the default for many enterprise and consumer workflows. | How image, audio, or document inputs change evaluation, latency, and user-experience design. |
| Smaller specialized models | Many teams now balance frontier-model quality against cost, control, and deployment flexibility. | When to use a smaller task-shaped model, PEFT, or routing instead of always calling the largest model. |
| Evaluation and governance | As products scale, trust and measurement become harder than demo quality. | Offline and online evaluation, hallucination control, guardrails, escalation, and monitoring. |
| Inference optimization | Cost and latency increasingly define whether an LLM product is viable. | Quantization, batching, caching, structured outputs, and output-budget discipline. |
| Tool use and agents | Real products increasingly combine language models with APIs, databases, and workflow engines. | Planning versus execution, tool selection, state management, and human-in-the-loop controls. |


## Mind Map of This Handbook

这张思维导图刻意保持高层次，为读者提供各章节如何关联的可视化索引，使 PEFT 或 serving 等后续话题感觉像是同一系统的延伸，而非不相关的面试琐事。

注意思维导图如何从表示层移向系统层。当答案将机制层面的理解与产品行为和生产约束联系起来时，才会真正令人信服。

<div align="center">
<img src="./assets/figure-1.2.png">
</div>


## Bonus: Resume Structure for LLM and GenAI Roles

**Bonus Layer:** 一份优秀的语言模型简历应该读起来像一份工程系统文档，而不是一堆热词的堆砌。招聘人员和面试官寻找的是真实工作负载所有权的证据：evaluation、retrieval 质量、agent 编排、可靠性、安全性和可衡量的影响。

这部分内容放在这里是因为职业定位应该与本书所教授的技术栈保持一致：你构建了什么、如何衡量它，以及你承担了哪些权衡。

下表将 LLM 简历的宏观概念转化为具体的章节，将其作为证据清单来阅读，每一行都回答了招聘人员关于你的背景是否与现代 GenAI 工作清晰对应的问题。

第一张表展示了结构，第二张表更深入一层，展示了如何撰写听起来像工程证据而非泛泛参与的简历条目。第二张简历表之所以包含在内，是因为许多好项目因措辞不当而被低估。使用下面的递进示例，看看同样的工作在明确了系统设计、权衡和影响之后如何变得更有说服力。

将两张简历表都作为备考工具使用，它们强化了面试中"系统、选择、约束、指标和结果"的答题模式。

Table 1.2: A premium resume structure for LLM, RAG, and GenAI engineering roles

| Section | What it should prove | Strong content pattern |
|---------|----------------------|-------------------------|
| Headline and summary | Clear role alignment | One-line positioning such as "AI engineer building production LLM, RAG, and evaluation systems" plus 3–4 high-value specialties. |
| Core skills | Technical depth without keyword stuffing | Group by LLM stack, retrieval stack, serving stack, cloud and observability, and evaluation rather than listing tools randomly. |
| Experience bullets | Measurable ownership | Start with the system built, state the decision or optimization, then quantify relevance, latency, cost, reliability, or adoption impact. |
| Projects | Proof of initiative | Include one flagship project with architecture, evaluation loop, failure controls, and deployment shape. |
| Public signal | Market credibility | Add GitHub, technical writing, talks, or products only when they reinforce the same narrative. |


**What Strong Candidates Sound Like:**

**面试中可以说的一句话：** "LLM 产品是围绕模型构建的系统，因此最优秀的候选人能够解释从 tokenization 和 embeddings 到 retrieval、adaptation、serving、evaluation 以及每一层商业价值的完整路线图。"

Table 1.3: A practical formula for writing better experience bullets

| Category | Text |
|----------|------|
| Weak bullet | "Worked on chatbot and retrieval pipeline." |
| Better bullet | "Built a hybrid BM25 plus vector retrieval pipeline for an internal support assistant, improving grounded answer hit rate and reducing hallucination-heavy responses during evaluation." |
| Best bullet | "Designed a hybrid BM25 plus vector retrieval pipeline with reranking and citation checks for an internal support assistant, increasing grounded answer hit rate by 18% while reducing escalation volume and keeping median response time within product SLOs." |
| Why it works | It shows system design, decision quality, measurable outcome, and engineering constraint management in one compact line. |


# Chapter 2: Tokens, Tokenization, and Context Windows

## Overview

**Chapter overview:**

- 本章解释 large language model 如何将原始文本转换为可处理的单元。在模型能够分类、检索、摘要或回答问题之前，它必须先对输入进行 tokenize，将 token 映射为 ID，并将其放入有限的 context window 中。这些设计选择影响模型行为、多语言鲁棒性、内存占用、延迟和价格。
- 从词级词汇表到 subword tokenization 的转变，使现代语言模型在开放词汇任务上实用性大幅提升。BPE 和 SentencePiece 等 subword 方法通过将文本分解为可复用的单元（而非要求固定的完整词典），改善了对罕见词、专有名词、拼写变体和多语言文本的处理（Sennrich et al., 2016；Kudo & Richardson, 2018）。


**Interview Anchor:**

- **面试官真正在考察什么**：你能否将 tokenization 与价格、延迟、多语言行为、截断风险和 prompt 设计联系起来，而不是将其视为词汇表的琐碎知识。
- **强答案模式**：将 token 定义为真正的计算单元，解释为何 token 不等于单词，然后将 token 数量与 context 预算、retrieval 分块和输出规划联系起来。
- **常见失误**：候选人常说"模型读取单词"。优秀候选人会提到 subword、special token，以及输出 token 消耗同等有限窗口和预算这一事实。


**INTERVIEW CHEATSHEET:**

- **要传递的信号**：Token 数量控制成本、延迟、截断，以及有多少证据能与指令并排放置。
- **最佳示例**：解释为什么 JSON payload、源代码块或从 PDF 复制的文本消耗的 token 数量往往远超人类预期。
- **追问角度**：提及 BPE 或 SentencePiece，然后将 tokenization 直接与分块大小和 context window 预算联系起来。
- **高级候选人的加分点**：谈论预留输出预算，以及避免因分块过大造成的 retrieval 浪费。
- **红旗警示**：说"128k context"意味着 128k 个单词，或假设更长的 prompt 自动带来更好的答案。

本章开头的可视化内容聚焦于 token 流、context 预算和 prompt 机制，展示了 token 数量在模型生成答案之前很久就已成为工程约束。

下面的代码示例刻意保持简单，因为核心课题是预算管理。它展示了输出预留如何改变固定窗口内 retrieval 和工具 context 的实际可用空间。

```python
def reserve_context(total_window: int, prompt_tokens: int, output_budget: int) -> int:
    """Return how many tokens remain for retrieval and tool context."""
    remaining = total_window - prompt_tokens - output_budget
    return max(remaining, 0)


window = 128000
prompt = 1800
completion_budget = 1200
retrieval_budget = reserve_context(window, prompt, completion_budget)
print({"retrieval_budget": retrieval_budget})
```


**What Strong Candidates Sound Like:**

**面试中可以说的一句话：** "Tokenization 是人类语言转化为模型计算的地方，因此它悄然驱动着成本、延迟、retrieval 分块、多语言鲁棒性，以及最终答案是否能放入窗口。"

下图将 tokenization 章节锚定在端到端流程中，展示 tokenization 不是孤立的预处理步骤，它直接塑造了模型实际推理所用的 context。

<div>
<img src="./assets/figure-2.1">
</div>

下方的 tokenization 对比表旨在磨练权衡语言。与其死记名称，不如关注每种策略如何改变词汇行为、多语言处理和下游成本。

Table 2.1: A practical comparison of tokenization strategies

| Approach | Strength | Limitation |
|----------|----------|------------|
| Whitespace / word-level | Simple to inspect | Breaks on rare words, morphology, and many multilingual cases |
| Byte-pair encoding | Strong open-vocabulary behavior with compact vocabularies | Learned merges may be unintuitive to humans |
| SentencePiece | Language-independent, trainable from raw text, reproducible packaging | Still requires evaluation because segmentation choices affect cost and behavior |


## Q&A-01

> Q1. 什么是 token，为什么它是 LLM 中真正的计算单元？

**Answer:**

- Token 是 LLM 实际读取和预测的单元。在实践中，token 通常不是完整的单词，它们可能是整个单词、词片段、标点符号、空白模式，甚至是字符片段，具体取决于 tokenizer。模型从不直接看到原始句子，它看到的是一串 token ID。
- Token 之所以重要，是因为几乎所有工程限制都以 token 为单位表达：context 长度、成本、吞吐量、延迟、retrieval 分块大小和输出预算。当 API 说模型支持 128k context 时，意味着 128k 个 token，而非 128k 个单词。在面试中，最强的答案是将 tokenization 与建模和运营都联系起来：它是人类文本与机器计算之间的桥梁。


## Q&A-02

> Q2. 为什么 token 与单词不能简单对应？

**Answer:**

- 人类语言是混乱的。单词包含前缀、后缀、标点、缩写、表情符号、代码片段和多语言模式，这些都不符合"一个单词等于一个单元"的简洁规则。因此 tokenizer 将文本分解为对模型统计上高效的片段，而非语言学上完美的单元。
- 这就是为什么看起来很短的短语可能消耗很多 token，而看起来很长的短语可能消耗更少。这也是为什么从 PDF、源代码、JSON 复制的 prompt，或没有空格分隔单词的语言，可能会意外地膨胀。在生产中，这影响 prompt 预算和成本估算。在面试中，提及 tokenization 优化的是表示效率，而非人类可读性。


## Q&A-03

> Q3. Byte-pair encoding 如何帮助现代语言模型？

**Answer:**

- BPE 从小单元开始，反复将频繁出现的符号对合并为更大的 subword 单元。随着时间推移，"ing"、"tion"或特定领域的片段等常见模式成为可复用的词汇项。这让模型能够紧凑地表示频繁文本，同时仍能组合式地处理罕见词。
- 关键优势是开放词汇行为。模型不会在遇到未见过的词时失败，而是可以将其分解为已知片段。这一思想在神经机器翻译中颇具影响力，后来成为语言模型 pipeline 的标准，因为它在不爆炸词汇表大小的情况下改善了对罕见和未见形式的处理（Sennrich et al., 2016）。


## Q&A-04

> Q4. 什么是 SentencePiece，何时优于经典的基于空格的 tokenization？

**Answer:**

- SentencePiece 是一个 tokenizer 框架，它直接从原始文本学习 subword 单元，而不假设文本已经按单词分割。这使其对多语言和语言无关的 pipeline 特别有用，尤其是在空格不可靠或缺失的情况下。
- 其实用价值在于可复现性和可移植性。Tokenizer 规则、归一化行为和词汇表被打包成一个模型 artifact，因此训练和推理在不同系统间保持一致。在面试中，强答案是：SentencePiece 在需要对混合语言语料库、嘈杂文本或大规模训练 pipeline 进行鲁棒的端到端 tokenization 时非常有用（Kudo & Richardson, 2018）。


## Q&A-05

> Q5. 什么是 context window？

**Answer:**

- Context window 是模型在一次前向传播中能够 attend 的最大 token 数量。它包括 system prompt、用户输入、检索到的 context、工具结果、对话历史和生成输出预算。如果总量超过限制，某些内容必须被截断、摘要或丢弃。
- 在真实系统中，context window 不像记忆宫殿，更像一张工作桌：只有放在桌上的东西才能被主动使用。这就是为什么 context 管理在聊天系统、agent 循环和 RAG pipeline 中是一个一等工程问题。


## Q&A-06

> Q6. 为什么 tokenization 直接影响成本和延迟？

**Answer:**

- LLM API 通常按处理或生成的 token 计费，transformer attention 的成本随序列长度增长。更多 token 意味着更多计算、更大内存压力和更高延迟。两个对人类看起来相似的 prompt，因为 tokenize 方式不同，运行时间可能有实质性差异。
- 这就是为什么有经验的工程师在发布 prompt 之前会测量 token 数量，尤其是对于长 retrieval context、结构化输出或链式工作流。一个有用的面试观点是：token 效率不仅是财务问题，它还影响用户体验、吞吐量，以及系统在负载下是否保持稳定。


## Q&A-07

> Q7. 当输入超过模型可接受的长度时会发生什么？

**Answer:**

- 如果输入超过 context 限制，系统必须截断、滑窗、摘要、压缩或选择性检索。如果处理不当，模型可能会错过关键指令或证据，产生自信但不完整的答案。
- 主要的工程教训是：长 context 并不能消除对 retrieval 或 context 管理的需求。在生产中，你很少想盲目地将所有内容塞入 prompt。你需要排序、分块选择和内存策略，以确保最相关的信息在预算内得以保留。


## Q&A-08

> Q8. 截断、滑动窗口和摘要有什么区别？

**Answer:**

- 截断直接丢弃 token，通常从前端或后端删除。它简单但有风险，因为可能删除模型所需的关键指令或证据。滑动窗口以重叠片段处理长文本，使模型能够看到局部邻域而不消耗整个文档。
- 摘要将早期内容压缩为更短的表示。这比硬截断更能保留意图，但也引入了抽象损失。在面试中，清晰地解释权衡：截断成本最低，滑动窗口保留局部细节，摘要保留要点但牺牲精确措辞。


## Q&A-09

> Q9. 为什么 special token 对模型行为很重要？

**Answer:**

- Special token 充当结构标记。它们可能表示序列开始、序列结束、填充、分隔符边界、指令轮次、图像占位符或工具使用边界。即使用户从未看到它们，它们也塑造了模型对序列的解释方式。
- 许多微妙的 bug 来自在 fine-tuning 或推理期间对这些标记的错误处理。例如，如果角色边界的表示方式与模型预期不符，聊天格式可能会失败。强面试答案提及：tokenization 不仅仅是分割文本，还涉及编码结构。


## Q&A-10

> Q10. 工程师应如何在生产 LLM 系统中规划 token 预算？

**Answer:**

- 好的 token 预算从为最昂贵且最不可协商的项目预留空间开始：系统指令、必需工具、guardrail、输出长度和顶级检索段落。其他所有内容应根据价值竞争剩余空间。这就是为什么 retrieval 排序、对话摘要和响应长度上限很重要。
- 一个实用规则是从最大安全预算向后设计 prompt，而不是从理想化的 prompt 向前推进。在面试中，展示你的运营思维：估算平均和尾部 token 使用量，限制输出，监控溢出事件，并将 token 预算视为可靠性控制而非事后考虑。

这个 tokenizer 示例让 token 计数变得具体，将早期理论与工程师在 prompt 意外变得昂贵或溢出模型窗口时使用的精确检查方式联系起来。

```python
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "RAG systems trade token budget for grounding quality."
encoded = tokenizer(text, add_special_tokens=True)

print(encoded["input_ids"])
print("token_count =", len(encoded["input_ids"]))
```


# Chapter 3: Embeddings and Semantic Representations

## Overview

**Chapter overview:**

- Embedding 将离散文本转换为密集向量，使语义相似的项目在向量空间中彼此靠近。它们是语义搜索、聚类、重排序、推荐以及许多 retrieval-augmented generation 系统的核心。Tokenization 回答"如何分割文本"，而 embedding 回答"如何用数字表示含义"。
- 关于表示学习的实践文献表明，可以训练出有用的语义向量，使相关的句子、文档或图文对占据 embedding 空间中的邻近区域。Sentence-BERT 使句子级相似度搜索比成对 cross-encoding 效率大幅提升，而后来的多模态工作（如 CLIP）将同样的思想扩展到文本和图像（Reimers & Gurevych, 2019；Radford et al., 2021）。


**Interview Anchor:**

- **面试官真正在考察什么**：你是否理解 embedding 是为几何和任务效用优化的数值表示，而不仅仅是从 API 调用的花哨向量。
- **强答案模式**：解释 embedding 捕获什么，为什么相似性是几何的而非词汇的，以及这如何影响搜索、聚类、推荐、重排序和 evaluation。
- **常见失误**：不要暗示所有 embedding 模型可以互换。提及任务不匹配、领域漂移，以及 retrieval 质量与生成质量之间的差异。


**INTERVIEW CHEATSHEET:**

- **要传递的信号**：Embedding 将语义关系压缩到向量空间，使距离可以代表含义。
- **最佳示例**：即使措辞与源文本不同，retrieval 查询也应能检索到相关文档。
- **追问角度**：提及 cosine similarity、dense retrieval，以及为什么 embedding 质量取决于训练目标和领域适配性。
- **高级候选人的加分点**：区分第一阶段召回与第二阶段重排序和 evaluation。
- **红旗警示**：将向量搜索视为保证真实性的机制，而非概率相关性阶段。

这张 embedding 图旨在使语义搜索 pipeline 具体化。重要的是，向量空间之所以有用，是因为后续阶段（如搜索、聚类和排序）可以高效地在其上运行。

<div align="center">
<img src="./assets/figure-3.1.png">
</div>

下面的 cosine similarity 代码片段保持了 embedding 讨论的具体性，展示了即使上游模型很复杂，语义搜索最终也归结为数值比较。

```python
from math import sqrt


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    return dot / (na * nb)


query = [0.30, 0.22, 0.91]
doc_a = [0.28, 0.20, 0.89]
doc_b = [0.10, 0.77, 0.14]
print(cosine(query, doc_a), cosine(query, doc_b)
```


**What Strong Candidates Sound Like:**

**面试中可以说的一句话：** "Embedding 之所以有用，是因为它们使语义接近度可测量，从而让 retrieval、聚类和推荐系统能够超越精确关键词匹配进行扩展。"


## Q&A-01

> Q1. 什么是 embedding？

**Answer:**

- Embedding 是一个密集数值向量，以保留有用关系的方式表示 token、句子、文档、图像或其他对象。模型不是将文本视为 ID 查找，而是将其映射到一个连续空间，在那里距离和方向可以编码语义相似性。
- 简单来说，embedding 让机器能够在不仅依赖精确关键词匹配的情况下比较含义。这就是它们成为语义搜索和 retrieval 核心的原因。强面试答案解释两个层面：embedding 是学习到的数值表示，其价值来自于将语义比较转化为向量运算。


## Q&A-02

> Q2. 为什么 embedding 使语义搜索成为可能？

**Answer:**

- 语义搜索之所以有效，是因为相关项目不需要共享完全相同的词语，只要它们的 embedding 落在彼此附近即可。关于"physician salary growth"的查询仍然可以检索到关于"doctor compensation trends"的内容，因为向量编码的是相关含义而非严格的词汇重叠。
- 这使 embedding 对问答、支持搜索和长尾用户措辞特别有用。需要注意的是，dense similarity 也可能检索到概念上相邻但错误的内容，因此 embedding 搜索强大但不自动精确。这就是为什么生产系统通常会添加元数据过滤器、词汇匹配或重排序。


## Q&A-03

> Q3. Token embedding、sentence embedding 和 document embedding 有什么区别？

**Answer:**

- Token embedding 在模型输入层表示单个 token 的身份，是模型内部处理的一部分，通常不直接用于搜索。Sentence embedding 将整个句子压缩为一个为语义比较设计的向量。Document embedding 在更大范围内做同样的事，通常使用 pooling 或分块级聚合。
- 重要的面试观点是将表示与任务匹配。Token embedding 非常适合内部语言建模，但与 retrieval embedding 不同。对于搜索和聚类，你通常需要在该级别明确训练以保留语义相似性的句子或分块 embedding。


## Q&A-04

> Q4. 为什么工程师经常对 embedding 进行 L2 归一化？

**Answer:**

- 归一化将向量缩放到单位长度，使相似性主要取决于方向而非原始大小。这使 cosine similarity 和点积行为更加一致，并且通常在向量范数在样本间差异较大时提高 retrieval 稳定性。
- 在实践中，归一化也简化了索引行为和阈值设置。没有归一化，一个向量可能因为大小而非含义而主导比较。在面试中，提及归一化不是魔法，它是一个可以提高可比性的设计选择，但正确的选择取决于 embedding 模型的训练方式以及索引如何计算相似性。


## Q&A-05

> Q5. 何时应该使用 cosine similarity 而非 dot product？

**Answer:**

- **核心思想**：当你希望相似性反映语义方向而非原始向量大小时，使用 cosine similarity。Cosine 比较向量之间的角度，而 dot product 混合了角度和长度，因此高范数向量即使在含义上并非最接近的匹配，也可能胜出。
- **cosine 通常是更安全选择的情况**：当 embedding 范数在样本间变化时，当模型文档推荐基于 cosine 的 retrieval 时，或者当你希望排序行为对训练或预处理中引入的尺度差异不那么敏感时，优先选择 cosine。
- **专家注意**：如果 embedding 经过 L2 归一化，cosine similarity 和 dot product 在排序上等价，因为每个向量都有单位长度。在生产中，真正的规则是一致性：使用你的 embedding 模型、向量索引和离线 evaluation pipeline 所设计的指标。不匹配可能悄然改变 retrieval 质量，即使 embedding 本身从未改变。


## Q&A-06

> Q6. 什么是 embedding 空间中的 hubness 和 anisotropy？

**Answer:**

- Hubness 指某些向量倾向于作为太多查询的最近邻出现。Anisotropy 意味着向量空间分布不均匀，许多 embedding 聚集在相似的方向上。这两种效应共同降低 retrieval 质量，因为索引不断返回通用项目。
- 在面试中不需要过度理论化。清晰的答案是：并非所有 embedding 空间对最近邻搜索都同样健康。如果你看到宽泛或重复文档的过度检索，应调查归一化、fine-tuning 质量、负样本采样和重排序，而不是假设向量数据库有问题。


## Q&A-07

> Q7. Dense 表示和 sparse 表示有什么区别？

**Answer:**

- Dense 表示是连续向量，大多数维度携带非零值。Sparse 表示是高维信号，只有少数维度是活跃的，如词袋或 BM25 风格的词汇匹配。Dense 方法更好地捕获语义相似性；sparse 方法更好地保留精确术语证据。
- 在生产中，这不是哲学选择，通常是召回率和精确率的权衡。Dense 搜索可以找到概念相关的文本，而 sparse 搜索可以防止遗漏精确术语、名称、代码或产品 ID。这就是为什么混合 retrieval 已成为许多企业系统的默认选择。


## Q&A-08

> Q8. Bi-encoder 和 cross-encoder 有什么区别？

**Answer:**

- Bi-encoder 独立编码查询和候选文本，然后比较它们的向量。这使 retrieval 快速且可扩展，因为候选向量可以预先计算。Cross-encoder 将查询和候选一起处理，允许更丰富的交互，但推理成本高得多。
- 一个有用的心智模型是：bi-encoder 是快速拉出候选列表的图书管理员，而 cross-encoder 是仔细阅读每个候选项与查询的审阅者。在面试中，强调常见模式：使用 bi-encoder 进行召回，使用 cross-encoder 进行重排序。


## Q&A-09

> Q9. Embedding 维度如何影响系统设计？

**Answer:**

- 更高维度的 embedding 可以捕获更丰富的区别，但也会增加存储、内存带宽和索引成本。更低维度的 embedding 更便宜更快，但可能在复杂领域中损失 retrieval 保真度。
- 因此，正确的维度是系统决策，而非仅仅是模型选择。在面试中，提及完整的权衡：向量大小影响索引占用、延迟、缓存效率和重新 embedding 的迁移成本。优秀的工程师不只问"什么最准确"，还问"什么能在真实流量下扩展"。


## Q&A-10

> Q10. 在生产中使用 embedding 模型之前，如何评估它？

**Answer:**

- 在 embedding 实际支持的任务上评估它。对于 retrieval，测量 recall at k、mean reciprocal rank、normalized discounted cumulative gain 和下游答案质量。对于聚类，检查纯度或人工可解释性。对于推荐，评估邻居是否有实质意义，而不仅仅是数值上接近。最强的面试答案是：仅靠离线向量相似性是不够的。
- Embedding 应在它们将要支持的完整 pipeline 中测试。句子级基准测试很有用，但最终问题始终是 embedding 是否改善了与业务相关的 retrieval 或决策质量（Reimers & Gurevych, 2019）。


# Chapter 4: Transformer Architecture, Attention, and Positional Reasoning

## Overview

**Chapter overview:**

- Transformer 是现代 large language model 的架构骨干。其核心洞见是：序列建模可以围绕 attention 而非循环来构建，从而在仍能建模长程依赖的同时实现训练并行化。这一变革重塑了 NLP，后来也影响了多模态、视觉和音频系统（Vaswani et al., 2017）。
- 理解 transformer 意味着理解 token embedding 如何通过 self-attention、feed-forward 块、残差连接和位置信息进行混合。面试通常测试候选人能否在多个层面解释架构：直觉层面、数学层面和系统层面。


**Interview Anchor:**

- **面试官真正在考察什么**：你能否清晰地解释 attention，使团队信任你能推理模型行为、context 混合和扩展权衡？
- **强答案模式**：将 self-attention 描述为加权信息路由，然后将其与并行序列处理、长程依赖和位置编码联系起来。
- **常见失误**：避免神秘化的措辞，如"模型同时理解一切"。解释机制，然后描述工程后果。


**INTERVIEW CHEATSHEET:**

- **要传递的信号**：Attention 让每个 token 通过动态加权其他 token 来构建上下文感知表示。
- **最佳示例**：代词或否定词的相关性通常取决于几个位置之前的 token。
- **追问角度**：提及 multi-head attention、位置编码，以及为什么序列长度影响内存和计算。
- **高级候选人的加分点**：将 attention 既解释为建模概念，也解释为系统瓶颈。
- **红旗警示**：将 attention 权重与完整的可解释性混淆，或默认将其视为因果解释。


**What Strong Candidates Sound Like:**

**面试中可以说的一句话：** "Attention 之所以强大，是因为它让每个 token 计算序列的上下文敏感视图，但这种灵活性也使长 context 的服务成本高昂。"

下方的 attention 图从系统层面简化了 transformer 的工作原理，应将其理解为信息混合的流程，而非研究符号中展示的每个矩阵运算。

<div>
<img src="./assets/figure-4.1">
</div>


## Q&A-1

> Q1. 为什么 transformer 是如此重大的突破？

**Answer:**

- 在 transformer 之前，许多序列模型依赖循环，逐步处理 token。这使训练更慢，长程依赖追踪更困难。Transformer 用 attention 取代了循环，允许每个 token 直接权衡序列中其他相关 token，同时实现更多并行计算。
- 因此，这一突破既是算法上的，也是运营上的。它同时提升了建模能力和硬件利用率。在面试中，不要只回答"因为 attention"。要说架构扩展性更好，在现代加速器上训练更快，并成为当前 LLM 的基础（Vaswani et al., 2017）。


## Q&A-2

> Q2. 用简单的话解释什么是 self-attention？

**Answer:**

- Self-attention 是一种机制，让每个 token 查看同一序列中的其他 token，并决定哪些对构建其表示最重要。表示"bank"这个词的 token 可以 attend 到附近的"river"或"loan"，并相应地改变其含义。
- 一个好的心智模型是：每个 token 都在问"在更新我的理解之前，我应该参考哪些其他 token？"这就是为什么 self-attention 对歧义消解、共指和长距离依赖如此强大。在面试中，清晰胜过术语：解释 attention 构建上下文敏感的含义。


## Q&A-3

> Q3. Query、key 和 value 向量在 attention 中扮演什么角色？

**Answer:**

- Query 表示当前 token 正在寻找什么。Key 表示每个 token 作为可寻址信号提供什么。Value 表示一旦确定相关性后被混合的内容。Attention 分数来自 query 与 key 的匹配，value 的加权组合成为新的表示。
- 实际上，query-key 相似性决定谁重要，value 决定什么信息被向前传递。面试官喜欢这个问题，因为它揭示你是真正理解 attention 还是只记住了词汇。最好的答案同时解释匹配步骤和内容聚合步骤。


## Q&A-4

> Q4. 为什么 transformer 使用多个 attention head？

**Answer:**

- 多个 head 允许模型并行学习几种类型的关系。一个 head 可能专注于局部语法，另一个专注于实体引用，另一个专注于话语结构，另一个专注于位置模式。每个 head 有自己的投影空间，因此模型不被迫采用一种通用的相关性概念。
- 关键思想是专业化。Multi-head attention 在不要求一个单一 attention 模式完成所有工作的情况下增加了表示丰富性。在面试中，避免说"更多 head 总是更好"。更多 head 增加灵活性，但其价值取决于模型大小、任务和训练质量。


## Q&A-5

> Q5. 为什么 transformer 需要位置编码或位置 embedding？

**Answer:**

- Attention 本身是置换不变的。如果去掉位置信息，模型知道哪些 token 存在，但不知道它们出现的顺序。位置编码注入序列顺序，使模型能够区分"dog bites man"和"man bites dog"。
- 现代系统使用不同的位置策略，包括学习的 embedding 和 rotary position embedding。面试安全答案是：位置信号是必需的，因为 attention 本身不编码顺序。没有它们，模型将失去语言的核心结构之一（Su et al., 2021）。


## Q&A-6

> Q6. Encoder-only、decoder-only 和 encoder-decoder transformer 有什么区别？

**Answer:**

- Encoder-only 模型针对理解任务（如分类或 retrieval）进行优化，因为它们可以对输入使用双向 attention。Decoder-only 模型通过从先前 token 预测下一个 token 来自回归地生成文本。Encoder-decoder 模型将输入编码与输出生成分离，常用于翻译和序列到序列任务。
- 在面试中，好的答案将架构映射到工作负载。BERT 是 encoder 风格，GPT 风格模型是 decoder-only，T5 是 encoder-decoder。重点不仅仅是分类法，而是理解为什么 attention 模式的变化决定了模型最适合做什么（Devlin et al., 2019；Raffel et al., 2020）。


## Q&A-7

> Q7. Feed-forward 块、残差路径和 layer normalization 各自贡献什么？

**Answer:**

- Self-attention 在 token 之间混合信息，但逐位置的 feed-forward 网络在混合之后对每个 token 表示进行非线性变换。残差连接通过让每个块学习改进而非完全替换来保留梯度流并帮助稳定深层网络。Layer normalization 通过将激活保持在可管理范围内来提高训练稳定性。
- 强答案是：transformer 不仅仅是 attention，它是 attention 加上反复的稳定化和变换机制。面试官经常问这个问题，以了解你是否理解为什么架构作为一个堆栈工作，而不仅仅是一个聪明的 attention 技巧。


## Q&A-8

> Q8. 为什么 transformer 扩展性好，但在长序列上变得昂贵？

**Answer:**

- Transformer 扩展性好，因为 attention 可以在 token 之间并行计算，这有效地映射到现代硬件。但标准 self-attention 成对比较 token，因此随着序列长度增加，计算和内存快速增长。
- 这就是为什么长 context 不是免费的。工程师为此付出延迟、吞吐量和内存压力的代价。在面试中，将架构与系统联系起来：使 transformer 占主导地位的同一设计也为 context 优化、稀疏 attention 思想、批处理策略和 KV caching 创造了强烈动机。


## Q&A-9

> Q9. Causal masking 和双向 attention 有什么区别？

**Answer:**

- Causal masking 防止 token attend 到未来的 token，这对于自回归生成中的 next-token 预测至关重要。双向 attention 允许 token 使用左右两侧的 context，这对于理解任务（如 masked language modeling 或分类）很有用。
- 更深层的观点是：mask 定义了信息流。改变 attention mask 改变了模型在训练和推理期间被允许知道什么。这就是为什么架构和目标在 transformer 设计中紧密耦合。


## Q&A-10

> Q10. 工程师应该了解哪些常见的 transformer 失效模式？

**Answer:**

- 常见失效模式包括：长 context 上的 attention 扩散、位置退化、嘈杂 prompt 导致的 context 稀释、retrieval 薄弱时的幻觉，以及解码配置不当时的不稳定输出。这些都不意味着 transformer 坏了，而是意味着周边系统必须很好地管理其局限性。
- 在面试中，这是高级候选人脱颖而出的地方。不要止步于架构图，解释 transformer 行为如何与 token 预算、retrieval 质量、训练数据和服务约束相互作用。这展示了系统层面的理解，而非仅仅是模型琐事。





# Chapter 5: Pretraining Objectives, Model Families, and Classical Comparisons

## Overview

**Chapter overview:**

- 现代语言模型并非凭空出现，它们经历了一系列设计转变：从 n-gram 统计到分布式表示，从循环序列模型到 transformer，再从窄任务模型到广泛预训练的 foundation model。理解预训练目标很重要，因为它塑造了模型天然擅长的事情。BERT 风格的目标强调双向表示学习，而 GPT 风格的目标强调 next-token 生成和开放式续写（Devlin et al., 2019；Brown et al., 2020）。
- 本章还澄清了面试中经常混淆的模型家族术语。Autoregressive、masked、generative、discriminative、sequence-to-sequence 和 foundation model 等术语指的是不同的比较维度。强答案将这些维度分开，而不是将它们视为可互换的标签。


**Interview Anchor:**

- **面试官真正在考察什么**：你能否按目标和用例比较模型家族，而不仅仅是按品牌名称。
- **强答案模式**：解释预训练目标，描述它鼓励的行为类型，然后将其与下游优势、劣势和 adaptation 选项联系起来。
- **常见失误**：避免在抽象层面上将 autoregressive 和 masked 模型视为更好或更差。将它们与任务适配性联系起来。


**INTERVIEW CHEATSHEET:**

- **要传递的信号**：目标塑造了模型高效学习的内容：续写、双向 context 使用、指令遵循或迁移行为。
- **最佳示例**：比较 GPT 风格的 next-token 预测与 BERT 风格的 masked 预测，并解释为什么它们在不同场景中表现出色。
- **追问角度**：在讨论任务统一时提及 T5 风格的 text-to-text 框架。
- **高级候选人的加分点**：在一个答案中比较训练目标、推理行为和 adaptation 成本。
- **红旗警示**：当问题实际上是关于架构或训练目标时，使用公司特定的名称。


**What Strong Candidates Sound Like:**

**面试中可以说的一句话：** "从学习目标出发比较模型家族更容易，因为目标悄然决定了模型在下游使用中的高效性。"

这张模型家族表为预训练章节提供了一个紧凑的参考点，帮助读者在面试问题深入细节之前比较目标选择和架构家族。

<p align="center"><strong>Table 5.1</strong> — <em>A compact map of major model families</em></p>

| Family idea | Typical objective | Best mental model | 
|-------------|-------------------|-------------------| 
| Autoregressive LM | Predict the next token | Excellent for free-form generation and continuation | 
| Masked LM | Recover hidden tokens from surrounding context | Strong for representation learning and understanding tasks |
| Seq2Seq model | Map one sequence into another | Useful when input and output roles are clearly distinct |
| Foundation model | Broad pretraining, then adaptation | A general base model reused across many downstream tasks |


## Q&A-1

> Q1. 什么定义了语言模型，为什么称之为"大"？

**Answer:**

- 语言模型估计 token 序列的概率。简单来说，它学习哪个 token 最可能出现在下一个位置，或者哪个 token 最适合某个 context，具体取决于训练目标。之所以称为"大"，是因为现代版本使用非常大的参数量、数据集和计算预算进行训练，使其能够内化关于语言和许多下游任务的广泛统计规律。
- 强面试答案将大小与能力联系起来，但也与成本联系起来。"大"不仅意味着更多参数，还意味着更长的训练运行、更复杂的基础设施、更大的 context 管理问题，以及幻觉、分布偏移和高部署成本等新失效模式。


## Q&A-2

> Q2. Autoregressive 和 masked 模型有什么区别？

**Answer:**

- Autoregressive 模型学习在给定先前 token 的情况下预测下一个 token。它们在生成期间从左到右读取文本，天然适合续写、对话、摘要和编码辅助。Masked 模型则隐藏一些 token，并学习从左右两侧的 context 中恢复它们。这使它们在表示学习、分类和面向 retrieval 的理解任务中表现强劲。
- 解释差异最清晰的方式是将生成与表示分开。Autoregressive 目标训练模型续写序列，Masked 目标训练模型构建丰富的内部 context 表示。两者都很强大，但它们为模型准备了不同的默认优势。


## Q&A-3

> Q3. 什么是 masked language modeling，它教会模型什么？

**Answer:**

- Masked language modeling（MLM）随机隐藏一部分 token，并要求模型从周围 context 中预测它们。因为隐藏的 token 可以依赖于它之前和之后出现的词，模型学习双向上下文表示，而非纯粹的从左到右生成策略。
- 在面试中，说 MLM 有价值是因为它教授上下文理解而非仅仅是 next-token 续写。这就是为什么 BERT 风格的预训练在搜索、排序、分类和句子对任务中如此有效（Devlin et al., 2019）。


## Q&A-4

> Q4. 什么是 next sentence prediction，为什么它在历史上很重要？

**Answer:**

- Next sentence prediction（NSP）是一个预训练任务，模型判断一个句子是否自然地跟随另一个句子。在原始 BERT 公式中，它帮助模型学习句子对之间的粗粒度话语关系，这对自然语言推理和问答等任务特别有用（Devlin et al., 2019）。
- 今天，NSP 作为通用方案的重要性不如以前，但作为历史里程碑仍然重要。后来的工作表明，一些句子级任务可以在没有单独 NSP 损失的情况下学习，但面试官仍然会问到它，因为它说明了预训练目标如何塑造下游行为。


## Q&A-5

> Q5. 语言模型如何处理词汇表外的词？

**Answer:**

- 现代语言模型通常通过使用 subword tokenization 来避免硬性的词汇表外问题。它们不要求每个完整单词都存在于词汇表中，而是将不熟悉的词分解为更小的已知片段。因此，罕见的生物医学术语或新产品名称仍然可以作为熟悉 subword 单元的序列被处理。
- 实践教训是：OOV 处理从字典设计转移到了 tokenization 设计。模型可能对新术语的含义了解不多，但仍然可以摄取和操作文本，因为 tokenizer 可以将其分解为已知片段。


## Q&A-6

> Q6. 什么是 sequence-to-sequence 模型，它在哪里最有用？

**Answer:**

- Sequence-to-sequence（Seq2Seq）模型将一个序列映射到另一个序列，通常具有不同的长度和不同的表面形式。翻译、摘要和结构化转换任务是经典示例，因为它们有清晰的源序列和目标序列。
- 面试质量的答案是：Seq2Seq 是一种任务框架，而非单一架构。较旧的 Seq2Seq 系统使用带 attention 的循环网络；较新的通常使用 encoder-decoder transformer。核心思想仍然是在保留正确信息的同时将输入序列转换为目标序列。


## Q&A-7

> Q7. 为什么 transformer 取代了许多基于 RNN 的 Seq2Seq 系统？

**Answer:**

- Transformer 取代了许多循环 Seq2Seq 模型，因为 self-attention 更好地处理长程依赖，并且在训练期间允许更多并行计算。循环模型必须逐步处理 token，这减慢了训练速度，使长距离信号传播更困难。Transformer 让每个 token 在同一层中 attend 到每个其他相关 token，这同时提升了规模和性能（Vaswani et al., 2017）。
- 在面试中，将这与运营联系起来，而不仅仅是准确性。更快的并行训练使利用更大的数据集和模型成为可能，这对 foundation 规模系统的兴起至关重要。


## Q&A-8

> Q8. Foundation model 和特定任务模型有什么区别？

**Answer:**

- Foundation model 在大型多样化语料库上广泛预训练，以便后来可以适应许多任务。特定任务模型通常针对更窄的工作（如情感分类、实体提取或特定领域 retrieval）进行训练或 fine-tuning。权衡是广度与专业化。
- 强答案是：foundation model 将工作从反复的逐任务训练转移到 adaptation、prompting、retrieval 或轻量级调整。它们之所以强大，正是因为一个 base model 可以支持许多产品，但这种广度也在控制、安全和成本方面带来挑战。


## Q&A-9

> Q9. Generative 和 discriminative 模型有什么区别？

**Answer:**

- Generative 模型学习建模或近似数据本身的生成方式，这允许它们生成新样本，如文本续写。Discriminative 模型专注于将输入映射到标签或决策，如预测评论是正面还是负面。在实践中，界限并不总是绝对的，因为强大的 generative 模型通常可以通过 prompting 执行 discriminative 任务。
- 清晰的面试答案是：generative 模型通常更灵活，而 discriminative 模型对于窄任务通常更高效且更容易校准。哪个更好取决于产品是否需要开放式生成或严格控制的预测。


## Q&A-10

> Q10. LLM 与传统统计语言模型有什么不同？

**Answer:**

- 传统统计语言模型（如 n-gram 模型）从局部 token 计数估计概率，通常依赖短的固定历史。Large language model 则学习分布式表示，使用可以捕获更长更丰富 context 的深层架构。这让它们能够超越记忆的计数进行泛化，并跨许多任务迁移。
- 有用的面试框架是：经典语言模型主要是带平滑的查找表，而现代 LLM 是表示学习器。经典系统仍然可解释且廉价，但无法匹配基于 transformer 的 LLM 的上下文灵活性、推理行为和迁移学习能力。


# Chapter 6: Classification with Large Language Models

## Overview

**Chapter overview:**

- 尽管许多团队最初将 LLM 视为聊天系统，但它们也是强大的分类引擎。它们可以通过 prompting 直接分配标签，为审计生成理由，并快速适应新的分类体系。尽管如此，并非每个分类工作负载都应该交给通用生成器。某些工作负载仍然倾向于使用更小的 discriminative 模型或混合 pipeline。
- 强工程答案比较各种方法，而不是将 LLM 分类视为通用升级。正确的选择取决于类别复杂性、数据量、解释需求、延迟目标，以及标签空间是否频繁变化。


**Interview Anchor:**

- **面试官真正在考察什么**：你能否以清晰的业务理由在 prompting、zero-shot 分类、few-shot 分类和专用分类器之间做出选择？
- **强答案模式**：从标签稳定性和成本容忍度出发，然后解释何时 generative 模型足够，何时更小的监督模型是更好的生产选择。
- **常见失误**：候选人在使用 generative 输出进行分类时经常忘记校准、类别不平衡和 schema 强制执行。


**INTERVIEW CHEATSHEET:**

- **要传递的信号**：分类设计取决于标签清晰度、数量、漂移、可解释性和每次预测的价格。
- **最佳示例**：当标签经常变化时使用 prompted LLM，但当数量巨大且标签集稳定时使用紧凑分类器。
- **追问角度**：提及置信度阈值、结构化输出和对模糊类别的人工审查。
- **高级候选人的加分点**：区分产品实验与生产吞吐量经济学。
- **红旗警示**：假设最强的 generative 模型总是最好的分类器。

下方的分类表将模型选择转化为运营标准，按行阅读作为标签稳定性、迭代速度和成本概况的决策工具。

<p align="center"><strong>Table 6.1</strong> — <em>Choosing the right classification strategy in practice</em></p>

| Approach | Best when | Operational note |
|----------|-----------|------------------|
| Prompted LLM | Labels are evolving or nuanced | Easy to iterate, but cost and consistency need control. |
| Few-shot LLM | Edge cases matter and examples improve framing | Helpful for pilot phases and policy-heavy tasks. |
| Fine-tuned classifier | Labels are stable and volume is high | Better throughput and cost once the taxonomy settles. |
| Hybrid approach | You need automation plus human escalation | Useful when uncertain cases must be triaged safely. |


**What Strong Candidates Sound Like:**

**面试中可以说的一句话：** "真正的分类决策不仅仅是准确性，而是产品能够容忍多少标签漂移、歧义、规模和治理。"


## Q&A-1

> Q1. Generative LLM 如何执行分类？

**Answer:**

- Generative LLM 可以通过被 prompt 将输入映射到定义集合中的一个标签来执行分类。它不是学习专用的分类器头，而是使用其指令遵循和语言理解能力来生成目标类别，通常还附带理由或结构化输出。
- 这在类别用自然语言描述、输入混乱或样本稀少时特别有效。权衡是：除非仔细约束输出，否则 generative 分类可能比传统分类器更慢且更不稳定。


## Q&A-2

> Q2. 何时应该使用 prompting 而非 fine-tuning 进行分类？

**Answer:**

- 当分类体系经常变化、标注数据有限且需要快速行动时，使用 prompting。当解释质量很重要时，prompting 也很有吸引力，因为同一系统可以在一次传递中分类并证明其决策。
- 当标签稳定、数量大、延迟重要且需要更严格的一致性时，使用 fine-tuning。在面试中，强调 prompting 换取灵活性，而 fine-tuning 换取专业化。两者都不是天然优越的，更好的选择取决于任务的运营概况。


## Q&A-3

> Q3. Zero-shot 和 few-shot 分类有什么区别？

**Answer:**

- Zero-shot 分类只给模型标签定义或指令。Few-shot 分类还提供少量示例，展示输入应如何映射到类别。Few-shot 示例帮助模型更可靠地推断边界、边缘情况和格式期望。
- 好的面试答案指出：当标签微妙、重叠或特定于组织时，few-shot 示例特别有帮助。它们将 prompt 变成一个即时的小型训练信号。GPT-3 推广了这种 in-context learning 风格，精心选择的示例可以显著提升性能（Brown et al., 2020）。


## Q&A-4

> Q4. 如何为 LLM 分类器设计标签分类体系？

**Answer:**

- 标签分类体系应该相互可理解、在运营上有用，并尽可能不重叠。标签应该用清晰的边界、包含规则、排除规则和示例来定义。如果标签过于抽象或语义上纠缠，模型将反映这种歧义。
- 强生产答案是将分类体系设计视为产品设计，而非仅仅是建模。许多分类失败来自不清晰的类别定义，而非弱模型。如果人类无法一致地分类，LLM 不会为你修复本体论。


## Q&A-5

> Q5. 如何在基于 LLM 的分类中处理类别不平衡？

**Answer:**

- 类别不平衡可以通过更好的示例、有针对性的评估集、成本敏感的审查策略，或使用平衡或重新加权数据进行 fine-tuning 来解决。仅靠 prompt 的系统可能过度预测宽泛的多数类，除非 prompt 明确描述少数类情况和边缘条件。
- 在面试中，提及不平衡既是数据问题也是决策策略问题。在欺诈、安全或医疗分诊中，你可能比原始整体准确性更关心少数类召回率。正确的评估指标应该反映这一优先级。


## Q&A-6

> Q6. Multi-label 分类与 single-label 分类有什么不同？

**Answer:**

- 在 single-label 分类中，必须选择恰好一个类别。在 multi-label 分类中，多个标签可能同时适用。因此 prompt、schema 和评估策略必须改变。系统不是选择一个最佳标签，而是必须决定哪些标签超过包含阈值。
- 实践挑战是校准。Multi-label 输出需要更强的阈值设置、验证和审计逻辑，因为模型可能标注不足或过度标注。在面试中，强调 multi-label 工作不仅仅是 single-label 分类的小扩展，它改变了决策结构。


## Q&A-7

> Q7. 对于用 LLM 构建的分类系统，哪些指标最重要？

**Answer:**

- 准确率是起点，但精确率、召回率、F1、混淆矩阵和校准通常更具信息量。在不平衡设置中，macro-F1 或每类召回率可能远比总体准确率重要。对于人工审查工作流，你可能还关心弃权率和审查员推翻率。
- 高级候选人通过将指标与业务风险联系起来而脱颖而出。如果错误的假阴性代价高昂，优化召回率。如果假阳性触发痛苦的人工审查，优化精确率。最好的指标是与犯错成本对齐的那个。


## Q&A-8

> Q8. 如何估计 LLM 分类器的置信度？

**Answer:**

- 置信度可以通过约束标签概率、自一致性检查、辅助模型、校准集或跨 prompt 变体的一致性来估计。模型的原始口头置信度声明不够可靠，不能作为唯一的置信度信号。
- 好的面试答案是：尽可能从外部测量置信度。生产系统通常结合模型分数、retrieval 证据、schema 有效性和历史错误模式来决定何时自动路由与升级到人工审查。


## Q&A-9

> Q9. 分类 pipeline 何时应该包含人工介入？

**Answer:**

- 当决策影响重大、模糊、新颖或对合规敏感时，人工审查是合适的。当模型置信度低、证据冲突或类别经常混淆时也很有用。来自这些案例的人工反馈可以成为最有价值的训练和评估数据。
- 在面试中，将人工审查视为精确工具而非系统弱点的标志。成熟的设计自动路由简单案例，并将稀缺的审查员注意力保留给能创造最大风险降低的案例。


## Q&A-10

> Q10. LLM 分类系统中常见的生产失效模式是什么？

**Answer:**

- 常见失效包括：分类体系变更后的标签漂移、prompt 脆弱性、隐藏的格式错误、对少数类情况的处理不当，以及对模糊输入的虚假置信。另一个常见问题是当上游 retrieval 或预处理改变分类器看到的输入时的静默退化。
- 最强的答案是系统层面的：分类质量取决于 prompt、数据定义、评估集、路由策略和审查循环。如果你只监控单一准确率数字，你将错过系统成功或失败的真正原因。


# Chapter 7: Topic Modeling, Clustering, and Theme Discovery at Scale

## Overview

**Chapter overview:**

- 主题发现回答的问题与分类不同。分类将数据映射到已知的桶中；主题建模和聚类帮助揭示你事先未定义的模式。现代 LLM 系统通常结合 embedding、聚类和摘要来发现评论、工单、研究论文或支持聊天中的主题。
- 最好的面试答案同时解释问题的统计和产品两个方面：发现聚类只是第一步，更难的部分是命名它们、验证它们、监控漂移，并使它们对下游团队有用。


**Interview Anchor:**

- **面试官真正在考察什么**：你能否从原始文本移动到可解释的主题，而不将无监督发现与真实标签混淆。
- **强答案模式**：解释从表示到分组再到解释的 pipeline，然后描述人类如何验证聚类是否真正有用。
- **常见失误**：不要过度推销聚类名称为事实。主题发现是探索性的，严重依赖表示质量和评估设计。


**INTERVIEW CHEATSHEET:**

- **要传递的信号**：主题发现是表示加评估问题，而非仅仅是聚类算法选择。
- **最佳示例**：客户支持工单通常需要基于 embedding 的分组，然后人类才能命名主要主题。
- **追问角度**：提及降维、聚类可解释性，以及跨重新运行或新数据的稳定性。
- **高级候选人的加分点**：将探索性洞察生成与生产分类体系分配分开。
- **红旗警示**：将无监督聚类标签视为客观真理。

主题发现图帮助将无监督分析与生产工作流联系起来，展示主题建模在标签、验证和迭代围绕聚类步骤时最为强大。

<div align="center">
<img src="./assets/figure-7.1.png">
</div>


**What Strong Candidates Sound Like:**

**面试中可以说的一句话：** "好的主题发现 pipeline 将聚类视为必须被解释和验证的假设，而非来自算法的自动真理。"


## Q&A-1

> Q1. 主题建模与分类有什么不同？

**Answer:**

- 分类是有监督的，从预定义的标签集开始。主题建模通常是探索性的，试图从数据本身发现潜在主题。目标不是将每个项目强制放入现有分类体系，而是发现后来可以为其提供信息的结构。
- 在实践中，团队通常在构建正式分类体系之前使用主题发现。它帮助揭示反复出现的问题、隐藏的子群体和真实用户使用的语言。在面试中，说主题建模是关于模式发现，而分类是关于决策分配。


## Q&A-2

> Q2. 为什么基于 embedding 的聚类方法在主题发现中变得流行？

**Answer:**

- 基于 embedding 的方法在语义向量空间中表示每个文本单元，因此即使不共享精确关键词，聚类也可以将概念相关的项目分组。这对于客户反馈、支持日志和研究语料库特别有用，因为人们用许多不同的方式描述同一问题。
- 吸引力是实用的：embedding 提供更好的语义分组，LLM 然后可以总结或命名发现的聚类。这种组合通常比单独的传统主题词列表对产品团队更有用。


## Q&A-3

> Q3. 大规模主题发现的实用 pipeline 是什么？

**Answer:**

- 常见 pipeline 是：清理文本，选择分析单元，嵌入数据，可选地降维，聚类向量，提取代表性示例，最后使用 LLM 或人工审查者标记聚类。最后几步很重要，因为原始聚类不能自我解释。
- 在面试中，注意可扩展性取决于批处理、近似索引、增量更新和审查的采样策略。主题建模系统必须像数据产品一样设计，而不仅仅是笔记本实验。


## Q&A-4

> Q4. 为什么工程师经常在聚类之前降维？

**Answer:**

- 高维 embedding 空间对某些聚类算法来说可能嘈杂且难以清晰分区。降维可以揭示局部结构，去除空间噪声，使聚类在视觉上或算法上更容易分离。
- 权衡是：如果不小心应用，降维可能扭曲距离。强答案是：降维是工具，而非默认规则。当它改善聚类结构或可解释性时使用它，然后用代表性示例验证结果，而不是仅仅信任图表。


## Q&A-5

> Q5. 如何为主题发现选择聚类算法？

**Answer:**

- 选择取决于你期望数据的样子。K-means 假设大致球形的聚类并需要选择 k。基于密度的方法可以捕获不规则形状并分离噪声点，这对真实文本数据通常很有用。层次方法在你想要从粗到细探索时很有帮助。
- 在面试中，展示判断力而非品牌忠诚度。正确的算法取决于数据分布、规模和可解释性需求。没有任何聚类方法可以拯救弱 embedding 或定义不清的分析单元。


## Q&A-6

> Q6. 如何命名聚类，使业务团队能够实际使用它们？

**Answer:**

- 有用的聚类标签应该总结主题，而不仅仅是重复最频繁的 token。好的标签通常来自顶级术语、代表性示例以及看到足够证据来准确描述聚类的 LLM 或人工摘要者的组合。
- 最好的标签是可操作的。"结账时的账单摩擦"比"支付问题词"更有用。在面试中，提及聚类命名与其说是建模问题，不如说是人因工程问题。


## Q&A-7

> Q7. 如何处理随时间演变的主题？

**Answer:**

- 随着产品变化、事件发生和新语言进入语料库，主题会漂移。因此生产系统应该支持定期重新 embedding、增量聚类或时间切片分析，使团队能够看到主题是否在增长、缩小、分裂或合并。
- 这就是监控重要的地方。主题发现不是一次性报告。在面试中，展示你将主题演变理解为时间分析问题，而非仅仅是静态 NLP 任务。


## Q&A-8

> Q8. 如何评估发现的主题是否好？

**Answer:**

- 好的主题内部连贯、彼此不同，并且对决策者有用。自动测量可以帮助，但对代表性示例的人工检查仍然至关重要。如果一个聚类包含语义混合的示例，即使指标看起来可以接受，标签也可能具有误导性。
- 强答案是：评估应该结合统计连贯性与分析师有用性。问题不仅仅是"聚类是否存在"，而是"产品、运营或研究团队能否根据它们采取行动"。


## Q&A-9

> Q9. LLM 如何改善主题建模工作流？

**Answer:**

- LLM 在聚类之后特别有用。它们可以标记主题、总结代表性示例、比较相邻聚类，并从大型语料库生成人类可读的洞察。一旦发现了潜在主题，它们还可以帮助引导分类体系。
- 需要注意的是：即使底层聚类混乱，LLM 生成的摘要也可能听起来清晰。在面试中，说 LLM 改善解释和报告，但聚类质量仍然必须针对原始示例进行验证。


## Q&A-10

> Q10. 团队在大规模运行主题建模时常见的错误是什么？

**Answer:**

- 常见错误包括：使用错误的分析单元、聚类嘈杂的样板文本、过度解释弱可视化，以及将自动生成的标签视为真理。另一个错误是忽略时间漂移，假设相同的聚类无限期保持稳定。
- 最强的面试答案是强调验证。主题发现应该被视为迭代的意义建构。目标是你可以信任的洞察，而非仅仅是令人印象深刻的图表。


# Chapter 8: Retrieval Foundations for Large Language Model Systems

## Overview

**Chapter overview:**

- Retrieval 是静态模型知识与新鲜、特定领域信息之间的核心桥梁。Retrieval-augmented generation 引入了一种实用方式，将存储在模型权重中的参数记忆与存储在索引、文档和知识库中的非参数记忆结合起来（Lewis et al., 2020）。
- 本章聚焦于找到正确证据的机制：词汇 retrieval、dense retrieval、混合搜索、分块、排序、过滤和评估。优秀候选人理解：好的 LLM 答案通常始于好的 retrieval 问题。


**Interview Anchor:**

- **面试官真正在考察什么**：你能否将 retrieval 解释为具有召回率、精确率、分块、重排序、元数据和评估的信息系统，而不仅仅是向量数据库选择。
- **强答案模式**：从 retrieval 存在的原因开始，然后逐步讲解索引、分块、召回、重排序、元数据过滤器和离线评估。
- **常见失误**：候选人通常只谈向量存储，而忘记分块、文档卫生和重排序主导质量。


**INTERVIEW CHEATSHEET:**

- **要传递的信号**：Retrieval 质量取决于表示、分块、过滤、排序、新鲜度和评估。
- **最佳示例**：同一个模型，根据知识库的分块和排序方式，可能看起来非常出色或非常糟糕。
- **追问角度**：提及第一阶段召回与第二阶段重排序，以及文档级元数据过滤器。
- **高级候选人的加分点**：将 retrieval 作为具有自己指标和回归测试的可测量子系统来讨论。
- **红旗警示**：说"我们使用了向量"，好像这本身就能解释有据可查的质量。

Retrieval 评分卡的包含使讨论保持在可测量的系统行为上，将章节从模糊的相关性讨论转移到你实际可以评估的维度。

Table 8.1: A compact retrieval scorecard to discuss in interviews

| Metric | What it checks | Why it matters |
|--------|----------------|----------------|
| Recall at k | Relevant evidence appears in the candidate set | Low recall means the generator never sees the right facts. |
| Precision at k | Returned context is mostly useful | High noise wastes context window and increases hallucination risk. |
| MRR or nDCG | Ranking quality among retrieved chunks | Strong reranking improves answerability without reindexing everything. |
| Freshness checks | Recent documents are retrievable when needed | Prevents stale answers in policy and operational domains. |

这个分块辅助函数展示了为什么重叠是 retrieval 设计选择而非格式便利。实现刻意保持简小，使读者能够专注于召回率权衡。

```python
def chunk_text(tokens, chunk_size=400, overlap=60):
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokens[start:end])
        if end == len(tokens):
            break
        
        start = end - overlap
    
    return chunks
```


**What Strong Candidates Sound Like:**

**面试中可以说的一句话：** "Retrieval 是有据可查的 LLM 质量通常赢得或失去的地方，因为生成器只能推理我们成功浮现和排序的证据。"

下方的 retrieval pipeline 图将分块、索引、排序和生成链接到一条有据可查的路径，使解释召回问题和幻觉问题从哪里开始分歧变得更容易。

<div align="center">
<img src="./assets/figure-8.1.png">
</div>

这张 retrieval 质量表将注意力集中在系统通常失败的地方，行被设计为直接映射到关于分块、排序、元数据和信任的调试对话。

Table 8.2: Where retrieval quality is won or lost

| Component | Main question | Typical failure |
|-----------|----------------|-----------------|
| Chunking | What unit should be retrieved? | Chunks too broad or too thin |
| Embeddings / lexical search | Can the system find likely evidence? | Semantic misses or exact-match misses |
| Metadata filters | Is the search happening in the right slice? | Wrong tenant, wrong date, wrong scope |
| Reranking | Are the best passages near the top? | Useful evidence buried too low |
| Prompt assembly | Does the model see enough clean support? | Context noise overwhelms the answer |


## Q&A-1

> Q1. 什么是 retrieval-augmented generation（RAG）？

**Answer:**

- RAG 是一种模式，系统首先检索相关的外部信息，然后将该信息作为生成的 context 提供给语言模型。目标是改善事实依据、支持引用，并使知识更新无需重新训练 base model 即可实现。
- 核心洞见是：并非所有知识都应该存在于模型权重中。外部 retrieval 给系统提供了更新鲜、更可审计的证据。在面试中，定义目的和好处：RAG 改善可控性与改善准确性同样重要（Lewis et al., 2020）。


## Q&A-2

> Q2. 词汇 retrieval 和 dense retrieval 有什么区别？

**Answer:**

- 词汇 retrieval 匹配明确的术语或短语，因此在措辞重叠重要时表现强劲。Dense retrieval 使用 embedding，因此即使查询和文档使用不同的表面形式，也可以检索语义相关的内容。
- 权衡很直接：词汇 retrieval 保护精确匹配，而 dense retrieval 改善语义召回。在企业系统中，你通常两者都需要，因为用户以概念方式提问，而文档以运营方式编写。


## Q&A-3

> Q3. 为什么混合 retrieval 通常比只使用一种方法更好？

**Answer:**

- 混合 retrieval 结合词汇和 dense 信号，使系统能够同时受益于精确术语和语义相似性。这有助于在同一 pipeline 中处理缩写、错误代码、产品名称、法律条款和用户释义。
- 强面试答案是：混合搜索减少了每种方法的盲点。单独的 dense retrieval 可能遗漏罕见标识符；单独的词汇 retrieval 可能遗漏释义。两者结合通常改善第一阶段召回。


## Q&A-4

> Q4. 为什么分块在 RAG 中如此重要？

**Answer:**

- 分块决定了 retrieval 的单元。如果分块太大，retrieval 变得嘈杂，因为每个命中包含太多不相关的文本。如果分块太小，答案可能失去解释所需的周围 context。好的分块与源材料的结构对齐。
- 在面试中，说分块既是召回率又是精确率决策。它塑造了 retriever 能找到什么，以及一旦检索到段落后生成器能理解什么。


## Q&A-5

> Q5. 元数据过滤器如何改善 retrieval 质量？

**Answer:**

- 元数据过滤器使用产品、地区、日期、语言、权限范围或文档类型等结构化属性缩小搜索空间。这帮助系统在排序语义相关性之前就从正确的邻域检索。
- 实践教训是：retrieval 质量不仅仅关于更好的 embedding。结构化约束可以廉价且可靠地完成大量工作。优秀候选人经常将元数据过滤作为生产搜索中回报最高的改进之一提及。


## Q&A-6

> Q6. 什么是向量数据库，它解决什么问题？

**Answer:**

- 向量数据库存储 embedding 并支持高效的最近邻搜索。它被构建为在规模上找到与查询向量最相似的向量，通常同时结合元数据过滤器和复制、持久性、监控等运营特性。
- 重要的面试观点是：向量存储是基础设施，而非智能。它使 retrieval 可行且快速，但相关性仍然取决于其上层的 embedding 模型、分块策略和排序逻辑。


## Q&A-7

> Q7. 为什么生产系统依赖近似最近邻搜索？

**Answer:**

- 精确最近邻搜索在大型索引上变得昂贵，因为每个查询都需要与太多候选进行比较。近似方法以少量召回率换取更好的速度和可扩展性，这通常是真实系统中正确的权衡。
- 在面试中，最好的答案是运营性的。ANN 存在是因为搜索系统必须以低延迟服务真实流量。问题不是近似是否在哲学上纯粹，而是它是否在生产速度下保留了足够的相关性。


## Q&A-8

> Q8. 什么是重排序，为什么它有用？

**Answer:**

- 重排序将更昂贵的相关性模型应用于第一个 retriever 返回的候选列表。初始 retriever 最大化速度和召回率；重排序器改善排序，使最好的证据到达生成器。
- 常见模式是 bi-encoder retrieval 后跟 cross-encoder 重排序。这给你向量搜索的可扩展性和更丰富的查询-文档交互的精确率。在面试中，将重排序解释为第二阶段质量过滤器。


## Q&A-9

> Q9. 查询重写如何帮助 retrieval？

**Answer:**

- 查询重写通过将用户的原始问题转换为与索引内容更好对齐的形式来改善 retrieval。系统可能扩展缩写、规范化术语、添加关键词、消歧实体，或将一个复杂查询拆分为多个专注的 retrieval 意图。
- 关键思想是：用户不会自然地用索引友好的语言说话。强答案是：查询重写通常是在不重新 embedding 语料库的情况下改善召回率的最便宜方式之一。


## Q&A-10

> Q10. 哪些离线指标对 retrieval 质量最重要？

**Answer:**

- Recall at k 测量相关证据是否出现在候选列表中。Mean reciprocal rank 和 nDCG 测量相关项目是否出现在顶部附近。对于答案生成系统，段落级相关性也应该与端到端有据可查的答案质量挂钩。
- 最强的面试答案是：retrieval 指标不应该与生成结果隔离。离线看起来强劲但向生成器提供嘈杂证据的 retriever，仍然可能在用户任务上失败。



















