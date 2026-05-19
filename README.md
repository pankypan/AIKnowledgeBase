# AI Knowledge Base

> 一份面向 **AI 算法工程师(应用型)** 的中文学习笔记与知识图谱，沉淀从「数学基础 → 机器学习 → 深度学习 → 大模型 → AI 系统 → Agent」的完整学习路径。

## 项目介绍

本仓库是个人在学习与实践 AI 过程中持续整理的 **结构化知识库**，目标是把零散的论文、书籍、课程、博客与工程经验，归纳为可复用、可检索、可迭代的 Markdown 笔记。

### 适合的读者

- 想系统补齐 **机器学习数学基础** 的初学者
- 希望沿着「经典 ML → DL → LLM → Agent」路径成长的 **应用型 AI 算法工程师**
- 需要快速回顾某个算法 / 模型 / 工程概念的 **在岗工程师**（如面试、复盘、查阅）
- 关注大模型生态（LLM、RAG、Function Calling、MCP、A2A、Agent Skills 等）的 **从业者与研究者**


### 工业界"算法工程师"的真实画像

在大多数人的认知中，算法工程师 = 读论文 + 写论文 + 设计新算法。但在工业界，AI 相关岗位实际是一个从"研究"到"工程"的连续谱：

```
                        偏研究 ◄──────────────────────► 偏工程
                        人数少                           人数多

┌──────────────────────────────────────────────────────────────────────────┐
│  研究科学家 / Research Scientist                                          │
│  PhD，发顶会论文，设计新模型架构                                            │
│  占比：≈5%（大厂研究院、AI 创业公司核心团队、高校）                            │
├──────────────────────────────────────────────────────────────────────────┤
│  研究型算法工程师                                                          │
│  硕博，复现 + 改进论文，偶尔发论文                                           │
│  占比：≈10%（大厂核心算法团队）                                             │
├──────────────────────────────────────────────────────────────────────────┤
│  ⭐ 应用型算法工程师                                                      │
│  本硕均可，读论文 → 选型适配 → 落地优化 → 实验验证 → 业务交付                   │
│  占比：≈30%（所有 AI 公司都需要，岗位需求量大）                               │
├──────────────────────────────────────────────────────────────────────────┤
│  ⭐ AI 开发工程师 / MLOps / AI Infra                                         │
│  工程为主，框架调用 + 系统搭建 + 模型部署 + 数据管线 + 业务落地                 │
│  占比：≈55%（需求量最大，与上层存在大量交叉）                                  │
└──────────────────────────────────────────────────────────────────────────┘

注：比例为估算，实际因公司类型和阶段不同而差异较大。
    大模型时代，第三层与第四层的边界正在快速模糊——
    越来越多岗位同时要求"懂算法原理"和"能工程落地"。
```


**应用型算法工程师的真实日常**：很多人以为算法工程师整天在读论文、推公式。实际上，应用型算法工程师的时间分配更接近这样：

| 工作内容 | 占比 | 典型活动 |
|---------|------|---------|
| 方案选型与落地优化 | 35-40% | 读论文/开源方案 → 评估可行性 → 适配业务 → 调参调优 |
| 数据与特征工程 | 20-25% | 数据清洗、标注质量把控、构造训练/评测数据集 |
| 实验设计与效果验证 | 15-20% | 对比实验、消融分析、A/B 测试、撰写实验报告 |
| 工程化与部署交付 | 10-15% | 模型服务化、性能优化、与开发团队联调上线 |
| 跨团队沟通与汇报 | 5-10% | 向业务方解释算法方案、对齐需求、评审方案可行性 |
| 原创性研究与论文 | 0-5% | 锦上添花，多数团队不做硬性要求 |

> **关键认知**：工业界不缺写论文的人，缺的是能把论文变成生产力的人。
> 注意"数据工程"和"跨团队沟通"占据了近 1/3 的时间——这是很多转型者容易低估的部分。


### 知识体系总览

仓库按主题划分为 6 大模块，每个模块对应一个 `KBxx-*` 目录：

| 模块 | 目录 | 主要内容 |
| --- | --- | --- |
| 1. 数学基础 | [`KB01-Mathmatics`](./KB01-Mathmatics) | 3Blue1Brown 线性代数本质、《Mathematics for Machine Learning》全书章节笔记（线代、解析几何、矩阵分解、向量微积分、概率分布、连续优化、线性回归、降维、GMM、SVM 等） |
| 2. 机器学习 | [`KB02-MachineLearning`](./KB02-MachineLearning) | 李航《统计学习方法》算法笔记、时间序列预测（ARIMA / GBDT / XGBoost / Wavenet / Seq2Seq / Transformer 等）、线性规划与遗传算法等其它算法 |
| 3. 深度学习 | [`KB03-DeepLearning`](./KB03-DeepLearning) | 深度学习入门笔记、李宏毅《深度学习》课程章节笔记、CNN / RNN 模型族小结 |
| 4. 大语言模型 | [`KB04-LLM`](./KB04-LLM) | 《Build a LLM from Scratch》、LangChain、LLM 面试题、NVIDIA GPU 与算力租用、多模态数据预处理、应用型 AI 算法工程师修炼路线 |
| 5. AI 系统 | [`KB05-AISystem`](./KB05-AISystem) | AISystem 体系、CUDA 编程、TensorRT 推理部署等底层与工程化内容 |
| 6. LLM Agent | [`KB06-LLMAgent`](./KB06-LLMAgent) | Token、Agent 构建模式、Context Engineering、Agent Skills、Function Calling、MCP、A2A 协议、RAG、OpenClaw 架构等 Agent 前沿主题 |

### 内容特点

- **中文优先**：以中文整理为主，关键术语保留英文原文，便于检索与对照
- **以 Markdown 组织**：纯文本，方便阅读、搜索、二次编辑与版本管理
- **理论 + 工程并重**：既有数学推导与算法原理，也有 GPU / CUDA / TensorRT / 推理部署等工程实践
- **持续更新**：标记为 `# Todo` / `# TODO` 的章节为待补充内容，会随学习进度持续完善

### 如何使用

1. 直接从下方目录跳转到感兴趣的章节阅读
2. 推荐学习路径：`KB01 数学基础` → `KB02 机器学习` → `KB03 深度学习` → `KB04 LLM` → `KB05 AI 系统` → `KB06 LLM Agent`
3. 已有基础的读者可直接按主题检索，例如想了解 Agent 生态可直接进入 [`KB06-LLMAgent`](./KB06-LLMAgent)
4. 欢迎通过 Issue / PR 反馈错误、补充资料或交流学习心得

---


## 1 Mathematics for Machine Learning

### 3Blue1Brown-线性代数的本质

- [3Blue1Brown-线性代数的本质01](./KB01-Mathmatics/3Blue1Brown-线性代数的本质01.md)
- [3Blue1Brown-线性代数的本质02](./KB01-Mathmatics/3Blue1Brown-线性代数的本质02.md)
- [3Blue1Brown-线性代数的本质03](./KB01-Mathmatics/3Blue1Brown-线性代数的本质03.md)


### MML-机器学习数学基础

- [CH02-Linear Algebra(线性代数)](./KB01-Mathmatics/CH02-LinearAlgebra.md)
- [CH03-Analytic Geometry(解析几何)](./KB01-Mathmatics/CH03-AnalyticGeometry.md)
- [CH04-Matrix Decompositions(矩阵分解)](./KB01-Mathmatics/CH04-MatrixDecompositions.md)
- [CH05-Vector Calculus(向量微积分)](./KB01-Mathmatics/CH05-VectorCalculus.md)
- [CH06-Probability and Distribution(概率与分布)](./KB01-Mathmatics/CH06-ProbabilityAndDistribution.md)
- [CH07-Continuous Optimization(连续优化)](./KB01-Mathmatics/CH07-ContinuousOptimization.md)
- [CH08-When Models Meet Data(模型与数据相遇)](./KB01-Mathmatics/CH08-WhenModelsMeetData.md)
- [CH09-Linear Regression(线性回归)](./KB01-Mathmatics/CH09-LinearRegression.md)
- [CH10-Dimensionality Reduction(降维)](./KB01-Mathmatics/CH10-DimensionalityReduction.md)
- [CH11-Density Estimation With Gaussian Mixture Models(密度估计)](./KB01-Mathmatics/CH11-DensityEstimationWithGaussianMixtureModels.md)
- [CH12-Classification With Support Vector Machines(支持向量机)](./KB01-Mathmatics/CH12-ClassificationWithSupportVectorMachines.md)




## 2 Machine Learning

### 统计机器学习算法

- [CH01-统计学习及监督学习概论](./KB02-MachineLearning/StatisticalLearningMethods/CH01-统计学习及监督学习概论.md)
- [CH04-朴素贝叶斯算法](./KB02-MachineLearning/StatisticalLearningMethods/CH04-朴素贝叶斯算法.md)
- [CH07-支持向量机](./KB02-MachineLearning/StatisticalLearningMethods/CH07-支持向量机.md)  # Todo
- [CH10-隐马尔可夫模型](./KB02-MachineLearning/StatisticalLearningMethods/CH10-隐马尔可夫模型.md)  # Todo
- [CH13-无监督学习概论](./KB02-MachineLearning/StatisticalLearningMethods/CH13-无监督学习概论.md)  # Todo
- [CH14-聚类方法](./KB02-MachineLearning/StatisticalLearningMethods/CH14-聚类方法.md)  # Todo
- [CH15-奇异值分解](./KB02-MachineLearning/StatisticalLearningMethods/CH15-奇异值分解.md)  # Todo
- [CH16-主成分分析](./KB02-MachineLearning/StatisticalLearningMethods/CH16-主成分分析.md)  # Todo
- [CH19-马尔可夫链蒙特卡罗法](./KB02-MachineLearning/StatisticalLearningMethods/CH19-马尔可夫链蒙特卡罗法.md)  # Todo
- [CH20-潜在狄利克雷分配](./KB02-MachineLearning/StatisticalLearningMethods/CH20-潜在狄利克雷分配.md)  # Todo
- [CH21-PageRank算法](./KB02-MachineLearning/StatisticalLearningMethods/CH21-PageRank算法.md)  # Todo





### 时间序列预测算法

- [时序问题预测算法总结](./KB02-MachineLearning/TimeSeriesForecastingAlgos/时序问题预测算法总结.md)
- **传统时序建模**: [ARIMA](./KB02-MachineLearning/TimeSeriesForecastingAlgos/ARIMA.md)
- **机器学习模型方法**: 1.[GBDT](./KB02-MachineLearning/TimeSeriesForecastingAlgos/GBDT.md); 2. [XGBoost](./KB02-MachineLearning/TimeSeriesForecastingAlgos/XGBoost.md)
- **深度学习模型方法**：
  - Wavenet: [Wavenet原理与实现](https://zhuanlan.zhihu.com/p/28849767)
  - 1D-CNN: [1D Convolutional Neural Networks and Applications: A Survey](https://arxiv.org/abs/1905.03554)
  - LSTM: [RNN01-相关模型总结](./KB03-DeepLearning/RNN01-相关模型总结.md)
  - Seq2Seq: 1. [Seq2Seq原理详解](https://www.cnblogs.com/liuxiaochong/p/14399416.html); 2. [D2L-序列到序列学习](https://zh-v2.d2l.ai/chapter_recurrent-modern/seq2seq.html)
  - Transformer: 1. [Transformer for TimeSeries时序预测算法详解](https://zhuanlan.zhihu.com/p/391337035); 2. [D2L-Transformer](https://zh-v2.d2l.ai/chapter_attention-mechanisms/transformer.html)





### 其它算法

- [线性规划](./KB02-MachineLearning/OtherAlgos/LinearProgramming.md)
- [遗传算法(GA)](./KB02-MachineLearning/OtherAlgos/GeneticAlgorithm.md)  # TODO








## 3 Deep Learning

### 深度学习入门

- [深度学习入门-笔记1](./KB03-DeepLearning/DeepLearningPrime/深度学习入门-笔记1.md)
- [深度学习入门-笔记2](./KB03-DeepLearning/DeepLearningPrime/深度学习入门-笔记2.md)




### 深度学习（理论）

- [ch02_实践方法论](./KB03-DeepLearning/LeeDLNotes/ch02_实践方法论.md)
- [ch03_深度学习基础](./KB03-DeepLearning/LeeDLNotes/ch03_深度学习基础.md)
- [ch04_卷积神经网络](./KB03-DeepLearning/LeeDLNotes/ch04_卷积神经网络.md)
- [ch05_循环神经网络](./KB03-DeepLearning/LeeDLNotes/ch05_循环神经网络.md)
- [ch06_自注意力机制](./KB03-DeepLearning/LeeDLNotes/ch06_自注意力机制.md)  # TODO
- [ch07_Transformer](./KB03-DeepLearning/LeeDLNotes/ch07_Transformer.md)  # TODO
- [ch08To09_生成模型And扩散模型](./KB03-DeepLearning/LeeDLNotes/ch08To09_生成模型And扩散模型.md)  # TODO
- [ch10_自监督学习](./KB03-DeepLearning/LeeDLNotes/ch10_自监督学习.md)  # TODO



### CV & NLP

**CV**：

- [CV01-CNN相关模型总结](./KB03-DeepLearning/CNN01-相关模型总结.md)
- [CV02-计算机视觉基础](./KB03-DeepLearning/CNN02-计算机视觉基础.md)  # TODO
- [CV03-现代视觉模型与Kaggle实战]()  # TODO


**NLP**:

- [NLP01-RNN相关模型总结](./KB03-DeepLearning/RNN01-相关模型总结.md)
- [NLP02-自然语言基础](./KB03-DeepLearning/RNN02-自然语言基础.md)  # TODO
- [NLP03-BERT与预训练模型]()  # TODO





## 4 LLM

### 修炼路线, 数据预处理, 算力指南

- [应用型AI算法工程师修炼路线](./KB04-LLM/应用型AI算法工程师修炼路线.md)
- [Text-Audio-Image 模型数据预处理](./KB04-LLM/Text-Audio-Image模型数据预处理.md)
- [NVIDIA-GPU-分类与国内算力租用指南](./KB04-LLM/NVIDIA-GPU-分类与国内算力租用指南.md)



### 从零构建LLM

- [Text Data Processing](./KB04-LLM/BuildLLMFromScratch/ch02_WorkingWithTextData.md)
- [Coding Attention Mechanisms](./KB04-LLM/BuildLLMFromScratch/ch03_CodingAttentionMechanisms.md)
- [Implementing a GPT Model](./KB04-LLM/BuildLLMFromScratch/ch04_ImplementingAGPTModel.md)
- [Pretraining on Unlabeled Data](./KB04-LLM/BuildLLMFromScratch/ch05_PretrainingOnUnlabeledData.md)
- [Finetuning for Classification](./KB04-LLM/BuildLLMFromScratch/ch06_FinetuningForClassification.md)
- [Finetuning to Follow Instructions](./KB04-LLM/BuildLLMFromScratch/ch07_FinetuningToFollowInstructions.md)




### LLM 微调指南

- [The Ultimate Guide to Fine-Tuning LLMs from Basics to Breakthroughs](https://arxiv.org/html/2408.13296v1#Ch1.S1) # TODO



### 其它笔记
- [LangChain](./KB04-LLM/LangChain.md)
- [LLM Interview P1](./KB04-LLM/LLMInterview_P1.md)





## 5 AI System

### AI System 体系

- [AI芯片体系结构-AI计算体系](./KB05-AISystem/AISystem/二(AI芯片体系结构)-1.AI计算体系.md)
- [AI芯片体系结构-AI芯片基础](./KB05-AISystem/AISystem/二(AI芯片体系结构)-2.AI芯片基础.md)
- [AI编译原理-传统编译器](./KB05-AISystem/AISystem/三(AI编译原理)-1.传统编译器.md)
- [AI推理系统-推理系统介绍](./KB05-AISystem/AISystem/四(AI推理系统)-1.推理系统介绍.md)




### Pytorch实用教程

- [Pytorch实用教程](https://github.com/TingsongYu/PyTorch-Tutorial-2nd)  # Pytorch基础、产业应用、推理部署



### 其它笔记

- [CUDA编程](./KB05-AISystem/CUDA编程/CUDA编程.md)
- [TensorRT](./KB05-AISystem/TensorRT/TensorRT.md)
  






## 6 LLM Agent

- [Note001-LLM中的Token详解](./KB06-LLMAgent/Note001-LLM中的Token详解.md)
- [Note002-Agent的概念、原理与构建模式](./KB06-LLMAgent/Note002-Agent的概念、原理与构建模式.md)
- [Note003-ContextEngineering详解](./KB06-LLMAgent/Note003-ContextEngineering详解.md)
- [Note004-AgentSkills从使用到原理](./KB06-LLMAgent/Note004-AgentSkills从使用到原理.md)
- [Note005-FunctionCalling详解](./KB06-LLMAgent/Note005-FunctionCalling详解.md)
- [Note006-MCP终极指南](./KB06-LLMAgent/Note006-MCP终极指南.md)
- [Note007-A2A协议概览](./KB06-LLMAgent/Note007-A2A协议概览.md)
- [Note008-RAG](./KB06-LLMAgent/Note008-RAG.md)
- [Note009-OpenClaw系统架构详解](./KB06-LLMAgent/Note009-OpenClaw系统架构详解.md)


