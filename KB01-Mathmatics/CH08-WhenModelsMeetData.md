# When Models Meet Data(模型与数据相遇)

## 8.1 数据，模型与学习

### 8.1.1 数据作为向量

我们假设数据可以被计算机读取，并以数值格式进行充分表示。数据被假定为表格形式，其中每一行代表一个特定的实例或示例，每一列代表一个特定的特征。

即使我们拥有表格格式的数据，也仍然需要做出选择以获得数值表示。

<div align="center">
    <img src="https://datawhalechina.github.io/math-for-ai/attachments/1723897317189.png" alt="表8.2" >
    <center style="color: blue;">表8.2来自一个虚构的人力资源数据库的示例数据，被转换为数字格式。</center>
</div>

在本书的这一部分，我们将使用$N$来表示数据集中的示例数量，并用小写字母$n=1,\ldots,N$对示例进行索引。我们假设给定了一组数值数据，表示为一个向量数组（表8.2）。
- 每一行都是一个特定的个体$x_n$，在机器学习中通常被称为示例或数据点。下标$n$表示这是数据集中总共$N$个示例中的第$n$个示例。
- 每一列代表示例的一个特定特征，我们用$d=1,\ldots,D$对特征进行索引。

请记住，**数据以向量的形式表示**，这意味着每个示例（每个数据点）都是一个 **$D$维向量**。表格的方向源自数据库社区，但对于某些机器学习算法，将示例表示为列向量更为方便。

<div align="center">
    <img src="https://datawhalechina.github.io/math-for-ai/attachments/8.1.png" alt="图8.1" width="500">
    <center style="color: blue;">图8.1用于线性回归的玩具数据。来自表8.2最右边两列的训练数据 $(x_n，y_n)$ 对。我们感兴趣的是一个60岁（岁的x=60）的工资，用垂直虚线表示，这不是训练数据的一部分。</center>
</div>

让我们考虑基于表8.2中的数据，根据年龄预测年薪的问题。这被称为监督学习问题，其中每个示例 $x_n$（即年龄）都与一个标签
$$
y_n
$$

（即薪资）相关联。标签$y_n$还有其他各种名称，包括目标、响应变量和注释。数据集被写为一组示例-标签对$\{(\boldsymbol x_1,y_1),\ldots,(\boldsymbol x_n,y_n),\ldots,(\boldsymbol x_N,y_N)\}$。示例表 $\{x_1,\ldots,x_N\}$ 经常被串联起来，并写为 $X\in\mathbb{R}^{N\times D}$。




### 8.1.2 模型的含义

一旦我们将数据以适当的向量形式表示，我们就可以着手构建预测函数（称为预测器）。“模型”的含义。本书提出了两种主要方法：
- 将预测器视为**函数**
- 将预测器视为**概率模型**



#### 8.1.2.1 模型作为函数


**预测器是一个函数**，当给定一个特定的输入示例（在我们的情况下，是一个特征向量）时，会产生一个输出。目前，我们将输出视为一个单一的数字，即一个实值标量输出。这可以写为
$$
f:\mathbb{R}^{D}\to\mathbb{R} \tag{8.1}
$$

其中输入向量 $x$ 是 $D$ 维的（具有$D$个特征），然后函数 $f$ 应用于它（写为$f(x)$）并返回一个实数。

图8.2展示了一个可能的函数，该函数可用于计算输入值$x$的预测值。

<div align="center">
    <img src="https://datawhalechina.github.io/math-for-ai/attachments/8.2.png" alt="图8.2" width="500">
    <center style="color: blue;">图8.2示例函数（黑色实对角线）及其在x = 60处的预测，即f（60）= 100。</center>
</div>

在本书中，我们不考虑所有函数的一般情况，因为这会涉及泛函分析。相反，我们考虑线性函数的特殊情况
$$
f(\boldsymbol{x})=\boldsymbol{\theta}^\top\boldsymbol{x}+\theta_0
$$


其中 $\boldsymbol{\theta}$ 和 $\theta_0$ 是未知的参数。



#### 8.1.2.2 模型作为概率分布

**预测器也可以被视为概率模型**，即描述可能函数分布的模型。与确定性函数直接输出一个预测值不同，概率模型给出关于输出的概率分布，可以写为
$$
p(y \mid \boldsymbol{x}) \tag{8.2}
$$

其中$p(y \mid \boldsymbol{x})$表示给定输入$\boldsymbol{x}$时输出$y$的条件概率分布。我们通常认为数据是对某些真实潜在效应的有噪声观测，并希望通过机器学习从噪声中识别出信号，这要求我们有一种量化噪声和不确定性的语言。概率论提供了这样一种语言。

图8.3展示了函数预测不确定性的高斯分布图示。

<div align="center">
    <img src="https://datawhalechina.github.io/math-for-ai/attachments/8.3.png" alt="图8.3" width="500">
    <center style="color: blue;">图8.3示例函数（黑色实对角线）及其在x = 60处的预测不确定性（绘制为高斯曲线）。</center>
</div>

从图8.3可以看出，在$x=60$处的预测并非一个确定的值，而是服从一个高斯分布。对于线性模型，我们可以将这种带有噪声的预测具体写为
$$
p(y \mid \boldsymbol{x}, \boldsymbol{\theta}) = \mathcal{N}\!\left(y \mid \boldsymbol{\theta}^\top \boldsymbol{x} + \theta_0,\; \sigma^2\right)
$$

其中 $\boldsymbol{\theta}^\top \boldsymbol{x} + \theta_0$ 是均值（即线性预测函数），$\sigma^2$ 是观测噪声的方差。这意味着对于任意输入 $\boldsymbol{x}$，输出 $y$ 以线性函数值为中心、按方差 $\sigma^2$ 波动，从而将确定性预测推广为概率预测。

#### 8.1.2.3 函数 VS 概率分布

上面两小节分别介绍了两种建模视角，它们的核心区别在于：**确定性 vs 概率性**。

| | 确定性函数 | 概率模型 |
|---|---|---|
| 公式 | $f(\boldsymbol{x})=\boldsymbol{\theta}^\top\boldsymbol{x}+\theta_0$ | $p(y \mid \boldsymbol{x}, \boldsymbol{\theta}) = \mathcal{N}(y \mid \boldsymbol{\theta}^\top \boldsymbol{x} + \theta_0,\; \sigma^2)$ |
| 输出 | 一个确定的数值 | 一个概率分布 |
| 关系 | — | 以确定性函数的值为均值，附加噪声方差 $\sigma^2$ |

换言之，确定性函数是概率模型的"骨架"：概率模型在确定性函数给出的预测值上，套了一个高斯分布来刻画不确定性，$\sigma^2$ 越大，不确定性越高。

**示例：预测房价。** 假设参数 $\theta = 2$，$\theta_0 = 10$，噪声方差 $\sigma^2 = 9$，输入面积 $x = 50$（平方米）。

- **确定性函数**给出一个精确值：
$$
f(50) = 2 \times 50 + 10 = 110 \text{（万元）}
$$

- **概率模型**给出一个分布：
$$
p(y \mid x=50) = \mathcal{N}(y \mid 110,\; 9)
$$

前者的含义是"房价**就是** 110 万"；后者的含义是"房价**大约** 110 万，但会在大约 101\~119 万之间波动（$\pm 3\sigma$）"。对应到图上，确定性函数给出的是黑色直线上的一个点，概率模型给出的则是该点处的那条橙色高斯曲线。


### 8.1.3 学习是寻找参数

学习的目标是找到好的参数，使预测器在未见过的数据上表现良好。机器学习算法从概念上分为三个阶段：

1. **训练（参数估计）**：根据训练数据调整模型参数。
2. **预测（推断）**：用训练好的模型对新数据做出预测。
3. **模型选择（超参数调整）**：选择模型结构或调整超参数。

其中训练阶段是核心，根据模型类型的不同，有不同的参数估计策略：

| 模型类型 | 训练策略 | 章节 |
|---|---|---|
| 非概率模型（确定性函数） | **经验风险最小化**：定义损失函数，将训练转化为优化问题 | 8.2 |
| 概率模型（点估计） | **最大似然估计（MLE）**：寻找使观测数据概率最大的参数；**最大后验估计（MAP）**：在 MLE 基础上引入先验，寻找后验概率最大的参数 | 8.3 |
| 概率模型（分布估计） | **贝叶斯推断**：不做点估计，而是求参数的完整后验分布 | 8.4 |

> **点估计 vs 分布估计**
>
> - **点估计**：训练的结果是参数的一个"最佳值" $\boldsymbol{\theta}^*$（一个点）。模型训练完成后，参数就固定了，预测时直接使用 $p(y\mid\boldsymbol{x}, \boldsymbol{\theta}^*)$。优点是简单高效，缺点是丢弃了参数的不确定性——我们无从得知对这个 $\boldsymbol{\theta}^*$ 有多大信心。
> - **分布估计**：训练的结果是参数的一个完整概率分布 $p(\boldsymbol{\theta}\mid\mathcal{X},\mathcal{Y})$（后验分布）。它告诉我们每组参数值的可信程度。预测时需要对所有可能的 $\boldsymbol{\theta}$ 做积分（边缘化）：$p(y\mid\boldsymbol{x},\mathcal{X},\mathcal{Y}) = \int p(y\mid\boldsymbol{x},\boldsymbol{\theta})\,p(\boldsymbol{\theta}\mid\mathcal{X},\mathcal{Y})\,d\boldsymbol{\theta}$。优点是能量化预测的不确定性，缺点是计算代价通常远高于点估计。

在实践中，大多数训练方法都可以视为数值优化——通过爬山法等迭代算法寻找目标函数的最优值。



## 8.2 经验风险最小化

机器学习中的“学习”部分实质上就是基于训练数据来估计参数。在本节中，我们考虑**预测器是一个函数**的情况，而概率模型的情况将在第8.3节中讨论。

**经验风险最小化（Empirical Risk Minimization, ERM）** 是指：在给定训练数据上，选择使平均损失最小的预测器。它包含两个核心要素：
1. **函数假设类**：确定预测器的候选集合（8.2.1）
2. **损失函数**：度量预测值与真实标签的匹配程度（8.2.2）



### 8.2.1 函数假设类

假设我们得到 $N$ 个样本 $x_n\in\mathbb{R}^D$ 和对应的标量标签 $y_n\in\mathbb{R}$。我们考虑监督学习的设置，其中我们获得样本对 $(x_1,y_1),\ldots,(x_N,y_N)$。基于这些数据，我们希望估计一个预测器 $f(\cdot,\boldsymbol{\theta}):\mathbb{R}^D\to\mathbb{R}$，它通过参数 $\theta$ 进行参数化。我们希望能够找到一个好的参数 $\theta^{*}$，以便很好地拟合数据，即
$$
f(\boldsymbol{x}_{n},\boldsymbol{\theta}^{*})\approx y_{n}\quad \forall n=1,\ldots,N\:.
\tag{8.3}
$$

在本节中，我们使用符号 $\hat{y}_n=f(x_n,\theta^*)$ 来表示预测器的输出。


> **例8.1**
>
> 我们引入普通最小二乘回归问题来说明经验风险最小化。当标签$y_n$是实数值时，预测器函数类的一个流行选择是仿射函数集。我们通过向$x_n$添加一个额外的单位特征$x^{(0)}=1$，即$x_n=[1,x_n^{(1)},x_n^{(2)},\ldots,x_n^{(D)}]^\top$，来更简洁地表示仿射函数。相应地，机器学习参数向量是$\boldsymbol{\theta}=[\theta_0,\theta_1,\theta_2,\ldots,\theta_D]^\top$，这使得我们可以将预测器写为线性函数
> $$ f(x_n,\boldsymbol{\theta})=\boldsymbol{\theta}^\top\boldsymbol{x}_n$$
> 
> 这个线性预测器等价于仿射模型
> $$f(\boldsymbol{x}_n,\boldsymbol{\theta})=\theta_0+\sum_{d=1}^D\theta_dx_n^{(d)}$$
>
> 预测器以表示单个样本 $x_n$ 的特征向量为输入，并产生实数值输出，即 $f:\mathbb{R}^{D+1}\to\mathbb{R}$。
> 


给定这类函数，我们想要寻找一个好的预测器。现在，我们转向 **经验风险最小化** 的第二个要素：**如何测量预测器与训练数据的匹配程度**。



### 8.2.2 训练损失函数

机器学习中的一个常见假设是，样本集 $(x_1,y_1),\ldots,(\boldsymbol{x}_N,y_N)$ 是独立同分布的。独立一词

1. 意味着两个数据点 $(\boldsymbol x_i,y_i)$ 和 $(\boldsymbol x_j,y_j)$ 在统计上互不依赖
2. 意味着经验均值是总体均值的良好估计
3. 意味着我们可以使用训练数据上损失的经验均值

对于给定的训练集，
$$
\{(\boldsymbol x_1,y_1),\ldots,(\boldsymbol x_N,y_N)\}
$$

我们引入以下符号：

1. 示例矩阵
$$
\boldsymbol{X} :=[x_1,\ldots,x_N]^\top\in\mathbb{R}^{N\times D}
$$

2. 标签向量
$$
\boldsymbol{y} := [y_1,\ldots,y_N]^\top\in\mathbb{R}^N
$$

使用这种矩阵符号，平均损失由下式给出：
$$
\boldsymbol{R}_{\mathrm{emp}}(f,\boldsymbol{X},\boldsymbol{y})=\frac{1}{N}\sum_{n=1}^{N}\ell(y_{n},\hat{y}_{n}) \tag{8.4}
$$

其中 $\hat{y}_n=f(\boldsymbol{x}_n,\boldsymbol{\theta})$。式（8.4）被称为经验风险，它取决于三个参数：预测器$f$ 和 数据$\boldsymbol{X}, \boldsymbol{y}$。这种学习策略通常被称为经验风险最小化。



> **例 8.2（最小二乘损失）**
>
> 继续最小二乘回归的示例，我们指定使用 平方损失$$\ell(y_n,\hat{y}_n)=(y_n-\hat{y}_n)^2$$
> 
> 来衡量训练过程中犯错的代价。我们希望最小化经验风险（8.4），即数据上损失的平均值
> $$\min_{\boldsymbol{\theta}\in\mathbb{R}^D}\frac{1}{N}\sum_{n=1}^N(y_n-f(\boldsymbol{x}_n,\boldsymbol{\theta}))^2$$
> 
> 其中我们用预测器 $\hat{y}_n=f(\boldsymbol{x}_n,\boldsymbol{\theta})$进行了替换。
> 
> 通过选择线性预测器 $f(\boldsymbol{x}_n,\boldsymbol{\theta})=\boldsymbol{\theta}^\top\boldsymbol{x}_n$，我们得到优化问题
> $$\min_{\boldsymbol{\theta}\in\mathbb{R}^D}\frac{1}{N}\sum_{n=1}^N(y_n-\boldsymbol{\theta}^\top\boldsymbol{x}_n)^2$$
>
> 这个方程可以等价地用矩阵形式表示
> $$\min_{\boldsymbol{\theta}\in\mathbb{R}^{D}}\frac{1}{N}\left\|\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\theta}\right\|^{2}$$
>
> 这被称为最小二乘问题。通过求解正规方程，我们可以得到一个闭式解析解。
> 

我们并不关心仅在训练数据上表现良好的预测器。相反，**我们寻求的是在未见的测试数据上表现良好（风险低）的预测器**。更正式地说，我们感兴趣的是找到一个预测器$f$（参数固定），该预测器能够 **最小化预期风险**
$$
\boldsymbol{R}_{\mathrm{true}}(f)=\mathbb{E}_{x,y}[\ell(y,f(\boldsymbol{x}))]\:. \tag{8.5}
$$

其中$y$是标签，$f(x)$是基于样本$x$的预测。符号$\boldsymbol{R}_{\text{true}}(f)$表示如果我们拥有无限量的数据，这就是真正的风险。该期望是针对所有可能的数据和标签的（无限）集合。

从我们通常希望最小化预期风险的愿望中，产生了两个实际问题，我们将在以下两个小节中讨论：
- 我们应该如何 **改变训练过程** 以使其 **具有良好的泛化能力**？
- 我们如何 **从（有限）数据中估计预期风险**？




### 8.2.3 正则化以减少过拟合

事实证明，经验风险最小化可能导致“过拟合”，即预测器过于紧密地拟合训练数据，而不能很好地泛化到新数据（Mitchell, 1997）。

这种在训练集上平均损失很小但在测试集上平均损失很大的普遍现象，往往发生在我们拥有少量数据和复杂假设类时。

对于特定的预测器$f$（参数固定），当过拟合现象发生时，来自训练数据的风险估计 $\boldsymbol{R}_\mathrm{emp}(f,X_\mathrm{train},y_\mathrm{train})$ 会低估预期风险 $\boldsymbol{R}_\mathrm{true}(f)$。

由于我们使用测试集上的经验风险$\boldsymbol{R}_{\mathrm{emp}}(f,\boldsymbol{X}_{\mathrm{test}},\boldsymbol{y}_{\mathrm{test}})$来估计预期风险$\boldsymbol{R}_\mathrm{true}(f)$，如果测试风险远大于训练风险，这就是过拟合的迹象。

因此，我们需要通过引入惩罚项来以某种方式偏向寻找经验风险最小化的最小化器，这使得优化器更难返回一个过于灵活的预测器。**在机器学习中，这个惩罚项被称为正则化。** 正则化是在经验风险最小化的准确解与解的大小或复杂性之间做出妥协的一种方式。

**正则化** 是一种方法，用于阻止优化问题中出现复杂或极端的解决方案。最简单的正则化策略是通过添加一个仅涉及$\theta$的惩罚项。


> **示例 8.3（正则化最小二乘法）**
>
> 对于，前一个示例中的最小二乘问题
> $$\min_{\boldsymbol{\theta}}\frac{1}{N}\left\|\boldsymbol{y}-\boldsymbol{X}\boldsymbol{\theta}\right\|^2$$
> 
> 替换为“正则化”问题：
> $$
> \min_{\boldsymbol{\theta}} \frac{1}{N} \left\|\boldsymbol{y}- \boldsymbol{X}\boldsymbol{\theta} \right\|^{2}+ \lambda\left\|\boldsymbol{\theta}\right\|^{2}
> $$
>
> 其中，附加项 $\|\boldsymbol{\theta}\|^2$ 被称为正则化项，而参数 $\lambda$ 被称为正则化参数。正则化参数在训练集上的损失最小化和参数 $\boldsymbol{\theta}$ 的幅度之间进行了权衡。
> 


正则化项有时被称为惩罚项，它促使向量 $\boldsymbol{\theta}$ 更接近原点。


 

### 8.2.4 交叉验证以评估泛化性能

#### 动机：训练集与验证集的矛盾

我们在上一节中提到，通过将预测器应用于测试数据来估计泛化误差以衡量其性能。这些测试数据有时也被称为**验证集**——从可用训练数据中保留出来的一个子集。

然而这里存在一个实际矛盾：
- 我们希望用**尽可能多**的数据来训练模型（要求验证集小）
- 但验证集太小会导致性能估计**噪声大**（高方差）

**交叉验证**正是为解决这一矛盾而设计的方法。

#### K 折交叉验证的流程

K 折交叉验证将数据集分成 $K$ 个不重叠的部分，每次用其中 $K-1$ 个部分作为训练集 $\mathcal{R}$，剩余 1 个部分作为验证集 $\mathcal{V}$。轮流选择验证集并对 $K$ 次结果取平均，见图 8.4。

![1723900862136](https://datawhalechina.github.io/math-for-ai/attachments/8.4.png)

<center style="color: blue;">图8.4k倍交叉验证。数据集被分为K = 5个块，其中K−1作为训练集（蓝色），一个作为验证集（橙色孵化）。</center>

具体步骤如下：

1. 将数据集划分为 $K$ 个不重叠的子集：$\mathcal{D}=\mathcal{R}\cup\mathcal{V}$（$\mathcal{R}\cap\mathcal{V}=\emptyset$）
2. 对每个分区 $k$，在训练集 $\mathcal{R}^{(k)}$ 上训练得到预测器 $f^{(k)}$
3. 在验证集 $\mathcal{V}^{(k)}$ 上计算经验风险 $R(f^{(k)},\mathcal{V}^{(k)})$（例如 RMSE）
4. 遍历所有 $K$ 种分区，计算平均泛化误差

交叉验证近似于期望泛化误差：
$$
\mathbb{E}_{\mathcal{V}}[R(f,\mathcal{V})]\approx\frac{1}{K}\sum_{k=1}^{K}R(f^{(k)},\mathcal{V}^{(k)})\:.
\tag{8.6}
$$

其中 $R(f^{(k)},\mathcal{V}^{(k)})$ 是预测器 $f^{(k)}$ 在验证集 $\mathcal{V}^{(k)}$ 上的风险。

> **近似误差的两个来源：**
> 1. 有限的训练集 → 学到的 $f^{(k)}$ 不是最优预测器
> 2. 有限的验证集 → 对风险 $R(f^{(k)},\mathcal{V}^{(k)})$ 的估计不够准确

#### 计算代价与实际考虑

K 折交叉验证的**主要缺点**是需要训练模型 $K$ 次，当单次训练成本很高时可能难以承受。此外在实践中，还需要搜索超参数（如正则化参数），这可能导致训练次数随超参数数量指数增长。可以使用**嵌套交叉验证**（第 8.6.1 节）来搜索良好的超参数。

不过，交叉验证天然适合并行化（embarrassingly parallel）——各折之间互不依赖，在拥有足够计算资源（云计算、服务器集群）的情况下，所需时间不会比单次评估更长。



## 8.3 参数估计

在 8.2 节中，预测器是一个确定性函数 $f(\boldsymbol{x})$，我们通过最小化损失来训练它。本节切换到**概率模型**的视角：我们不再直接输出一个预测值，而是为每组参数 $\boldsymbol{\theta}$ 指定一个条件概率分布
$$
p(y \mid \boldsymbol{x}, \boldsymbol{\theta})
$$

来描述"在给定输入 $\boldsymbol{x}$ 和参数 $\boldsymbol{\theta}$ 时，输出 $y$ 有多大可能"。这同一个数学表达式，根据"谁固定、谁变化"的不同，有两种身份：

| | 条件概率分布 | 似然函数 |
|---|---|---|
| 表达式 | $p(y \mid \boldsymbol{x}, \boldsymbol{\theta})$ | $p(y \mid \boldsymbol{x}, \boldsymbol{\theta})$ |
| 固定的 | $\boldsymbol{\theta}$（参数已知） | $y, \boldsymbol{x}$（数据已观测） |
| 变化的 | $y$（输出取各种值） | $\boldsymbol{\theta}$（遍历参数空间） |
| 性质 | 对 $y$ 积分 = 1（合法概率分布） | 对 $\boldsymbol{\theta}$ 积分 $\neq$ 1（不是概率分布） |
| 回答的问题 | "给定模型，数据会是什么样？" | "给定数据，哪组参数更合理？" |

当我们把数据视为已观测、让参数变化时，这个表达式就称为**似然函数**——它是连接参数与数据的桥梁，也是本节所有方法的出发点。例如，在 8.1.2.2 中我们已经见过一个具体的似然：高斯似然 $p(y \mid \boldsymbol{x}, \boldsymbol{\theta}) = \mathcal{N}(y \mid \boldsymbol{\theta}^\top \boldsymbol{x} + \theta_0, \sigma^2)$。

有了似然函数，"训练"就变成了：**找到使观测数据在该分布下概率最大的参数**。这就是下面要介绍的最大似然估计和最大后验估计。




### 8.3.1 最大似然估计 (MLE)

#### 核心思想

**最大似然估计（Maximum Likelihood Estimation, MLE）** 的思想是：在所有可能的参数 $\boldsymbol{\theta}$ 中，选择那个使观测数据出现概率最大的参数值。

我们先给出 MLE 的一般定义，再写出监督学习中的对应形式。

**一般定义**：对于由随机变量 $\boldsymbol{x}$ 表示的观测数据（此处 $\boldsymbol{x}$ 泛指所有观测数据，不限于监督学习中的输入特征）和由参数 $\boldsymbol{\theta}$ 参数化的概率密度 $p(\boldsymbol{x}\mid\boldsymbol{\theta})$，定义**负对数似然**：
$$
\mathcal{L}_{\boldsymbol{x}}(\boldsymbol{\theta})=-\log p(\boldsymbol{x}\mid\boldsymbol{\theta}) \tag{8.7}
$$

**监督学习形式**：当数据为输入-输出对 $(\boldsymbol{x}, y)$ 时，似然变为条件概率 $p(y\mid\boldsymbol{x},\boldsymbol{\theta})$，负对数似然相应写为：
$$
\mathcal{L}_{\boldsymbol{x}}(\boldsymbol{\theta})=-\log p(y\mid\boldsymbol{x},\boldsymbol{\theta}) \tag{8.8}
$$

符号 $\mathcal{L}_x(\boldsymbol{\theta})$ 强调参数 $\boldsymbol{\theta}$ 在变化，而数据 $x$ 是固定的。当上下文清楚时，我们简写为 $\mathcal{L}(\boldsymbol{\theta})$。

如 8.3 节开头的对比表所示，MLE 采用"似然视角"：数据已经固定，我们在参数空间中寻找最可能生成该数据的那组参数。



#### 从单个似然到整体似然

在监督学习设置中，我们获得样本对 
$$
\{(\boldsymbol{x}_1, y_1),\ldots,(\boldsymbol{x}_N, y_N)\}, \quad \boldsymbol x_n\in\mathbb{R}^D, y_n\in\mathbb{R}
$$

目标是为特定参数设置 $\boldsymbol{\theta}$ 指定给定样本条件下标签的条件概率分布
$$
p(y_n \mid \boldsymbol{x}_n, \boldsymbol{\theta})
$$

假设样本集 $(x_1,y_1),\ldots,(x_N,y_N)$ 是**独立同分布**的：

- **独立**：整体似然可以分解为各个样本似然的乘积
- **同分布**：每个样本遵循相同的分布，共享相同的参数

因此，整体似然为：
$$
p(\mathcal{Y}\mid\mathcal{X},\boldsymbol{\theta})=\prod_{n=1}^{N}p(y_{n}\mid\boldsymbol{x}_{n},\boldsymbol{\theta}) \tag{8.9}
$$

取负对数后，乘积变为求和（对优化更友好）：

$$
\mathcal{L}_{\text{MLE}}(\boldsymbol{\theta})=-\log p(\mathcal{Y}\mid\mathcal{X},\boldsymbol{\theta})=-\sum_{n=1}^{N}\log p(y_{n}\mid\boldsymbol{x}_{n},\boldsymbol{\theta}) \tag{8.10}
$$

MLE 估计就是最小化该负对数似然：

$$\boldsymbol{\theta}_{\text{MLE}} = \arg\min_{\boldsymbol{\theta}}\; \mathcal{L}_{\text{MLE}}(\boldsymbol{\theta}) \tag{8.11}$$

> **注意：** 尽管在 $p(y_n|x_n,\boldsymbol{\theta})$ 中 $\boldsymbol{\theta}$ 位于条件符号右侧，但负对数似然 $\mathcal{L}(\boldsymbol{\theta})$ 仍然是 $\boldsymbol{\theta}$ 的函数——我们正是要对它做最小化。
> 
> **备注：** 式(8.10)中的负号是历史惯例——我们想要最大化似然，但数值优化文献习惯最小化目标函数，所以取负号统一为最小化问题。



### 8.3.2 最大后验估计 (MAP)

#### 从 MLE 到 MAP：引入先验知识

MLE 只看数据本身，但如果我们对参数 $\boldsymbol{\theta}$ 有**先验知识**（例如"参数值不应过大"），就可以将其纳入估计过程。

最大后验估计（Maximum A Posteriori, MAP）的做法是：在似然函数的基础上，再乘以一个参数的**先验分布** $p(\boldsymbol{\theta})$，然后利用贝叶斯定理计算后验分布。

**一般定义**：

$$p(\boldsymbol{\theta}\mid\boldsymbol{x})=\frac{p(\boldsymbol{x}\mid\boldsymbol{\theta})p(\boldsymbol{\theta})}{p(\boldsymbol{x})}\:. \tag{8.12}$$

**监督学习形式**：

$$p(\boldsymbol{\theta}\mid\mathcal{X},\mathcal{Y})=\frac{p(\mathcal{Y}\mid\mathcal{X},\boldsymbol{\theta})p(\boldsymbol{\theta})}{p(\mathcal{Y}\mid\mathcal{X})}\:. \tag{8.13}$$




#### MAP 的优化目标

由于分母不依赖于 $\boldsymbol{\theta}$，优化时可以忽略，得到：

**一般定义**：

$$p(\boldsymbol{\theta}\mid\boldsymbol{x})\propto p(\boldsymbol{x}\mid\boldsymbol{\theta})p(\boldsymbol{\theta})\:. \tag{8.14}$$

**监督学习形式**：

$$p(\boldsymbol{\theta}\mid\mathcal{X},\mathcal{Y})\propto p(\mathcal{Y}\mid\mathcal{X},\boldsymbol{\theta})p(\boldsymbol{\theta})\:. \tag{8.15}$$

即：**后验 ∝ 似然 × 先验**。

MAP 估计就是最小化负对数后验（而非负对数似然）：

$$
\mathcal{L}_{\text{MAP}}(\boldsymbol{\theta}) = -\log p(\mathcal{Y}\mid\mathcal{X},\boldsymbol{\theta}) - \log p(\boldsymbol{\theta}) \tag{8.16}
$$

$$\boldsymbol{\theta}_{\text{MAP}} = \arg\min_{\boldsymbol{\theta}}\; \mathcal{L}_{\text{MAP}}(\boldsymbol{\theta}) \tag{8.17}$$



#### MAP 与正则化的关系

MAP 估计可以被看作连接**非概率世界**和**概率世界**的桥梁：

| 视角 | 方法 | 额外项的含义 |
|---|---|---|
| 非概率（8.2.3节） | 正则化 | 惩罚项，使参数偏向原点 |
| 概率（本节） | MAP | 先验分布，编码对参数的先验信念 |

两者在数学形式上往往等价（例如高斯先验 ↔ L2 正则化），但 MAP 明确承认了先验分布的需求。不过它仍然只产生参数的**点估计**，而非完整的后验分布（后者将在 8.4 节讨论）。




### 8.3.3 MLE Vs MAP

为了直观理解两种方法的异同，我们用一个监督学习的例子走完完整的求解过程。

#### 问题设定

在监督学习设置中，我们获得 $N$ 个样本对：

$$
\{(\boldsymbol{x}_1, y_1), \ldots, (\boldsymbol{x}_N, y_N)\}, \quad \boldsymbol{x}_n \in \mathbb{R}^3,\; y_n \in \mathbb{R}
$$

其中 $\boldsymbol{x}_n$ 是 3 维特征向量（例如房屋的面积、楼层、房龄），$y_n$ 是对应的标签（例如房价）。

我们使用线性模型 $\hat{y}_n = \boldsymbol{x}_n^\top \boldsymbol{\theta}$，其中参数 $\boldsymbol{\theta} \in \mathbb{R}^3$。假设观测噪声为高斯分布，即似然为：

$$
p(y_n \mid \boldsymbol{x}_n, \boldsymbol{\theta}) = \mathcal{N}(y_n \mid \boldsymbol{x}_n^\top \boldsymbol{\theta},\; \sigma^2)
$$

噪声方差 $\sigma^2$ 已知。将所有样本的输入堆叠为矩阵 $\boldsymbol{X} \in \mathbb{R}^{N \times 3}$，标签堆叠为向量 $\boldsymbol{y} \in \mathbb{R}^N$。

#### MLE 求解

负对数似然为：

$$
\mathcal{L}(\boldsymbol{\theta}) = -\sum_{n=1}^N \log \mathcal{N}(y_n \mid \boldsymbol{x}_n^\top\boldsymbol{\theta}, \sigma^2) = \frac{1}{2\sigma^2}\sum_{n=1}^N (y_n - \boldsymbol{x}_n^\top\boldsymbol{\theta})^2 + \text{const}
$$

写成矩阵形式：

$$
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{2\sigma^2}\|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\theta}\|^2 + \text{const}
$$

对 $\boldsymbol{\theta}$ 求导并令其为零：

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\theta}} = -\frac{1}{\sigma^2}\boldsymbol{X}^\top(\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\theta}) = \boldsymbol{0} \quad \Longrightarrow \quad \boldsymbol{\theta}_{\text{MLE}} = (\boldsymbol{X}^\top\boldsymbol{X})^{-1}\boldsymbol{X}^\top\boldsymbol{y}
$$

结论：**MLE 的解就是经典的最小二乘解**，完全由数据决定。

#### MAP 求解

在 MLE 的基础上，假设我们对 $\boldsymbol{\theta}$ 有先验信念——认为参数不应过大，具体表示为零均值高斯先验：

$$
p(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta} \mid \boldsymbol{0},\; \sigma_0^2 \boldsymbol{I})
$$

MAP 的目标函数为负对数后验（忽略常数项）：

$$
\mathcal{L}_{\text{MAP}}(\boldsymbol{\theta}) = \underbrace{\frac{1}{2\sigma^2}\|\boldsymbol{y} - \boldsymbol{X}\boldsymbol{\theta}\|^2}_{\text{负对数似然（数据项）}} + \underbrace{\frac{1}{2\sigma_0^2}\|\boldsymbol{\theta}\|^2}_{\text{负对数先验（正则项）}}
$$

对 $\boldsymbol{\theta}$ 求导并令其为零：

$$
\frac{1}{\sigma^2}\boldsymbol{X}^\top(\boldsymbol{X}\boldsymbol{\theta} - \boldsymbol{y}) + \frac{1}{\sigma_0^2}\boldsymbol{\theta} = \boldsymbol{0}
$$

解得：

$$
\boldsymbol{\theta}_{\text{MAP}} = \left(\boldsymbol{X}^\top\boldsymbol{X} + \frac{\sigma^2}{\sigma_0^2}\boldsymbol{I}\right)^{-1}\boldsymbol{X}^\top\boldsymbol{y}
$$

对比 MLE 的解 $(\boldsymbol{X}^\top\boldsymbol{X})^{-1}\boldsymbol{X}^\top\boldsymbol{y}$，MAP 多了一项 $\frac{\sigma^2}{\sigma_0^2}\boldsymbol{I}$——这正是**岭回归（Ridge Regression）** 中的正则化项，正则化系数 $\lambda = \frac{\sigma^2}{\sigma_0^2}$。

#### 核心区别总结

| | MLE | MAP |
|---|---|---|
| 优化目标 | $-\log p(\mathcal{Y}\mid\mathcal{X},\boldsymbol{\theta})$ | $-\log p(\mathcal{Y}\mid\mathcal{X},\boldsymbol{\theta}) - \log p(\boldsymbol{\theta})$ |
| 本例结果 | $(\boldsymbol{X}^\top\boldsymbol{X})^{-1}\boldsymbol{X}^\top\boldsymbol{y}$ | $(\boldsymbol{X}^\top\boldsymbol{X} + \lambda\boldsymbol{I})^{-1}\boldsymbol{X}^\top\boldsymbol{y}$ |
| 数据量 $N \to \infty$ 时 | 两者趋同（数据主导，先验被"淹没"） | |
| 数据量 $N$ 小或特征共线性时 | $\boldsymbol{X}^\top\boldsymbol{X}$ 可能奇异，解不稳定 | $+\lambda\boldsymbol{I}$ 保证可逆，解更稳定 |
| 等价的非概率视角 | 普通最小二乘 | 岭回归（L2 正则化） |

**一句话总结**：MLE 让数据说了算；MAP 是数据与先验信念的折中——数据越多，先验的影响越小；数据越少，先验越重要。当先验为均匀分布（即 $\sigma_0^2 \to \infty$，"我没有任何偏好"）时，$\lambda \to 0$，MAP 退化为 MLE。







### 8.3.4 模型拟合

#### 什么是模型拟合？

**模型拟合** 是指：给定数据集，优化参数 $\boldsymbol{\theta}$ 以最小化某个损失函数（如负对数似然），使参数化模型尽可能接近生成数据的真实模型。

- MLE 和 MAP 是两种常用的模型拟合算法
- 参数化定义了一个我们可以操作的 **模型类** $M_{\boldsymbol{\theta}}$


<div align="center">
    <img src="https://datawhalechina.github.io/math-for-ai/attachments/8.7.png" alt="图8.7" width="500">
    <br>
    <span style="color: blue;">图8.7 模型拟合。在参数化的模型 $M_{\theta}$ 类中，我们优化模型参数 $\theta$，以最小化到真（未知）模型 $M^*$ 的距离。</span>
</div>


如图 8.7 所示，假设数据来自未知的 真实模型 $M^*$，我们从 $M_{\boldsymbol{\theta}_0}$ 开始搜索，优化后得到最佳参数 $\boldsymbol{\theta}^*$。根据模型类与真实模型的匹配程度，会出现三种情况。



#### 三种拟合情况

<div align="center">
    <img src="https://datawhalechina.github.io/math-for-ai/attachments/8.8.png" alt="图8.8" width="700">
    <br>
    <span style="color: blue;">图8.8 分别展示了 (a) 过拟合、(b) 欠拟合、(c) 拟合良好 的回归示例。</span>
</div>

| 情况 | 模型类特点 | 典型表现 | 示例 |
|---|---|---|---|
| **过拟合** | 过于丰富（参数过多） | 训练误差极低，但泛化误差高；模型捕捉了噪声中的虚假信号 | 用七次多项式拟合线性数据 |
| **欠拟合** | 不够丰富（参数过少） | 训练误差和泛化误差都高；模型表达能力不足 | 用直线拟合正弦数据 |
| **拟合良好** | 恰到好处 | 模型类刚好足够描述数据，泛化性能好 | 用线性模型拟合线性数据 |




#### 实践中的应对策略

在实践中，我们经常使用非常丰富的模型类（如深度神经网络，参数众多）。为减轻过拟合，可以采用：
- **正则化**（第8.2.3节）
- **先验分布**（第8.3.2节）

如何系统地选择模型类，将在第 8.6 节中讨论。




## 8.4 概率建模与推断

在机器学习中，我们经常需要对未来事件做出预测或制定决策。为此，我们构建**概率模型**来描述观测数据的生成过程，并通过**推断**从数据中学习模型的未知量。

> **引例：抛硬币。** 假设我们想描述抛硬币的结果（正面/反面）：
> 1. 定义参数 $\mu$（出现"正面"的概率），作为伯努利分布的参数
> 2. 从 $p(x\mid\mu)=\text{Ber}(\mu)$ 中抽取结果 $x\in\{\text{head}, \text{tail}\}$
>
> 参数 $\mu$ 未知且无法直接观测，我们需要根据观察结果来学习它。

### 8.4.1 概率模型

概率模型将实验中的不确定部分表示为概率分布。它通过概率论提供了一套统一且一致的工具集（随机变量，第6章），用于建模、推断、预测和模型选择。

#### 联合分布是核心

在概率建模中，观测变量 $\boldsymbol{x}$ 和隐藏参数 $\boldsymbol{\theta}$ 的**联合分布** $p(\boldsymbol{x},\boldsymbol{\theta})$ 至关重要，因为从中可以导出所有我们需要的量：

<mark>**联合分布**</mark>：
$$
\underbrace{p(\boldsymbol{x}, \boldsymbol{\theta})}_{\text{联合分布}} = \underbrace{p(\boldsymbol{x}\mid\boldsymbol{\theta})}_{\text{似然}} \cdot \underbrace{p(\boldsymbol{\theta})}_{\text{先验}} \tag{8.18}
$$

<mark>**边缘似然**</mark>：($\xrightarrow{\text{对 }\boldsymbol{\theta}\text{ 积分}}$)
$$
\underbrace{p(\boldsymbol{x})}_{\text{边缘似然}} = \int p(\boldsymbol{x}\mid\boldsymbol{\theta})\,p(\boldsymbol{\theta})\,d\boldsymbol{\theta} \tag{8.19}
$$

<mark>**后验分布**</mark>：
$$
\underbrace{p(\boldsymbol{\theta}\mid\boldsymbol{x})}_{\text{后验}} = \frac{p(\boldsymbol{x}, \boldsymbol{\theta})}{p(\boldsymbol{x})} = \frac{p(\boldsymbol{x}\mid\boldsymbol{\theta})\,p(\boldsymbol{\theta})}{p(\boldsymbol{x})} \tag{8.20}
$$

| 导出量 | 计算方式 | 用途 |
|---|---|---|
| 联合 = 似然 × 先验 | 乘积规则（第6.3节） | 定义模型结构 |
| 边缘似然 $p(\boldsymbol{x})$ | 对 $\boldsymbol{\theta}$ 积分（求和规则） | 模型选择（第8.6节） |
| 后验 $p(\boldsymbol{\theta}\mid\boldsymbol{x})$ | 联合分布 / 边缘似然 | 参数学习 |

只有联合分布才同时具备这些性质。因此，**概率模型由其所有随机变量的联合分布来指定**。

### 8.4.2 贝叶斯推断

#### 动机：点估计的局限性

在 8.3 节中，MLE 和 MAP 都产生参数的**点估计** $\boldsymbol{\theta}^*$，然后用 $p(\boldsymbol{x}\mid\boldsymbol{\theta}^*)$ 进行预测。但这种做法存在问题：

- 丢弃了参数的不确定性信息
- 决策系统的目标函数（如平方误差、分类错误率）往往与似然函数不同
- 后验分布的完整形态对稳健决策至关重要

#### 贝叶斯推断的核心公式

贝叶斯推断寻找参数的**完整后验分布**（而非单个点），通过贝叶斯定理实现：

$$
p(\boldsymbol{\theta}\mid\mathcal{X})=\frac{p(\mathcal{X}\mid\boldsymbol{\theta})p(\boldsymbol{\theta})}{p(\mathcal{X})} \tag{8.21}
$$

$$
p(\mathcal{X})=\int p(\mathcal{X}\mid\boldsymbol{\theta})p(\boldsymbol{\theta})\mathrm{d}\boldsymbol{\theta} \tag{8.22}
$$

核心思想：利用贝叶斯定理**反转**参数与数据的关系——从"参数 → 数据"（似然）得到"数据 → 参数"（后验）。

#### 贝叶斯预测

有了后验分布，预测时可以将参数不确定性传播到数据上：

$$
p(\boldsymbol{x})=\int p(\boldsymbol{x}\mid\boldsymbol{\theta})p(\boldsymbol{\theta})\mathrm{d}\boldsymbol{\theta}=\mathbb{E}_{\boldsymbol{\theta}}[p(\boldsymbol{x}\mid\boldsymbol{\theta})] \tag{8.23}
$$

预测不再依赖于某个特定的 $\boldsymbol{\theta}$ 值，而是对所有合理参数值取平均（由 $p(\boldsymbol{\theta})$ 加权）。

#### 点估计 vs 贝叶斯推断

| | 点估计（MLE/MAP） | 贝叶斯推断 |
|---|---|---|
| 输出 | 参数的单一最优值 $\boldsymbol{\theta}^*$ | 参数的后验分布 $p(\boldsymbol{\theta}\mid\mathcal{X})$ |
| 核心计算 | 优化问题 | 积分问题 |
| 预测方式 | $p(\boldsymbol{x}\mid\boldsymbol{\theta}^*)$（直接代入） | $\int p(\boldsymbol{x}\mid\boldsymbol{\theta})p(\boldsymbol{\theta}\mid\mathcal{X})\mathrm{d}\boldsymbol{\theta}$（再解一个积分） |
| 优势 | 计算简单 | 有原则地整合先验知识、量化不确定性 |

#### 实践中的挑战

贝叶斯推断在数学上有原则，但面临计算挑战：如果不选择共轭先验（第6.6.1节），式(8.22)和(8.23)中的积分在解析上不可处理。此时需要近似方法：

- **随机近似**：马尔可夫链蒙特卡洛（MCMC）
- **确定性近似**：拉普拉斯近似、变分推断、期望传播

尽管如此，贝叶斯推断已成功应用于大规模主题建模、点击率预测、数据高效强化学习、在线排名系统、推荐系统、贝叶斯优化等领域。




### 8.4.3 贝叶斯推断案例

我们沿用 8.3.3 节完全相同的问题设定（线性回归 + 高斯似然 + 高斯先验），但这次不求点估计，而是求参数的**完整后验分布**，并用它做预测。

#### 问题设定（与 8.3.3 相同）

- 数据：$\{(\boldsymbol{x}_n, y_n)\}_{n=1}^N$，$\boldsymbol{x}_n \in \mathbb{R}^3$，$y_n \in \mathbb{R}$
- 似然：$p(y_n \mid \boldsymbol{x}_n, \boldsymbol{\theta}) = \mathcal{N}(y_n \mid \boldsymbol{x}_n^\top\boldsymbol{\theta},\; \sigma^2)$
- 先验：$p(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta} \mid \boldsymbol{0},\; \sigma_0^2\boldsymbol{I})$

#### 第一步：写出后验分布

由于高斯似然 × 高斯先验仍然是高斯分布（共轭性，见第 6.6.1 节），后验分布有解析形式：

$$
p(\boldsymbol{\theta} \mid \mathcal{X}, \mathcal{Y}) = \mathcal{N}(\boldsymbol{\theta} \mid \boldsymbol{\mu}_{\text{post}},\; \boldsymbol{\Sigma}_{\text{post}})
$$

其中：

$$
\boldsymbol{\Sigma}_{\text{post}} = \left(\frac{1}{\sigma^2}\boldsymbol{X}^\top\boldsymbol{X} + \frac{1}{\sigma_0^2}\boldsymbol{I}\right)^{-1}
$$

$$
\boldsymbol{\mu}_{\text{post}} = \boldsymbol{\Sigma}_{\text{post}} \left(\frac{1}{\sigma^2}\boldsymbol{X}^\top\boldsymbol{y}\right)
$$

> **注意**：后验均值 $\boldsymbol{\mu}_{\text{post}}$ 与 MAP 点估计 $\boldsymbol{\theta}_{\text{MAP}}$ 数值上相同（这是高斯后验的特殊性质——众数 = 均值）。但贝叶斯推断还额外给出了后验协方差 $\boldsymbol{\Sigma}_{\text{post}}$，它量化了我们对参数估计的不确定性。

#### 第二步：贝叶斯预测

对于新输入 $\boldsymbol{x}_*$，点估计方法直接给出 $\hat{y}_* = \boldsymbol{x}_*^\top \boldsymbol{\theta}^*$。而贝叶斯方法对所有可能的 $\boldsymbol{\theta}$ 做积分：

$$
p(y_* \mid \boldsymbol{x}_*, \mathcal{X}, \mathcal{Y}) = \int p(y_* \mid \boldsymbol{x}_*, \boldsymbol{\theta})\, p(\boldsymbol{\theta} \mid \mathcal{X}, \mathcal{Y})\, d\boldsymbol{\theta}
$$

由于被积函数是两个高斯的乘积，积分结果仍为高斯：

$$
p(y_* \mid \boldsymbol{x}_*, \mathcal{X}, \mathcal{Y}) = \mathcal{N}\left(y_* \;\middle|\; \boldsymbol{x}_*^\top \boldsymbol{\mu}_{\text{post}},\; \sigma^2 + \boldsymbol{x}_*^\top \boldsymbol{\Sigma}_{\text{post}}\, \boldsymbol{x}_*\right)
$$

预测分布的方差由两部分组成：

| 来源 | 对应项 | 含义 |
|---|---|---|
| 观测噪声 | $\sigma^2$ | 即使知道真实参数，数据本身也有随机波动 |
| 参数不确定性 | $\boldsymbol{x}_*^\top \boldsymbol{\Sigma}_{\text{post}}\, \boldsymbol{x}_*$ | 我们对参数的估计不够确定所带来的额外不确定性 |

#### 第三步：与点估计对比

| | MLE / MAP | 贝叶斯推断 |
|---|---|---|
| 参数估计 | 单个值 $\boldsymbol{\theta}^*$ | 完整分布 $\mathcal{N}(\boldsymbol{\mu}_{\text{post}}, \boldsymbol{\Sigma}_{\text{post}})$ |
| 预测均值 | $\boldsymbol{x}_*^\top \boldsymbol{\theta}^*$ | $\boldsymbol{x}_*^\top \boldsymbol{\mu}_{\text{post}}$（与 MAP 相同） |
| 预测方差 | 仅 $\sigma^2$（固定噪声） | $\sigma^2 + \boldsymbol{x}_*^\top \boldsymbol{\Sigma}_{\text{post}}\, \boldsymbol{x}_*$（更大、更诚实） |
| 数据稀疏区域 | 仍然自信地给出预测 | 预测方差增大，表示"我不确定" |
| 数据 $N \to \infty$ | — | $\boldsymbol{\Sigma}_{\text{post}} \to \boldsymbol{0}$，退化为点估计 |

**一句话总结**：贝叶斯推断的预测均值和 MAP 一样好，但它额外提供了**校准过的不确定性**——在数据充足的区域给出自信预测，在数据稀疏的区域诚实地说"我不知道"。



### 8.4.4 隐变量模型

#### 为什么需要隐变量？

在实际应用中，除了模型参数 $\boldsymbol{\theta}$ 外，引入额外的**隐变量** $\boldsymbol{z}$ 作为模型的一部分是有用的。隐变量与模型参数不同——它们不显式地对模型进行参数化，而是描述数据的生成过程。

引入隐变量的好处：
- 提高模型的**可解释性**（描述数据生成过程）
- 简化模型结构，减少参数数量
- 使模型能够表达更丰富的分布族

典型的隐变量模型包括：

| 模型 | 用途 | 章节 |
|---|---|---|
| 主成分分析（PCA） | 降维 | 第10章 |
| 高斯混合模型（GMM） | 密度估计/聚类 | 第11章 |
| 隐马尔可夫模型（HMM） | 时间序列建模 | — |
| 动态系统 | 状态估计 | — |

#### 隐变量模型的生成过程

用 $\boldsymbol{x}$ 表示数据，$\boldsymbol{\theta}$ 表示模型参数，$\boldsymbol{z}$ 表示隐变量，条件分布为：

$$p(\boldsymbol{x}\mid\boldsymbol{z},\boldsymbol{\theta}) \tag{8.24}$$

对隐变量放置先验 $p(\boldsymbol{z})$，然后通过边缘化得到似然：

$$p(\boldsymbol{x}\mid\boldsymbol{\theta})=\int p(\boldsymbol{x}\mid\boldsymbol{z},\boldsymbol{\theta})p(\boldsymbol{z})\mathrm{d}\boldsymbol{z}\:, \tag{8.25}$$

这个似然不再依赖于隐变量 $\boldsymbol{z}$，只是数据 $\boldsymbol{x}$ 和模型参数 $\boldsymbol{\theta}$ 的函数。有了它，就可以直接使用 MLE、MAP 或贝叶斯推断来估计参数。

#### 隐变量模型中的推断

**参数后验**（给定数据，推断模型参数）：

$$p(\boldsymbol{\theta}\mid\mathcal{X})=\frac{p(\mathcal{X}\mid\boldsymbol{\theta})p(\boldsymbol{\theta})}{p(\mathcal{X})} \tag{8.26}$$

**隐变量后验**（给定数据和参数，推断隐变量）：

$$p(\boldsymbol{z}\mid\mathcal{X},\boldsymbol{\theta})=\frac{p(\mathcal{X}\mid\boldsymbol{z},\boldsymbol{\theta})p(\boldsymbol{z})}{p(\mathcal{X}\mid\boldsymbol{\theta})}\:, \tag{8.27}$$

其中分母 $p(\mathcal{X}\mid\boldsymbol{\theta})$ 由式 (8.25) 给出。

> **计算挑战：** 除非选择 $p(\boldsymbol{x}\mid\boldsymbol{z},\boldsymbol{\theta})$ 的共轭先验 $p(\boldsymbol{z})$，否则式 (8.25) 中的边缘化在解析上不可处理，需要求助于近似方法。同时边缘化隐变量和模型参数在一般情况下也不可能解析完成。

> **备注：** 在后续章节中，我们有时不严格区分"隐变量 $\boldsymbol{z}$"和"模型参数 $\boldsymbol{\theta}$"，因为两者都是不可观测的。原则上，对任何参数设置先验并积分，就将其转变为随机变量。在第10章和第11章中将同时出现两种隐藏变量。

我们可以利用概率模型中所有元素都是随机变量这一事实，来定义一种统一的图形化表示语言。在第 8.5 节中，我们将介绍有向图模型，用于简洁地描述概率模型的结构。



### 8.4.5 进一步阅读

概率模型的综合参考文献包括 Bishop (2006)、Barber (2012)、Murphy (2012)。主要计算方法有：
- **采样方法**：MCMC（Gilks et al., 1996; Brooks et al., 2011）
- **变分推断**：Jordan et al. (1999); Blei et al. (2017)
- **隐变量模型中的贝叶斯推断**：Moustaki et al. (2015); Paquet (2008)

近年来，**概率编程**（Probabilistic Programming）成为一个快速发展的方向：将程序中的变量视为随机变量，由编译器自动处理贝叶斯推断规则，使用户能够便捷地编写概率模型。


## 8.5 有向图模型

有向图模型（贝叶斯网络）是一种用图形语言描述概率模型结构的方法，能紧凑地表示随机变量之间的依赖关系。

概率图模型的优势：
- 可视化概率模型结构
- 启发新型统计模型设计
- 通过检查图形直接洞察条件独立性等属性
- 将推断和学习的复杂计算表达为图形操作

### 8.5.1 图形语义

在有向图模型中，**节点**代表随机变量，**有向边（箭头）** 代表条件概率关系。例如从 $a$ 指向 $b$ 的箭头表示 $p(b\mid a)$。

**从联合分布构造图模型**的步骤：
1. 为所有随机变量创建节点
2. 对于每个条件分布，从其所依赖的变量节点出发添加有向箭头

> **示例**：联合分布
> $$p(a,b,c)=p(c\mid a,b)\,p(b\mid a)\,p(a) \tag{8.28}$$
> 表明 $c$ 依赖于 $a,b$；$b$ 依赖于 $a$；$a$ 独立。

**从图模型读取联合分布**：联合分布等于各节点条件概率的乘积，每个节点仅以其父节点为条件。一般形式为

$$p(\boldsymbol{x})=\prod_{k=1}^K p(x_k\mid \text{Pa}_k) \tag{8.29}$$

其中 $\text{Pa}_k$ 表示 $x_k$ 的父节点集合。

**观测与隐变量的图形表示**：观测变量用阴影节点表示；当多个变量独立同分布时，可使用**板块符号**（plate notation）将重复结构压缩在一个矩形框内，框内变量重复 $N$ 次。

### 8.5.2 条件独立性与 d-分离

**d-分离**（d-separation）是仅通过查看图形即可判断条件独立性的关键工具。

给定不相交节点集 $\mathcal{A}$、$\mathcal{B}$、$\mathcal{C}$，判断

$$\mathcal{A}\perp\mathcal{B}\mid\mathcal{C} \tag{8.30}$$

的方法：考虑从 $\mathcal{A}$ 到 $\mathcal{B}$ 的所有路径（忽略箭头方向），若路径上存在节点使下列任一条件成立，则该路径**被阻塞**：

- **头到尾或尾到尾**：箭头在该节点汇聚为链式或发散式，且该节点在 $\mathcal{C}$ 中
- **头到头（对撞）**：箭头在该节点对撞（v-结构），且该节点及其所有后代均**不在** $\mathcal{C}$ 中

若所有路径均被阻塞，则 $\mathcal{A}$ 与 $\mathcal{B}$ 被 $\mathcal{C}$ d-分离，联合分布满足 $\mathcal{A}\perp\mathcal{B}\mid\mathcal{C}$。

### 8.5.3 拓展阅读

概率图模型主要分为三类：
- **有向图模型**（贝叶斯网络）
- **无向图模型**（马尔可夫随机场）
- **因子图**

图模型支持基于图的推理算法（如消息传递），广泛应用于计算机视觉、编码理论、信号处理、因果推断等领域。







## 8.6 模型选择

模型选择关注的核心问题是：如何在模型复杂度与数据拟合度之间取得平衡，以获得良好的泛化性能。更复杂的模型更具表达力，但更易过拟合；我们需要机制来评估模型对未见数据的泛化能力。这一思想也被称为**奥卡姆剃刀原则**——在能合理解释数据的前提下，优先选择最简单的模型。

### 8.6.1 嵌套交叉验证

在第 8.2.4 节交叉验证的基础上，嵌套交叉验证引入两层循环：

- **内层循环**：在验证集上估计特定模型/超参数选择的性能，近似泛化误差的期望

$$\mathbb{E}_{\mathcal{V}}[\boldsymbol{R}(\mathcal{V}\mid M)]\approx\frac{1}{K}\sum_{k=1}^{K}\boldsymbol{R}(\mathcal{V}^{(k)}\mid M) \tag{8.31}$$

其中 $\boldsymbol{R}(\mathcal{V}\mid M)$ 是模型 $M$ 在验证集 $\mathcal{V}$ 上的经验风险。对所有候选模型重复此过程，选择最优模型。

- **外层循环**：在测试集上估计内层选出的最佳模型的泛化性能。

用于选择最佳模型的数据称为**验证集**，用于估计泛化性能的数据称为**测试集**。

### 8.6.2 贝叶斯模型选择

贝叶斯框架下的模型选择自动体现了奥卡姆剃刀原则（"自动奥卡姆剃刀"）。其核心直觉：简单模型只能预测少量数据集，但在其适用范围内预测能力更强；复杂模型覆盖面广但单点预测力被稀释。

考虑有限模型集 $M=\{M_1,\ldots,M_K\}$，每个模型 $M_k$ 拥有参数 $\theta_k$。贝叶斯模型选择的层次生成过程为

$$
M_k\sim p(M),\quad \theta_k\sim p(\theta\mid M_k),\quad \mathcal{D}\sim p(\mathcal{D}\mid\theta_k) \tag{8.32}
$$

给定训练集 $\mathcal{D}$，模型上的后验为

$$
p(M_k\mid\mathcal{D})\propto p(M_k)\,p(\mathcal{D}\mid M_k) \tag{8.33}
$$

其中**模型证据**（边缘似然）定义为

$$
p(\mathcal{D}\mid M_k)=\int p(\mathcal{D}\mid\boldsymbol{\theta}_k)\,p(\boldsymbol{\theta}_k\mid M_k)\,\mathrm{d}\boldsymbol{\theta}_k \tag{8.34}
$$

模型参数 $\theta_k$ 在此已被积分掉。最优模型的 MAP 估计为

$$
M^*=\arg\max_{M_k} p(M_k\mid\mathcal{D}) \tag{8.35}
$$

若模型先验均匀，则 MAP 估计等价于选择证据最大的模型。

**似然 vs 边缘似然**：似然易过拟合，而边缘似然因为参数已被边缘化，自动在模型复杂度和数据拟合度之间进行权衡。

### 8.6.3 贝叶斯因子

比较两个模型 $M_1,M_2$ 时，后验比率可分解为

$$
\underbrace{\frac{p(M_1\mid\mathcal{D})}{p(M_2\mid\mathcal{D})}}_{\text{后验比率}}=\underbrace{\frac{p(M_1)}{p(M_2)}}_{\text{先验比率}}\;\underbrace{\frac{p(\mathcal{D}\mid M_1)}{p(\mathcal{D}\mid M_2)}}_{\text{贝叶斯因子}} \tag{8.36}
$$

**贝叶斯因子**衡量数据被 $M_1$ 预测得比 $M_2$ 好多少。若模型先验均匀，后验比率即为贝叶斯因子；贝叶斯因子 $>1$ 则选 $M_1$，否则选 $M_2$。

计算边缘似然通常需要求解不可解析的积分 (8.34)，实践中常用数值积分、蒙特卡洛方法或共轭先验等近似/特殊技巧。

### 8.6.4 信息准则

当使用最大似然估计时，可用信息准则在数据拟合与模型复杂度间做权衡：

**赤池信息准则（AIC）**：

$$
\text{AIC}=\log p(\boldsymbol{x}\mid\boldsymbol{\theta})-M \tag{8.37}
$$

其中 $M$ 为模型参数数量，惩罚项校正过拟合偏差。

**贝叶斯信息准则（BIC）**：

$$
\text{BIC}\approx\log p(\boldsymbol{x}\mid\boldsymbol{\theta})-\frac{1}{2}M\log N \tag{8.38}
$$

其中 $N$ 为数据点数量。BIC 对模型复杂度的惩罚比 AIC 更重。两者均选择值最大的模型。







