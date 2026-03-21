# Probability and Distribution(概率与分布)

## 6.1 概率空间的构建

概率论旨在定义一个数学结构来描述实验结果的随机性。利用这种概率的数学结构，目标是进行自动化推理，从这个意义上说，概率是对逻辑推理的泛化（Jaynes, 2003）。


### 哲学问题

概率的哲学基础以及它应该如何以某种方式（Jaynes, 2003）E. T. Jaynes（1922-1998）确定了三个数学标准，这些标准必须适用于所有可能性：

1. 可能性的程度由实数表示。

2. 这些数字必须基于常识的规则。

3. 所得推理必须是一致的，其中“一致”一词包含以下三层含义：

   (a) 一致性或无矛盾性：当可以通过不同方式达到相同结果时，在所有情况下都必须找到相同的可能性值。
   (b) 诚实性：必须考虑所有可用数据。
   (c) 可再现性：如果我们对两个问题的知识状态相同，那么我们必须为它们分配相同程度的可能性。


在机器学习和统计学中，概率有两种主要解释：**贝叶斯解释**和**频率解释**
- **贝叶斯解释**使用概率来指定用户对某个事件发生的不确定性程度。它有时被称为“主观概率”或“信念程度”。
- **频率解释**则考虑感兴趣事件相对于发生事件总数的相对频率。当数据无限时，某事件的概率被定义为该事件的相对频率。



### 概率与随机变量

现代概率论基于Kolmogorov提出的一组公理（Grinstead and Snell, 1997; Jaynes, 2003），这些公理引入了**样本空间**、**事件空间**和**概率测度**这三个概念。概率空间模型用于模拟具有随机结果的现实世界过程（称为实验）。

1. **样本空间 $\Omega$**. 样本空间是实验所有可能结果的集合，通常表示为 $\Omega$。
   > 例如，连续两次抛硬币的样本空间为 $\{hh, tt, ht, th\}$，其中“h”表示“正面”，“t”表示“反面”。
2. **事件空间 A**. 事件空间是实验潜在结果的集合。
   > 如果实验结束时我们可以观察到某个特定结果 $\omega\in\Omega$ 是否在 $A$ 中，则样本空间 $\Omega$ 的子集 $A$ 就属于事件空间 $\mathcal{A}$。事件空间 $A$ 是通过考虑 $\Omega$ 的子集集合获得的，对于离散概率分布（第6.2.1节），$\mathcal{A}$ 通常是 $\Omega$ 的幂集。
3. **概率方程 $P$**. 一个映射或者说函数，对于每个事件 $A\in\mathcal{A}$，我们关联一个数 $P(A)$，它衡量了事件发生的概率或信念程度。
   > $P(A)$ 被称为 $A$ 的概率。

**随机变量**的定义，对任意实数 $x$，$\{ \omega: X(\omega) \leq x \} \in \mathcal{A}$ 的实值函数为随机变量。首先，**随机变量是个函数**，一般用大写字母表示 $X$，通过这样的映射 $X$ 把一个事件 $\omega$ 变成一个实数 $x$ ，然后我们又可以通过概率空间，得到不同事件发生的概率 $0 \leq P(X(\omega)) \leq 1$。

[随机变量](https://zhuanlan.zhihu.com/p/150295256)



### 统计学

**概率论(Probability theory)** 和 **统计学(Statistics)** 经常被放在一起讨论，但它们关注的是 **不确定性的不同方面**。对比它们的一种方式是考虑所研究的问题类型。
- 使用**概率论**，我们可以考虑某个过程的模型，其中潜在的不确定性通过随机变量来捕捉，并利用概率规则来推导出所发生的事情。
- 在**统计学**中，我们观察到某件事情已经发生，并试图找出解释这些观察结果的潜在过程。

机器学习的目标更接近统计学，即构建一个能够充分表示数据生成过程的模型。我们可以利用概率规则来获得某些数据的“最佳拟合”模型。




## 6.2 离散概率与连续概率

### 离散概率

当目标空间是离散的时，我们可以将多个随机变量的概率分布想象为填充一个（多维）数字数组。图6.2给出了一个示例。联合概率的目标空间是每个随机变量目标空间的笛卡尔积。我们将联合概率定义为两个值共同出现的条目
$$P(X=x_i,Y=y_j)=\frac{n_{ij}}{N}\:, \tag{6.9}$$

其中 $n_{ij}$ 是状态为 $x_i$ 和 $y_j$ 的事件数，$N$ 是事件的总数。**联合概率**是两个事件交集的概率，即 $P(X=x_i, Y=y_j) = P(X=x_i \cap Y=y_j)$。

图6.2展示了离散概率分布的**概率质量函数（PMF）**。对于两个随机变量 $X$ 和 $Y$，$X=x$ 且 $Y=y$ 的概率（简略地）写为 $p(x,y)$，并称为联合概率。我们可以将概率视为一个函数，它接受状态 $x$ 和 $y$ 并返回一个实数，这就是我们写 $p(x,y)$ 的原因。无论随机变量 $Y$ 的值如何，$X$ 取值 $x$ 的边缘概率（简略地）写为 $p(x)$。

我们用 $X\sim p(x)$ 来表示随机变量 $X$ 根据 $p(x)$ 分布。如果我们只考虑 $X=x$ 的情况，那么 $Y=y$ 的实例比例（条件概率）简略地写为 $p(y \mid x)$。

![](https://datawhalechina.github.io/math-for-ai/attachments/6.2.png)

<center>图6.2具有随机变量X和y的离散二变量概率质量函数的可视化。此图改编自Bishop（2006）。</center>




### 连续概率

**定义6.1（概率密度函数）**。如果函数$f:\mathbb{R}^D\to\mathbb{R}$满足以下条件，则称为**概率密度函数（PDF）**：

1. $\forall x\in \mathbb{R} ^D: f( \boldsymbol{x}) \geqslant 0$
2. 其积分存在，且

$$
\int_{\mathbb{R}^D}f(\boldsymbol{x})\mathrm{d}\boldsymbol{x}=1\:. \tag{6.15}
$$

对于离散随机变量的**概率质量函数（PMF）**，(6.15)中的积分被替换为求和(6.12)。

请注意，概率密度函数是任何非负且积分为1的函数。我们通过以下方式将随机变量$X$与该函数$f$相关联：
$$
P(a\leqslant X\leqslant b)=\int_{a}^{b}f(x)\mathrm{d}x\:, \tag{6.16}
$$

其中$a,b\in\mathbb{R}$且$x\in\mathbb{R}$是连续随机变量$X$的结果。通过考虑向量$x\in\mathbb{R}$，类似地定义$x\in\mathbb{R}^D$的状态。这种关联(6.16)称为随机变量$X$的概率法或分布。

**备注**。与离散随机变量不同，连续随机变量$X$取特定值$x$的概率$P(X=x)$为零。这就像在(6.16)中尝试指定一个区间，其中$a=b$。





**定义6.2（累积分布函数）**。具有状态$x\in\mathbb{R}^D$的多元实值随机变量$X$的累积分布函数（CDF）由下式给出：

$$
F_X(\boldsymbol{x})=P(X_1\leqslant x_1,\ldots,X_D\leqslant x_D)\:, \tag{6.17}
$$

其中$X=[X_1,\ldots,X_D]^\top$，$\boldsymbol{x}=[x_1,\ldots,x_D]^\top$，且右侧表示随机变量$X_i$取值小于或等于$x_i$的概率。

**累积分布函数（CDF）**也可以表示为概率密度函数$f(x)$的积分，即
$$
F_{X}(\boldsymbol{x})=\int_{-\infty}^{x_{1}}\cdots\int_{-\infty}^{x_{D}}f(z_{1},\ldots,z_{D})\mathrm{d}z_{1}\cdots\mathrm{d}z_{D}\:. \tag{6.18}
$$

**备注**。我们重申，在讨论分布时，实际上有两个不同的概念。第一个是pdf（用$f(x)$表示），它是一个非负且积分为1的函数。第二个是随机变量$\bar{X}$的法则，即将随机变量$X$与pdf $f(x)$相关联。

![](https://datawhalechina.github.io/math-for-ai/attachments/6.3.png)
<center>图6.3(a)离散分布和(b)连续均匀分布的例子。有关分布的详细信息，请参见示例6.3。</center>





### 离散分布与连续分布的对比

> **例6.3**
>
> 我们考虑均匀分布的两个例子，其中每个状态发生的可能性都相等。这个例子说明了离散概率分布和连续概率分布之间的一些差异。
>
> 设$Z$是一个具有三个状态$\{z=-1.1, z=0.3, z=1.5\}$的离散均匀随机变量。其概率质量函数可以用概率值的表格来表示：
>
> ![1723859872288](https://datawhalechina.github.io/math-for-ai/attachments/1723859872288.png)
>
> 或者，我们可以将其视为一个图形（图6.3(a)），其中我们使用了一个事实，即状态可以位于$x$轴上，而$y$轴表示特定状态的概率。图6.3(a)中的$y$轴被故意延长，以便与图6.3(b)中的$y$轴相同。
>
> 设$X$是一个在范围$0.9\leqslant X\leqslant1.6$内取值的连续随机变量，如图6.3(b)所示。请注意，密度的高度可以大于1。但是，它必须满足
>
> $$\int_{0.9}^{1.6}p(x)\mathrm{d}x=1\:. \tag{6.19}$$


![1723859895678](https://datawhalechina.github.io/math-for-ai/attachments/1723859895678.png)
<center>表6.1：概率分布的命名法。</center>





## 6.3 加法规则, 乘法规则与贝叶斯公式

### 加法规则

$$
p(\boldsymbol{x})=\left\{\begin{array}{ll}\displaystyle\sum_{\boldsymbol{y}\in\mathcal{Y}}p(\boldsymbol{x},\boldsymbol{y})&\quad\text{如果}\:\boldsymbol{y}\:\text{是离散的}\\\\\displaystyle\int_{\mathcal{Y}}p(\boldsymbol{x},\boldsymbol{y})\mathrm{d}\boldsymbol{y}&\quad\text{如果}\:\boldsymbol{y}\:\text{是连续的}\end{array}\right., \tag{6.20}
$$


其中$\mathcal{Y}$是随机变量$Y$的目标空间的状态。这意味着我们对随机变量$Y$的状态集$y$进行求和（或积分）。加法规则也被称为边缘化属性。加法规则将联合分布与边缘分布联系起来。一般来说，当联合分布包含两个以上的随机变量时，加法规则可以应用于随机变量的任何子集，从而得到可能包含多个随机变量的边缘分布。




### 乘法规则

它通过以下方式将联合分布与条件分布联系起来：

$$
p(\boldsymbol{x},\boldsymbol{y})=p(\boldsymbol{y}\mid\boldsymbol{x})p(\boldsymbol{x})\:, \tag{6.22}
$$

乘法规则可以理解为，任何两个随机变量的联合分布都可以分解为（写成乘积形式）另外两个分布。这两个因子分别是第一个随机变量的边缘分布$p(x)$，以及给定第一个随机变量时第二个随机变量的条件分布$p(\boldsymbol{y}\mid\boldsymbol{x})$。由于在$p(x,y)$中随机变量的顺序是任意的，乘法规则也意味着$p(\boldsymbol{x},\boldsymbol{y})=p(\boldsymbol{x}\mid\boldsymbol{y})p(\boldsymbol{y})$。




### 贝叶斯定理

**贝叶斯定理（也称为贝叶斯规则或贝叶斯定律）**: 在机器学习和贝叶斯统计中，我们经常在观察到其他随机变量的情况下，对未观察到的（潜在的）随机变量进行推断。假设我们对一个未观察到的随机变量$x$有一些先验知识$p(x)$，以及$x$与我们可以观察到的第二个随机变量$y$之间的某种关系$p(\boldsymbol{y}\mid\boldsymbol{x})$。如果我们观察到了$y$，我们可以使用贝叶斯定理根据观察到的$y$的值来得出关于$x$的一些结论。



$$
\underbrace{p(\boldsymbol{x}\mid\boldsymbol{y})}_{\text{后验}}=\frac{\overbrace{p(\boldsymbol{y}\mid\boldsymbol{x})}^{\text{似然度}}\overbrace{p(\boldsymbol{x})}^{\text{先验}}}{\underbrace{p(\boldsymbol{y})}_{\text{证据}}}
\tag{6.23}
$$

式（6.22）中乘法规则的直接结果，因为

$$
p(\boldsymbol{x},\boldsymbol{y})=p(\boldsymbol{x}\mid\boldsymbol{y})p(\boldsymbol{y}) \tag{6.24}
$$

以及

$$
p(\boldsymbol{x},\boldsymbol{y})=p(\boldsymbol{y}\mid\boldsymbol{x})p(\boldsymbol{x}) 
\tag{6.25}
$$

所以

$$
p(\boldsymbol{x}\mid\boldsymbol{y})p(\boldsymbol{y})=p(\boldsymbol{y}\mid\boldsymbol{x})p(\boldsymbol{x})\iff p(\boldsymbol{x}\mid\boldsymbol{y})=\frac{p(\boldsymbol{y}\mid\boldsymbol{x})p(\boldsymbol{x})}{p(\boldsymbol{y})}\:. \tag{6.26}
$$

在（6.23）中，$p(x)$是先验，它包含了我们在观察到任何数据之前对未观察到的（潜在的）变量$x$的主观先验知识。我们可以选择任何对我们有意义的先验，但至关重要的是要确保先验在所有可能的$x$上都有非零的概率密度函数（或概率质量函数），即使它们非常罕见。

**似然度$p(\boldsymbol{y}\mid x)$** 描述了$x$和$y$之间的关系，在离散概率分布的情况下，它是如果我们知道潜在变量$x$，则数据$y$出现的概率。请注意，似然度有时并不被视为$x$上的分布，而只是$y$上的分布（MacKay, 2003）。

**后验$p(x\mid y)$** 是贝叶斯统计中我们感兴趣的量，因为它准确地表达了我们所关心的内容，即观察到$y$之后我们对$x$的了解。

$$
p(\boldsymbol{y}):=\int p(\boldsymbol{y}\mid\boldsymbol{x})p(\boldsymbol{x})\mathrm{d}\boldsymbol{x}=\mathbb{E}_X[p(\boldsymbol{y}\mid\boldsymbol{x})] \tag{6.27}
$$

是边缘似然/证据。式(6.27)的右侧使用了期望算子，我们将在第6.4.1节中定义它。根据定义，边缘似然是对(6.23)式的分子关于隐变量$x$的积分。因此，边缘似然与$x$无关，并且它确保了后验$p(\boldsymbol{x}\mid\boldsymbol{y})$是归一化的。边缘似然也可以被解释为在先验$p(x)$下的期望似然。





## 6.4 汇总统计量与独立性

### 均值与协方差

均值和（协）方差通常用于描述概率分布的性质（期望值和离散程度）。

**定义 6.3（期望值）**：对于单变量连续随机变量$X\sim p(x)$的函数$g: \mathbb{R} \to \mathbb{R}$，其期望值定义为

$$
\operatorname{E}_X[g(x)]=\int_{\mathcal{X}}g(x)p(x)\mathrm{d}x \tag{6.28}
$$

相应地，对于离散随机变量$X\sim p(x)$的函数$g$，其期望值定义为

$$
\operatorname{E}_{X}[g(x)]=\sum_{x\in\mathcal{X}}g(x)p(x) \tag{6.29}
$$

其中，$X$是随机变量$X$所有可能结果（目标空间）的集合。

在本节中，我们认为离散随机变量的结果是数值型的。这可以通过观察函数$g$以实数作为输入来看出。

**备注**：我们将多元随机变量$X$视为单变量随机变量$[X_1,\ldots,X_D]^\top$的有限向量。对于多元随机变量，我们逐元素地定义期望值



**定义 6.3**定义了符号$\mathbb{E}_X$的含义，作为指示我们应对概率密度（对于连续分布）或对所有状态求和（对于离散分布）取积分的算子。均值的定义（定义6.4）是期望值的一个特例，通过选择$g$为恒等函数获得。


**定义 6.4（均值）**：随机变量$X$，其状态$x\in\mathbb{R}^D$，的均值是一个平均值，定义为

$$
\operatorname{E}_X[\boldsymbol{x}]=\begin{bmatrix}\operatorname{E}_{X_1}[x_1]\\\vdots\\\operatorname{E}_{X_D}[x_D]\end{bmatrix}\in\mathbb{R}^D
\tag{6.31}
$$

其中

$$
\operatorname{E}_{X_d}[x_d]:=\left\{\begin{array}{ll}\int_{\mathcal{X}}x_dp(x_d)\mathrm dx_d&\text{如果}X\text{是连续随机变量}\\\sum_{x_i\in\mathcal{X}}x_ip(x_d=x_i)&\text{如果}X\text{是离散随机变量}\end{array}\right.
\tag{6.32}
$$

对于$d=1,\ldots,D$，下标$d$表示$x$的相应维度。积分和求和是针对随机变量$X$的目标空间状态$\chi$进行的。




> **例 6.4**
>
> 考虑图 6.4 中所示的二维分布：
>
> ![1723861918526](https://datawhalechina.github.io/math-for-ai/attachments/6.4.png)
>
> <center>图6.4一个二维数据集的平均值、模式和中位数及其边缘密度的说明。</center>
>
> $$p(x)=0.4\mathcal{N}\left(\boldsymbol{x}\:\bigg|\begin{bmatrix}10\\2\end{bmatrix},\begin{bmatrix}1&0\\0&1\end{bmatrix}\right)+0.6\mathcal{N}\left(\boldsymbol{x}\:\bigg|\begin{bmatrix}0\\0\end{bmatrix},\begin{bmatrix}8.4&2.0\\2.0&1.7\end{bmatrix}\right).$$
>
> $(6.33)$
>
> 我们将在第 6.5 节中定义高斯分布 $\mathcal{N}(\mu,\sigma^2)$。同时，还展示了该分布在每个维度上的对应边缘分布。观察到该分布是双峰的（有两个众数），但其中一个边缘分布是单峰的（有一个众数）。水平方向上的双峰一元分布说明了均值和中位数可能彼此不同。尽管我们可能会想要将二维中位数定义为每个维度上中位数的串联，但由于我们无法定义二维点的顺序，这变得困难。当我们说“无法定义顺序”时，我们的意思是存在多种方式来定义关系 <，使得 $\begin{bmatrix}3\\0\end{bmatrix}<\begin{bmatrix}2\\3\end{bmatrix}$ 这样的关系不是唯一的。


> **备注**：期望值（定义 6.3）是一个线性算子。例如，给定一个实值函数 $f(\boldsymbol{x})=ag(\boldsymbol{x})+bh(\boldsymbol{x})$，其中 $a,b\in\mathbb{R}$ 且 $x\in\mathbb{R}^D$，我们得到
> $$
>\begin{aligned}\operatorname{E}_{X}[f(\boldsymbol{x})]&=\int f(\boldsymbol{x})p(\boldsymbol{x})\mathrm{d}\boldsymbol{x}\\&=\int[ag(\boldsymbol{x})+bh(\boldsymbol{x})]p(\boldsymbol{x})\mathrm{d}\boldsymbol{x}\\&=a\int g(\boldsymbol{x})p(\boldsymbol{x})\mathrm{d}\boldsymbol{x}+b\int h(\boldsymbol{x})p(\boldsymbol{x})\mathrm{d}\boldsymbol{x}\\&=a\operatorname{E}_{X}[g(\boldsymbol{x})]+b\operatorname{E}_{X}[h(\boldsymbol{x})]\:.\end{aligned}
>$$
> 对于两个随机变量，我们可能希望描述它们之间的对应关系。协方差直观地表示了随机变量之间依赖性的概念。



**定义 6.5（协方差（单变量））**：两个单变量随机变量 $X,Y\in\mathbb{R}$ 之间的协方差由它们各自偏离各自均值的乘积的期望值给出，即

$$
\mathrm{Cov}_{X,Y}[x,y]:=\mathrm{E}_{X,Y}\big[(x-\mathrm{E}_{X}[x])(y-\mathrm{E}_{Y}[y])\big].
\tag{6.35}
$$

> **备注**：当与期望值或多变量随机协方差相关的随机变量通过其参数明确时，下标通常会被省略（例如，E$_X[x]$ 通常简写为 E$[x])$。

通过使用期望的线性性质，定义 6.5 中的表达式可以重写为乘积的期望值减去期望值的乘积，即

$$
\mathrm{Cov}[x,y]=\mathrm{E}[xy]-\mathrm{E}[x]\mathrm{E}[y]\:. \tag{6.36}
$$

变量与其自身的协方差 Cov$[x,x]$ 称为**方差**，记作 $\mathcal{V}_X[x]$。方差的平方根称为**标准差**，通常记作 $\sigma(x)$。协方差的概念可以推广到多变量随机变量。



**定义 6.6（协方差（多变量））**：如果我们考虑两个多变量随机变量 $X$ 和 $Y$，其状态分别为 $x\in\mathbb{R}^D$ 和 $y\in\mathbb{R}^E$，则 $X$ 和 $Y$ 之间的协方差定义为

$$
\mathrm{Cov}[\boldsymbol{x},\boldsymbol{y}]=\mathrm{E}[\boldsymbol{x}\boldsymbol{y}^{\top}]-\mathrm{E}[\boldsymbol{x}]\mathrm{E}[\boldsymbol{y}]^{\top}=\mathrm{Cov}[\boldsymbol{y},\boldsymbol{x}]^{\top}\in\mathbb{R}^{D\times E}\:.
\tag{6.37}
$$

定义 6.6 可以应用于两个参数中的相同多变量随机变量，这导致了一个有用的概念，它直观地捕获了随机变量的“散布”。对于多变量随机变量，方差描述了随机变量各个维度之间的关系。




**定义 6.7（方差）**：随机变量 $X$ 的方差，其状态为 $x\in\mathbb{R}^D$，均值向量为 $\mu\in\mathbb{R}^D$，定义为

$$
\begin{aligned}\mathbb{V}_{X}[\boldsymbol{x}]&=\mathrm{Cov}_{X}[\boldsymbol{x},\boldsymbol{x}]\\&=\mathbb{E}_{X}[(\boldsymbol{x}-\boldsymbol{\mu})(\boldsymbol{x}-\boldsymbol{\mu})^{\top}]=\mathbb{E}_{X}[\boldsymbol{x}\boldsymbol{x}^{\top}]-\mathbb{E}_{X}[\boldsymbol{x}]\mathbb{E}_{X}[\boldsymbol{x}]^{\top}\\&=\begin{bmatrix}\mathrm{Cov}[x_1,x_1]&\mathrm{Cov}[x_1,x_2]&\ldots&\mathrm{Cov}[x_1,x_D]\\\mathrm{Cov}[x_2,x_1]&\mathrm{Cov}[x_2,x_2]&\ldots&\mathrm{Cov}[x_2,x_D]\\\vdots&\vdots&\ddots&\vdots\\\mathrm{Cov}[x_D,x_1]&\ldots&\ldots&\mathrm{Cov}[x_D,x_D]\end{bmatrix}.\end{aligned}
\tag{6.38}
$$

(6.38)中的$D\times D$矩阵被称为多元随机变量$X$的**协方差矩阵**。协方差矩阵是对称的且是半正定的，它向我们揭示了数据的分布情况。在其对角线上，协方差矩阵包含了边缘分布的方差

$$
p(x_i)=\int p(x_1,\ldots,x_D)\mathrm{d}x_{\setminus i}\:,\tag{6.39}
$$

其中“$\setminus i$”表示“除了变量$i$之外的所有变量”。非对角线上的元素是$i,j=1,\ldots,D,i\neq j$时的交叉协方差项$\text{Cov}[x_i,x_j]$。

![](https://datawhalechina.github.io/math-for-ai/attachments/6.5.png)

<center>图6.5二维数据集沿每个轴（彩色线）具有相同的均值和方差，但具有不同的协方差。</center>

> **备注**。在本书中，我们通常假设协方差矩阵是正定的，以便更好地理解。因此，我们不讨论导致半正定（低秩）协方差矩阵的特殊情况。

当我们想要比较不同随机变量对之间的协方差时，发现每个随机变量的方差都会影响协方差的值。协方差的归一化版本被称为相关系数。




**定义6.8（相关系数）**。两个随机变量$X,Y$之间的相关系数由

$$
\text{corr}[x,y]=\frac{\text{Cov}[x,y]}{\sqrt{\text{V}[x]\text{V}[y]}}\in[-1,1]\:.
\tag{6.40}
$$

相关系数矩阵是标准化随机变量$x/\sigma(x)$的协方差矩阵。换句话说，在相关系数矩阵中，每个随机变量都被其标准差（方差的平方根）除。

协方差（和相关系数）表明了两个随机变量之间的关系；见图6.5。正相关$\text{corr}[x,y]$意味着当$x$增长时，$y$也预期会增长。负相关则意味着当$x$增加时，$y$会减小。





### 经验均值和协方差

**定义6.9（经验均值和协方差）**。经验均值向量是每个变量观测值的算术平均值，定义为

$$
\bar{\boldsymbol{x}}:=\frac{1}{N}\sum_{n=1}^{N}\boldsymbol{x}_{n}\:, \tag{6.41}
$$

其中$x_n\in\mathbb{R}^D$。


**经验协方差矩阵**是一个$D\times D$矩阵

$$
\boldsymbol{\Sigma}:=\frac{1}{N}\sum_{n=1}^{N}(\boldsymbol{x}_{n}-\bar{\boldsymbol{x}})(\boldsymbol{x}_{n}-\bar{\boldsymbol{x}})^{\top}.
\tag{6.42}
$$

为了计算特定数据集的统计量，我们将使用实现（观测值）$x_1,\ldots,x_N$，并使用(6.41)和(6.42)。经验协方差矩阵是对称的、半正定的（见第3.2.3节）。






### 方差的三种表达式

**方差的标准定义**，对应于协方差（定义6.5）的定义，是随机变量$X$与其期望值$\mu$之间的平方偏差的期望值，即

$$
\mathrm V_X[x]:=\mathrm E_X[(x-\mu)^2]\:. \tag{6.43}
$$


在经验上估计(6.43)中的方差时, 可以将(6.43)中的公式转换为所谓的**方差原始分数公式**：

$$
\mathrm{V}_X[x]=\mathrm{E}_X[x^2]-\left(\mathrm{E}_X[x]\right)^2\:. \tag{6.44}
$$


**理解方差的第三种方式是**，它是所有观测对之间的成对差异之和。考虑随机变量$X$的实现的一个样本$x_1,\ldots,x_N$，我们计算每对$x_i$和$x_j$之间的平方差。通过展开平方，我们可以证明$N^2$个成对差异的总和是观测值的经验方差：

$$
\dfrac{1}{N^2}\sum_{i,j=1}^N(x_i-x_j)^2=2\left[\dfrac{1}{N}\sum_{i=1}^Nx_i^2-\left(\dfrac{1}{N}\sum_{i=1}^Nx_i\right)^2\right]\:. \tag{6.45}
$$




### 统计独立性

**定义6.10（独立性）**。两个随机变量$X,Y$是统计独立的当且仅当

$$
p(x,y)=p(x)p(y)\:. \tag{6.53}
$$

直观上，如果两个随机变量$X$和$Y$是独立的，那么知道$y$的值并不会给$x$提供任何额外的信息（反之亦然）。如果$X,Y$是（统计）独立的，那么

- $p(y\mid x) = p(y)$
- $p(x\mid y) = p(x)$
- $\mathrm{V}_{X,Y}[x+y]=\mathrm{V}_X[x]+\mathrm{V}_Y[y]$
- $\mathrm{Cov}_{X, Y}[x, y] = 0$

最后一点可能不总是成立的逆命题，即两个随机变量可以有协方差为零但并非统计独立。为了理解这一点，需要回顾协方差只衡量线性依赖关系。因此，非线性依赖的随机变量可能具有零协方差。

> **例6.5**
> 考虑一个均值为零的随机变量$X$（$\mathbb{E}_X[x]=0$）且
>
> $\mathbb{E}_X[x^3]=0$。令$y=x^2$（因此，$Y$依赖于$X$），并考虑$X$和$Y$之间的协方差（6.36）。但这给出
>
> $$
> \mathrm{Cov}[x,y]=\mathbb{E}[xy]-\mathbb{E}[x]\mathbb{E}[y]=\mathbb{E}[x^{3}]=0\:. \tag{6.54}
> $$




机器学习中另一个重要的概念是条件独立性。

**定义6.11（条件独立性）**。两个随机变量$X$和$Y$在给定$Z$的条件下是条件独立的当且仅当

$$
p(\boldsymbol{x},\boldsymbol{y}\mid\boldsymbol{z})=p(\boldsymbol{x}\mid\boldsymbol{z})p(\boldsymbol{y}\mid\boldsymbol{z})\quad\mathrm{for~all}\quad\boldsymbol{z}\in\mathcal{Z}\:, \tag{6.55}
$$

其中，$\mathcal{Z}$是随机变量$Z$的状态集。我们用$X\perp Y\mid Z$来表示给定$Z$时，$X$与$Y$是条件独立的。




### 随机变量的内积

回顾第3.2节中内积的定义。我们可以在随机变量之间定义内积，并在本节中简要描述。如果我们有两个不相关的随机变量$X,Y$，则

多变量随机变量可以

（6.58）此处原文似乎有误或遗漏，但基于上下文，我们可以理解为讨论的是不相关随机变量方差的可加性，即：

$$\mathrm{V}[X+Y]=\mathrm{V}[X]+\mathrm{V}[Y]\:.$$

由于方差是以平方单位衡量的，这看起来非常像直角三角形中的勾股定理$c^2=a^2+b^2$。接下来，我们探讨是否能为（6.58）中不相关随机变量的方差关系找到几何解释。

![](https://datawhalechina.github.io/math-for-ai/attachments/6.6.png)

<center>图6.6随机变量的几何形状。如果随机变量X和Y不相关，则它们是相应线性空间中的正交向量，并应用毕达哥拉斯定理。</center>

随机变量可以视为线性空间中的向量，我们可以定义内积以获得随机变量的几何性质（Eaton, 2007）。如果我们定义

（6.59）

$$\langle X,Y\rangle:=\mathrm{Cov}[X,Y]$$

对于均值为零的随机变量$X$和$Y$，我们得到了一个内积。可以看出，协方差是对称的、正定的，并且在任一参数上都是线性的。随机变量的“长度”是

$$\|X\|=\sqrt{\mathrm{Cov}[X,X]}=\sqrt{\mathrm{V}[X]}=\sigma[X]\:,$$

（6.60）

即其标准差。随机变量“越长”，其不确定性就越大；长度为0的随机变量是确定的。

如果我们查看两个随机变量$X,Y$之间的角度$\theta$，我们得到

（6.61）

$$\cos\theta=\frac{\langle X,Y\rangle}{\|X\|\:\|Y\|}=\frac{\mathrm{Cov}[X,Y]}{\sqrt{\mathrm{V}[X]\mathrm{V}[Y]}}\:,$$

这是两个随机变量之间的相关性（定义6.8）。这意味着，当我们从几何角度考虑时，可以将相关性视为两个随机变量之间角度的余弦值。根据定义3.7，我们知道$X\perp Y\Longleftrightarrow\langle X,Y\rangle=0$。在我们的情况下，这意味着$X$和$Y$是正交的当且仅当Cov$[X,Y]=0$，即它们是不相关的。图6.6说明了这种关系。

![](https://datawhalechina.github.io/math-for-ai/attachments/6.7.png)

<center>图6.7两个随机变量x1和x2的高斯分布。</center>







## 6.5 高斯分布








## 6.6 共轭性与指数族分布







## 6.7 变量变换/逆变换









