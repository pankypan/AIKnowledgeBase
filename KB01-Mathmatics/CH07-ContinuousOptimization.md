# Continuous Optimization(连续优化)

## 7.1 基于梯度下降的优化

### 梯度下降算法

现在考虑求解一个实值函数最小值的问题：
$$
\min\limits_{\boldsymbol{x}}~f(\boldsymbol{x}), \tag{7.4}
$$
其中 $f: \mathbb{R}^{d} \rightarrow \mathbb{R}$ 是一个函数，它刻画了我们手中的机器学习问题。我们假设函数 $f$ 是可微的，并且我们无法找到上述问题的解析解。

梯度下降是一个一阶优化算法。它的每次迭代都将估计点做一个正比于函数在该点处的负梯度向量的移动，以逐步找到一个局部最小值点。回顾第 5.1 节，梯度方向是函数值增长最快的方向。另一个有用的直观理解是考虑函数处于某个特定值处的那组线（即 $f(\boldsymbol{x})=c$ ，其中某个值 $c \in \mathbb{R}$ ），这些线被称为等高线。梯度方向与我们希望优化的函数的等高线方向正交。

让我们考虑多变量函数。想象一个曲面（由函数 $f(\boldsymbol{x})$ 描述），并设想一个球从某个特定位置 $\boldsymbol{x}_0$ 开始。当球被释放时，它会沿着最陡峭的下坡方向向下滚动。梯度下降利用了这样一个事实：从 $\boldsymbol{x}_0$ 出发，若朝着函数 $f$ 在 $\boldsymbol{x}_0$ 处负的梯度方向 $-\left((\nabla f)(\boldsymbol{x}_0)\right)^{\top}$ 移动，$f(\boldsymbol{x}_0)$ 的值将最快地减小。本书假设所涉及的函数都是可微的，并引导读者参考第 7.4 节中更一般的设置。于是假如我们考虑下面的更新：
$$
\boldsymbol{x}_{1} = \boldsymbol{x}_{0} - \gamma \big[ (\nabla f)(\boldsymbol{x}_{0}) \big] ^{\top} \tag{7.5}
$$
若 $\gamma \geqslant 0$ 是一个很小的 **步长**，就有 $f(\boldsymbol{x}_{1}) \leqslant f(\boldsymbol{x}_{0})$。注意我们在梯度的部分使用了转置记号，这是因为我们在本书中默认梯度是行向量——如果不转置的话维度对不上。

有了这个发现，我们就能提出一个简单的**梯度下降算法**：我们想要找到一个函数 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}, \boldsymbol{x} \mapsto f(\boldsymbol{x})$ 的局部最优解 $f(\boldsymbol{x}_{*})$ ，我们从一个初始估计 $\boldsymbol{x}_{0}$ 开始，然后按照下面的更新规则不断迭代
$$
\boldsymbol{x}_{i+1} = \boldsymbol{x}_{i} - \gamma_{i} \big[ (\nabla f)(\boldsymbol{x}_{i}) \big] ^{\top} \tag{7.6}
$$
假设我们每次迭代选择的步长足够合适，我们得到的序列就是一个下降的 “链”：$f(\boldsymbol{x}_{0}) \geqslant f(\boldsymbol{x}_{1}) \geqslant \cdots$ 它最终会趋于函数的局部最小值。


> **示例 7.1**
> 考虑下面的二维二次函数
> $$f\left(\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}\right) = \frac{1}{2}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}^{\top}\begin{bmatrix}2&1\\1&20\end{bmatrix}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix} - \begin{bmatrix}5\\3\end{bmatrix}^{\top}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}\tag{7.7}$$
> 它对 $\boldsymbol{x}$ 的梯度是 $$\nabla f\left(\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}\right) = \begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}^{\top}\begin{bmatrix}2&1\\1&20\end{bmatrix} - \begin{bmatrix}5\\3\end{bmatrix}^{\top}\tag{7.8}$$
> 如图 7.3 所示，我们从初始估计 $\boldsymbol{x}_{0} = [-3, -1]^{\top}$ 开始用公式 (7.6) 不断迭代，以得到一个收敛于函数最小值的估计值序列。可见 $\boldsymbol{x}_{0}$ 处的负梯度指向右上方，从而得到第二个估计 $\boldsymbol{x}_{1} = [-1.98, 1.21]^{\top}$ （令 $\gamma = 0.085$，并将 $\boldsymbol{x}_{0}$ 代入 (7.8) ）。再迭代一次，我们得到 $\boldsymbol{x}_{2} = [-1.32, -0.42]^{\top}$，以此类推。
> <center><img src="https://datawhalechina.github.io/math-for-ai/ch7/attachments/Pasted%20image%2020250630213059.png" alt="alt text" style="zoom:50%;"></center>
> <center>图 7.3 梯度下降算法的示例</center>


> **注释**
> 梯度下降算法趋近局部最小值的速度可以很慢，它的渐近收敛速度弱于很多其他算法。在面临一些性质不甚好的凸函数时，我们可以想象一个从很长但很窄的斜坡滚下的球：梯度下降的更新轨迹将会是像图 7.3 那样的锯齿形，每次更新的方向甚至会与该点与局部最小值点的直接连线几乎垂直。





**步长（学习率）**，前文提到，步长大小在梯度下降算法中十分重要：
- 如果步长太小，梯度下降的速度会很慢；
- 如果步长太大，梯度下降算法有可能射出原本的 “峡谷” 区域，难以收敛，甚至发散。





### 动量梯度下降

如图 7.3 所示，如果优化曲面的曲率使得某些区域的性质不好，梯度下降的收敛速度可能会非常慢。曲率使得梯度下降更新在 ”峡谷“ 两侧跳跃，只能一小步一小步地接近最优值。为提高收敛性，我们可以赋予梯度下降一些 "记忆"。


动量梯度下降（Rumelhart et al., 1986）是一种引入与上一次迭代的相关项的方法。这种记忆可以抑制振荡并使得梯度更新更加平滑。我们像之前一样考虑一个很重的滚动的球，动量项就模拟了它的惯性——很难轻易改变运动方向。这个方法也同时通过记忆梯度的更新以实现移动平均。

具体而言，基于动量的方法会储存第 $i$ 次迭代的更新 $\Delta \boldsymbol{x}_{i}$，然后加在第 $i+1$ 次的梯度更新上；这相当于将第 $i$ 次迭代和第 $i+1$ 次迭代中得到的梯度做线性组合：
$$
\begin{align}
\boldsymbol{x}_{i+1} &= \boldsymbol{x}_{i} - \gamma_{i} \big[ (\nabla f)(\boldsymbol{x}_{i}) \big] ^{\top} + \alpha\Delta \boldsymbol{x}_{i} \tag{7.11}\\
\Delta \boldsymbol{x}_{i} &= \boldsymbol{x}_{i} - \boldsymbol{x}_{i-1} = \alpha\Delta \boldsymbol{x}_{i-1} - \gamma_{i-1}\big[ (\nabla f)(\boldsymbol{x}_{i-1}) \big] ^{\top}, \tag{7.12}
\end{align}
$$
其中 $\alpha \in [0, 1]$。有时我们只知道梯度的一个估计值，此时上面的动量项作为移动平均会帮我们抹除梯度估计中的噪声，因此十分有用。




### 随机梯度下降
精确地计算梯度十分费时费力，但我们往往可以找到更快速地计算梯度估计值的方法 —— 只要我们估计的梯度和真实的梯度方向大致相同。

**随机梯度下降**（SGD）是一种用于最小化可被写成一系列可微函数的目标函数，并给出梯度的随机估计的梯度下降算法。”随机“ 一词指的是我们每次更新不知道梯度的真实值，而只有一个**带噪声的梯度估计值**。如果限制梯度估计值的分布，在理论上我们依然可以保证 SGD 的收敛性。

在机器学习中，给定 $n = 1, \dots, N$ 个数据点，我们通常将每个数据的损失 $L_{n}$ 的求和作为目标函数：
$$
L(\boldsymbol{\theta}) = \sum\limits_{n=1}^{N} L_{n}(\boldsymbol{\theta})\tag{7.13}
$$
其中 $\boldsymbol{\theta}$ 是我们关心的参数向量 —— 我们要找出最小化 $L$ 的参数 $\boldsymbol{\theta}$。第九章中我们将见到来自回归问题的 **负对数似然函数**，它是每个数据的负对数似然函数的求和：
$$
L(\boldsymbol{\theta}) = -\sum\limits_{n=1}^{N} \log p(y_{n}|\boldsymbol{x}_{n}, \boldsymbol{\theta}) \tag{7.14}
$$
其中 $\boldsymbol{x}_{n} \in \mathbb{R}^{D}$ 是训练中的输入数据，$y_{n}$ 是训练中的目标数据，$\boldsymbol{\theta}$ 是回归模型的参数。

前文提到，经典的梯度下降是一个 ”整批“ 的优化方法，这是说每次我们都要选一个合适的 $\gamma_{i}$，并用 **所有的** 训练集来完成下面的迭代：
$$
\boldsymbol{\theta}_{i+1} + \boldsymbol{\theta}_{i} = \gamma_{i}\big[ \nabla L(\boldsymbol{\theta}_{i}) \big] ^{\top} = \boldsymbol{\theta}_{i} - \gamma_{i}\sum\limits_{n=1}^{N} \big[ \nabla L_{n}(\boldsymbol{\theta}_{i}) \big] ^{\top}\tag{7.15}
$$
计算上面对所有 $L_{n}$ 的梯度之和是个大工程。当训练集很大，或是没有显式的梯度可以求解的时候，这么做显然是极其昂贵的。

考虑 (7.15) 中的一项 $\displaystyle \sum\limits_{n=1}^{N} [\nabla L_{n}(\boldsymbol{\theta})]$，我们可以通过只算一小部分 $L_{n}$ 的梯度之和来降低计算成本。相较于用上全部 $L_{n}, n = 1, \dots, N$ 的经典梯度下降算法，我们只选择小部分 $L_{n}$ ，这样我们就得到了 **小批次梯度下降**；该算法最极端的情况是每次只考虑一个 $L_{n}$。我们这么做是有道理的：我们只需要拿到一个对真实梯度的 **无偏估计**，而公式 (7.15) 中的 $\displaystyle \sum\limits_{n=1}^{N} [\nabla L_{n}(\boldsymbol{\theta})]$ 事实上就是对梯度期望值 (见 6.4.1) 的经验估计，因此任何对梯度的无偏估计都可以拿来用。不论我们的小批次中的数据量是多少它都是对梯度的无偏估计，SGD 也总会收敛。

> **注释**
> 在相对较弱的假设下，如果学习率以适当的幅度逐步降低，SGD **几乎必然 (almost surely)** 收敛到局部最优解。 (Bottu, 1998)

> **译者注**
> 几乎必然是一个专有名词，它属于概率论，指的是事件发生的概率为 $1$，或 Lebesgue 测度为 $1$；有时也简记为 a.s.

我们为什么要估计梯度的值呢？主要的原因是实践中的 CPU 和 GPU 的存储空间或是计算时间有限。我们可以考虑不同大小的批次。较大的批次不但可以利用高效的矩阵算法快速计算结果，还会给出梯度更加精确的估计，降低了参数更新的方差，算法的收敛也会更稳定。相比之下较小的批次可以更快的算出，但牺牲了估计的精确性，这可能会让我们陷入更差的局部最优而无法脱离。





## 7.2 约束优化和 Lagrange 乘子

在前一节中，我们讨论了如何求解函数的最小化问题：
$$
\min\limits_{\boldsymbol{x}}~f(\boldsymbol{x}), \tag{7.16}
$$
其中 $f: \mathbb{R}^{D} \rightarrow \mathbb{R}$。但在本节中，我们得面对额外的“约束条件”，具体来说，对于实值函数 $g_i: \mathbb{R}^{D} \rightarrow \mathbb{R}$（$i=1,\ldots, m$），我们考虑如下的约束优化问题（如图 7.4）：
$$
\begin{align}
\min\limits_{\boldsymbol{x}}~&~f(\boldsymbol{x})\\
\text{subject to}~&~g_{i}(\boldsymbol{x}) \leqslant 0\quad \text{for all}\quad i = 1, \dots, m
\end{align} \tag{7.17}
$$

<center><img src="https://datawhalechina.github.io/math-for-ai/ch7/attachments/Pasted%20image%2020250701133410.png" alt="alt text" style="zoom:50%;"></center>
<center>图7.4 约束优化图示</center>

这里有个值得注意的细节：函数 $f$ 和 $g_i$ 在一般情况下可能非凸（non-convex），不过别急，我们将在下一节讨论凸优化这个“乖孩子”。  

一种直观但不太实用的方法是使用 **示性函数（indicator function）** 将约束问题 (7.17) 转化为无约束形式：
$$
J(\boldsymbol{x}) = f(\boldsymbol{x}) + \sum\limits_{i=1}^{m} \boldsymbol{1}[g_{i}(\boldsymbol{x})] \tag{7.18}
$$
其中
$$
\boldsymbol{1}(z) = \begin{cases}
0 & z \leqslant 0\\
\infty & \text{otherwise}
\end{cases}. \tag{7.19}
$$
这招儿就像给违反约束的行为判了“无期徒刑”，理论上能给出相同解，但实际优化起来十分困难。我们可以用**Lagrangre 乘数法（Lagrange multipliers）解决这个问题：它的妙招是把阶跃函数松弛为线性函数。**

我们为问题 (7.17) 引入 **Lagrangre 函数（Lagrangian）**，通过Lagrangre 乘数 $\lambda_i \geqslant 0$ 将每个不等式约束松弛化（Boyd and Vandenberghe, 2004, 第四章）：
$$
\begin{align}
\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}) &= f(\boldsymbol{x}) + \sum\limits_{i=1}^{m} \lambda_{i}g_{i}(\boldsymbol{x}) \tag{7.20a}\\
&= f(\boldsymbol{x}) + \boldsymbol{\lambda}^{\top}\boldsymbol{g}(\boldsymbol{x})\tag{7.20b}
\end{align}
$$
这里，我们把所有约束 $g_i(x)$ 打包成一个向量 $\boldsymbol{g}(x)$，所有乘数也塞进向量，得到 $\boldsymbol{\lambda} \in \mathbb{R}^{m}$。  

现在我们引入 **Lagrangre 对偶性 (Lagrangian duality)** 。优化中的对偶思想，本质是把原变量（primal variables）$\boldsymbol{x}$ 的问题，转换成另一组对偶变量（dual variables）$\boldsymbol{\lambda}$ 的问题。本节我们聚焦Lagrangre 对偶，除此之外我们将在 7.3.3 节介绍 Legendre-Fenchel 对偶。 

> **定义 7.1** 我们称 (7.17) 中的问题 $$\begin{align}\min\limits_{\boldsymbol{x}}~&~f(\boldsymbol{x})\\\text{subject to}~&~g_{i}(\boldsymbol{x}) \leqslant 0\quad\text{for all}\quad i = 1, \dots, m\end{align} \tag{7.21}$$
> 为**原问题**（primal problem），对应原变量 $\boldsymbol{x}$。其关联的**Lagrangre 对偶问题**（Lagrangian dual problem）是$$\begin{align}\min\limits_{\boldsymbol{\lambda} \in \mathbb{R}^{m}}~&~\mathfrak{D}(\boldsymbol{\lambda})\\\text{subject to}~&~\boldsymbol{\lambda} \geqslant \boldsymbol{0}.\end{align} \tag{7.22}$$
> 其中  $\boldsymbol{\lambda}$ 是对偶变量， $\displaystyle \mathfrak{D}(\boldsymbol{\lambda}) = \min_{\boldsymbol{x} \in \mathbb{R}^d} \mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda})$。  

> **注释**
> 
> 在定义 7.1 的讨论中，我们用到两个独立有趣的概念（Boyd and Vandenberghe, 2004）
> 
> 第一个概念叫做 **极小极大不等式（minimax inequality）**：对任意双变量函数 $\varphi(\boldsymbol{x}, \boldsymbol{y})$，有  $$\max\limits_{\boldsymbol{y}}~\min\limits_{\boldsymbol{x}}~\phi(\boldsymbol{x}, \boldsymbol{y}) \leqslant \min\limits_{\boldsymbol{x}}~\max\limits_{\boldsymbol{y}}~\phi(\boldsymbol{x}, \boldsymbol{y}). \tag{7.23}$$ 可以考虑下面的不等式来证明 $$\forall \boldsymbol{x}, \boldsymbol{y}\quad \min\limits_{\boldsymbol{x}}~\phi(\boldsymbol{x}, \boldsymbol{y}) \leqslant \max\limits_{\boldsymbol{y}}~\phi(\boldsymbol{x}, \boldsymbol{y}).\tag{7.24}$$ 显然，左边的式子对 $\boldsymbol{y}$ 取 $\max$ 就对应 (7.23) 的左边；类似地操作我们也能得到右边。
> 
> 第二个概念是 **弱对偶性（weak duality）**，这是说我们在 (7.23) 证明了了的 "原问题值总大于等于对偶值"，更多细节见 (7.27)。

回忆一下，(7.18) 中的 $J(\boldsymbol{x})$ 与Lagrangre 函数的关键区别，是我们把指示函数松弛成了线性函数。因此，当 $\boldsymbol{\lambda} \geqslant 0$ 时，Lagrangre  $\mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda})$ 是 $J(\boldsymbol{x})$ 的下界。于是，$\mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda})$ 对 $\boldsymbol{\lambda}$ 的最大化给出
$$
J(\boldsymbol{x}) = \max\limits_{\boldsymbol{\lambda} \geqslant \boldsymbol{0}}~\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}). \tag{7.25}
$$
同时原问题是最小化 $J(\boldsymbol{x})$ 
$$
\min\limits_{\boldsymbol{x} \in \mathbb{R}^{d}}~\max\limits_{\boldsymbol{\lambda} \geqslant \boldsymbol{0}}~\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}). \tag{7.26}
$$
由极小极大不等式 (7.23)，交换最小和最大顺序会得到更小值，也就是所谓的弱对偶性：
$$
\min\limits_{\boldsymbol{x}\in \mathbb{R}^{d}}~\max\limits_{\boldsymbol{\lambda} \geqslant \boldsymbol{0}}~\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}) \geqslant \max\limits_{\boldsymbol{\lambda} \geqslant \boldsymbol{0}}~\min\limits_{\boldsymbol{x} \in \mathbb{R}^{d}}~\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}). \tag{7.27}
$$
其中右侧里面正是对偶目标函数 $\mathfrak{D}(\boldsymbol{\lambda})$。

与原优化问题（带约束）相比，$\displaystyle \min_{\boldsymbol{x} \in \mathbb{R}^{d}} \mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda})$ 对给定 $\boldsymbol{\lambda}$ 是无约束问题。如果这个子问题容易求解，那整体问题就变简单了！观察 (7.20b)，$\mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda})$ 关于 $\boldsymbol{\lambda}$ 是仿射（affine）的，因此 $\displaystyle \min_{\boldsymbol{x} \in \mathbb{R}^{d}} \mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda})$ 是 $\boldsymbol{\lambda}$ 的仿射函数的逐点最小值，故 $\mathfrak{D}(\boldsymbol{\lambda})$ 是凹函数——即使 $f(\cdot)$ 和 $g_i(\cdot)$ 非凸。外部最大化问题（对 $\boldsymbol{\lambda}$）是凹函数的最大化，可高效求解

假设 $f(\cdot)$ 和 $g_i(\cdot)$ 可微，我们通过微分Lagrangre 函数求解对偶问题：对 $\boldsymbol{x}$ 求导、设导数为零、解最优值。第7.3.1和7.3.2节将讨论两个具体例子（$f$ 和 $g_i$ 为凸时）。  

> **注释 （等式约束）**
> 考虑 (7.17) 添加等式约束 $$\begin{align}\min\limits_{\boldsymbol{x}}~&f(\boldsymbol{x})\\\text{subject to}~&~g_{i}(\boldsymbol{x}) \leqslant 0 \quad \forall i = 1, \dots, m\\&~h_{j}(\boldsymbol{x}) = 0\quad \forall j  = 1, \dots, n.\end{align} \tag{7.28}$$ 我们可以用两个不等式约束模拟等式约束：对每个 $h_j(\boldsymbol{x})=0$，等价替换为 $h_j(\boldsymbol{x}) \leqslant 0$ 和 $h_j(\boldsymbol{x}) \geqslant 0$。结果Lagrangre 乘数将无约束。
> 因此，在 (7.28) 中，我们仅约束不等式乘数为非负，而等式乘数则没有约束。





## 7.3 凸优化

### 凸函数

我们将目光聚焦于一类能**保证全局最优解**的特殊优化问题。当目标函数 $f(\cdot)$ 是凸函数，且约束函数 $g(\cdot)$ 和 $h(\cdot)$ 定义的集合为凸集时，这类问题称为**凸优化问题**。凸优化问题具有 **强对偶性**：对偶问题的最优解与原问题完全一致。虽然机器学习文献常模糊凸函数与凸集的界限，但上下文通常能提供明确指引。 

> **定义 7.2（凸集）** 
> 若集合 $\mathcal{C}$ 满足：对任意 $x, y\in \mathcal{C}$ 和标量 $\theta \in [0,1]$，有 $$\theta x + (1-\theta)y \in \mathcal{C}. \tag{7.29}$$ 则称 $\mathcal{C}$ 为凸集。

**凸集中两点之间的线段总是位于凸集中。**下图给出了凸集的一个典型例子和反例。

<center><img src="https://datawhalechina.github.io/math-for-ai/ch7/attachments/Pasted%20image%2020250701182000.png" alt="alt text" style="zoom:50%;"></center>
<center>图 7.5, 图 7.6 凸集（左）和非凸集合（右）</center>
凸函数定义和凸集很像，它的定义是函数上两点的连线一定位于函数曲线的上方。


> **定义 7.3（凸函数）**
> 考虑函数 $f: \mathbb{R}^{D} \rightarrow \mathbb{R}$，且 $f$ 的定义域为凸集。如果对定义域中的所有 $\boldsymbol{x}, \boldsymbol{y}$ 和任意标量 $0 \leqslant \theta \leqslant 1$，都有
> $$
> f(\theta \boldsymbol{x} + (1-\theta)\boldsymbol{y}) \leqslant \theta f(\boldsymbol{x}) + (1-\theta)f(\boldsymbol{y}).\tag{7.30}
> $$
> 则它被称为是 **凸函数** 。

> **注释**
> 一个 **凹函数** 一定是某个 **凸函数** 的负数




### 凸性判断

**使用梯度判断凸性：**
如果函数 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ 是可微的，我们还可以根据其梯度 $\nabla_{\boldsymbol{x}}f(\boldsymbol{x})$（见 5.2）来判断其凸性。这样的函数是凸的，当且仅当对任意定义域中的 $\boldsymbol{x}$ 和 $\boldsymbol{y}$，都有
$$
f(\boldsymbol{y}) \geqslant f(\boldsymbol{x}) + \nabla_{\boldsymbol{x}}f(\boldsymbol{x})^{\top}(\boldsymbol{y} - \boldsymbol{x}). \tag{7.31}
$$
进一步地，如果我们知道 $f$ 是二阶可微的，也就是在定义域中的每一点都存在 Hesse 矩阵 (5.147)，则该函数是凸的当且仅当 $\nabla_{\boldsymbol{x}}^{2}f(\boldsymbol{x})$ 是半正定的 (Boyd and Vandengerghe, 2004)。




> **示例 7.3（熵）**
> 
> 负熵函数 $f(x) = x\log_{2}x$ 在 $x > 0$ 上是凸函数，如图 7.8 所示。为了说明先前提到的凸函数的定义，我们选择 $x = 2$ 和 $x = 4$ 两个位置检查。需要注意的是，要证明该函数的凸性，只选择两个点不够，我们要检查所有的 $x \in \mathbb{R}$。
> 
> <center><img src="https://datawhalechina.github.io/math-for-ai/ch7/attachments/Pasted%20image%2020250701164942.png" alt="alt text" style="zoom:50%;"></center>
> 
> 让我们回忆定义 7.3，考虑它们的中间位置 ( $\theta = 0.5$ )，那么公式 (7.30) 的左边是 $f(0.5 \cdot 2 + 0.5 \cdot 4) = 3\log_{2}3 \approx 4.75$，右边是 $0.5(2\log_{2}2) + 0.5(4\log_{2}4) = 1 + 4 = 5$，这符合凸函数的定义。
> 由于 $f(x)$ 是可微的，我们也可以使用公式 (7.31) 对其图形进行判定。首先我们计算 $f(x)$ 的导数：$$\nabla_{x}\log(x\log_{2}x) = 1 \cdot \log_{2}x + x \cdot \frac{1}{x\log_{e}2} = \log_{2}x + \frac{1}{\log_{e}2}. \tag{7.32}$$ 我们同样使用 $x = 2$ 和 $x = 4$ 两点，公式 (7.31) 的左侧是 $f(4) = 8$，右侧是 $$\begin{align}f(\boldsymbol{x}) + \nabla_{\boldsymbol{x}}^{\top}(\boldsymbol{y} - \boldsymbol{x}) &= f(2) + \nabla f(2) \cdot (4 - 2) \tag{7.33a}\\&= 2 + \left( 1 + \frac{1}{\log_{e}2} \right) \cdot 2 \approx \frac{6}{9} \tag{7.33b}\end{align}$$

我们可以通过多种方法检查一个函数是否是凸函数。实际操作中我们通常通过保持凸性的变换来检查某个函数或集合是不是凸的。尽管细节有很大不同，但这仍然是我们在第二章中为线性空间引入的闭包思想。

> **示例 7.4（凸函数的非负线性组合）**
> 若干凸函数的非负线性组合还是凸函数。我们首先观察到，如果 $f$ 是凸函数，那么对于任意非负实数 $\alpha$，函数 $\alpha f$ 也是凸的。这个证明很简单，只需要将公式 (7.3) 的左右两侧都乘上 $\alpha$ 即可。
> 考虑两个凸函数 $f_{1}$ 和 $f_{2}$，根据凸函数定义我们有 $$\begin{align}f_{1}(\theta \boldsymbol{x} + (1 - \theta)\boldsymbol{y}) &\leqslant \theta f_{1}(\boldsymbol{x}) + (1-\theta)f_{1}(\boldsymbol{y}) \tag{7.34}\\f_{2}(\theta \boldsymbol{x} + (1 - \theta)\boldsymbol{y}) &\leqslant \theta f_{2}(\boldsymbol{x}) + (1-\theta)f_{2}(\boldsymbol{y}). \tag{7.35}\end{align}$$ 两式相加，有$$\begin{align}f_{1}(\theta \boldsymbol{x} &+ (1 - \theta)\boldsymbol{y}) + f_{2}(\theta \boldsymbol{x} + (1 - \theta)\boldsymbol{y}) \\ &\leqslant \theta f_{1}(\boldsymbol{x}) + (1-\theta)f_{1}(\boldsymbol{y}) + \theta f_{2}(\boldsymbol{x}) + (1-\theta)f_{2}(\boldsymbol{y}) \end{align}\tag{7.36}$$ 其中不等式右边还可以进一步整理为 $$\theta \Big[ f_{1}(\boldsymbol{x}) + f_{2}(\boldsymbol{x}) \Big] + (1-\theta)\Big[ f_{1}(\boldsymbol{y}) + f_{2}(\boldsymbol{y}) \Big] , \tag{7.37}$$ 这样我们就证明了 $f_{1} + f_{2}$ 是凸的。结合这两个事实，我们有对于任意的 $\alpha, \beta \geqslant 0$，$\alpha f_{1} + \beta f_{2}$ 是凸函数。对于三个及以上函数的非负线性组合，证明方法类似。

> **注释**
> 公式 (7.30) 中的不等式又称为 **Jensen 不等式**。事实上，这一整类用于求凸函数非负加权和的不等式都称为 Jensen 不等式。



### 凸优化问题

总的来说，被称为 **凸优化** 的约束优化问题的长相如下：
$$\begin{align}\min\limits_{\boldsymbol{x}}~&~f(\boldsymbol{x})\\\text{subject to}~&~g_{i}(\boldsymbol{x}) \leqslant 0\quad \forall i = 1, \dots, m\\&~h_{j}(\boldsymbol{x}) = 0\quad \forall j = 1, \dots, n,\end{align}\tag{7.38}$$
其中 $f(\boldsymbol{x})$ 和所有的 $g_{i}(\boldsymbol{x})$ 都是凸函数，所有的 $h_{j}(\boldsymbol{x}) = 0$ 都对应着凸集。下面的内容我们将讨论两个常用并以研究透了的凸优化问题。





### 线性规划
我们首先考虑所有函数都是线性函数这一特殊情况：
$$\begin{align}\min\limits_{\boldsymbol{x} \in \mathbb{R}^{d}}~&~\boldsymbol{c}^{\top}\boldsymbol{x}\\\text{subject to}~&~\boldsymbol{A}\boldsymbol{x} \leqslant \boldsymbol{b},\end{align}\tag{7.39}$$
其中 $\boldsymbol{A} \in \mathbb{R}^{m \times d}$，$\boldsymbol{b} \in \mathbb{R}^{m}$。这样的问题称为 **线性规划**。

> **注释**
> 线性规划是工业中最常用的一类方法 

它有 $d$ 个变量和 $m$ 个线性约束，它的 Lagrangre 函数是
$$\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}) = \boldsymbol{c}^{\top}\boldsymbol{x} + \boldsymbol{\lambda}^{\top}(\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}), \tag{7.40}$$
其中 $\boldsymbol{\lambda} \in \mathbb{R}^{m}$ 是非负的 Lagrangre 乘子组成的向量，稍微整理一下，得到
$$\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}) = (\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda})^{\top}\boldsymbol{x} - \boldsymbol{\lambda}^{\top}\boldsymbol{b}. \tag{7.41}$$
求 $\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda})$ 对 $\boldsymbol{x}$ 的导数，并令其为零，我们得到
$$\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda} = \boldsymbol{0}. \tag{7.42}$$
因此我们得到对偶 Lagrangre 函数 $\mathfrak{D}(\boldsymbol{\lambda}) = -\boldsymbol{\lambda}^{\top}\boldsymbol{b}$，我们需要最大化 $\mathfrak{D}(\boldsymbol{\lambda})$。除了前文中 $\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda})$ 要为零，我们还需要保持 $\boldsymbol{\lambda} \geqslant \boldsymbol{0}$，这就得到下面的对偶优化问题
$$\begin{align}\max\limits_{\boldsymbol{\lambda} \in \mathbb{R}^{m}}~&~-\boldsymbol{b}^{\top}\boldsymbol{\lambda}\\\text{subject to}~&~\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda} = \boldsymbol{0}\\&~\boldsymbol{\lambda} \geqslant \boldsymbol{0}\end{align} \tag{7.43}$$
> **注解**
> 一般主问题是一个最小化的问题，对偶问题则是一个最大化的问题。

这还是一个线性优化问题，但变元的数量是 $m$。我们可以依据实际情况选择是解原问题 (7.39) 还是解对偶问题 (7.43)，就看是原问题中的变元数量 $d$ 更小还是原问题中约束数量 $m$ 更小，哪个小选哪个。

> **示例 7.5（线性规划）**
> 考虑下面的二变元线性规划问题 $$\begin{align}\min\limits_{\boldsymbol{x} \in \mathbb{R}^{2}}~&~\begin{bmatrix}5\\3\end{bmatrix}^{\top}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}\\[0.5em]\text{subject to}~&~\begin{bmatrix}2&2\\2&-4\\-2&1\\0&-1\\0&1\end{bmatrix}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix} \leqslant \begin{bmatrix}33\\8\\5\\-1\\8\end{bmatrix}\end{align} \tag{7.44}$$如图 7.9。
> <center><img src="https://datawhalechina.github.io/math-for-ai/ch7/attachments/Pasted%20image%2020250701172750.png" alt="alt text" style="zoom:80%;"></center>
> 由图可知目标函数是线性的 —— 它的等高线是直线。问题的约束集合在图中由不同颜色的实直线表示，可行域由灰色阴影表示，这意味着最优解（红色五角星）必须在灰色阴影区域（在此例中，也包括其边缘）。





### 二次规划
现在考虑目标函数是凸的二次函数，而约束是仿射函数的情形：
$$\begin{align}\min\limits_{\boldsymbol{x} \in \mathbb{R}^{d}}~&~ \frac{1}{2}\boldsymbol{x}^{\top}\boldsymbol{Q}\boldsymbol{x} + \boldsymbol{c}^{\top}\boldsymbol{x}\\\text{subject to}~&~\boldsymbol{A}\boldsymbol{x} \leqslant \boldsymbol{b},\end{align}\tag{7.45}$$
其中 $\boldsymbol{A} \in \mathbb{R}^{m \times d}, \boldsymbol{b} \in \mathbb{R}^{m}, \boldsymbol{c} \in \mathbb{R}^{d}$。目标函数中的矩阵 $\boldsymbol{Q} \in \mathbb{R}^{d \times d}$ 是正定的，因此目标函数是凸的。这样的问题叫做 **二次规划**。它有 $d$ 个变量， $m$ 个线性约束。

> **示例 7.6（二次规划）**
> 考虑下面的二变元二次规划问题 $$\begin{align}\min\limits_{\boldsymbol{x}\in \mathbb{R}^{2}}~&~ \frac{1}{2}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}^{\top}\begin{bmatrix}2&1\\1&4\end{bmatrix}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix} + \begin{bmatrix}5\\3\end{bmatrix}^{\top}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}\tag{7.46}\\\text{subject to}~&~\begin{bmatrix}1&0\\-1&0\\0&1\\0&-1\end{bmatrix}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix} \leqslant \begin{bmatrix}1\\1\\1\\1\end{bmatrix}\tag{7.47}\end{align}$$ 
> 
> <center><img src="https://datawhalechina.github.io/math-for-ai/ch7/attachments/Pasted%20image%2020250701172628.png" alt="alt text" style="zoom:50%;"></center>
> 
> 由图可知，目标函数是二次的，矩阵 $\boldsymbol{Q}$ 是半正定的，因此我们看到的目标函数等高线是一系列椭圆。可行域是灰色区域，最优解由红色五角星表示。

二次规划的 Lagrangre 函数整理一下之后是
$$\begin{align}\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}) &= \frac{1}{2}\boldsymbol{x}^{\top}\boldsymbol{Q}\boldsymbol{x} + \boldsymbol{c}^{\top}\boldsymbol{x} + \boldsymbol{\lambda}^{\top}(\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b}) \tag{7.48a}\\&= \frac{1}{2}\boldsymbol{x}^{\top}\boldsymbol{Q}\boldsymbol{x} + (\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda})^{\top}\boldsymbol{x} -\boldsymbol{\lambda}^{\top}\boldsymbol{b}, \tag{7.48b}\end{align}$$
求它对 $\boldsymbol{x}$ 的导数并令其为零，我们有
$$\boldsymbol{Q}\boldsymbol{x} + (\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda}) = \boldsymbol{0}. \tag{7.49}$$
假设 $\boldsymbol{Q}$ 是可逆的，得到
$$\boldsymbol{x} = -\boldsymbol{Q}^{-1}(\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda}). \tag{7.50}$$
把 (7.50) 代入最初的 Lagrangre 函数 $\mathfrak{L}(\boldsymbol{x} ,\boldsymbol{\lambda})$，我们得到 Lagrangre 对偶函数
$$\mathfrak{D}(\boldsymbol{\lambda}) = \frac{1}{2}(\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda})^{\top}\boldsymbol{Q}^{-1}(\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda}) - \boldsymbol{\lambda}^{\top}\boldsymbol{b}. \tag{7.51}$$
于是二次规划的对偶优化问题就是
$$\begin{align}\max\limits_{\boldsymbol{\lambda} \in \mathbb{R}^{m}}~&~ \frac{1}{2}(\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda})^{\top}\boldsymbol{Q}{-1}(\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda}) - \boldsymbol{\lambda}^{\top}\boldsymbol{b}\\\text{subject to}~&~ \boldsymbol{\lambda} \geqslant \boldsymbol{0}.\end{align} \tag{7.52}$$
我们将在第十二章的机器学习内容中再次见到二次规划。






### Legendre-Fenchel 变换和凸共轭

让我们不考虑约束，重新回顾 7.2 节中的对偶概念。关于凸集的一个有用事实是，它可以用它的支撑超平面等价地描述。如果一个超平面与凸集相交，并且凸集只包含在它的一侧，则该超平面称为凸集的支撑超平面。回想一下，我们可以 “填充” 凸函数来获得上镜图，它是一个凸集。因此，我们也可以用它们的支撑超平面来描述凸函数。此外，观察到支撑超平面刚好与凸函数相切，实际上是该函数在该点的切线。回想一下，函数 $f (\boldsymbol{x})$ 在给定点 $\boldsymbol{x}_0$ 的切线是该函数在该点的梯度的求值 $\displaystyle \left. \frac{\mathrm{d}f(\boldsymbol{x})}{\mathrm{d}\boldsymbol{x}} \right|_{\boldsymbol{x} = \boldsymbol{x}_{0}}$ 。总而言之，由于凸集可以用其支撑超平面等效地描述，因此凸函数也可以用其梯度的函数等效地描述。**Legendre 变换**形式化地表达了这一概念。

> **注解**
> 物理系学生常常在学习经典力学中的 Lagrangre 量和 Hamilton 量的时候接触 Legendre 变换的

我们从最一般的定义开始，但它的形式有些违反直觉。我们先来看一些特殊情况，以便将定义与上一段描述的直觉联系起来。Legendre-Fenchel 变换是从凸可微函数 $f(\boldsymbol{x})$ 到依赖于切线 $s(\boldsymbol{x}) = \nabla_{\boldsymbol{x}}f(\boldsymbol{x})$ 的函数的变换（在傅里叶变换的意义上）。值得强调的是，这是函数 $f (\cdot)$ 的变换，而不是变量 $\boldsymbol{x}$ 或在 $\boldsymbol{x}$ 处求值的函数的变换。Legendre-Fenchel 变换也称为凸共轭（关于凸共轭的原因，我们很快就会看到），并且与对偶性密切相关（Hiriart-Urruty and Lemar´echal, 2001, 第五章）。

> **定义 7.4（凸共轭）**
> 函数 $f: \mathbb{R}^{D} \rightarrow \mathbb{R}$ 的 **凸共轭** 是$$f^{*}(\boldsymbol{s}) = \sup_{\boldsymbol{x} \in \mathbb{R}^{D}} \Big[ \left\langle \boldsymbol{s}, \boldsymbol{x} \right\rangle - f(\boldsymbol{x}) \Big] . \tag{7.53}$$

注意下文中提到的凸共轭并不需要函数 $f$ 是凸的或是可微的。定义 7.4 中，我们用的是抽象的内积记号（见 3.2），但下文中我们将继续使用有限维向量之间的标准内积（$\left\langle \boldsymbol{s}, \boldsymbol{x} \right\rangle = \boldsymbol{s}^{\top}\boldsymbol{x}$），以避免一些不必要的麻烦

> **注解**
> 画图能帮我们更好理解凸共轭的定义

为了从几何角度理解定义 7.4 的内容，考虑一个简单的一元可微的凸函数，例如 $f(x) = x^{2}$。注意我们考虑的是一元函数，超平面就是一条直线。考虑直线 $y = sx + c$ —— 我们可以用支撑超平面描述凸函数，因此让我们尝试用支撑超平面来描述函数 $f(x)$。固定直线的梯度 $s \in \mathbb{R}$ ，对于 $f$ 的图上的每个点 $(x_0, f(x_0))$，找到 $c$ 的最小值，使直线仍然经过 $(x_0, f(x_0))$ 相交。请注意，$c$ 的最小值是斜率为 $s$ 的直线刚好和函数 $f(x) = x^{2}$ 相切的位置。通过 $(x_0, f(x_0))$ 且梯度为 $s$ 的直线由  
$$y - f(x_{0}) = s(x - x_{0}). \tag{7.54}$$
给出。这条直线的 $y$ 轴截距为 $−sx_0 + f(x_0)$。因此，当 $y = sx + c$ 与 $f$ 的图像相交时，$c$ 的最小值为
$$\inf_{x_{0}} \Big[ -sx_{0} + f(x_{0}) \Big]. \tag{7.55} $$
按照惯例，前述凸共轭定义为其负值。本段的推理并不依赖于我们选择一维凸可微函数这一事实，并且对于 $f:\mathbb{R}^{D} \rightarrow \mathbb{R}$ 成立，它们是非凸且不可微的。

> **注解**
> 像 $f(x) = x^{2}$ 这样的可微凸函数是一个很好的特殊情况，我们不需要求上确界，且每个可微的凸函数和它的 Legendre 变换一一对应。让我们一步步导出这个结果。考虑可微凸函数 $f$，和 $(x_{0}, f(x_{0}))$ 处的切线 $$f(x_{0}) = sx_{0} + c. \tag{7.56}$$ 回忆可微凸函数 $f$ 和其梯度 $\nabla_{x}f(x)$ 的性质，我们有 $x = \nabla_{x}f(x_{0})$，整理上式，得到 $$-c = sx_{0} - f(x_{0}). \tag{7.57}$$ 注意 $c$ 随着 $x_{0}$（也即 $s$）的变化而变化，我们可以将其写为 $$f^{*}(s) \coloneqq sx_{0} - f(x_{0}). \tag{7.58}$$ 将 (7.58) 与定义 7.4 对比，容易发现前者是一个不带上确界的特殊情况。

凸共轭函数有不少良好的性质。例如对于凸函数，它的共轭的共轭是它本身。同样地，$f(x)$ 处的切线斜率是 $s$ 而 $f^{*}(s)$ 处的斜率是 $x$。下面的两个例子给出凸共轭在机器学习中的常见应用。

> **示例 7.7（凸共轭）**
> 为了展示凸共轭的应用，考虑下面的二次规划问题 $$f(\boldsymbol{y}) = \frac{\lambda}{2}\boldsymbol{y}^{\top}\boldsymbol{K}^{-1}\boldsymbol{y}\tag{7.59}$$其中 $\boldsymbol{K} \in \mathbb{R}^{n \times n}$ 是一个正定矩阵。我们定义主变量是 $\boldsymbol{y} \in \mathbb{R}^{n}$，对偶变量是 $\boldsymbol{\alpha} \in \mathbb{R}^{n}$。
> 根据定义 7.4，我们有 $$f^{*}(\boldsymbol{\alpha}) = \sup_{\boldsymbol{y}\in \mathbb{R}^{n}} \left[  \left\langle \boldsymbol{y}, \boldsymbol{\alpha} \right\rangle - \frac{\lambda}{2}\boldsymbol{y}^{\top}\boldsymbol{K}^{-1}\boldsymbol{y} \right]. \tag{7.60} $$由于该上确界中的函数是可微的，我们可以通过令其对 $\boldsymbol{y}$ 的梯度$$\displaystyle \frac{ \partial \left[  \left\langle \boldsymbol{y}, \boldsymbol{\alpha} \right\rangle - \frac{\lambda}{2}\boldsymbol{y}^{\top}\boldsymbol{K}^{-1}\boldsymbol{y} \right] }{ \partial \boldsymbol{y} } = (\boldsymbol{\alpha} - \lambda \boldsymbol{K}^{-1}\boldsymbol{y})^{\top}\tag{7.61}$$为零，也即当 $\displaystyle \boldsymbol{y} = \frac{1}{\lambda}\boldsymbol{K}\boldsymbol{\alpha}$，得到其最大值，也就是 $$f^{*}(\boldsymbol{\alpha}) = \frac{1}{\lambda}\boldsymbol{\alpha}^{\top}\boldsymbol{K}^{-1}\boldsymbol{\alpha} - \frac{\lambda}{2}\left( \frac{1}{\lambda}\boldsymbol{K}\boldsymbol{\alpha} \right)^{\top}\boldsymbol{K}^{-1}\left( \frac{1}{\lambda}\boldsymbol{K}\boldsymbol{\alpha} \right) = \frac{1}{2\lambda}\boldsymbol{\alpha}^{\top}\boldsymbol{K}\boldsymbol{\alpha}. \tag{7.62}$$

> **示例 7.8**
> 机器学习中，我们常用一系列函数（例如每条训练数据的损失函数 $\ell: \mathbb{R} \rightarrow \mathbb{R}$）的和作为目标。下面我们推导损失函数 $\ell(t)$ 之和的凸共轭，这同时展示了凸共轭在向量变元函数情况下的应用。令 $\displaystyle \mathcal{L}(\boldsymbol{t}) = \sum\limits_{i=1}^{n} \ell_{i}(t_{i})$，于是 $$\begin{align}\mathcal{L}^{*}(\boldsymbol{z}) &= \sup_{\boldsymbol{t} \in \mathbb{R}^{n}} \left[ \left\langle \boldsymbol{z}, \boldsymbol{t} \right\rangle -\sum\limits_{i=1}^{n} \ell_{i}(t_{i}) \right] \tag{7.63a}\\&= \sup_{\boldsymbol{t} \in \mathbb{R}^{n}} \sum\limits_{i=1}^{n}\left[ z_{i}t_{i} - \ell_{i}(t_{i})  \right] & \text{内积定义} \tag{7.63b}\\&= \sum\limits_{i=1}^{n} \sup_{\boldsymbol{t} \in \mathbb{R}^{n}} [z_{i}t_{i} - \ell_{i}(t_{i})] \tag{7.63c}\\&= \sum\limits_{i=1}^{n} \ell^{*}_{i}(z_{i}) & \text{共轭定义} \tag{7.63d}\end{align}$$

回忆在 7.2 节中，我们使用 Lagrangre 乘子导出原问题的对偶优化问题。进一步地，凸优化问题具有强对偶性：对偶问题的解就是原问题的解。本节中介绍的Legendre-Fenchel 变换也可以用来求对偶优化问题，特别地，当目标函数是可微且凸时，Legendre-Fenchel 变换中的上确界是唯一的。为了进一步说明这两个方法之间的联系，考虑下面带线性等式约束的凸优化问题。

> **示例 7.9**
> 考虑凸函数 $f(\boldsymbol{x})$，$g(\boldsymbol{x})$，实矩阵 $\boldsymbol{A}$，并假设方程 $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{y}$ 中的向量和矩阵形状匹配。于是
> $$\min\limits_{\boldsymbol{x}}~f(\boldsymbol{A}\boldsymbol{x}) + g(\boldsymbol{x}) = \min\limits_{\boldsymbol{A}\boldsymbol{x} = \boldsymbol{y}}~f(\boldsymbol{y}) + g(\boldsymbol{x}). \tag{7.64}$$ 引入约束 $\boldsymbol{A}\boldsymbol{x} = \boldsymbol{y}$ 和 Lagrangre 乘子 $\boldsymbol{u}$，有 $$\begin{align}\min\limits_{\boldsymbol{A}\boldsymbol{x} = \boldsymbol{y}}~f(\boldsymbol{y}) + g(\boldsymbol{x}) &=\min\limits_{\boldsymbol{x}, \boldsymbol{y}}~\max\limits_{\boldsymbol{u}}~f(\boldsymbol{y}) + g(\boldsymbol{x}) + (\boldsymbol{A}\boldsymbol{x} - \boldsymbol{y})^{\top}\boldsymbol{u} \tag{7.65a}\\&= \max\limits_{\boldsymbol{u}}~\min\limits_{\boldsymbol{x}, \boldsymbol{y}}~f(\boldsymbol{y}) + g(\boldsymbol{x}) + (\boldsymbol{A}\boldsymbol{x} -\boldsymbol{y})^{\top}\boldsymbol{u} \tag{7.65b}\end{align}$$ 其中最后一步可以交换 $\max$ 和 $\min$ 是因为 $f(\boldsymbol{y})$ 和 $g(\boldsymbol{x})$ 是凸函数。展开点积这一项，然后分开 $\boldsymbol{x}$ 和 $\boldsymbol{y}$ 的项，得到 $$\begin{align}&\max\limits_{\boldsymbol{u}}~\min\limits_{\boldsymbol{x}, \boldsymbol{y}}~f(\boldsymbol{y}) + g(\boldsymbol{x}) + (\boldsymbol{A}\boldsymbol{x} -\boldsymbol{y})^{\top}\boldsymbol{u} \tag{7.66a}\\=&\max\limits_{\boldsymbol{u}}~\Big[ \min\limits_{\boldsymbol{y}}~-\boldsymbol{y}^{\top}\boldsymbol{u} + f(\boldsymbol{y}) \Big] + \Big[ \min\limits_{\boldsymbol{x}}~(\boldsymbol{A}\boldsymbol{x})^{\top}\boldsymbol{u} + g(\boldsymbol{x}) \Big] \tag{7.66b}\\=& \max\limits_{\boldsymbol{u}}~\Big[ \min\limits_{\boldsymbol{y}}~-\boldsymbol{y}^{\top}\boldsymbol{u} + f(\boldsymbol{y}) \Big] + \Big[ \min\limits_{\boldsymbol{x}}~\boldsymbol{x}^{\top}\boldsymbol{A}^{\top}\boldsymbol{u} + g(\boldsymbol{x}) \Big] \tag{7.66c}\\\end{align}$$ 回忆凸共轭的定义（定义 7.4）以及（实）点积的对称性，我们有 $$\begin{align}&\max\limits_{\boldsymbol{u}}~\Big[ \min\limits_{\boldsymbol{y}}~-\boldsymbol{y}^{\top}\boldsymbol{u} + f(\boldsymbol{y}) \Big] + \Big[ \min\limits{\boldsymbol{x}}~\boldsymbol{x}^{\top}\boldsymbol{A}^{\top}\boldsymbol{u} + g(\boldsymbol{x}) \Big] \tag{7.67a}\\=& \max\limits_{\boldsymbol{u}}~-f^{*}(\boldsymbol{y}) - g^{*}(-\boldsymbol{A}^{\top}\boldsymbol{u}). \tag{7.67b}\end{align}$$ 于是我们就证明了 $$\min\limits_{\boldsymbol{x}}~f(\boldsymbol{A}\boldsymbol{x}) + g(\boldsymbol{x}) = \max\limits_{\boldsymbol{u}}~-f^{*}(\boldsymbol{u}) - g^{*}(-\boldsymbol{A}^{\top}\boldsymbol{u}). \tag{7.68}$$

事实上，Legendre-Fenchel 共轭在可表示为凸优化的机器学习中非常有用。特别地，对于独立作用于每个数据的损失函数，共轭损失函数是推导对偶问题的便捷方法。












