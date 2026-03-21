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




