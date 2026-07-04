# Continuous Optimization(连续优化)

## 7.1 什么是优化问题

机器学习的核心流程可以概括为：**定义损失函数 $L(\boldsymbol{\theta})$ → 求解使其最小化的参数 $\boldsymbol{\theta}^{*}$**，即

$$
\boldsymbol{\theta}^{*} = \arg\min_{\boldsymbol{\theta}}~L(\boldsymbol{\theta}) \tag{7.1}
$$

这就是一个**优化问题**。求解方式有两种：

| | 解析解（analytical） | 数值解（numerical） |
|---|---|---|
| 方法 | 令 $\nabla f = 0$，解方程 | 迭代算法逐步逼近 |
| 要求 | $f$ 结构简单、有闭式解 | $f$ 可微（或有梯度估计） |
| 典型场景 | 线性回归的正规方程 | 神经网络的梯度下降 |

实际的机器学习问题通常高维、非凸且复杂，解析解几乎不存在，因此本章的重点是**数值求解方法**。

不同的优化问题有不同的结构（无约束 / 有约束 / 凸），而这些结构上的差异决定了我们应当选择怎样的求解方法。

> **注释（优化 ≠ 学习）**
> 优化的目标是 $\min_{\boldsymbol{\theta}} f(\boldsymbol{\theta})$，即把训练损失压到最低；
> 而深度学习（统计推断）的真正目标是**减少泛化误差**。单纯地将训练损失最小化到底反而可能导致 **过拟合**，因此实践中还需要正则化、早停等手段。
> 
> 本章聚焦于优化问题本身的数学结构与求解方法，**过拟合** 与 **泛化** 的讨论详见深度学习的相关章节。
> 





## 7.2 优化问题的定义与分类

### 无约束优化问题

最简单的优化问题是**无约束优化**：寻找一个实值函数的最小值，没有任何额外限制。形式化地，考虑

$$
\min\limits_{\boldsymbol{x}}~f(\boldsymbol{x}) \tag{7.2}
$$

其中 $f: \mathbb{R}^{d} \rightarrow \mathbb{R}$ 是一个目标函数，它刻画了我们手中的机器学习问题。我们假设函数 $f$ 是可微的，并且我们无法找到上述问题的解析解。这种情况在机器学习中非常常见——例如神经网络的损失函数通常没有闭式解。



### 约束优化问题

当我们需要在某些限制条件下寻找最优解时，就得到了**约束优化问题**的一般形式：

$$
\begin{align}\min\limits_{\boldsymbol{x}}~&f(\boldsymbol{x})\\\text{subject to}~&~g_{i}(\boldsymbol{x}) \leqslant 0 \quad \forall i = 1, \dots, m\\&~h_{j}(\boldsymbol{x}) = 0\quad \forall j  = 1, \dots, n.\end{align} \tag{7.3}
$$

其中 $f$、$g_i$、$h_j$ 在一般情况下可能非凸（non-convex）。当它们满足凸性条件时，就得到凸优化问题（见 7.3 节）。


一个示例如下图7.1 所示：
<center>
    <img src="https://datawhalechina.github.io/math-for-ai/ch7/attachments/Pasted%20image%2020250701133410.png" alt="alt text" style="zoom:50%;">
</center> 

<center>图7.1 约束优化图示</center>






## 7.3 凸优化问题

我们将目光聚焦于一类能**保证全局最优解**的特殊优化问题。当

1. 目标函数 $f(\cdot)$ 是凸函数
2. 且约束函数 $g(\cdot)$ 和 $h(\cdot)$ 定义的集合为凸集

时，这类问题称为 **凸优化问题**。


> **定义 7.1（凸集）**
> 
> 若集合 $\mathcal{C}$ 满足：对任意 $x, y\in \mathcal{C}$ 和标量 $\theta \in [0,1]$，有 
> $$\theta x + (1-\theta)y \in \mathcal{C} \tag{7.4}$$
> 
> 则称 $\mathcal{C}$ 为凸集。
> 

 **凸集中两点之间的线段总是位于凸集中。** 下图给出了凸集的一个典型例子和反例。

<center><img src="https://datawhalechina.github.io/math-for-ai/ch7/attachments/Pasted%20image%2020250701182000.png" alt="alt text" style="zoom:50%;"></center>

<center>图 7.2 凸集（左）和非凸集合（右）</center>



> **定义 7.2（凸函数）**
> 
> 考虑函数 $f: \mathbb{R}^{D} \rightarrow \mathbb{R}$，且 $f$ 的定义域为凸集。如果对定义域中的所有 $\boldsymbol{x}, \boldsymbol{y}$ 和任意标量 $0 \leqslant \theta \leqslant 1$，都有
> $$
> f(\theta \boldsymbol{x} + (1-\theta)\boldsymbol{y}) \leqslant \theta f(\boldsymbol{x}) + (1-\theta)f(\boldsymbol{y}) \tag{7.5}
> $$
> 
> 则它被称为是 **凸函数** 。
> 

凸函数定义和凸集很像，它的定义是**函数上两点的连线一定位于函数曲线的上方**。一个 **凹函数** 一定是某个 **凸函数** 的负数




### 凸性判断

**使用梯度判断凸性**：如果函数 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$ 是可微的，我们还可以根据其梯度 $\nabla_{\boldsymbol{x}}f(\boldsymbol{x})$ 来判断其凸性。这样的函数是凸的，当且仅当对任意定义域中的 $\boldsymbol{x}$ 和 $\boldsymbol{y}$，都有

$$
f(\boldsymbol{y}) \geqslant f(\boldsymbol{x}) + \nabla_{\boldsymbol{x}}f(\boldsymbol{x})^{\top}(\boldsymbol{y} - \boldsymbol{x}) \tag{7.6}
$$


进一步地，如果我们知道 $f$ 是二阶可微的，也就是在定义域中的每一点都存在 Hesse 矩阵，则该函数是凸的当且仅当 $\nabla_{\boldsymbol{x}}^{2}f(\boldsymbol{x})$ 是半正定的 (Boyd and Vandengerghe, 2004)。




凸性在运算下的保持性质也很重要：凸函数的非负线性组合仍是凸函数，即对 $\alpha, \beta \geqslant 0$，若 $f_1, f_2$ 凸则 $\alpha f_1 + \beta f_2$ 凸。

> **注释**: 公式 (7.5) 中的不等式又称为 **Jensen 不等式**。事实上，这一整类用于求凸函数非负加权和的不等式都称为 Jensen 不等式。
> 


### 凸优化问题一般形式

总的来说，被称为 **凸优化** 的约束优化问题的长相如下：
$$
\begin{align}\min\limits_{\boldsymbol{x}}~&~f(\boldsymbol{x})\\\text{subject to}~&~g_{i}(\boldsymbol{x}) \leqslant 0\quad \forall i = 1, \dots, m\\&~h_{j}(\boldsymbol{x}) = 0\quad \forall j = 1, \dots, n \end{align}\tag{7.7}
$$

其中 $f(\boldsymbol{x})$ 和所有的 $g_{i}(\boldsymbol{x})$ 都是凸函数，所有的 $h_{j}(\boldsymbol{x}) = 0$ 都对应着凸集。凸优化问题具有 **强对偶性**：对偶问题的最优解与原问题完全一致。


下面的内容我们将讨论两个常用并已研究透了的 **凸优化问题**

- **线性规划** (Linear Programming, LP)
- **二次规划** (Quadratic Programming, QP)

---


### 经典凸优化问题（线性规划）

**线性规划**：我们首先考虑所有函数都是线性函数这一特殊情况：
$$
\begin{align}\min\limits_{\boldsymbol{x} \in \mathbb{R}^{d}}~&~\boldsymbol{c}^{\top}\boldsymbol{x}\\\text{subject to}~&~\boldsymbol{A}\boldsymbol{x} \leqslant \boldsymbol{b}\end{align}\tag{7.8}
$$

其中 $\boldsymbol{A} \in \mathbb{R}^{m \times d}$，$\boldsymbol{b} \in \mathbb{R}^{m}$。这样的问题称为 **线性规划**，它有 $d$ 个变量和 $m$ 个线性约束。

> **注释**: 线性规划是工业中最常用的一类方法 
> 

<center style="font-weight: bold;">示例 7.1（线性规划）</center>

考虑下面的二变元线性规划问题 
$$
\begin{align}\min\limits_{\boldsymbol{x} \in \mathbb{R}^{2}}~&~\begin{bmatrix}5\\3\end{bmatrix}^{\top}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}\\[0.5em]\text{subject to}~&~\begin{bmatrix}2&2\\2&-4\\-2&1\\0&-1\\0&1\end{bmatrix}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix} \leqslant \begin{bmatrix}33\\8\\5\\-1\\8\end{bmatrix}\end{align} \tag{7.9}
$$

如图 7.3,

<center>
    <img src="https://datawhalechina.github.io/math-for-ai/ch7/attachments/Pasted%20image%2020250701172750.png" alt="alt text" style="zoom:80%;">
</center>

<center>图 7.3 线性规划的示例</center>

由图可知目标函数是线性的 —— 它的等高线是直线。问题的约束集合在图中由不同颜色的实直线表示，可行域由灰色阴影表示，这意味着最优解（红色五角星）必须在灰色阴影区域（在此例中，也包括其边缘）。






### 经典凸优化问题（二次规划）

**二次规划**：现在考虑目标函数是凸的二次函数，而约束是仿射函数的情形：
$$
\begin{align}\min\limits_{\boldsymbol{x} \in \mathbb{R}^{d}}~&~ \frac{1}{2}\boldsymbol{x}^{\top}\boldsymbol{Q}\boldsymbol{x} + \boldsymbol{c}^{\top}\boldsymbol{x}\\\text{subject to}~&~\boldsymbol{A}\boldsymbol{x} \leqslant \boldsymbol{b} \end{align}\tag{7.10}
$$

其中 $\boldsymbol{A} \in \mathbb{R}^{m \times d}, \boldsymbol{b} \in \mathbb{R}^{m}, \boldsymbol{c} \in \mathbb{R}^{d}$。目标函数中的矩阵 $\boldsymbol{Q} \in \mathbb{R}^{d \times d}$ 是正定的，因此目标函数是凸的。这样的问题叫做 **二次规划**。它有 $d$ 个变量， $m$ 个线性约束。

<center style="font-weight: bold;">示例 7.2（二次规划）</center>

考虑下面的二变元二次规划问题 
$$
\begin{align}\min\limits_{\boldsymbol{x}\in \mathbb{R}^{2}}~&~ \frac{1}{2}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}^{\top}\begin{bmatrix}2&1\\1&4\end{bmatrix}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix} + \begin{bmatrix}5\\3\end{bmatrix}^{\top}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}\tag{7.11}\\\text{subject to}~&~\begin{bmatrix}1&0\\-1&0\\0&1\\0&-1\end{bmatrix}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix} \leqslant \begin{bmatrix}1\\1\\1\\1\end{bmatrix}\tag{7.12}\end{align}
$$ 

<center>
    <img src="https://datawhalechina.github.io/math-for-ai/ch7/attachments/Pasted%20image%2020250701172628.png" alt="alt text" style="zoom:50%;">
</center>

<center>图 7.4 二次规划的示例</center>

由图可知，目标函数是二次的，矩阵 $\boldsymbol{Q}$ 是半正定的，因此我们看到的目标函数等高线是一系列椭圆。可行域是灰色区域，最优解由红色五角星表示。





## 7.4 求解优化问题——问题变换

对于约束优化问题，直接求解通常很困难，需要先将其变换为更易处理的形式。本章介绍两种变换工具：

- **Lagrange 乘子法**：通过引入对偶变量，将带约束优化问题转化为对偶问题（凸优化）；
- **Legendre-Fenchel 变换**：当目标函数具有特殊结构（如可分离的损失函数之和）时，提供更便捷的对偶问题构造方式。

两者的目的都是把原问题重新表述为更容易求解的等价（或近似）问题，但并非所有优化问题都适合进行变换：

| 适用条件 | Lagrange 乘子法 | Legendre-Fenchel 变换 |
|---|---|---|
| 约束 | 必须有显式约束 $g_i(\boldsymbol{x}) \leqslant 0$ | 适合有线性等式约束的问题 |
| 可微性 | $f$、$g_i$ 需可微（三步法要求对 $\boldsymbol{x}$ 求导） | 不要求原函数可微或凸 |
| 解析可解 | 令导数为零后需能解出 $\boldsymbol{x}^*(\boldsymbol{\lambda})$ | 上确界需可计算 |
| 变换结果 | 对偶问题一定是凸的，但弱对偶间隙可能很大 | 凸函数时强对偶成立 |

典型的**不适合**变换的场景包括：深度学习的 $\min \text{Loss}$（无显式约束、高度非线性无法解出 $\boldsymbol{x}^*$）；目标函数结构过于复杂导致对偶函数没有解析形式。

以 Lagrange 乘子法为例，其完整流程为：

1. **构造对偶问题**：写出 Lagrangian $\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda})$，通过三步法（对 $\boldsymbol{x}$ 求导→解出 $\boldsymbol{x}^*(\boldsymbol{\lambda})$→代回 $\mathfrak{L}$）得到对偶函数 $D(\boldsymbol{\lambda})$ 及对偶优化问题；
2. **求解对偶问题**：$D(\boldsymbol{\lambda})$ 一定是凹函数，因此 $\max D(\boldsymbol{\lambda})$ 是凸优化，可用求解算法（见 7.5 节）高效求解，得到最优对偶变量 $\boldsymbol{\lambda}^*$；
3. **恢复原变量**：将 $\boldsymbol{\lambda}^*$ 代入 $\boldsymbol{x}^*(\boldsymbol{\lambda})$，得到原问题的最优解 $\boldsymbol{x}^*$ 及最优值 $f(\boldsymbol{x}^*)$（若强对偶成立，则 $f(\boldsymbol{x}^*) = D(\boldsymbol{\lambda}^*)$）。







### Lagrange 乘子法

####  示性函数

在 7.2 节中，我们定义了约束优化问题。现在我们来讨论如何求解它。

一种直观但不太实用的方法是使用 **示性函数（indicator function）** 将约束问题 (7.3) 转化为无约束形式：
$$
J(\boldsymbol{x}) = f(\boldsymbol{x}) + \sum\limits_{i=1}^{m} \boldsymbol{1}[g_{i}(\boldsymbol{x})] \tag{7.13}
$$

其中
$$
\boldsymbol{1}(z) = \begin{cases}
0 & z \leqslant 0\\
\infty & \text{otherwise}
\end{cases}. \tag{7.14}
$$

这招儿就像给违反约束的行为判了"无期徒刑"，理论上能给出相同解，但实际优化起来十分困难。



#### Lagrangre 乘数法

我们可以用**Lagrangre 乘数法（Lagrange multipliers）解决这个问题：它的妙招是把阶跃函数松弛为线性函数。**

我们为问题 (7.13) 引入 **Lagrangre 函数（Lagrangian）**，通过Lagrangre 乘数 $\lambda_i \geqslant 0$ 将每个不等式约束松弛化：
$$
\begin{align}
\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}) &= f(\boldsymbol{x}) + \sum\limits_{i=1}^{m} \lambda_{i}g_{i}(\boldsymbol{x}) \tag{7.15a}\\
&= f(\boldsymbol{x}) + \boldsymbol{\lambda}^{\top}\boldsymbol{g}(\boldsymbol{x})\tag{7.15b}
\end{align}
$$

这里，我们把所有约束 $g_i(x)$ 打包成一个向量 $\boldsymbol{g}(x)$，所有乘数也塞进向量，得到 $\boldsymbol{\lambda} \in \mathbb{R}^{m}$。

---



#### 对偶性问题

现在我们引入 **Lagrangre 对偶性 (Lagrangian duality)** 。优化中的对偶思想，本质是把原变量（primal variables）$\boldsymbol{x}$ 的问题，转换成另一组对偶变量（dual variables）$\boldsymbol{\lambda}$ 的问题。

> **定义 7.3** 我们称
> $$\begin{align}\min\limits_{\boldsymbol{x}}~&~f(\boldsymbol{x})\\\text{subject to}~&~g_{i}(\boldsymbol{x}) \leqslant 0\quad\text{for all}\quad i = 1, \dots, m\end{align} \tag{7.16}$$
> 
> 为**原问题**（primal problem），对应原变量 $\boldsymbol{x}$。
> 
> 其关联的**Lagrangre 对偶问题**（Lagrangian dual problem）是
> $$\max\limits_{\boldsymbol{\lambda} \geqslant \boldsymbol{0}}~D(\boldsymbol{\lambda}) = \max\limits_{\boldsymbol{\lambda} \geqslant \boldsymbol{0}}~\min\limits_{\boldsymbol{x} \in \mathbb{R}^{d}}~\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}) \tag{7.17}$$
> 
> 其中 $\boldsymbol{\lambda} \in \mathbb{R}^m$ 是对偶变量，$\displaystyle D(\boldsymbol{\lambda}) = \min_{\boldsymbol{x} \in \mathbb{R}^d} \mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda})$ 为对偶目标函数。约束被统一到了 Lagrangian 中，原来的带约束优化变为对 $\boldsymbol{\lambda}$ 的无约束（仅需 $\boldsymbol{\lambda} \geqslant \boldsymbol{0}$）最大化。  

**为什么要转化为对偶问题？** Lagrangian 对偶带来的变化和优势可以概括为：

1. **空间转换**：$\boldsymbol{x} \in \mathbb{R}^d$ → $\boldsymbol{\lambda} \in \mathbb{R}^m$，维度可能更低
2. **约束简化**：原来复杂的约束 $g_i(\boldsymbol{x}) \leqslant 0$ 被吸收进函数，只剩 $\boldsymbol{\lambda} \geqslant \boldsymbol{0}$
3. **性质改善**：对偶问题一定是凸优化（凹函数最大化），即使原问题非凸。这是因为 $\mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda})$ 关于 $\boldsymbol{\lambda}$ 是仿射的（观察 (7.15b)），因此 $\displaystyle D(\boldsymbol{\lambda}) = \min_{\boldsymbol{x} \in \mathbb{R}^{d}} \mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda})$ 是一族仿射函数的逐点最小值，故 $D(\boldsymbol{\lambda})$ 是凹函数，可高效求解。

代价就是弱对偶性——对偶问题的最优值只是原问题最优值的一个下界，不一定完全等价（除非强对偶性成立）。

**如何求解对偶问题？** 假设 $f(\cdot)$ 和 $g_i(\cdot)$ 可微：

1. 对 $\boldsymbol{x}$ 求导：计算 $\dfrac{\partial \mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda})}{\partial \boldsymbol{x}}$，并令其为零；
2. 解出 $\boldsymbol{x}$：从上述方程中解出 $\boldsymbol{x}$ 关于 $\boldsymbol{\lambda}$ 的表达式 $\boldsymbol{x}^{*}(\boldsymbol{\lambda})$；
3. 代回 $\mathfrak{L}$：将 $\boldsymbol{x}^{*}(\boldsymbol{\lambda})$ 代入 $\mathfrak{L}$，消去 $\boldsymbol{x}$，得到仅关于 $\boldsymbol{\lambda}$ 的对偶函数 $D(\boldsymbol{\lambda})$，然后最大化 $D(\boldsymbol{\lambda})$。

---

**对偶性的论证**

但对偶问题的解只是原问题的近似——两者之间存在 **弱对偶性（weak duality）**，即对偶问题的最优值总是小于等于原问题的最优值。下面我们论证这一点。

回忆一下，(7.13) 中的 $J(\boldsymbol{x})$ 与Lagrangre 函数的关键区别，是我们把指示函数松弛成了线性函数。因此，当 $\boldsymbol{\lambda} \geqslant 0$ 时，Lagrangre 函数 $\mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda})$ 是 $J(\boldsymbol{x})$ 的下界。于是，$\mathfrak{L}(\boldsymbol{x},\boldsymbol{\lambda})$ 对 $\boldsymbol{\lambda}$ 的最大化给出
$$
J(\boldsymbol{x}) = \max\limits_{\boldsymbol{\lambda} \geqslant \boldsymbol{0}}~\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}). \tag{7.18}
$$

同时原问题是最小化 $J(\boldsymbol{x})$
$$
\min\limits_{\boldsymbol{x} \in \mathbb{R}^{d}}~\max\limits_{\boldsymbol{\lambda} \geqslant \boldsymbol{0}}~\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}). \tag{7.19}
$$

由 **极小极大不等式（minimax inequality）**，交换最小和最大顺序会得到更小值：
$$
\min\limits_{\boldsymbol{x}\in \mathbb{R}^{d}}~\max\limits_{\boldsymbol{\lambda} \geqslant \boldsymbol{0}}~\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}) \geqslant \max\limits_{\boldsymbol{\lambda} \geqslant \boldsymbol{0}}~\min\limits_{\boldsymbol{x} \in \mathbb{R}^{d}}~\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}). \tag{7.20}
$$

其中左侧是原问题，右侧正是对偶问题（其中 $\min$ 部分即对偶目标函数 $D(\boldsymbol{\lambda})$）。这就是弱对偶性。

> **注释（极小极大不等式）**
> 
> 对任意双变量函数 $\phi(\boldsymbol{x}, \boldsymbol{y})$，有
> $$\max\limits_{\boldsymbol{y}}~\min\limits_{\boldsymbol{x}}~\phi(\boldsymbol{x}, \boldsymbol{y}) \leqslant \min\limits_{\boldsymbol{x}}~\max\limits_{\boldsymbol{y}}~\phi(\boldsymbol{x}, \boldsymbol{y}). \tag{7.21}$$
> 
> 可以考虑下面的不等式来证明
> $$\forall \boldsymbol{x}, \boldsymbol{y}\quad \min\limits_{\boldsymbol{x}}~\phi(\boldsymbol{x}, \boldsymbol{y}) \leqslant \max\limits_{\boldsymbol{y}}~\phi(\boldsymbol{x}, \boldsymbol{y}).\tag{7.22}$$
> 
> 显然，左边的式子对 $\boldsymbol{y}$ 取 $\max$ 就对应 (7.21) 的左边；类似地操作我们也能得到右边。



#### 线性规划的 Lagrange 对偶

线性规划 (7.8) 的 Lagrangre 函数是
$$
\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}) = \boldsymbol{c}^{\top}\boldsymbol{x} + \boldsymbol{\lambda}^{\top}(\boldsymbol{A}\boldsymbol{x} - \boldsymbol{b})
$$

其中 $\boldsymbol{\lambda} \in \mathbb{R}^{m}$ 是非负的 Lagrangre 乘子组成的向量，稍微整理一下，得到
$$
\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}) = (\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda})^{\top}\boldsymbol{x} - \boldsymbol{\lambda}^{\top}\boldsymbol{b}
$$

按前文三步法求解对偶问题：

1. **对 $\boldsymbol{x}$ 求导**：$\dfrac{\partial \mathfrak{L}}{\partial \boldsymbol{x}} = \boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda}$。注意 $\mathfrak{L}$ 关于 $\boldsymbol{x}$ 是线性的，在 $\boldsymbol{x} \in \mathbb{R}^d$ 无界的情况下，只有系数为零时 $\min_{\boldsymbol{x}} \mathfrak{L}$ 才存在有限值，因此必须有
$$
\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda} = \boldsymbol{0} \tag{7.23}
$$

2. **解出 $\boldsymbol{x}$**：由于 $\mathfrak{L}$ 关于 $\boldsymbol{x}$ 是线性（而非二次）的，(7.23) 并不能解出 $\boldsymbol{x}^*(\boldsymbol{\lambda})$，而是直接消去了 $\boldsymbol{x}$——满足 (7.23) 时，$(\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda})^{\top}\boldsymbol{x} = \boldsymbol{0}^{\top}\boldsymbol{x} = 0$。

3. **代回 $\mathfrak{L}$**：第一项消失，得到对偶函数 $D(\boldsymbol{\lambda}) = -\boldsymbol{\lambda}^{\top}\boldsymbol{b}$。连同 (7.23) 作为约束和 $\boldsymbol{\lambda} \geqslant \boldsymbol{0}$，得到对偶优化问题
$$
\begin{array}{c}
\max _{\boldsymbol{\lambda} \in \mathbb{R}^{m}}-\boldsymbol{b}^{\top} \boldsymbol{\lambda} \\
\text { subject to } \boldsymbol{c}+\boldsymbol{A}^{\top} \boldsymbol{\lambda}=\mathbf{0} \\
\boldsymbol{\lambda} \geqslant \mathbf{0}
\end{array}
$$


> **注解**：一般原问题是一个最小化问题，对偶问题则是一个最大化问题。
> 

这还是一个线性优化问题，但变元的数量是 $m$。我们可以依据实际情况选择是解 原问题 还是解 对偶问题，就看是原问题中的变元数量 $d$ 更小还是原问题中约束数量 $m$ 更小，哪个小选哪个。


#### 二次规划的 Lagrange 对偶

二次规划 (7.10) 的 Lagrangre 函数整理一下之后是
$$
\begin{aligned}
\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda}) & =\frac{1}{2} \boldsymbol{x}^{\top} \boldsymbol{Q} \boldsymbol{x}+\boldsymbol{c}^{\top} \boldsymbol{x}+\boldsymbol{\lambda}^{\top}(\boldsymbol{A} \boldsymbol{x}-\boldsymbol{b}) \\
& =\frac{1}{2} \boldsymbol{x}^{\top} \boldsymbol{Q} \boldsymbol{x}+\left(\boldsymbol{c}+\boldsymbol{A}^{\top} \boldsymbol{\lambda}\right)^{\top} \boldsymbol{x}-\boldsymbol{\lambda}^{\top} \boldsymbol{b}
\end{aligned}
$$

按三步法求解：

1. **对 $\boldsymbol{x}$ 求导**：$\dfrac{\partial \mathfrak{L}}{\partial \boldsymbol{x}} = \boldsymbol{Q}\boldsymbol{x} + (\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda}) = \boldsymbol{0}$。与线性规划不同，$\mathfrak{L}$ 关于 $\boldsymbol{x}$ 是二次的，所以令导数为零就是在求真正的驻点（极小值点）。

2. **解出 $\boldsymbol{x}^*(\boldsymbol{\lambda})$**：假设 $\boldsymbol{Q}$ 可逆，得到
$$
\boldsymbol{x}^*(\boldsymbol{\lambda}) = -\boldsymbol{Q}^{-1}(\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda})
$$

1. **代回 $\mathfrak{L}$**：将 $\boldsymbol{x}^*(\boldsymbol{\lambda})$ 代入 $\mathfrak{L}(\boldsymbol{x}, \boldsymbol{\lambda})$，得到对偶函数
$$
D(\boldsymbol{\lambda}) = -\frac{1}{2}(\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda})^{\top}\boldsymbol{Q}^{-1}(\boldsymbol{c} + \boldsymbol{A}^{\top}\boldsymbol{\lambda}) - \boldsymbol{\lambda}^{\top}\boldsymbol{b}
$$

于是二次规划的对偶优化问题就是
$$
\begin{aligned}
\max _{\boldsymbol{\lambda} \in \mathbb{R}^{m}} & -\frac{1}{2}\left(\boldsymbol{c}+\boldsymbol{A}^{\top} \boldsymbol{\lambda}\right)^{\top} \boldsymbol{Q}^{-1}\left(\boldsymbol{c}+\boldsymbol{A}^{\top} \boldsymbol{\lambda}\right)-\boldsymbol{\lambda}^{\top} \boldsymbol{b} \\
\text { subject to } \boldsymbol{\lambda} & \geqslant \mathbf{0}
\end{aligned}
$$




### Legendre-Fenchel 变换和凸共轭

让我们不考虑约束，重新回顾 Lagrange 对偶中的对偶概念。

> **注解**
> 物理系学生常常在学习经典力学中的 Lagrange 量和 Hamilton 量的时候接触 Legendre 变换。

---

#### 记号约定

- $\boldsymbol{x}, \boldsymbol{s} \in \mathbb{R}^{D}$ 均为**列向量**
- 内积写作 $\boldsymbol{s}^{\top}\boldsymbol{x}$，不使用抽象的 $\langle \cdot, \cdot \rangle$ 记号
- $\nabla f(\boldsymbol{x}) \in \mathbb{R}^{D}$ 为**梯度（列向量）**
- $f^{*}$ 表示 $f$ 的**凸共轭（Legendre-Fenchel 变换）**

---

#### 核心思想：用切线族重新"编码"函数

**问题**：描述一个凸函数 $f$，除了给出 $f(\boldsymbol{x})$ 的解析式之外，还有什么等价方式？

**关键观察**：凸集可由其**支撑超平面**完全确定。凸函数 $f$ 的上镜图 $\mathrm{epi}(f) = \{(\boldsymbol{x}, t) : t \geq f(\boldsymbol{x})\}$ 是一个凸集，因此也可以用支撑超平面来描述。而上镜图的支撑超平面恰好对应着 $f$ 在各点的**切超平面**。

**结论**：我们可以放弃自变量 $\boldsymbol{x}$，改用**切超平面的斜率** $\boldsymbol{s}$ 作为新自变量，用**截距**作为新函数值，从而用一族切超平面来等价地描述原曲面。这就是 Legendre-Fenchel 变换的本质。

---

#### 一元情形的几何直觉

以 $f(x) = x^2$ 为例，逐步建立直觉。

**Step 1. 写出过图上一点的直线**

对于 $f$ 图上的点 $(x_0, f(x_0))$，斜率为 $s$ 的直线为：
$$y - f(x_0) = s(x - x_0) \tag{7.24}$$

整理为 $y = sx + c$ 的形式，得到 $y$ 轴截距：
$$c = f(x_0) - sx_0 \tag{7.25}$$

**Step 2. 找支撑直线（切线）**

固定斜率 $s$，我们想找到一条斜率为 $s$ 的直线，使得它在 $f$ 的图像**下方**并且刚好与 $f$ 相切。这等价于让截距 $c$ 取最小值：
$$c_{\min} = \inf_{x_0} \big[ f(x_0) - sx_0 \big]$$

**Step 3. 定义凸共轭为截距的负值**

按照惯例，定义凸共轭为 $-c_{\min}$（即截距取负后求最大值）：
$$f^{*}(s) = \sup_{x_0} \big[ sx_0 - f(x_0) \big]$$

**几何意义**：$f^{*}(s)$ 就是斜率为 $s$ 的切线在纵轴上的截距的**负值**。换言之，Legendre-Fenchel 变换的本质是：**以切线斜率 $s$ 为新自变量，以 $sx - f(x)$ 的最大值为新函数值，从而用切线族重新编码原函数**。

---

#### 形式定义

上述一元推导可以直接推广到多元情形。

> **定义 7.4（凸共轭 / Legendre-Fenchel 变换）**
> 函数 $f: \mathbb{R}^{D} \rightarrow \mathbb{R}$ 的 **凸共轭** 定义为
> $$\boxed{f^{*}(\boldsymbol{s}) = \sup_{\boldsymbol{x} \in \mathbb{R}^{D}} \Big[ \boldsymbol{s}^{\top}\boldsymbol{x} - f(\boldsymbol{x}) \Big]} \tag{7.26}$$

**几点说明**：

1. 凸共轭的定义**不要求** $f$ 是凸的或可微的。即使 $f$ 非凸，$f^{*}$ 也总是凸函数（因为它是关于 $\boldsymbol{s}$ 的一族仿射函数的逐点上确界）。
2. $\boldsymbol{s}$ 称为**对偶变量**，对应于原函数的"斜率"或"梯度"方向。
3. 这是**函数 $f(\cdot)$ 的变换**（类似于傅里叶变换），而不是对变量 $\boldsymbol{x}$ 的变换。

---

#### 可微凸函数的特殊情况：经典 Legendre 变换

当 $f$ **严格凸且可微**时，上确界在唯一一点取到，推导大幅简化。

**Step 1. 一阶最优性条件**

固定 $\boldsymbol{s}$，令 $h(\boldsymbol{x}) = \boldsymbol{s}^{\top}\boldsymbol{x} - f(\boldsymbol{x})$。由于 $f$ 严格凸，$h$ 严格凹，其最大值点 $\boldsymbol{x}^{*}$ 唯一，由一阶条件确定：
$$\nabla_{\boldsymbol{x}} h = \boldsymbol{s} - \nabla f(\boldsymbol{x}^{*}) = \boldsymbol{0} \quad \Longrightarrow \quad \boxed{\boldsymbol{s} = \nabla f(\boldsymbol{x}^{*})} \tag{7.27}$$

**Step 2. 代入得到凸共轭表达式**

将 $\boldsymbol{x}^{*}$ 代回定义：
$$\boxed{f^{*}(\boldsymbol{s}) = \boldsymbol{s}^{\top}\boldsymbol{x}^{*} - f(\boldsymbol{x}^{*}), \quad \text{其中 } \boldsymbol{s} = \nabla f(\boldsymbol{x}^{*})} \tag{7.28}$$

此时不需要求上确界，因为最优解由 $\boldsymbol{s} = \nabla f(\boldsymbol{x}^{*})$ 唯一确定。

**Step 3. 对偶梯度关系**

由于 $f$ 严格凸，映射 $\boldsymbol{x} \mapsto \boldsymbol{s} = \nabla f(\boldsymbol{x})$ 可逆。可以证明：
$$\boxed{\boldsymbol{s} = \nabla f(\boldsymbol{x}), \qquad \boldsymbol{x} = \nabla f^{*}(\boldsymbol{s})} \tag{7.29}$$

这就是原函数和共轭函数之间**完美的对偶关系**：$f$ 的梯度给出 $\boldsymbol{s}$，$f^{*}$ 的梯度反过来恢复 $\boldsymbol{x}$。

---

#### 凸共轭的基本性质

凸共轭具有以下重要性质：

**性质 1（Young-Fenchel 不等式）**：对任意 $\boldsymbol{x}$ 和 $\boldsymbol{s}$，
$$\boxed{f(\boldsymbol{x}) + f^{*}(\boldsymbol{s}) \geq \boldsymbol{s}^{\top}\boldsymbol{x}}$$

等号成立当且仅当 $\boldsymbol{s} \in \partial f(\boldsymbol{x})$（$\boldsymbol{s}$ 属于 $f$ 在 $\boldsymbol{x}$ 处的次微分）。这直接由 $\sup$ 的定义得出。

**性质 2（对合性）**：若 $f$ 是闭真凸函数，则
$$\boxed{f^{**} = f}$$

即共轭的共轭是其自身。

**性质 3（梯度互逆）**：若 $f$ 严格凸可微，则 $f^{*}(\boldsymbol{s})$ 在 $\boldsymbol{s}$ 处的梯度恰好是 $\boldsymbol{x}$（如 (7.29) 所述）。

---

#### 应用示例

下面两个例子展示凸共轭在机器学习中的典型用法。

> **示例 7.7（二次函数的凸共轭）**
> 
> 设 $f(\boldsymbol{y}) = \dfrac{\lambda}{2}\boldsymbol{y}^{\top}\boldsymbol{K}^{-1}\boldsymbol{y}$，其中 $\boldsymbol{K} \in \mathbb{R}^{n \times n}$ 正定，$\lambda > 0$。
> 
> 主变量 $\boldsymbol{y} \in \mathbb{R}^{n}$，对偶变量 $\boldsymbol{\alpha} \in \mathbb{R}^{n}$。
> 
> **Step 1**：写出定义
> $$f^{*}(\boldsymbol{\alpha}) = \sup_{\boldsymbol{y}\in \mathbb{R}^{n}} \left[ \boldsymbol{\alpha}^{\top}\boldsymbol{y} - \frac{\lambda}{2}\boldsymbol{y}^{\top}\boldsymbol{K}^{-1}\boldsymbol{y} \right]$$
> 
> **Step 2**：一阶最优性条件（对 $\boldsymbol{y}$ 求梯度，令其为零）
> $$\nabla_{\boldsymbol{y}} \left[ \boldsymbol{\alpha}^{\top}\boldsymbol{y} - \frac{\lambda}{2}\boldsymbol{y}^{\top}\boldsymbol{K}^{-1}\boldsymbol{y} \right] = \boldsymbol{\alpha} - \lambda \boldsymbol{K}^{-1}\boldsymbol{y} = \boldsymbol{0}$$
> 
> 解得：$\boldsymbol{y}^{*} = \dfrac{1}{\lambda}\boldsymbol{K}\boldsymbol{\alpha}$
> 
> **Step 3**：代回计算
> $$\begin{aligned}
> f^{*}(\boldsymbol{\alpha}) &= \boldsymbol{\alpha}^{\top}\left(\frac{1}{\lambda}\boldsymbol{K}\boldsymbol{\alpha}\right) - \frac{\lambda}{2}\left(\frac{1}{\lambda}\boldsymbol{K}\boldsymbol{\alpha}\right)^{\top}\boldsymbol{K}^{-1}\left(\frac{1}{\lambda}\boldsymbol{K}\boldsymbol{\alpha}\right) \\
> &= \frac{1}{\lambda}\boldsymbol{\alpha}^{\top}\boldsymbol{K}\boldsymbol{\alpha} - \frac{1}{2\lambda}\boldsymbol{\alpha}^{\top}\boldsymbol{K}\boldsymbol{\alpha}
> \end{aligned}$$
> 
> **结论**：
> $$\boxed{f^{*}(\boldsymbol{\alpha}) = \frac{1}{2\lambda}\boldsymbol{\alpha}^{\top}\boldsymbol{K}\boldsymbol{\alpha}}$$

> **示例 7.8（可分离函数的凸共轭 = 各分量共轭之和）**
> 
> 机器学习中，总损失常常是各样本损失之和。设 $\displaystyle \mathcal{L}(\boldsymbol{t}) = \sum_{i=1}^{n} \ell_{i}(t_{i})$，则：
> 
> $$\begin{aligned}
> \mathcal{L}^{*}(\boldsymbol{z})
> &= \sup_{\boldsymbol{t} \in \mathbb{R}^{n}} \left[ \boldsymbol{z}^{\top}\boldsymbol{t} - \sum_{i=1}^{n} \ell_{i}(t_{i}) \right] & & \text{（定义）}\\
> &= \sup_{\boldsymbol{t}} \sum_{i=1}^{n} \left[ z_{i} t_{i} - \ell_{i}(t_{i}) \right] & & \text{（展开内积）}\\
> &= \sum_{i=1}^{n} \sup_{t_{i}} \left[ z_{i} t_{i} - \ell_{i}(t_{i}) \right] & & \text{（各 $t_i$ 独立，sup 可分离）}\\
> &= \sum_{i=1}^{n} \ell_{i}^{*}(z_{i}) & & \text{（凸共轭定义）}
> \end{aligned}$$
> 
> **结论**：$\boxed{\mathcal{L}^{*}(\boldsymbol{z}) = \sum_{i=1}^{n} \ell_{i}^{*}(z_{i})}$，即可分离函数的共轭等于各分量共轭之和。

---

#### Lagrange 对偶与 Legendre-Fenchel 变换的联系

前文中，我们用 Lagrange 乘子推导了对偶问题。本节的 Legendre-Fenchel 变换提供了另一条通往对偶问题的路径。两者的关系可以概括为：

$$\text{Lagrange 对偶} \xrightarrow{\text{min-max 交换 + 分离变量}} \text{凸共轭的形式}$$

下面通过一个完整的例子说明这一联系。

> **示例 7.9（通过凸共轭推导对偶问题）**
> 
> **问题**：设 $f$, $g$ 为凸函数，$\boldsymbol{A}$ 为实矩阵（维度匹配），求解
> $$\min_{\boldsymbol{x}}~f(\boldsymbol{A}\boldsymbol{x}) + g(\boldsymbol{x})$$
> 
> **Step 1：引入辅助变量，建立 Lagrangian**
> 
> 令 $\boldsymbol{y} = \boldsymbol{A}\boldsymbol{x}$，原问题等价于：
> $$\min_{\boldsymbol{x},\, \boldsymbol{y}}~f(\boldsymbol{y}) + g(\boldsymbol{x}) \quad \text{s.t.} \quad \boldsymbol{A}\boldsymbol{x} = \boldsymbol{y}$$
> 
> 引入 Lagrange 乘子 $\boldsymbol{u}$，写出 Lagrangian：
> $$L(\boldsymbol{x}, \boldsymbol{y}, \boldsymbol{u}) = f(\boldsymbol{y}) + g(\boldsymbol{x}) + \boldsymbol{u}^{\top}(\boldsymbol{A}\boldsymbol{x} - \boldsymbol{y})$$
> 
> **Step 2：交换 min-max（强对偶性保证）**
> 
> 由于 $f$, $g$ 凸且约束为线性，强对偶成立：
> $$\min_{\boldsymbol{x}, \boldsymbol{y}} \max_{\boldsymbol{u}} L = \max_{\boldsymbol{u}} \min_{\boldsymbol{x}, \boldsymbol{y}} L$$
> 
> **Step 3：分离变量**
> 
> 展开 $\boldsymbol{u}^{\top}(\boldsymbol{A}\boldsymbol{x} - \boldsymbol{y}) = \boldsymbol{x}^{\top}\boldsymbol{A}^{\top}\boldsymbol{u} - \boldsymbol{y}^{\top}\boldsymbol{u}$，将 $\boldsymbol{x}$ 和 $\boldsymbol{y}$ 的项分开：
> $$\min_{\boldsymbol{x}, \boldsymbol{y}} L = \underbrace{\min_{\boldsymbol{y}} \left[ f(\boldsymbol{y}) - \boldsymbol{y}^{\top}\boldsymbol{u} \right]}_{\text{关于 } \boldsymbol{y}} + \underbrace{\min_{\boldsymbol{x}} \left[ g(\boldsymbol{x}) + \boldsymbol{x}^{\top}\boldsymbol{A}^{\top}\boldsymbol{u} \right]}_{\text{关于 } \boldsymbol{x}}$$
> 
> **Step 4：用凸共轭定义识别结果**
> 
> 回忆 $f^{*}(\boldsymbol{s}) = \sup_{\boldsymbol{x}} [\boldsymbol{s}^{\top}\boldsymbol{x} - f(\boldsymbol{x})]$，等价地 $\inf_{\boldsymbol{x}} [f(\boldsymbol{x}) - \boldsymbol{s}^{\top}\boldsymbol{x}] = -f^{*}(\boldsymbol{s})$。于是：
> 
> - $\displaystyle \min_{\boldsymbol{y}} \left[ f(\boldsymbol{y}) - \boldsymbol{y}^{\top}\boldsymbol{u} \right] = -\sup_{\boldsymbol{y}} \left[ \boldsymbol{u}^{\top}\boldsymbol{y} - f(\boldsymbol{y}) \right] = -f^{*}(\boldsymbol{u})$
> - $\displaystyle \min_{\boldsymbol{x}} \left[ g(\boldsymbol{x}) + \boldsymbol{x}^{\top}\boldsymbol{A}^{\top}\boldsymbol{u} \right] = -\sup_{\boldsymbol{x}} \left[ (-\boldsymbol{A}^{\top}\boldsymbol{u})^{\top}\boldsymbol{x} - g(\boldsymbol{x}) \right] = -g^{*}(-\boldsymbol{A}^{\top}\boldsymbol{u})$
> 
> **结论**：
> $$\boxed{\min_{\boldsymbol{x}}~f(\boldsymbol{A}\boldsymbol{x}) + g(\boldsymbol{x}) = \max_{\boldsymbol{u}}~\big[-f^{*}(\boldsymbol{u}) - g^{*}(-\boldsymbol{A}^{\top}\boldsymbol{u})\big]}$$
> 
> 左边是**原问题**（关于 $\boldsymbol{x}$ 的最小化），右边是**对偶问题**（关于 $\boldsymbol{u}$ 的最大化），对偶目标完全用凸共轭表达。

**总结**：Legendre-Fenchel 共轭为推导对偶问题提供了系统化的工具。特别地，对于可分离的损失函数（如示例 7.8），先计算各分量的共轭 $\ell_i^{*}$，再组合起来，是一种简洁高效的推导对偶问题的方法。









## 7.5 求解优化问题——求解算法

无论是原问题（无约束优化）还是变换后的对偶问题（凸优化），最终都需要具体的求解算法来找到最优解。

### 基于梯度下降的优化

#### 梯度下降算法

梯度下降是一个一阶优化算法。它的每次迭代都将估计点做一个正比于函数在该点处的负梯度向量的移动，以逐步找到一个局部最小值点。

让我们考虑多变量函数。想象一个曲面（由函数 $f(\boldsymbol{x})$ 描述），并设想一个球从某个特定位置 $\boldsymbol{x}_0$ 开始。当球被释放时，它会沿着最陡峭的下坡方向向下滚动。

梯度下降利用了这样一个事实：从 $\boldsymbol{x}_0$ 出发，若朝着函数 $f$ 在 $\boldsymbol{x}_0$ 处负的梯度方向 $-\left((\nabla f)(\boldsymbol{x}_0)\right)^{\top}$ 移动，$f(\boldsymbol{x}_0)$ 的值将最快地减小（假设所涉及的函数都是可微的）。于是假如我们考虑下面的更新：
$$
\boldsymbol{x}_{1} = \boldsymbol{x}_{0} - \gamma \big[ (\nabla f)(\boldsymbol{x}_{0}) \big] ^{\top} \tag{7.30}
$$

若 $\gamma \geqslant 0$ 是一个很小的 **步长**，就有 $f(\boldsymbol{x}_{1}) \leqslant f(\boldsymbol{x}_{0})$。

> 注意我们在梯度的部分使用了转置记号，这是因为我们在本书中默认梯度是行向量——如果不转置的话维度对不上。

有了这个发现，我们就能提出一个简单的**梯度下降算法**：我们想要找到一个函数 $f: \mathbb{R}^{n} \rightarrow \mathbb{R}, \boldsymbol{x} \mapsto f(\boldsymbol{x})$ 的局部最优解 $f(\boldsymbol{x}_{*})$ ，我们从一个初始估计 $\boldsymbol{x}_{0}$ 开始，然后按照下面的更新规则不断迭代
$$
\boldsymbol{x}_{i+1} = \boldsymbol{x}_{i} - \gamma_{i} \big[ (\nabla f)(\boldsymbol{x}_{i}) \big] ^{\top} \tag{7.31}
$$

假设我们每次迭代选择的 **步长（学习率）** 足够合适，我们得到的序列就是一个下降的 "链"：$f(\boldsymbol{x}_{0}) \geqslant f(\boldsymbol{x}_{1}) \geqslant \cdots$ 它最终会趋于函数的局部最小值。

**步长（学习率）** $\gamma_{i}$ 的大小在梯度下降算法中十分重要：

- 如果步长太小，梯度下降的速度会很慢；
- 如果步长太大，梯度下降算法有可能射出原本的 "峡谷" 区域，难以收敛，甚至发散。

---

<center style="font-weight: bold;">示例 7.3</center>

考虑下面的二维二次函数
$$
f\left(\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}\right) = \frac{1}{2}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}^{\top}\begin{bmatrix}2&1\\1&20\end{bmatrix}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix} - \begin{bmatrix}5\\3\end{bmatrix}^{\top}\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}\tag{7.32}
$$

它对 $\boldsymbol{x}$ 的梯度是
$$
\nabla f\left(\begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}\right) = \begin{bmatrix}x_{1}\\x_{2}\end{bmatrix}^{\top}\begin{bmatrix}2&1\\1&20\end{bmatrix} - \begin{bmatrix}5\\3\end{bmatrix}^{\top}\tag{7.33}
$$

如图 7.5 所示，我们从初始估计 $\boldsymbol{x}_{0} = [-3, -1]^{\top}$ 开始用公式 (7.31) 不断迭代，以得到一个收敛于函数最小值的估计值序列。

可见 $\boldsymbol{x}_{0}$ 处的负梯度指向右上方，从而得到第二个估计 $\boldsymbol{x}_{1} = [-1.98, 1.21]^{\top}$ （令 $\gamma = 0.085$，并将 $\boldsymbol{x}_{0}$ 代入 (7.33) ）。再迭代一次，我们得到 $\boldsymbol{x}_{2} = [-1.32, -0.42]^{\top}$，以此类推。

<center>
    <img src="https://datawhalechina.github.io/math-for-ai/ch7/attachments/Pasted%20image%2020250630213059.png" alt="alt text" style="zoom:50%;">
</center>

<center>图 7.5 梯度下降算法的示例</center>


> **注释**：梯度下降算法趋近局部最小值的速度可以很慢，它的渐近收敛速度弱于很多其他算法。
> 
> 在面临一些性质不甚好的凸函数时，我们可以想象一个从很长但很窄的斜坡滚下的球：梯度下降的更新轨迹将会是像图 7.5 那样的锯齿形，每次更新的方向甚至会与该点与局部最小值点的直接连线几乎垂直。这会导致梯度下降的收敛速度非常慢。
> 


#### 动量梯度下降

如图 7.5 所示，如果优化曲面的曲率使得某些区域的性质不好，梯度下降的收敛速度可能会非常慢。曲率使得梯度下降更新在 "峡谷" 两侧跳跃，只能一小步一小步地接近最优值。为提高收敛性，我们可以赋予梯度下降一些 "记忆"。


动量梯度下降（Rumelhart et al., 1986）是一种引入与上一次迭代的相关项的方法。这种记忆可以抑制振荡并使得梯度更新更加平滑。我们像之前一样考虑一个很重的滚动的球，动量项就模拟了它的惯性——很难轻易改变运动方向。这个方法也同时通过记忆梯度的更新以实现移动平均。

具体而言，基于动量的方法会储存第 $i$ 次迭代的更新 $\Delta \boldsymbol{x}_{i}$，然后加在第 $i+1$ 次的梯度更新上；这相当于将第 $i$ 次迭代和第 $i+1$ 次迭代中得到的梯度做线性组合：
$$
\begin{align}
\boldsymbol{x}_{i+1} &= \boldsymbol{x}_{i} - \gamma_{i} \big[ (\nabla f)(\boldsymbol{x}_{i}) \big] ^{\top} + \alpha\Delta \boldsymbol{x}_{i} \tag{7.34}\\
\Delta \boldsymbol{x}_{i} &= \boldsymbol{x}_{i} - \boldsymbol{x}_{i-1} = \alpha\Delta \boldsymbol{x}_{i-1} - \gamma_{i-1}\big[ (\nabla f)(\boldsymbol{x}_{i-1}) \big] ^{\top}, \tag{7.35}
\end{align}
$$

其中 $\alpha \in [0, 1]$。有时我们只知道梯度的一个估计值，此时上面的动量项作为移动平均会帮我们抹除梯度估计中的噪声，因此十分有用。




#### 随机梯度下降
精确地计算梯度十分费时费力，但我们往往可以找到更快速地计算梯度估计值的方法 —— 只要我们估计的梯度和真实的梯度方向大致相同。

**随机梯度下降**（SGD）是一种用于最小化可被写成一系列可微函数的目标函数，并给出梯度的随机估计的梯度下降算法。"随机" 一词指的是我们每次更新不知道梯度的真实值，而只有一个**带噪声的梯度估计值**。如果限制梯度估计值的分布，在理论上我们依然可以保证 SGD 的收敛性。

在机器学习中，给定 $n = 1, \dots, N$ 个数据点，我们通常将每个数据的损失 $L_{n}$ 的求和作为目标函数：
$$
L(\boldsymbol{\theta}) = \sum\limits_{n=1}^{N} L_{n}(\boldsymbol{\theta})\tag{7.36}
$$

其中 $\boldsymbol{\theta}$ 是我们关心的参数向量 —— 我们要找出最小化 $L$ 的参数 $\boldsymbol{\theta}$。可以考虑使用**负对数似然函数**，它是每个数据的负对数似然函数的求和：
$$
L(\boldsymbol{\theta}) = -\sum\limits_{n=1}^{N} \log p(y_{n}|\boldsymbol{x}_{n}, \boldsymbol{\theta}) \tag{7.37}
$$

其中 $\boldsymbol{x}_{n} \in \mathbb{R}^{D}$ 是训练中的输入数据，$y_{n}$ 是训练中的目标数据，$\boldsymbol{\theta}$ 是回归模型的参数。

前文提到，经典的梯度下降是一个 "整批" 的优化方法，这是说每次我们都要选一个合适的 $\gamma_{i}$，并用 **所有的** 训练集来完成下面的迭代：
$$
\boldsymbol{\theta}_{i+1} + \boldsymbol{\theta}_{i} = \gamma_{i}\big[ \nabla L(\boldsymbol{\theta}_{i}) \big] ^{\top} = \boldsymbol{\theta}_{i} - \gamma_{i}\sum\limits_{n=1}^{N} \big[ \nabla L_{n}(\boldsymbol{\theta}_{i}) \big] ^{\top}\tag{7.38}
$$

计算上面对所有 $L_{n}$ 的梯度之和是个大工程。当训练集很大，或是没有显式的梯度可以求解的时候，这么做显然是极其昂贵的。

考虑 (7.38) 中的一项 $\displaystyle \sum\limits_{n=1}^{N} [\nabla L_{n}(\boldsymbol{\theta})]$，我们可以通过只算一小部分 $L_{n}$ 的梯度之和来降低计算成本。

相较于用上全部 $L_{n}, n = 1, \dots, N$ 的经典梯度下降算法，我们只选择小部分 $L_{n}$ ，这样我们就得到了 **小批次梯度下降**；该算法最极端的情况是每次只考虑一个 $L_{n}$。

我们这么做是有道理的：

- 我们只需要拿到一个对真实梯度的 **无偏估计**
- 而公式 (7.38) 中的 $\displaystyle \sum\limits_{n=1}^{N} [\nabla L_{n}(\boldsymbol{\theta})]$ 事实上就是对梯度期望值的经验估计，因此任何对梯度的无偏估计都可以拿来用。
- 不论我们的小批次中的数据量是多少它都是对梯度的无偏估计，SGD 也总会收敛。

> **注释**：在相对较弱的假设下，如果学习率以适当的幅度逐步降低，SGD **几乎必然 (almost surely)** 收敛到局部最优解。 (Bottu, 1998)
>



我们为什么要估计梯度的值呢？主要的原因是实践中的 CPU 和 GPU 的存储空间或是计算时间有限。我们可以考虑不同大小的批次。

- 较大的批次不但可以利用高效的矩阵算法快速计算结果，还会给出梯度更加精确的估计，降低了参数更新的方差，算法的收敛也会更稳定。
- 相比之下较小的批次可以更快的算出，但牺牲了估计的精确性，这可能会让我们陷入更差的局部最优而无法脱离。







### 内点法

梯度下降及其变体是**一阶方法**，适用于大规模无约束优化（如深度学习）。但在 7.4 节中，我们通过问题变换得到的对偶问题通常是中小规模的**带约束凸优化**，此时有更高效的专用求解器——**内点法（Interior Point Method）** 就是其中的代表。

#### 基本思想

内点法的核心思想是：不直接处理不等式约束 $\boldsymbol{\lambda} \geqslant \boldsymbol{0}$，而是用一个**障碍函数（barrier function）** 将约束融入目标函数中，使得迭代点始终保持在可行域的**内部**（这也是"内点"一词的由来）。

具体来说，对于带约束问题
$$
\min_{\boldsymbol{x}}~f(\boldsymbol{x}) \quad \text{subject to} \quad g_i(\boldsymbol{x}) \leqslant 0, \quad i = 1, \dots, m
$$

内点法构造如下的近似问题
$$
\min_{\boldsymbol{x}}~f(\boldsymbol{x}) - t \sum_{i=1}^{m} \log(-g_i(\boldsymbol{x})) \tag{7.39}
$$

其中 $t > 0$ 是障碍参数，$-\log(-g_i(\boldsymbol{x}))$ 是**对数障碍函数**：当 $g_i(\boldsymbol{x})$ 趋近于 $0$（即接近约束边界）时，障碍项趋向 $+\infty$，从而"惩罚"迭代点靠近边界的行为。

#### 求解流程

内点法通过逐步减小 $t$ 来逼近原问题的解：

1. 选择初始可行点 $\boldsymbol{x}_0$（满足所有 $g_i(\boldsymbol{x}_0) < 0$）和初始障碍参数 $t_0$；
2. 固定 $t$，用牛顿法求解无约束问题 (7.39)，得到近似解 $\boldsymbol{x}^*(t)$；
3. 减小 $t$（例如 $t \leftarrow t / \mu$，$\mu > 1$），使障碍项的影响减弱；
4. 重复步骤 2-3，直到 $t$ 足够小，此时 $\boldsymbol{x}^*(t)$ 收敛到原问题的最优解。

#### 与梯度下降的对比

| | 梯度下降 | 内点法 |
|---|---|---|
| 阶数 | 一阶（只用梯度） | 二阶（用梯度 + Hessian） |
| 约束处理 | 无约束问题 | 带约束凸优化 |
| 收敛速度 | 线性收敛，较慢 | 超线性收敛，较快 |
| 适用规模 | 大规模（百万级参数） | 中小规模（对偶问题） |
| 典型应用 | 深度学习 | LP/QP 对偶问题、SVM |

在 7.4 节的 Lagrange 对偶流程中，第 2 步"求解对偶问题"就可以使用内点法：对偶问题是凸优化，规模通常不大（变量数为约束数 $m$），内点法能快速给出高精度解。

---

## 附录：Legendre-Fenchel 变换的完整推导

本附录给出凸共轭理论的自包含推导，补充正文 7.3 节中省略的细节。

### 记号约定

- $x, u \in \mathbb{R}^n$ 均为**列向量**；$dx, du$ 为对应的微分
- $\nabla f(x) \in \mathbb{R}^n$ 为**梯度（列向量）**
- 内积显式写作 $a^{\top} b$，绝不省略转置

---

### A.1 从全微分推导经典 Legendre 变换

**Step 1. 原函数的全微分**

设 $f: \mathbb{R}^n \to \mathbb{R}$ 可微，其全微分为：
$$df(x) = \nabla f(x)^{\top} dx$$

令 $u = \nabla f(x)$，则：
$$df = u^{\top} dx \tag{A.1}$$

**Step 2. 对 $u^{\top} x$ 应用乘积法则**

$u^{\top} x$ 是标量，对其求全微分：
$$d(u^{\top} x) = (du)^{\top} x + u^{\top} dx \tag{A.2}$$

**Step 3. 消去 $u^{\top} dx$，识别新函数**

将 (A.1) 代入 (A.2) 并移项：
$$(du)^{\top} x = d(u^{\top} x - f) \tag{A.3}$$

右边恰好是某个以 $u$ 为自变量的函数的全微分。

**Step 4. 定义 Legendre 共轭**

假设 $f$ **严格凸**，则 $x \mapsto u = \nabla f(x)$ 可逆，记逆映射为 $x(u)$。定义：
$$f^{*}(u) \coloneqq u^{\top} x(u) - f(x(u)) \tag{A.4}$$

**Step 5. 验证对偶梯度关系**

对 (A.4) 求全微分，应用乘积法则与链式法则（记 $J_x = \partial x / \partial u$）：
$$df^{*} = (du)^{\top} x + u^{\top} J_x\, du - \nabla f(x)^{\top} J_x\, du$$

由于 $u = \nabla f(x)$，后两项相消：
$$df^{*} = x^{\top} du \tag{A.5}$$

与梯度定义 $df^{*} = \nabla f^{*}(u)^{\top} du$ 比较，得：
$$\boxed{x = \nabla f^{*}(u)} \tag{A.6}$$

**结论**：原函数与共轭函数之间满足对偶梯度关系：
$$\boxed{u = \nabla f(x), \qquad x = \nabla f^{*}(u)} \tag{A.7}$$

---

### A.2 几何意义（一元情形）

当 $n=1$ 时，$u = f'(x)$ 是切线斜率，此时：
$$f^{*}(u) = ux - f(x)$$

这恰好是 $f$ 在点 $x$ 处切线的**纵轴截距的负值**。Legendre 变换的本质：放弃自变量 $x$，改用切线斜率 $u$ 作为新自变量，用截距作为新函数值，从而用一族切线来重新"编码"原曲线。

---

### A.3 Fenchel 共轭：推广到非光滑/非凸函数

经典 Legendre 变换要求 $f$ 严格凸且可微。**Fenchel 共轭**（凸共轭）去掉这些假设。

设 $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ 是**真函数**（proper：不恒为 $+\infty$ 且从不取 $-\infty$），其有效域为 $\mathrm{dom}\, f = \{x : f(x) < +\infty\}$。

> **定义（Fenchel 共轭）**
> $$\boxed{f^{*}(y) = \sup_{x \in \mathrm{dom}\, f} \left\{ y^{\top} x - f(x) \right\}} \tag{A.8}$$

$f^{*}$ 总是**凸函数**（即使 $f$ 本身非凸），因为它是关于 $y$ 的一族仿射函数的逐点上确界。

**与经典 Legendre 变换的关系**：当 $f$ 严格凸可微时，$\sup$ 在 $y = \nabla f(x^{*})$ 唯一取到，(A.8) 退化为 (A.4)。

---

### A.4 Young-Fenchel 不等式

由 $\sup$ 的定义，对任意 $x \in \mathrm{dom}\, f$ 和任意 $y \in \mathbb{R}^n$：
$$f^{*}(y) \geq y^{\top} x - f(x)$$

移项得 **Young-Fenchel 不等式**：
$$\boxed{f(x) + f^{*}(y) \geq y^{\top} x} \tag{A.9}$$

等号成立 $\Longleftrightarrow$ $y \in \partial f(x)$（$y$ 属于 $f$ 在 $x$ 处的次微分）。

---

### A.5 二次共轭定理：$f^{**} = f$

设 $f$ 是**闭真凸函数**（closed proper convex，即 $\mathrm{epi}\, f$ 是闭集）。

> **定理**：$f^{**}(x) = f(x)$，即共轭的共轭恢复原函数。

**证明**：

**Step 1（$f^{**} \leq f$）**：由 Young-Fenchel 不等式，对任意 $y$：
$$x^{\top} y - f^{*}(y) \leq f(x)$$

对 $y$ 取上确界：$f^{**}(x) = \sup_y \{x^{\top} y - f^{*}(y)\} \leq f(x)$。

**Step 2（$f^{**} \geq f$）**：由于 $f$ 是闭真凸函数，由支撑超平面刻画：
$$f(x) = \sup_{x_0,\; g \in \partial f(x_0)} \left\{ f(x_0) + g^{\top}(x - x_0) \right\}$$

对任意 $x_0$ 和 $g \in \partial f(x_0)$：
$$f(x_0) + g^{\top}(x - x_0) = g^{\top} x - \underbrace{(g^{\top} x_0 - f(x_0))}_{\leq\, f^{*}(g)} \leq g^{\top} x - f^{*}(g) \leq f^{**}(x)$$

对所有 $x_0$, $g$ 取上确界得 $f(x) \leq f^{**}(x)$。结合 Step 1：$\boxed{f^{**} = f}$。

**可微情形的简化**：若 $f$ 严格凸可微，$y = \nabla f(x_0)$ 建立双射，则
$$f^{**}(x) = \sup_{x_0} \left\{ f(x_0) + \nabla f(x_0)^{\top}(x - x_0) \right\}$$

由凸函数一阶条件，$\sup$ 在 $x_0 = x$ 时取到，值为 $f(x)$。

---

### A.6 凸共轭的运算规则

#### (a) 可分离函数

设 $f(x_1, x_2) = f_1(x_1) + f_2(x_2)$，则：
$$\boxed{f^{*}(y_1, y_2) = f_1^{*}(y_1) + f_2^{*}(y_2)} \tag{A.10}$$

推导：各变量独立，$\sup$ 可逐变量分离。

#### (b) 缩放与平移

设 $g(x) = af(x) + b$（$a > 0$），则：
$$\boxed{g^{*}(y) = a\, f^{*}\!\left(\frac{y}{a}\right) - b} \tag{A.11}$$

#### (c) 线性变换

设 $g(x) = f(Ax + b)$，$A$ **可逆**，则：
$$\boxed{g^{*}(y) = f^{*}(A^{-\top} y) - b^{\top} A^{-\top} y} \tag{A.12}$$

推导：令 $z = Ax + b$，则 $x = A^{-1}(z-b)$，代入 $y^{\top}x$ 并换元即得。

> **注**：若 $A$ 不可逆，则需使用 Moore-Penrose 伪逆并附加值域约束条件。

