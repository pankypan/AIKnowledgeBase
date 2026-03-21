# 解析几何(Analytic Geometry)

## 3.1 范数

当我们考虑几何意义下的向量，也就是原点出发的有向线段时，其长度显然是原点到有向线段终点之间的直线距离。
下面我们将使用范数的概念讨论向量的长度。

> **定义 3.1**（范数）一个*范数*是线性空间$V$上的一个函数：
> $$
> \begin{align} \| \cdot \|: V &\rightarrow \mathbb{R} \tag{3.1}\\ x &\mapsto \| x \|, \tag{3.2}\end{align}
> $$
> 它给出每个线性空间中每个向量$x$的实值*长度*$\| x \| \in \mathbb{R}$，且对于任意的$x, y \in V$以及$\lambda \in \mathbb{R}$，满足下面的条件：
> * （绝对一次齐次）$\| \lambda x\| = |\lambda| \|x\|$，
> * （三角不等式）$\|x + y\| \leqslant \|x\| + \|y\|$，
> * （半正定）$\|x\| \geqslant 0$，当且仅当$x = 0$时取等


> **示例 3.1**（曼哈顿范数）
> $\mathbb{R}^{n}$上的*曼哈顿范数*（又叫$\mathscr{l}_{1}$范数）的定义如下：
> $$
> \|x\|_{1} := \sum\limits_{i=1}^{n} | x_{i} |, \tag{3.3}
> $$
> 其中$| \cdot |$是绝对值函数。 图 3.3 的左侧显示了平面$\mathbb{R}^{2}$上所有满足$\| x\| =  1$的点集。


> **示例 3.2** （ Euclid 范数）
> 向量$x \in \mathbb{R}^{n}$的* Euclid 范数*（又叫$\mathscr{l}_{2}$范数）定义如下：
> $$
> \|x\|_{2} := \sqrt{ \sum\limits_{i=1}^{n} x_{i}^{2} } = \sqrt{ x^{\top}x }, \tag{3.4}
> $$
> 它计算向量$x$从原点出发到终点的 Euclid 距离（译者注：也就是我们通常意义下的距离）。图 3.3 的右侧显示了$\mathbb{R}^{2}$平面上所有满足$\|x\|_{2} = 1$的点集。

<center>
<img src="https://datawhalechina.github.io/math-for-ai/ch3/attachments/Pasted%20image%2020250225195053.png" style="zoom: 40%;" alt="曼哈顿范数和 Euclid 范数的几何表示" />
</center>
<center>图 3.3：平面上满足向量在不同范数的度量下值为1的情况：左侧为曼哈顿范数，右侧为 Euclid 范数</center>





## 3.2 内积

引入内积的一个主要目的是确认两个向量是否*正交*。

### 3.2.1 点积

我们已经熟悉一些特殊形式的点积，如标量积或$\mathbb{R}^{n}$中的点积，由下面的式子给出：
$$
x^{\top}y = \sum\limits_{i=1}^{n} x_{i}y_{i}. \tag{3.5}
$$
在本书中，我们称这样的内积形式为*点积*。



### 3.2.2 一般的点积

> **定义 3.2**
> 设$V$为线性空间，双线性映射$\Omega:  V \times V \rightarrow \mathbb{R}$将两个$V$中的向量映射到一个实数，则
> * 若对所有$x, y \in V$，都有$\Omega(x, y) = \Omega(y, x)$，也即两个变量可以调换顺序，则称$\Omega$为*对称*的
> * 若对所有$x \in V$，都有
> $$
> \forall x \in V \setminus \{ 0 \}: \Omega(x, x) > 0, ~~ ~~ \Omega(0, 0) = 0, \tag{3.8}
> $$
> 则称$\Omega$为*正定*的。


> **定义 3.3**
> 设$V$为线性空间，双线性映射$\Omega:  V \times V \rightarrow \mathbb{R}$将两个$V$中的向量映射到一个实数，则
> * 对称且正定的双线性映射$\Omega$叫做$V$上的一个*内积*，并简写$\Omega(x, y)$为$\left\langle x, y \right\rangle$。
> * 二元组$(V, \left\langle \cdot, \cdot \right\rangle)$称为*内积空间*或*装配有内积的（实）线性空间*。特别地，如果内积采用（式 3.5）中定义的点积，则称$(V, \left\langle \cdot, \cdot \right\rangle)$为 Euclid 线性空间（译者注：简称欧氏空间）

本书中我们称这些空间为内积空间。




### 3.2.3 对称和正定矩阵

> **定义 3.4**（对称正定矩阵）
> $$
> \forall x \in V - \{ 0 \}: x^{\top}Ax > 0. \tag{3.11}
> $$
> 一个$n$级对称矩阵$A \in \mathbb{R}^{n \times n}$若满足（式 3.11），则叫做*对称正定矩阵*（或仅称为正定矩阵）。如果只满足将（式 3.11）中的不等号改成$\geqslant$的条件，则称为*对称半正定矩阵*


> **示例 3.4**（对称正定矩阵）
> 考虑下面两个矩阵
> $$A_{1} = \left[ \begin{matrix} 9 & 6 \\ 6 & 5 \end{matrix}\right] , \quad A_{2} = \left[  \begin{matrix} 9 & 6 \\ 6 & 3 \end{matrix} \right], \tag{3.12}$$
> 其中 $A_{1}$ 是对称且正定的，因为它不仅对称（译者注：这显而易见），而且对于任意 $x \in \mathbb{R}^{2} - \{ 0 \}$ 都有，
> $$
> \begin{align} x^{\top}A_{1}x &= \left[ \begin{matrix} x_{1} & x_{2} \end{matrix}\right]\left[ \begin{matrix} 9 & 6 \\ 6 & 5 \end{matrix}\right]\left[ \begin{matrix} x_{1} \\ x_{2}  \end{matrix}\right] \\\ &= 9x_{1}^{2} + 12x_{1}x_{2} + 5x_{2}^{2} \\ &= (3x_{1} + 2x_{2})^{2} + x_{2}^{2} > 0.\end{align} \tag{3.13}
> $$
> 相反地，$A_{2}$不是正定矩阵。如果取$x = [2, -3]^{\top}$，可以验证二次型$x^\top Ax$是负数。




假设$A \in \mathbb{R}^{n \times n}$是一个对称正定矩阵，则它可以定义一个在基$B$下的**内积**：
$$
\left\langle x, y \right\rangle = \hat{x}^{\top}A\hat{y}, \tag{3.15}
$$
其中$x, y \in V$。

> **定理 3.5**
> 考虑一个有限维实线性空间$V$及它的一个基（有序）$B$，双线性函数 $\left\langle \cdot, \cdot \right\rangle: V \times V \rightarrow R$是其上的一个内积<u>当且仅当</u>有一个对称正定矩阵$A \in \mathbb{R}^{n \times n}$，与之对应，即
> $$
> \left\langle x, y \right\rangle = \hat{x}^{\top} A \hat{y}.
> $$





## 3.3 向量长度和距离

> **定义 3.6**（距离和度量）
> 考虑一个内积空间$(V, \left\langle \cdot, \cdot \right\rangle)$，任取向量$x, y \in V$，称
> $$
> d(x, y) := \|x - y\| = \sqrt{ \left\langle x - y, x - y \right\rangle  } \tag{3.21}
> $$
> 为向量$x$和$y$之间的*距离*。如果我们选用点积作为$V$上的内积，则得出的距离称为* Euclid 距离*（也称*欧氏距离*）。这样的映射
> $$
> \begin{align} d: V \times V & \rightarrow \mathbb{R} \tag{3.22}\\ (x, y) & \mapsto d(x, y) \tag{3.23}\end{align}
> $$
> 称为*度量*。

> **注释**
> 和向量长度类似，确定向量之间的距离不一定需要内积，使用范数足矣。如果我们有由内积有道德范数，向量间的距离因选择的内积的不同而不同。


一个度量$d$满足下面三条性质：
1. （正定性）对任意的$x, y \in V$，$d(x, y) \geqslant 0$，当且仅当$x=y$时取等，
2. （对称性）对任意的$x, y \in V$，$d(x, y) = d(y, x)$，
3. （三角不等式）对任意的$x, y, z \in V$，$d(x, y) + d(y, z) \geqslant d(x, z)$。

> **注释**
> 第一次看到度量的定义时，读者会发现它和内积十分相似。但如果细致比对定义 3.3 和定义 3.6，我们会发现二者的“方向”截然相反。如果两向量$x, y \in V$的内积较大，则它们之间的度量较小，反之亦然。




## 3.4 向量夹角和正交




## 3.5 正交基



## 3.6 正交补




## 3.7 函数的内积



## 3.8 正交投影



## 3.9 旋转







