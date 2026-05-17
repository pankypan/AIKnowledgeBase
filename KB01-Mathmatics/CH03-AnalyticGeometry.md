# 解析几何(Analytic Geometry)

## 3.1 范数

当我们考虑几何意义下的向量，也就是原点出发的有向线段时，其长度显然是原点到有向线段终点之间的直线距离。
下面我们将使用范数的概念讨论向量的长度。

> **定义 3.1**（范数）一个*范数*是线性空间$V$上的一个函数：
> $$ \begin{align} \| \cdot \|: V &\rightarrow \mathbb{R} \tag{3.1}\\ x &\mapsto \| x \|, \tag{3.2}\end{align} $$
> 它给出每个线性空间中每个向量$x$的实值*长度*$\| x \| \in \mathbb{R}$，且对于任意的$x, y \in V$以及$\lambda \in \mathbb{R}$，满足下面的条件：
> * （绝对一次齐次）$\| \lambda x\| = |\lambda| \|x\|$，
> * （三角不等式）$\|x + y\| \leqslant \|x\| + \|y\|$，
> * （半正定）$\|x\| \geqslant 0$，当且仅当$x = 0$时取等


> **示例 3.1**（曼哈顿范数）
> $\mathbb{R}^{n}$上的*曼哈顿范数*（又叫$\mathscr{l}_{1}$范数）的定义如下：
> $$\|x\|_{1} := \sum\limits_{i=1}^{n} | x_{i} |, \tag{3.3}$$
> 其中$| \cdot |$是绝对值函数。 图 3.3 的左侧显示了平面$\mathbb{R}^{2}$上所有满足$\| x\| =  1$的点集。


> **示例 3.2** （ Euclid 范数）
> 向量$x \in \mathbb{R}^{n}$的* Euclid 范数*（又叫$\mathscr{l}_{2}$范数）定义如下：
> $$ \|x\|_{2} := \sqrt{ \sum\limits_{i=1}^{n} x_{i}^{2} } = \sqrt{ x^{\top}x }, \tag{3.4}$$
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
> $$\forall x \in V \setminus \{ 0 \}: \Omega(x, x) > 0, ~~ ~~ \Omega(0, 0) = 0, \tag{3.8}$$
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
> $$\begin{align} x^{\top}A_{1}x &= \left[ \begin{matrix} x_{1} & x_{2} \end{matrix}\right]\left[ \begin{matrix} 9 & 6 \\ 6 & 5 \end{matrix}\right]\left[ \begin{matrix} x_{1} \\ x_{2}  \end{matrix}\right] \\\ &= 9x_{1}^{2} + 12x_{1}x_{2} + 5x_{2}^{2} \\ &= (3x_{1} + 2x_{2})^{2} + x_{2}^{2} > 0.\end{align} \tag{3.13}$$
> 相反地，$A_{2}$不是正定矩阵。如果取$x = [2, -3]^{\top}$，可以验证二次型$x^\top Ax$是负数。




假设$A \in \mathbb{R}^{n \times n}$是一个对称正定矩阵，则它可以定义一个在基$B$下的**内积**：

$$
\left\langle x, y \right\rangle = \hat{x}^{\top}A\hat{y}, \tag{3.15}
$$

其中$x, y \in V$。

> **定理 3.5**
> 考虑一个有限维实线性空间$V$及它的一个基（有序）$B$，双线性函数 $\left\langle \cdot, \cdot \right\rangle: V \times V \rightarrow R$是其上的一个内积<u>当且仅当</u>有一个对称正定矩阵$A \in \mathbb{R}^{n \times n}$，与之对应，即
> $$\left\langle x, y \right\rangle = \hat{x}^{\top} A \hat{y}.$$





## 3.3 向量长度和距离

> **定义 3.6**（距离和度量）
> 考虑一个内积空间$(V, \left\langle \cdot, \cdot \right\rangle)$，任取向量$x, y \in V$，称
> $$d(x, y) := \|x - y\| = \sqrt{ \left\langle x - y, x - y \right\rangle  } \tag{3.21}$$
> 为向量$x$和$y$之间的*距离*。如果我们选用点积作为$V$上的内积，则得出的距离称为* Euclid 距离*（也称*欧氏距离*）。这样的映射
> $$\begin{align} d: V \times V & \rightarrow \mathbb{R} \tag{3.22}\\ (x, y) & \mapsto d(x, y) \tag{3.23}\end{align}$$
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

利用 Cauchy-Schwarz 不等式，内积空间中两个非零向量 $x, y$ 的夹角 $\omega \in [0, \pi]$ 由下式定义：

$$
\cos\omega = \frac{\left\langle x, y \right\rangle}{\|x\| \|y\|}. \tag{3.25}
$$

> **定义3.7（正交）**：两个向量 $x$ 和 $y$ **正交**当且仅当 $\left\langle x, y \right\rangle = 0$，记作 $x \perp y$。若同时满足 $\|x\| = \|y\| = 1$，则称它们**单位正交（orthonormal）**。零向量与任意向量正交。

注意：正交性依赖于所选的内积。在一个内积下正交的两个向量，在另一个内积下不一定正交。

> **定义3.8（正交矩阵）**：方阵 $A \in \mathbb{R}^{n \times n}$ 为**正交矩阵**，当且仅当 $AA^\top = A^\top A = I$，即 $A^{-1} = A^\top$。

正交矩阵的变换保持向量长度和夹角不变：

$$
\|Ax\|^2 = x^\top A^\top A x = x^\top x = \|x\|^2 \tag{3.31}
$$

正交矩阵对应的线性变换是**刚体变换**（旋转和/或翻转）。



## 3.5 正交基

> **定义3.9（正交基）**：$n$ 维线性空间 $V$ 的基 $\{b_1, \dots, b_n\}$ 若满足
> - $\left\langle b_i, b_j \right\rangle = 0$（$i \neq j$），则称为**正交基**；
> - 若还满足 $\left\langle b_i, b_i \right\rangle = 1$，则称为**标准正交基（ONB）**。

从一组非正交的基构造标准正交基的方法叫做 **Gram-Schmidt 正交化**（详见 3.8.3 节）。标准正交基在 PCA（第10章）和 SVM（第12章）中有重要应用。



## 3.6 正交补

设 $V$ 是 $D$ 维线性空间，$U \subset V$ 是 $M$ 维子空间，则 $U$ 的**正交补** $U^\perp$ 是 $(D-M)$ 维子空间，其中每个向量都与 $U$ 中所有向量正交。$V$ 中任意向量 $x$ 可唯一分解为

$$
x = \sum_{m=1}^{M} \lambda_m b_m + \sum_{j=1}^{D-M} \psi_j b_j^\perp \tag{3.36}
$$

其中 $(b_1, \dots, b_M)$ 是 $U$ 的基，$(b_1^\perp, \dots, b_{D-M}^\perp)$ 是 $U^\perp$ 的基。

典型应用：三维空间中平面 $U$ 的正交补是一维的，其基向量 $w$（$\|w\|=1$）即为平面的**法向量**。正交补也可用来描述高维空间中的**超平面**。



## 3.7 函数的内积

内积的概念可以从有限维向量推广到函数。两个函数 $u, v: \mathbb{R} \to \mathbb{R}$ 的内积定义为

$$
\left\langle u, v \right\rangle := \int_a^b u(x)v(x) \, \mathrm{d}x, \quad a, b < \infty. \tag{3.37}
$$

若结果为零，则 $u$ 和 $v$ 正交。例如 $\sin(x)$ 和 $\cos(x)$ 在 $[-\pi, \pi]$ 上正交。函数族 $\{1, \cos(x), \cos(2x), \dots\}$ 两两正交，它们张成的空间包含所有以 $[-\pi, \pi)$ 为周期的连续函数，这是 **Fourier 级数**的核心思想。



## 3.8 正交投影

投影是机器学习中数据压缩和降维的基础工具。正交投影将高维数据投影到低维特征空间，同时最小化信息损失。PCA、自编码器和线性回归都可从正交投影的角度理解。

> **定义3.10（投影）**：线性映射 $\pi: V \to U$（$U \subset V$）若满足 $\pi^2 = \pi$，则称为**投影**。等价地，投影矩阵 $P_\pi$ 满足 $P_\pi^2 = P_\pi$（幂等性）。

### 3.8.1 向一维子空间（直线）投影

将 $x \in \mathbb{R}^n$ 投影到由 $b$ 张成的直线 $U$ 上，由正交性条件 $\left\langle x - \pi_U(x), b \right\rangle = 0$ 推导：

1. **坐标**：$\lambda = \dfrac{b^\top x}{\|b\|^2}$（若 $\|b\|=1$ 则 $\lambda = b^\top x$）

2. **投影向量**：$\pi_U(x) = \lambda b = \dfrac{b^\top x}{\|b\|^2} b$，其长度为 $\|\pi_U(x)\| = |\cos\omega| \cdot \|x\|$

3. **投影矩阵**：$P_\pi = \dfrac{bb^\top}{\|b\|^2}$（秩 1 对称矩阵）

### 3.8.2 向一般子空间投影

将 $x \in \mathbb{R}^n$ 投影到 $m$ 维子空间 $U$（基矩阵 $B = [b_1, \dots, b_m] \in \mathbb{R}^{n \times m}$），同样由正交性条件 $B^\top(x - B\lambda) = 0$ 得到**正规方程**：

$$
B^\top B \lambda = B^\top x \quad \Longrightarrow \quad \lambda = (B^\top B)^{-1} B^\top x \tag{3.57}
$$

其中 $(B^\top B)^{-1} B^\top$ 称为 $B$ 的**伪逆**。由此得到：

- **投影向量**：$\pi_U(x) = B(B^\top B)^{-1} B^\top x$
- **投影矩阵**：$P_\pi = B(B^\top B)^{-1} B^\top$

> **注释**：若 $B$ 的列构成标准正交基（$B^\top B = I$），则公式简化为 $\pi_U(x) = BB^\top x$，$\lambda = B^\top x$，无需矩阵求逆。♦

投影还可用于无解方程组 $Ax = b$ 的**最小二乘估计**：将 $b$ 投影到 $A$ 的列空间，得到最接近的近似解。

### 3.8.3 Gram-Schmidt 正交化

利用投影，可以从任意基 $(b_1, \dots, b_n)$ 迭代构造标准正交基 $(u_1, \dots, u_n)$：

$$
u_1 := b_1, \quad u_k := b_k - \pi_{\text{span}\{u_1, \dots, u_{k-1}\}}(b_k), \quad k = 2, \dots, n \tag{3.67-3.68}
$$

每一步将 $b_k$ 减去它在前 $k-1$ 个正交向量张成空间上的投影，得到与之正交的分量。最后对所有 $u_k$ 归一化即得 ONB。

### 3.8.4 向仿射子空间投影

对仿射子空间 $L = x_0 + U$ 的投影，可转化为子空间投影：先平移去掉支点 $x_0$，投影到 $U$，再平移回去：

$$
\pi_L(x) = x_0 + \pi_U(x - x_0) \tag{3.72}
$$

从 $x$ 到 $L$ 的距离等于 $x - x_0$ 到 $U$ 的距离。此方法在第12章推导**分割超平面**时使用。



## 3.9 旋转

旋转是保长保角的正交变换（自同构），不变点为原点。

### 3.9.1 $\mathbb{R}^2$ 中的旋转

将标准基逆时针旋转角度 $\theta$，得到旋转矩阵：

$$
R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \tag{3.76}
$$

### 3.9.2 $\mathbb{R}^3$ 中的旋转

三维空间中有三个基本旋转（分别绕 $e_1, e_2, e_3$ 轴），例如绕 $e_3$ 轴旋转：

$$
R_3(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} \tag{3.79}
$$

绕 $e_1$、$e_2$ 轴的旋转矩阵类似，将旋转的 $2 \times 2$ 子块放在对应的两个坐标平面上，另一维保持不变。

### 3.9.3 $n$ 维旋转（Givens 旋转）

$n$ 维空间的旋转推广为 **Givens 旋转** $R_{i,j}(\theta)$：在单位矩阵 $I_n$ 的基础上，修改 $(i,i), (i,j), (j,i), (j,j)$ 四个位置为旋转子块，其他维度不变。当 $n=2$ 时退化为 $\mathbb{R}^2$ 中的旋转。

### 3.9.4 旋转的性质

- **保距**：$\|x - y\| = \|R_\theta(x) - R_\theta(y)\|$
- **保角**：旋转前后向量夹角不变
- 三维及更高维旋转**不满足交换律**（顺序重要）；只有二维旋转可交换，全体二维旋转关于乘法构成 Abel 群

