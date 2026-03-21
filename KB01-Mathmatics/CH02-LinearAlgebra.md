# 线性代数(Linear Algebra)
## 2.1 线性方程组
总的来说，对于一个实数域内的线性方程组，它要么**无解**，要么有**唯一解**，要么有**无穷个解**。

为了引出解线性方程组的符号方法，我们介绍一种有效的缩写方法。我们将系数 $a_{ij}$ 写作向量并将向量构造为矩阵。换而言之，我们将线性方程组改写为如下形式：

$$
\begin{align}
\left [\begin{matrix}a_{11} \\ \vdots \\a_{m1}\end{matrix}\right ]x_1
+\left [\begin{matrix}a_{12} \\ \vdots \\a_{m2}\end{matrix}\right ]x_2
+\cdots
+\left [\begin{matrix}a_{1n} \\ \vdots \\a_{mn}\end{matrix}\right ]x_n=
\left [\begin{matrix}b_{1} \\ \vdots \\b_{m}\end{matrix}\right ] \tag{2.9}\\
\Leftrightarrow
\left [
    \begin{matrix}
    a_{11}&\cdots&a_{1n} \\ 
    \vdots&\vdots&\vdots \\
    a_{m1}&\cdots&a_{mn}
    \end{matrix}
\right ]\left [
    \begin{matrix}
    x_{1} \\ 
    \vdots\\
    x_{n}
    \end{matrix}
\right ]=\left [
    \begin{matrix}
    b_{1} \\ 
    \vdots\\
    b_{m}
    \end{matrix}
\right ] \tag{2.10}
\end{align} 
$$

接下来，我们将对这些**矩阵**及其定义的运算规则作出进一步的探究。


## 2.2 矩阵
**定义2.1（矩阵）**：对于 $m,n\in \mathbb{Z}_{>0}$，一个形状为 $(m,n)$ 的实矩阵 $\boldsymbol{A}$ 是一个关于元素 $a_{ij}$ 的 $m\times n$ 元组，其中 $i=1,2,\dots,m, j=1,2,\dots,n$，按照 $m$ 行 $n$ 列的方式进行排布。

$$
\boldsymbol{A}=
\left [
    \begin{matrix}
    a_{11}&\cdots&a_{1n} \\ 
    \vdots&\vdots&\vdots \\
    a_{m1}&\cdots&a_{mn}
    \end{matrix}
\right ],
a_{ij}\in\mathbb{R} \tag{2.11}
$$

按照惯例，形状为 $(1, n)$ 的矩阵成为行向量，形状为 $(m, 1)$ 的矩阵称为列向量。

通过将矩阵的所有n列叠加成一个长向量，一个 $\boldsymbol{A}\in\R^{m\times n}$ 可以等价地表示为一个 $a\in\R^{mn}$，如图2.4所示。

![图2.4](https://datawhalechina.github.io/math-for-ai/ch2/attachments/2-4.png)




### 2.2.1 矩阵的加法与乘法
**矩阵加法**：两个矩阵 $\boldsymbol{A}\in\R^{m\times n}$， $\boldsymbol{B}\in\R^{m\times n}$ 的和被定义为两个矩阵按对应元素的相加得到的新矩阵，即：

$$
\boldsymbol{A}+\boldsymbol{B}=
\left [
    \begin{matrix}
    a_{11}+b_{11}&\cdots&a_{1n}+b_{1n} \\ 
    \vdots&\vdots&\vdots \\
    a_{m1}+b_{m1}&\cdots&a_{mn}+b_{mn}
    \end{matrix}
\right ]\in\R^{m\times n} \tag{2.12}
$$


**矩阵乘法**：对于矩阵$\boldsymbol{A}\in\R^{m\times n}$，$\boldsymbol{B}\in\R^{n\times k}$的乘积矩阵$\boldsymbol{C}=\boldsymbol{A}\boldsymbol{B}\in\R^{m\times k}$(注意这里矩阵的大小)中元素的计算法则为：

$$
c_{ij}=\sum_{l=1}^n a_{il}b_{lj},(i=1,2,\dots,m, j=1,2,\dots,k) \tag{2.13}
$$

也就是说，为了计算元素 $c_{ij}$ 我们用 $\boldsymbol{A}$ 的第i行和 $\boldsymbol{B}$ 的第j列元素逐项相乘并求和。

注意：矩阵只有在“相邻”的尺寸匹配时才可以相乘。

$$
\underbrace{\boldsymbol{A}}_{n\times k} \underbrace{\boldsymbol{B}}_{k\times m}=\underbrace{\boldsymbol{C}}_{n\times m} \tag{2.14}
$$

**Hadamard 积**：两个形状相同的矩阵 $\boldsymbol{A}\in\R^{m\times n}$， $\boldsymbol{B}\in\R^{m\times n}$ 的Hadamard积被定义为两个矩阵按对应元素的相乘得到的新矩阵，即：

$$
\boldsymbol{A}\odot\boldsymbol{B}=
\left [
    \begin{matrix}
    a_{11}b_{11}&\cdots&a_{1n}b_{1n} \\ 
    \vdots&\vdots&\vdots \\
    a_{m1}b_{m1}&\cdots&a_{mn}b_{mn}
    \end{matrix}
\right ]\in\R^{m\times n} \tag{2.15}
$$



**单位矩阵**：在$\R^{n\times n}$中，定义**单位矩阵**

$$
\boldsymbol{I}_n=\left [
    \begin{matrix}
    1&0&\cdots&0 \\
    0&1&\cdots&0\\
    \vdots&\vdots&\cdots&\vdots\\
    0&0&\cdots&1
    \end{matrix}
\right ]\in \mathbb{R}^{n\times n} \tag{2.17}
$$

为对角线上全部为1，其他位置全部为0的$n\times n$维矩阵。



**矩阵运算性质：**
- **结合律**：
$$
\forall \boldsymbol{A}\in\R^{m\times n}, \boldsymbol{B}\in\R^{n\times p}, \boldsymbol{C}\in\R^{p\times q}, (\boldsymbol{A}\boldsymbol{B})\boldsymbol{C}=\boldsymbol{A}(\boldsymbol{B}\boldsymbol{C})\tag{2.18}
$$
- **分配律**：
$$
\begin{align*}
\forall \boldsymbol{A},\boldsymbol{B}\in\R^{m\times n}, \boldsymbol{C},\boldsymbol{D}\in\R^{n\times p}, 
(\boldsymbol{A}+\boldsymbol{B})\boldsymbol{C}&=\boldsymbol{A}\boldsymbol{C}+\boldsymbol{B}\boldsymbol{C}, \tag{2.19a}\\
\boldsymbol{A}(\boldsymbol{C}+\boldsymbol{D})&=\boldsymbol{A}\boldsymbol{C}+\boldsymbol{A}\boldsymbol{D} \tag{2.19b}
\end{align*}
$$
- **与单位矩阵相乘**：
$$
\forall \boldsymbol{A}\in\R^{m\times n}, \boldsymbol{I}_m\boldsymbol{A}=\boldsymbol{A}\boldsymbol{I}_n=\boldsymbol{A} \tag{2.20}
$$



### 2.2.2 矩阵的逆与转置

**定义2.3（逆矩阵）**：考虑一个方阵 $\boldsymbol{A}\in \R^{n\times n}$，令矩阵 $\boldsymbol{B}$ 满足性质：$\boldsymbol{AB}=\boldsymbol{BA}=\boldsymbol{I}_n$，$\boldsymbol{B}$ 被称作 $\boldsymbol{A}$ 的**逆**并记作 $\boldsymbol{A}^{-1}$ 。



**定义2.4（转置）**：对于矩阵 $\boldsymbol{A}\in \mathbb{R}^{m\times n}$，满足 $b_{ij}=a_{ji}$ 的矩阵 $\boldsymbol{B}\in \mathbb{R}^{n\times m}$ 被称作 $\boldsymbol{A}$ 的转置。我们记 $\boldsymbol{B}=\boldsymbol{A}^{\top}$

总的来说，$\boldsymbol{A}^{\top}$可以通过把$\boldsymbol{A}$的行作为$\boldsymbol{A}^{\top}$的对应列得到



**有关逆与转置的重要性质：**

$$
\begin{align}
\boldsymbol{A}\boldsymbol{A}^{-1}=\boldsymbol{A}^{-1}\boldsymbol{A}=\boldsymbol{I} \tag{2.26}\\
(\boldsymbol{AB})^{-1}=\boldsymbol{B}^{-1}\boldsymbol{A}^{-1} \tag{2.27}\\
(\boldsymbol{A+B})^{-1}\neq \boldsymbol{A}^{-1}+\boldsymbol{B}^{-1}  \tag{2.28}\\
(\boldsymbol{A}^{\top})^{\top}=\boldsymbol{A}  \tag{2.29}\\
(\boldsymbol{A+B})^{\top}= \boldsymbol{A}^{\top}+\boldsymbol{B}^{\top}  \tag{2.30}\\
(\boldsymbol{AB})^{\top}=\boldsymbol{B}^{\top}\boldsymbol{A}^{\top}  \tag{2.31}
\end{align}
$$


**定义2.5（对称矩阵）**：一个矩阵 $\boldsymbol{A}\in \R^{n\times n}$ 若满足 $\boldsymbol{A}=\boldsymbol{A}^{\top}$，我们称其为 **对称矩阵** 。





### 2.2.3 矩阵的标量乘

令$\boldsymbol{A}\in\mathbb{R}^{m\times n}, \lambda \in \mathbb{R}$, 那么$\lambda\boldsymbol{A=K}$,$K_{ij}=\lambda a_{ij}$。

**性质：**
- **结合律1**
$$
(\lambda\psi)\boldsymbol{C}=\lambda(\psi\boldsymbol{C}), \boldsymbol{C}\in\R^{m\times n}
$$
- **结合律2**
$$
\lambda(\boldsymbol{B}\boldsymbol{C})=(\lambda\boldsymbol{B})\boldsymbol{C}=\boldsymbol{B}(\lambda\boldsymbol{C})=(\boldsymbol{B}\boldsymbol{C})\lambda, \boldsymbol{B}\in\R^{m\times n}, \boldsymbol{C}\in\R^{n\times k}
$$
- **转置**
$$
(\lambda\boldsymbol{C})^{\top}=\boldsymbol{C}^{\top}\lambda^{\top}=\boldsymbol{C}^{\top}\lambda=\lambda\boldsymbol{C}^{\top}
$$
因为对于$\forall \lambda \in \R$, $\lambda^{\top}=\lambda$
- **分配律**
$$
\begin{align*}
(\lambda+\psi)\boldsymbol{C}&=\lambda\boldsymbol{C}+\psi\boldsymbol{C},\\
\lambda(\boldsymbol{B}+\boldsymbol{C})&=\lambda\boldsymbol{B}+\lambda\boldsymbol{C}
\end{align*} \quad \boldsymbol{B}\in\R^{m\times n}, \boldsymbol{C}\in\R^{m\times n}, \\
$$



### 2.2.4 线性方程组的矩阵表示

考虑这样一个线性方程组：

$$
2x_1+3x_2+5x_3=1\\
4x_1-2x_2+7x_3=8\\
9x_1+5x_2-3x_3=2 \tag{2.35}
$$

利用矩阵乘法的规则，我们可以把这个方程组写成更紧凑的形式:

$$
\left [
    \begin{matrix}
    2&3&5\\
    4&-2&7\\
    9&5&-3
    \end{matrix}
\right ]\left [
    \begin{matrix}
    x_1\\x_2\\x_3
    \end{matrix}
\right ]=\left [
    \begin{matrix}
    1\\8\\2
    \end{matrix}
\right ] \tag{2.36}
$$

一般的，一个线性方程组可以缩写为矩阵形式 $\boldsymbol{Ax=b}$。




## 2.3 解线性方程组

### 2.3.1 特解和通解
考虑以下方程组：

$$
\begin{bmatrix}
1 & 0 & 8 & -4 \\
0 & 1 & 2 & 12
\end{bmatrix}
\begin{bmatrix}
x_1 \\ x_2 \\ x_3 \\ x_4
\end{bmatrix} =
\begin{bmatrix}
42 \\ 8
\end{bmatrix}.
\tag{2.38}
$$

一个解是 $[42, 8, 0, 0]^\top$。这个解被称为**特解**（或**特殊解**）

对于任意 $\lambda_1, \lambda_2 \in \mathbb{R}$。将所有内容放在一起，我们得到(2.38)方程组的所有解，称为**通解**：

$$
\left\{ x \in \mathbb{R}^4 : x =\begin{bmatrix}42 \\ 8 \\ 0 \\ 0\end{bmatrix}+ \lambda_1\begin{bmatrix}8 \\ 2 \\ -1 \\ 0\end{bmatrix}+ \lambda_2\begin{bmatrix}-4 \\ 12 \\ 0 \\ -1\end{bmatrix}, \lambda_1, \lambda_2 \in \mathbb{R}\right\}.\tag{2.43} 
$$


**注：** 我们遵循的一般方法包括以下三个步骤：
1.  找到一个特解 $Ax = b$。
2.  找到所有解 $Ax = 0$。
3.  将步骤1和步骤2的解结合起来得到通解。

通解和特解都不是唯一的。（译者注：假如矩阵 $A$ 的列向量线性相关，可能会有无穷多的特解和通解。）



### 2.3.2 初等变换

解线性方程组的关键是**初等变换**，这些变换可以保持方程组的解集不变，但可以将方程组转换为更简单的形式：
- 交换两个方程（矩阵中的行）。
- 将一个方程（行）乘以一个非零常数 $\lambda \in \mathbb{R} \setminus \{0\}$。
- 将两个方程（行）相加。


**注（主元和阶梯结构）：** 按照上述方法将方程组化简完成后，某一行从左边开始的第一个非零数字称为主元，它总是严格位于其上方行的主元的右边。因此，任何处于行阶梯形式的方程组总是具有“阶梯”结构。♦

**定义2.6（行阶梯形式）：** 如果一个矩阵满足以下条件，则称其处于行阶梯形式：
- 所有只包含零的行位于矩阵的底部；相应地，所有至少包含一个非零元素的行位于只包含零的行的上方。
- 只考虑非零行，从左边开始的第一个非零数字（也称为主元或领先系数）总是严格位于其上方行的主元的右边。

在其他文献中，有时要求主元是1。

**注（基本变量和自由变量）：** 在行阶梯形式中，对应于主元的变量称为基本变量，其他变量称为自由变量。例如，在(2.45)中，$x_1, x_3, x_4$ 是基本变量，而 $x_2, x_5$ 是自由变量。♦

**注（获得特解）：** 行阶梯形式使我们的生活更简单，当我们需要确定一个特解时。为此，我们用主元列表示方程组的右侧，使得 $b = \sum_{i=1}^P \lambda_i p_i$，其中 $p_i, i = 1, \dots, P$，是主元列。通过从最右边的主元列开始，向左工作，可以最简单地确定 $\lambda_i$。在前面的例子中，我们将尝试找到 $\lambda_1, \lambda_2, \lambda_3$，使得

$$
\lambda_1 \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 1 \\ 1 \\ 0 \\ 0 \end{bmatrix} + \lambda_3 \begin{bmatrix} -1 \\ -1 \\ 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 0 \\ -2 \\ 1 \\ 0 \end{bmatrix}. \tag{2.48} 
$$




### 2.3.3 负一技巧

介绍一个实用的技巧，用于读出齐次线性方程组 $Ax = 0$ 的解，其中 $A \in \mathbb{R}^{k \times n}$，$x \in \mathbb{R}^n$。首先，我们假设 $A$ 处于简化行阶梯形式，且没有只包含零的行，即

$$
A = \begin{bmatrix}
    0 & \cdots & 0 & \color{red}1 & * & \cdots & * & 0 & * & \cdots & * & 0 & * & \cdots & * \\
    \vdots & & \vdots & 0 & 0 & \cdots & 0 & \color{red}1 & * & \cdots & * & \vdots & \vdots & & \vdots \\
    \vdots & & \vdots & \vdots & \vdots & \cdots & \vdots & 0 & \vdots & \cdots & \vdots & \vdots & \vdots & & \vdots \\
    \vdots & & \vdots & \vdots & \vdots & \cdots & \vdots & \vdots & \vdots & \cdots & \vdots & 0 & \vdots & & \vdots \\
    0 & \cdots & 0 & 0 & 0 & \cdots & 0 & 0 & 0 & \cdots & 0 & \color{red}1 & * & \cdots & * \\
\end{bmatrix},
\tag{2.51} 
$$


其中 $*$ 可以是任意实数，条件是每行的第一个非零元素必须是1，且相应列中的所有其他元素必须是0。包含主元（以粗体标记）的列 $j_1, \dots, j_k$ 是标准单位向量 $e_1, \dots, e_k \in \mathbb{R}^k$。我们通过添加 $n - k$ 行的形式

$$
\begin{bmatrix}
0 & \cdots & 0 & -1 & 0 & \cdots & 0
\end{bmatrix}
\tag{2.52} 
$$

将这个矩阵扩展为 $n \times n$ 矩阵 $\tilde{A}$，使得 $\tilde{A}$ 的对角线上包含1或-1。然后，包含对角线上的-1的 $\tilde{A}$ 的列是齐次方程组 $Ax = 0$ 的解。更准确地说，这些列构成了 $Ax = 0$ 的解空间的一个基，我们稍后将称其为核或零空间


**例2.8（负一技巧）** 让我们重新审视已经处于简化REF的矩阵(2.49)：

$$
A = \begin{bmatrix}
1 & 3 & 0 & 0 & 3 \\
0 & 0 & 1 & 0 & 9 \\
0 & 0 & 0 & 1 & -4
\end{bmatrix}.
\tag{2.53} 
$$

我们通过在对角线上缺少主元的位置添加形式为(2.52)的行，将这个矩阵扩展为 $5 \times 5$ 矩阵：

$$
\tilde{A} = \begin{bmatrix}
1 & 3 & 0 & 0 & 3 \\
\color{red}0 & \color{red}-1 & \color{red}0 & \color{red}0 & \color{red}0 \\
0 & 0 & 1 & 0 & 9 \\
0 & 0 & 0 & 1 & -4 \\
\color{red}0 & \color{red}0 & \color{red}0 & \color{red}0 & \color{red}-1
\end{bmatrix}.
\tag{2.54} 
$$

从这个形式中，我们可以直接读出 $Ax = 0$ 的解，通过取 $\tilde{A}$ 中对角线上包含-1的列：

$$ 
\left\{ x \in \mathbb{R}^5 : x = \lambda_1 \begin{bmatrix} 3 \\ -1 \\ 0 \\ 0 \\ 0 \end{bmatrix} + \lambda_2 \begin{bmatrix} 3 \\ 0 \\ 9 \\ -4 \\ -1 \end{bmatrix}, \lambda_1, \lambda_2 \in \mathbb{R} \right\}, \tag{2.55} 
$$




### 2.3.4 计算逆矩阵

为了计算 $A^{-1}$，其中 $A \in \mathbb{R}^{n \times n}$，我们需要找到一个矩阵 $X$，使得 $AX = I_n$。然后，$X = A^{-1}$。我们可以将此写成一组同时线性方程 $AX = I_n$，其中我们解 $X = [x_1 | \dots | x_n]$。我们使用增广矩阵（译者注：就是把线性方程组中的所有数都写成一个矩阵，等号左边的系数矩阵放在左边，等号右边的数字放在右边）表示法来紧凑地表示这组方程组：

$$
[A \mid I_n] \Rightarrow \dots \Rightarrow [I_n \mid A^{-1}].
\tag{2.56} 
$$

这意味着，如果我们将增广方程组带入简化行阶梯形式，我们可以在方程组的右侧直接读出逆矩阵。因此，确定矩阵的逆等同于解线性方程组。

**可以通过高斯消元法计算逆矩阵**




### 2.3.5 解线性方程组的算法

在接下来的内容中，我们将简要讨论解决形式为 $Ax = b$ 的线性方程组的方法。我们假设存在解。如果没有解，我们需要诉诸于近似解，这在本章中没有涵盖。解决近似问题的一种方法是使用线性回归的方法，我们将在第9章中详细讨论。在特殊情况下，我们可能能够确定逆 $A^{-1}$，使得解 $Ax = b$ 为 $x = A^{-1}b$。然而，这只在 $A$ 是方阵且可逆的情况下才可能，然而这并不常见。否则在较弱的条件下（即 $A$ 需要具有线性独立的列），我们可以使用变换

$$
Ax = b \Leftrightarrow A^\top A x = A^\top b \Leftrightarrow x = (A^\top A)^{-1} A^\top b,
\tag{2.59} 
$$




## 2.4 线性空间

### 2.4.1 群


> **定义2.7（群）**
> 
> 考虑一个集合 $\mathcal{G}$ 和一个在 $G$ 上定义的二元运算 $\otimes: \mathcal{G} \times \mathcal{G} \to \mathcal{G}$。那么 $G := (\mathcal{G}, \otimes)$ 被称为一个**群**，如果满足以下条件：
> 
> 1. **关于 $\otimes$ 的封闭性**：$\forall x, y \in \mathcal{G}: x \otimes y \in \mathcal{G}$
> 2. **结合律**：$\forall x, y, z \in \mathcal{G}: (x \otimes y) \otimes z = x \otimes (y \otimes z)$
> 3. **单位元**：$\exists e \in \mathcal{G} \forall x \in \mathcal{G}: x \otimes e = x$ 且 $e \otimes x = x$
> 4. **逆元**：$\forall x \in \mathcal{G} \exists y \in \mathcal{G}: x \otimes y = e$ 且 $y \otimes x = e$，其中 $e$ 是单位元。我们通常用 $x^{-1}$ 表示 $x$ 的逆元。
> 
> 如果此外 $\forall x, y \in \mathcal{G}: x \otimes y = y \otimes x$，则 $G = (\mathcal{G}, \otimes)$ 是一个**Abel 群**（交换群）。

> **注释**
> 逆元是相对于操作 $\otimes$ 定义的，并不一定就是所谓的 $\displaystyle \frac{1}{x}$。



> **定义2.8（一般线性群）**
>
> 可逆矩阵 $A \in \mathbb{R}^{n \times n}$ 的集合关于矩阵乘法定义为(2.13)，称为**一般线性群**，记作 $GL(n, \mathbb{R})$。然而，由于矩阵乘法不是交换的，因此该群不是Abel 群。




### 2.4.2 线性空间

在讨论群时，我们研究了集合 $\mathcal{G}$ 和 $\mathcal{G}$ 上的内操作。接下来，我们将考虑包含内操作“+”和外操作“·”（标量乘法）的集合。我们可以将内操作视为一种加法形式，而外操作则为一种缩放形式。注意，内操作和外操作与内积和外积无关。

> **定义2.9（线性空间）**
> 
> 一个实值线性空间 $V$ 是这些资料 $(V, \mathbb{R}, +, \cdot)$，$V$ 上具有两个运算：
> 
> $$\begin{align}+: \mathcal{V} \times \mathcal{V} &\to \mathcal{V} & 加法 \tag{2.62} \\\cdot: \mathbb{R} \times \mathcal{V} &\to \mathcal{V} & 纯量乘法 \tag{2.63}\end{align}$$
> 
> 其中：
> 1.  $(\mathcal{V}, +)$ 是 Abel 群。
> 2.  分配律：
>    * $\forall \lambda \in \mathbb{R}, x, y \in V: \lambda \cdot (x + y) = \lambda \cdot x + \lambda \cdot y$
>    * $\forall \lambda, \psi \in \mathbb{R}, x \in V: (\lambda + \psi) \cdot x = \lambda \cdot x + \psi \cdot x$
> 3.  标量乘法的结合律：$\forall \lambda, \psi \in \mathbb{R}, x \in V: \lambda \cdot (\psi \cdot x) = (\lambda \psi) \cdot x$
> 4.  标量乘法单位元：$\forall x \in V: 1 \cdot x = x$

集合 $V$ 中的元素 $x$ 称为**向量**。集合 $V$ 中的单位元称为**零向量**，记作 $\boldsymbol{0}$ （在 Euclid 空间中可以写为 $[0, \dots, 0]^\top$ ），而运算 “$+$” 称为**向量加法**。集合 $\mathbb{R}$ 中的元素 $\lambda$ 称为**标量**，而运算 “$\cdot$” 称为**标量乘法**。注意，标量乘法与标量积不同，我们将在第3.2节中讨论。（译者注：在较早的资料中，常常将上面的数学对象为线性空间，而将有限维线性空间（Euclid 空间）称为向量空间。如今的资料中“向量空间”和“线性空间”的含义相同。）




### 2.4.3 向量子空间

接下来，我们引入向量子空间的概念。直观上，它们是包含在原始线性空间内的集合，具有这样的性质：当我们对子空间内的元素进行线性空间操作时，结果总是落在子空间中。在这个意义上，它们是“封闭的”。向量子空间是机器学习中的一个关键概念。

> **定义2.10（向量子空间）**
>
> 设 $V = (V, +, \cdot)$ 是一个线性空间，且 $U \subseteq V$，$U \neq \emptyset$。那么 $U = (U, +, \cdot)$ 称为 $V$ 的**向量子空间**（或**线性子空间**），如果 $U$ 是一个线性空间，且线性空间操作“+”和“$\cdot$”限制在 $U \times U$ 和 $\mathbb{R} \times U$ 上。我们用 $U \subseteq V$ 表示 $V$ 的子空间 $U$。

如果 $U \subseteq V$ 且 $V$ 是一个线性空间，那么 $U$ 自然地从 $V$ 继承了许多性质，因为这些性质对所有 $x \in V$ 都成立，特别是对所有 $x \in U \subseteq V$ 也成立。这包括Abel 群的性质、分配律、结合律和单位元。为了确定 $(U, +, \cdot)$ 是否是 $V$ 的子空间，我们仍需要证明：
1. $U \neq \varnothing$，特别是：$\boldsymbol{0} \in U$
2. $U$ 的封闭性：
   - 关于外操作：$\forall \lambda \in \mathbb{R}, \forall x \in U: \lambda x \in U$
   - 关于内操作：$\forall x, y \in U: x + y \in U$





## 2.5 线性无关

### 2.5.1 线性组合

**定义2.11（线性组合）**：考虑一个线性空间 $V$ 和有限个向量 $\boldsymbol{x}_1, \dots, \boldsymbol{x}_k \in V$。那么，每一个形如

$$
\boldsymbol{v} = \lambda_1 \boldsymbol{x}_1 + \cdots + \lambda_k \boldsymbol{x}_k = \sum_{i=1}^k \lambda_i \boldsymbol{x}_i \in V
\tag{2.65} 
$$

的向量 $\boldsymbol{v} \in V$，其中 $\lambda_1, \dots, \lambda_k \in \mathbb{R}$，称为向量 $\boldsymbol{x}_1, \dots, \boldsymbol{x}_k$ 的**线性组合**。

零向量 $\boldsymbol{0}$ 总是可以表示为 $k$ 个向量 $\boldsymbol{x}_1, \dots, \boldsymbol{x}_k$ 的线性组合，因为 $\boldsymbol{0} = \sum_{i=1}^k 0 \cdot \boldsymbol{x}_i$ 总是成立的。接下来，我们感兴趣的是向量集合的非平凡线性组合，即线性组合中的系数 $\lambda_i$ 不全为零的情况。



### 2.5.2 线性无关

**定义2.12（线性无关）**：设 $V$ 是一个线性空间，且 $\boldsymbol{x}_1, \dots, \boldsymbol{x}_k \in V$。如果存在一个非平凡的线性组合，使得

$$
\boldsymbol{0} = \sum_{i=1}^k \lambda_i \boldsymbol{x}_i
$$

其中至少有一个 $\lambda_i \neq 0$，则称向量 $\boldsymbol{x}_1, \dots, \boldsymbol{x}_k$ 是**线性相关的**。如果只有平凡解，即 $\lambda_1 = \cdots = \lambda_k = 0$，则称向量 $\boldsymbol{x}_1, \dots, \boldsymbol{x}_k$ 是**线性无关的**。

线性无关是线性代数中最重要的概念之一。直观上，一组线性无关的向量由没有冗余的向量组成，即如果我们从集合中移除任何一个向量，我们就会失去某些信息。在接下来的部分中，我们将更正式地形式化这种直觉。



> **注释**：以下性质有助于判断向量是否线性无关：
> - $k$ 个向量要么线性相关，要么线性无关。没有第三种可能。
> - 如果向量 $\boldsymbol{x}_1, \dots, \boldsymbol{x}_k$ 中至少有一个是零向量，则它们是线性相关的。
> - 如果两个向量相同，它们也是线性相关的。
> - 对于向量集合 $\{\boldsymbol{x}_1, \dots, \boldsymbol{x}_k : \boldsymbol{x}_i \neq \boldsymbol{0}, i = 1, \dots, k\}$，其中 $k \geq 2$，它们是线性相关的当且仅当（至少）其中一个向量是其他向量的线性组合。特别是，如果一个向量是另一个向量的倍数，即 $\boldsymbol{x}_i = \lambda \boldsymbol{x}_j$，$\lambda \in \mathbb{R}$，则集合 $\{\boldsymbol{x}_1, \dots, \boldsymbol{x}_k : \boldsymbol{x}_i \neq \boldsymbol{0}, i = 1, \dots, k\}$ 是线性相关的。



> **注释**：判断向量 $\boldsymbol{x}_1, \dots, \boldsymbol{x}_k \in V$ 是否线性无关的一个实用方法是使用高斯消元法：将所有向量作为矩阵 $A$ 的列向量，并进行高斯消元法，直到矩阵处于行阶梯形式（简化行阶梯形式在这里是不必要的）：
> - 主元列表示的向量是线性无关的。
> - 非主元列可以表示为它们左边主元列的线性组合。例如，行阶梯形式
>   \
> $$\begin{bmatrix}1 & 3 & 0  \\  0 & 0 & 2  \\  \end{bmatrix} \tag{2.66}$$
> 表明第一列和第三列是主元列。第二列是非主元列，因为它等于第一列的三倍。
> 所有列向量线性无关当且仅当所有列都是主元列。如果至少有一个非主元列，则列向量（因此对应的向量）是线性相关的。


> **注释**：考虑一个线性空间 $ V $ 和其中的 $ k $ 个线性无关的向量 $ \boldsymbol{b}_1, \dots, \boldsymbol{b}_k $，以及 $ m $ 个线性组合：$$\boldsymbol{x}_1 = \sum_{i=1}^k \lambda_{i1} \boldsymbol{b}_i, \quad \dots, \quad \boldsymbol{x}_m = \sum_{i=1}^k \lambda_{im} \boldsymbol{b}_i.\tag{2.70} $$定义 $ B = [\boldsymbol{b}_1, \dots, \boldsymbol{b}_k] $ 为矩阵，其列向量是线性无关的向量 $ \boldsymbol{b}_1, \dots, \boldsymbol{b}_k $，则可以更紧凑地表示为：$$\boldsymbol{x}_j = B \boldsymbol{\lambda}_j, \quad \boldsymbol{\lambda}_j = \begin{bmatrix}\lambda_{1j} \\ \vdots \\ \lambda_{kj}\end{bmatrix}, \quad j = 1, \dots, m,\tag{2.71} $$为了测试 $ \boldsymbol{x}_1, \dots, \boldsymbol{x}_m $ 是否线性无关，我们采用一般方法，测试 $ \sum_{j=1}^m \psi_j \boldsymbol{x}_j = \boldsymbol{0} $。根据(2.71)，我们有：$$\sum_{j=1}^m \psi_j \boldsymbol{x}_j = \sum_{j=1}^m \psi_j B \boldsymbol{\lambda}_j = B \sum_{j=1}^m \psi_j \boldsymbol{\lambda}_j.\tag{2.72} $$这意味着 $\{\boldsymbol{x}_1, \dots, \boldsymbol{x}_m\}$ 线性无关当且仅当列向量 $\{\boldsymbol{\lambda}_1, \dots, \boldsymbol{\lambda}_m\}$ 线性无关。


> **注释**：在线性空间 $V$ 中，$m$ 个线性组合的 $k$ 个向量 $\boldsymbol{x}_1, \dots, \boldsymbol{x}_k$ 是线性相关的，如果 $m > k$。




## 2.6 向量组的基与秩


### 2.6.1 生成集和基

**定义2.13（生成集和张成空间）**：考虑一个线性空间 $V = (V, +, \cdot)$ 和一组向量 $A = \{\boldsymbol{x}_1, \dots, \boldsymbol{x}_k\} \subseteq V$。如果 $V$ 中的每一个向量 $\boldsymbol{v}$ 都可以表示为 $\boldsymbol{x}_1, \dots, \boldsymbol{x}_k$ 的线性组合，则称 $A$ 是 $V$ 的一个**生成集**。所有 $A$ 中向量的线性组合的集合称为 $A$ 的**张成空间**，记作 $\text{span}[A]$ 或 $\text{span}[\boldsymbol{x}_1, \dots, \boldsymbol{x}_k]$。如果 $A$ 张成了线性空间 $V$，我们写作 $V = \text{span}[A]$ 或 $V = \text{span}[\boldsymbol{x}_1, \dots, \boldsymbol{x}_k]$。

生成集是能够张成向量（子）空间的向量集合，即每一个向量都可以表示为生成集中向量的线性组合。接下来，我们将更具体地描述最小的生成集，它张成了一个向量（子）空间。


**定义2.14（基）**：考虑一个线性空间 $V = (V, +, \cdot)$ 和 $A \subseteq V$。如果一个生成集 $A$ 是最小的，即不存在更小的集合 $\tilde{A} \subset A \subseteq V$ 能够张成 $V$，则称 $A$ 是 $V$ 的一个**基**。每一个线性无关的生成集都是最小的，因此称为 $V$ 的一个**基**。

设 $V = (V, +, \cdot)$ 是一个线性空间，$B \subseteq V$，$B \neq \emptyset$。那么，以下陈述是等价的：
- $B$ 是 $V$ 的一个基。
- $B$ 是一个最小的生成集。
- $B$ 是 $V$ 中一个最大的线性无关向量集合，即在 $B$ 中添加任何其他向量都将使其线性相关。
- $V$ 中的每一个向量 $\boldsymbol{x}$ 都可以表示为 $B$ 中向量的线性组合，并且每一个线性组合都是唯一的，即
  
  $$
  \boldsymbol{x} = \sum_{i=1}^k \lambda_i \boldsymbol{b}_i = \sum_{i=1}^k \psi_i \boldsymbol{b}_i \tag{2.77}
  $$ 其中 $\boldsymbol{b}_1, \dots, \boldsymbol{b}_k \in \mathcal{B}$，$\lambda_i, \psi_i \in \mathbb{R}$，我们立即有 $\lambda_i = \psi_i, i = 1, ..., k$。



> **注释**：每一个线性空间 $V$ 都有一个基 $B$。前面的例子表明，一个线性空间 $V$ 可以有多个基，即基不是唯一的。然而，所有基都包含相同数量的元素，即基向量（译者注：基中的向量称为基向量）。♦

我们只考虑有限维线性空间 $V$。在这种情况下，$V$ 的**维数**是 $V$ 的基向量的数量，记作 $\dim(V)$。如果 $U \subseteq V$ 是 $V$ 的一个子空间，则 $\dim(U) \leq \dim(V)$，且 $\dim(U) = \dim(V)$ 当且仅当 $U = V$。直观上，线性空间的维数可以被看作是该线性空间中独立方向的数量。线性空间的维数对应于其基向量的数量。

> **注释**：线性空间的维数并不一定是向量中的元素数量。例如，线性空间 $V = \text{span}\left\{\begin{bmatrix} 0 \\ 1 \end{bmatrix}\right\}$ 是一维的，尽管基向量包含两个元素。♦

> **注释**：一个子空间 $U = \text{span}[\boldsymbol{x}_1, \dots, \boldsymbol{x}_m] \subseteq \mathbb{R}^n$ 的基可以通过以下步骤找到：
> 1. 将张成向量作为矩阵 $A$ 的列。
> 2. 确定 $A$ 的行阶梯形式。
> 3. 与主元列对应的张成向量构成 $U$ 的一个基。



### 2.6.2 秩

矩阵 $A \in \mathbb{R}^{m \times n}$ 的线性无关列的数量等于线性无关行的数量，称为 $A$ 的**秩**，记作 $\text{rank}(A)$。

> **注释**：秩具有一些重要的性质：
> - $\text{rank}(A) = \text{rank}(A^\top)$，即列秩（线性无关列的数量）等于行秩（线性无关行的数量）。
> - $A$ 的列张成一个子空间 $U \subseteq \mathbb{R}^m$，其维数为 $\text{rank}(A)$。我们稍后将称这个子空间为像或值域。可以通过对 $A$ 应用高斯消元法来找到 $U$ 的一个基，以确定主元列。
> - $A$ 的行张成一个子空间 $W \subseteq \mathbb{R}^n$，其维数为 $\text{rank}(A)$。可以通过对 $A^\top$ 应用高斯消元法来找到 $W$ 的一个基。
> - 对于所有 $A \in \mathbb{R}^{n \times n}$，$A$ 是可逆的当且仅当 $\text{rank}(A) = n$。
> - 对于所有 $A \in \mathbb{R}^{m \times n}$ 和所有 $\boldsymbol{b} \in \mathbb{R}^m$，线性方程组 $A\boldsymbol{x} = \boldsymbol{b}$ 有解当且仅当 $\text{rank}(A) = \text{rank}([A \mid \boldsymbol{b}])$，其中 $[A \mid \boldsymbol{b}]$ 表示增广矩阵。
> - 对于 $A \in \mathbb{R}^{m \times n}$，齐次方程组 $A\boldsymbol{x} = \boldsymbol{0}$ 的解空间的维数为 $n - \text{rank}(A)$。我们稍后将称这个子空间为核或零空间。核的维数为 $n - \text{rank}(A)$。
> - 一个矩阵 $A \in \mathbb{R}^{m \times n}$ 具有**满秩**，如果其秩等于对于相同维度的矩阵可能的最大秩。这意味着满秩矩阵的秩是行数和列数中较小的那个，即 $\text{rank}(A) = \min(m, n)$。如果一个矩阵的秩不等于满秩，则称该矩阵是**秩亏的**。





## 2.7 线性映射

**定义2.15（线性映射）**：对于线性空间 $V, W$，一个映射 $\Phi: V \to W$ 称为**线性映射**（或线性空间同态/线性变换），如果

$$
\forall \boldsymbol{x}, \boldsymbol{y} \in V, \forall \lambda, \psi \in \mathbb{R}: \Phi(\lambda \boldsymbol{x} + \psi \boldsymbol{y}) = \lambda \Phi(\boldsymbol{x}) + \psi \Phi(\boldsymbol{y}). \quad \tag{2.87}
$$


**定义2.16（单射、满射、双射）**：考虑一个映射 $\Phi: V \to W$，其中 $V, W$ 可以是任意集合。那么 $\Phi$ 称为：
- **单射**（Injective）：如果 $\forall \boldsymbol{x}, \boldsymbol{y} \in V: \Phi(\boldsymbol{x}) = \Phi(\boldsymbol{y}) \Rightarrow \boldsymbol{x} = \boldsymbol{y}$。
- **满射**（Surjective）：如果 $\Phi(V) = W$。
- **双射**（Bijective）：如果它是单射且满射。


**定理2.17（Axler, 2015, 定理3.59）**：有限维线性空间 $V$ 和 $W$ 是同构的，当且仅当 $\dim(V) = \dim(W)$。

> **注释**：考虑线性空间 $V, W, X$。那么：
> - 对于线性映射 $\Phi: V \to W$ 和 $\Psi: W \to X$，映射 $\Psi \circ \Phi: V \to X$ 也是线性的。
> - 如果 $\Phi: V \to W$ 是一个同构，那么 $\Phi^{-1}: W \to V$ 也是一个同构。
> - 如果 $\Phi: V \to W$ 和 $\Psi: V \to W$ 是线性的，那么 $\Phi + \Psi$ 和 $\lambda \Phi$（$\lambda \in \mathbb{R}$）也是线性的。



### 2.7.1 线性映射的矩阵表示

任何 $n$ 维线性空间都与 $\mathbb{R}^n$ 同构（定理2.17）。我们考虑一个 $n$ 维线性空间 $V$ 的一个基 $\{\boldsymbol{b}_1, \dots, \boldsymbol{b}_n\}$。接下来，有序基的顺序将很重要。因此，我们写作

$$
B = (\boldsymbol{b}_1, \dots, \boldsymbol{b}_n) \quad \tag{2.89}
$$

并称这个 $n$-元组为 $V$ 的一个**有序基**。

**注（一些记号）**：现在我们有多个看起来相似易混淆的记号。因此我们在此重申：
- $B = (\boldsymbol{b}_1, \dots, \boldsymbol{b}_n)$ 是一个有序基；
- $B = \{\boldsymbol{b}_1, \dots, \boldsymbol{b}_n\}$ 是一个（无序）基；
- $B = [\boldsymbol{b}_1, \dots, \boldsymbol{b}_n]$ 是一个矩阵，其列向量是向量 $\boldsymbol{b}_1, \dots, \boldsymbol{b}_n$。♦



**定义2.18（坐标）**：考虑一个线性空间 $V$ 和 $V$ 的一个有序基 $B = (\boldsymbol{b}_1, \dots, \boldsymbol{b}_n)$。对于任意 $\boldsymbol{x} \in V$，我们得到一个唯一的表示（线性组合）

$$
\boldsymbol{x} = \alpha_1 \boldsymbol{b}_1 + \cdots + \alpha_n \boldsymbol{b}_n \quad \tag{2.90}
$$

的 $\boldsymbol{x}$ 关于 $B$。那么 $\alpha_1, \dots, \alpha_n$ 称为 $\boldsymbol{x}$ 关于 $B$ 的**坐标**，而向量

$$
\boldsymbol{\alpha} = \begin{bmatrix}
\alpha_1 \\ \vdots \\ \alpha_n
\end{bmatrix} \in \mathbb{R}^n \quad \tag{2.91}
$$

称为 $\boldsymbol{x}$ 关于有序基 $B$ 的**坐标向量**或**坐标表示**。



### 2.7.2 基变换



### 2.7.3 像和核





## 2.8 仿射空间



