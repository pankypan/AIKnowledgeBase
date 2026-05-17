# 3Blue1Brown「线性代数的本质」— 合集（第 12–16 章及附录）

> 说明：本文档由 `ch12`–`ch16` 合并而成；**文末附录**为全书快速复习（Ch1–Ch16，原《线性代数快速复习总结》）。数学术语与公式保持英文原文为主。  
> 正文范围：Cramer's rule、换基、特征值与特征向量、2×2 快速技巧、抽象向量空间。

## 本册目录

- [Ch.12 — Cramer's rule, explained geometrically（Cramer's rule 的几何解释）](#ch12)
- [Ch.13 — Change of basis（基变换）](#ch13)
- [Ch.14 — Eigenvectors and eigenvalues（特征向量与特征值）](#ch14)
- [Ch.15 — A quick trick for computing eigenvalues（快速求 $2 \times 2$ 特征值）](#ch15)
- [Ch.16 — Abstract vector spaces（抽象向量空间）](#ch16)
- [附录：全书快速复习（Ch1–Ch16）](#review)

---
<a id="ch12"></a>

## Ch.12 — Cramer's rule, explained geometrically（Cramer's rule 的几何解释）

> 来源：[3Blue1Brown — Cramer's rule, explained geometrically](https://www.3blue1brown.com/lessons/cramers-rule)（Chapter 12）  
> 说明：数学符号与术语保留英文原文。

### 目录（对应原文结构）

1. [Assume Determinant is Nonzero](#assume-determinant-is-nonzero)
   - [What about dot products with basis vectors?](#what-about-dot-products-with-basis-vectors)
   - [A Better Approach](#a-better-approach)
     - [Sanity check](#sanity-check)
     - [Two Dimension Questions](#two-dimension-questions)
   - [In three dimensions](#in-three-dimensions)

---

### Assume Determinant is Nonzero

- **问题**：解 $A\mathbf{x} = \mathbf{b}$，几何上即找哪个输入向量被矩阵变换映到已知输出 $\mathbf{b}$。
- **前提**：$\det(A) \neq 0$，变换可逆，解唯一；$\det(A)=0$ 时可能无解或无穷多解，本课不展开。

#### What about dot products with basis vectors?

- **误区**：希望用 $\mathbf{x}$ 与 $\hat{\imath},\hat{\jmath}$ 的 **dot product** 读出坐标，并认为变换后仍成立——一般**不成立**（点积一般不保持）。
- **特例**：**Orthonormal** 变换（常对应旋转、刚体运动）保持点积；此时可用输出与列向量的点积直接读坐标，但一般矩阵不行。

#### A Better Approach

- **坐标的新几何编码（2d）**：  
  - $y$：由 $\hat{\imath}$ 与未知 $\mathbf{x}$ 张成的平行四边形**有向面积**（底为 1，高为 $y$）。  
  - $x$：由 $\mathbf{x}$ 与 $\hat{\jmath}$ 张成的平行四边形有向面积。
- **3d 类比**：某坐标 = 与“其余标准基”张成的 **parallelepiped** 的**有向体积**（带 **right-hand rule** 定向）。
- **关键**：施加矩阵变换后，所有这些“面积/体积”按同一因子缩放，即 $\det(A)$。
- **Cramer's rule（2d）**：记 $A = [\mathbf{a}_1 \mid \mathbf{a}_2]$（两列为 $\hat{\imath},\hat{\jmath}$ 的像），则  
  - $x = \dfrac{\det([\mathbf{b} \mid \mathbf{a}_2])}{\det(A)}$（第一列换为 $\mathbf{b}$，第二列不变）；  
  - $y = \dfrac{\det([\mathbf{a}_1 \mid \mathbf{b}])}{\det(A)}$（第一列不变，第二列换为 $\mathbf{b}$）。
- **意义**：只用输出空间可见的数据（$A$ 的列与 $\mathbf{b}$）恢复未知坐标；大系统仍常用 **Gaussian elimination**，本课侧重**理论美感**与 determinant、线性方程组几何的统一。

##### Sanity check

- 代入具体数字验证公式与真实解一致。

##### Two Dimension Questions

- 原文含练习：用 **Cramer's rule** 求 $\mathbf{x}$。

#### In three dimensions

- **推广思路**：把某一坐标看成与除某一基向量外其余基向量及 $\mathbf{x}$ 张成的体积；变换后用 $\det(A)$ 统一缩放，得到用替换列后的 **determinant** 表示的坐标公式（鼓励自行推导）。

<a id="ch13"></a>

## Ch.13 — Change of basis（基变换）

> 来源：[3Blue1Brown — Change of basis](https://www.3blue1brown.com/lessons/change-of-basis)（Chapter 13）  
> 说明：数学符号与术语保留英文原文。

### 目录（对应原文结构）

1. [Alternate system](#alternate-system)
2. [The grid](#the-grid)
3. [Change of basis matrix](#change-of-basis-matrix)
   - [How this is a transformation](#how-this-is-a-transformation)
   - [Going from ours to hers](#going-from-ours-to-hers)
4. [Translating transformations](#translating-transformations)
   - [Follow Rotation](#follow-rotation)
   - [The process](#the-process)

---

### Alternate system

- **标准基**：$\hat{\imath},\hat{\jmath}$ 不仅给出数值，还隐含“第一数向右、第二数向上”等全部坐标约定；这种把向量变成数对的方式叫 **coordinate system**，基向量即 **basis**。
- **他人基**：例如朋友 Jennifer 用 $\vec{\mathbf{b}}_1, \vec{\mathbf{b}}_2$；同一几何向量在两套语言下**数值坐标不同**——像说不同语言，但指向同一对象。
- **要点**：在她眼中 $\vec{\mathbf{b}}_1,\vec{\mathbf{b}}_2$ 的坐标分别是 $[1,0]^T,[0,1]^T$；在我们眼中它们有各自的数值表示。

### The grid

- **网格是人为的**：画方格只是可视化所选基；空间本身没有内建网格；换基则网格随之改变（原点仍一致）。

### Change of basis matrix

- **她 → 我**：若她的坐标为 $(c_1,c_2)^T$，则几何向量为 $c_1\vec{\mathbf{b}}_1 + c_2\vec{\mathbf{b}}_2$；用**我们语言下的列**拼矩阵 $B = [\vec{\mathbf{b}}_1 \mid \vec{\mathbf{b}}_2]$，有  
  $\mathbf{v}_{\text{ours}} = B\,\mathbf{c}_{\text{hers}}$。  
  即：**以她的基为列的矩阵**把“她的坐标列向量”变到“我们的坐标列向量”。

#### How this is a transformation

- **线性变换视角**：该矩阵把 $\hat{\imath},\hat{\jmath}$ 分别映到她的基向量——几何上把我们的网格“拉成”她的网格。
- **易混点**：几何上像把我们的网格变过去，数值上却在做“她的语言 → 我们的语言”的翻译；可记为：先按**误读**（把我们的基向量线性组合系数当成她的意思），再经 $B$ 纠正为真实几何向量（用我们坐标写出）。

#### Going from ours to hers

- **我 → 她**：$\mathbf{c}_{\text{hers}} = B^{-1}\mathbf{v}_{\text{ours}}$。  
- **小矩阵逆**：$2 \times 2$ 可用公式 $M^{-1} = \frac{1}{\det(M)}\begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$；高维一般交给计算机。

### Translating transformations

- **问题**：同一几何变换（如逆时针 $90^\circ$ 旋转），在标准基下矩阵为 $R$；在 Jennifer 的基下应写什么矩阵？
- **错误直觉**：把 $R$ 的列“翻译成她的语言”并不够——需要的是：**她的基向量**变换后落在何处，并用**她的坐标**记录。

#### Follow Rotation

- 标准基下示例：$R = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$（$\hat{\imath},\hat{\jmath}$ 的像）。

#### The process

- **三步合成**：对“她的坐标”向量  
  1. 左乘 $B$ → 变到“我们坐标”；  
  2. 左乘 $R$（我们坐标下的变换）；  
  3. 左乘 $B^{-1}$ → 变回“她的坐标”。  
  故 Jennifer 坐标下的矩阵为 **$B^{-1} R B$**。
- **一般模式**：$A^{-1} M A$ 表示**换视角**（**empathy**）下观察同一变换 $M$；中间是“你看到的”，外侧是“坐标/基的改变”。
- **预告**：下一章 **eigenvectors** 与 **eigenvalues** 是换基的重要动机之一。

<a id="ch14"></a>

## Ch.14 — Eigenvectors and eigenvalues（特征向量与特征值）

> 来源：[3Blue1Brown — Eigenvectors and eigenvalues](https://www.3blue1brown.com/lessons/eigenvalues)（Chapter 14）  
> 说明：数学符号与术语保留英文原文。

### 目录（对应原文结构）

1. [An example](#an-example)
2. [3d rotational axis](#3d-rotational-axis)
3. [Notes on computation](#notes-on-computation)
   - [Zero Determinant](#zero-determinant)
   - [Characteristic polynomial](#characteristic-polynomial)
   - [Rotation](#rotation)
   - [Shear](#shear)
   - [Scaling](#scaling)
   - [Questions](#questions)
4. [Working in an eigenbasis](#working-in-an-eigenbasis)
   - [Change of basis](#change-of-basis)

---

### An example

- **定义直觉**：多数向量经矩阵变换会离开自己张成的直线（**span**）；**eigenvector** 经变换仍落在自身 span 上，效果等价于乘一个标量 **eigenvalue** $\lambda$（可拉伸、压扁或反向）。
- **例题**：矩阵把 $\hat{\imath}$ 方向整体拉伸 $\lambda=3$；另一斜向直线上的向量拉伸 $\lambda=2$；其余向量会被“拧离”原 span。
- **意义**：比单纯读列（基向量去哪）更能抓住变换的**内在几何**，且较少依赖坐标选择。

### 3d rotational axis

- **旋转**：若存在 **eigenvector**，其 span 即为 **axis of rotation**；此时 **eigenvalue** 为 $1$（旋转不改变长度）。

### Notes on computation

- **方程**：$A\vec{\mathbf{v}} = \lambda \vec{\mathbf{v}}$，改写为 $(A - \lambda I)\vec{\mathbf{v}} = \mathbf{0}$。
- **非零解条件**：需 **$\det(A - \lambda I) = 0$**（否则 $A-\lambda I$ 可逆，只有零解）。

#### Zero Determinant

- **旋钮直觉**：对角减去 $\lambda$，调 $\lambda$ 使行列式为零，对应 $A-\lambda I$ 把空间“压扁”，存在非零核向量即 **eigenvector**。

#### Characteristic polynomial

- **定义**：$\det(A - \lambda I)$ 作为 $\lambda$ 的多项式，即 **characteristic polynomial**；其根为 **eigenvalues**。
- **求 eigenvector**：对每个 $\lambda$，解 $(A-\lambda I)\mathbf{x}=\mathbf{0}$ 得特征空间（直线或更高维）。

#### Rotation

- **90° 旋转**：无实 **eigenvector**；特征多项式 $\lambda^2+1=0$，根为 $\pm i$，表示无实特征方向。

#### Shear

- **剪切**：$\begin{bmatrix}1&1\\0&1\end{bmatrix}$ 固定 $x$ 轴上向量，**eigenvalue** 均为 $1$，且（此例中）只有这些方向为 **eigenvectors**。

#### Scaling

- **全局均匀缩放**：可能只有一个 **eigenvalue**，但**所有向量**都是 **eigenvectors**。

#### Questions

- 原文含矩阵特征值选择题，用于巩固 **characteristic polynomial** 与直觉。

### Working in an eigenbasis

- **Diagonal matrix**：若基向量全是 **eigenvectors**，矩阵除对角线外全零，对角元为对应 **eigenvalues**；矩阵幂次极易计算（各坐标独立缩放幂次）。
- **一般情况**：若存在足够多独立 **eigenvectors** 张成全空间，可取 **eigenbasis**，把变换表成对角形。

#### Change of basis

- **公式**：若 $P$ 的列为 **eigenvectors**（在我们坐标下），则  
  $P^{-1} A P = D$（对角元为 **eigenvalues**）。  
  计算 $A^{100}$：可先在 **eigenbasis** 下算 $D^{100}$，再换回原坐标（$P D^{100} P^{-1}$）。
- **局限**：如 **shear** 没有足够的 **eigenvectors** 张成全空间，则不能全程对角化。

<a id="ch15"></a>

## Ch.15 — A quick trick for computing eigenvalues（快速求 $2 \times 2$ 特征值）

> 来源：[3Blue1Brown — A quick trick for computing eigenvalues](https://www.3blue1brown.com/lessons/quick-eigen)（Chapter 15）  
> 说明：数学符号与术语保留英文原文。

### 目录（对应原文结构）

1. [Examples](#examples)
2. [Relation to the quadratic formula](#relation-to-the-quadratic-formula)
3. [Last thoughts](#last-thoughts)

---

**要点摘要（原文在 Examples 之前的复习与三条事实，无独立章节标题）**

- **eigenvector / eigenvalue**：$A\vec{\mathbf{v}} = \lambda \vec{\mathbf{v}}$；等价于存在非零 $\vec{\mathbf{v}}$ 使 $(A-\lambda I)\vec{\mathbf{v}}=\mathbf{0}$，故 **$\det(A-\lambda I)=0$**。常规做法是展开 **characteristic polynomial** 再求根；本课对 $2 \times 2$ 给出更直接写法。
- **事实 1（Trace）**：$\operatorname{tr}(A) = a+d = \lambda_1 + \lambda_2$；两特征值平均 $m = \dfrac{a+d}{2}$ 可由对角立即读出。
- **事实 2（Determinant）**：$\det(A) = ad-bc = \lambda_1 \lambda_2$，记为 $p$。
- **事实 3（mean–product）**：两数若形如 $m \pm d$，则 $p=(m+d)(m-d)=m^2-d^2$，故 $\lambda_{1,2} = m \pm \sqrt{m^2 - p}$。

### Examples

- **流程**：先写 $m$（对角均值），再写 $\sqrt{m^2 - \det(A)}$ 中根号内第二项为行列式；快速得到 $\lambda_{1,2}$。
- **Pauli matrices（量子力学例子）**：对角均值常为 $0$，行列式为 $-1$，故特征值常为 $\pm 1$；对一般线性组合 $a\sigma_x+b\sigma_y+c\sigma_z$（归一化方向），技巧在脑中比展开 **characteristic polynomial** 更省事。

### Relation to the quadratic formula

- **等价性**：此法与求 **characteristic polynomial** 的根**同一回事**；优势是项（**trace**、**determinant**）直接来自矩阵，意义明确。
- **二次方程视角**：首一二次多项式两根之和、之积与系数关系，正是 mean–product 思路的抽象版。

### Last thoughts

- **目的**：少背一条“新魔法”，多巩固 **trace**、**determinant** 与特征根的联系；若要证明，可对一般 $2 \times 2$ 展开 $\det(A-\lambda I)$ 看系数含义。

<a id="ch16"></a>

## Ch.16 — Abstract vector spaces（抽象向量空间）

> 来源：[3Blue1Brown — Abstract vector spaces](https://www.3blue1brown.com/lessons/abstract-vector-spaces)（Chapter 16）  
> 说明：数学符号与术语保留英文原文。

### 目录（对应原文结构）

1. [Functions as vectors](#functions-as-vectors)
   - [Standard definition of linear](#standard-definition-of-linear)
   - [Derivative as linear transformation](#derivative-as-linear-transformation)
   - [The matrix of a derivative](#the-matrix-of-a-derivative)
2. [Abstract vector space](#abstract-vector-space)
3. [Conclusion](#conclusion)

---

### Functions as vectors

- **引言动机**：向量究竟是箭头、坐标列表，还是更一般的对象？换基后 **determinant**、**eigenvectors** 等几何量与坐标选择无关，提示“空间”先于坐标；下文用**函数**说明：满足加法与数乘的对象都可承载线性代数语言。
- **加法**：$(f+g)(x) = f(x)+g(x)$，与分量逐项相加类比（只是“坐标”有无穷多个）。
- **数乘**：$(cf)(x) = c\,f(x)$，与向量逐分量缩放类比。
- **结论**：只要加法、数乘合理，就可谈论 **linear transformation**、**null space**、**eigenvectors**（在函数空间常称 **eigenfunction**）、与 **dot product** 类比物 **inner product** 等。

#### Standard definition of linear

- **形式定义**：$L(\mathbf{v}+\mathbf{w})=L(\mathbf{v})+L(\mathbf{w})$，$L(s\mathbf{v})=sL(\mathbf{v})$（保持加法与数乘）。  
- **与几何版联系**：二维“网格线平行等距、过原点”是此定义在箭头世界中的**图示**。

#### Derivative as linear transformation

- **导数**：把函数映到函数；满足和法则与常数倍法则 ⇒ **linear**（在分析中常称 **operator**）。

#### The matrix of a derivative

- **多项式子空间**：选基 $1,x,x^2,x^3,\ldots$（无穷维，但每个多项式只有有限个非零坐标）。
- **坐标示例**：$x^2+3x+5$ 对应 $(5,3,1,0,\ldots)^T$（尾部无限零）。
- **导数的矩阵**：无限矩阵，次对角线上为正整数 $1,2,3,\ldots$（对基函数逐列求导后放入坐标）；矩阵–向量乘法与求导一致。
- **启示**：矩阵乘法与求导同属 **linear transformation** 家族。

### Abstract vector space

- **vector space**：任何对象集合，只要定义了合理的加法与数乘，并满足现代教材中的 **8 axioms**（原文图示列出），线性代数定理即可适用。
- **公理角色**：像“接口规范”——数学家只基于公理证明结论；具体实现可以是箭头、数组、函数或更怪的对象。
- **教学取舍**：入门用几何箭头很有效；高阶与教材则倾向抽象表述，因必须覆盖所有满足公理的模型。

### Conclusion

- **数学家式回答**：“向量是什么”不如问“对象是否构成 **vector space**”；形式不重要，关键是运算规则。
- **系列收尾**：若已理解前文直觉并完成练习，后续学习会更高效。
---

<a id="review"></a>

## 附录：全书快速复习（Ch1–Ch16）

> 由原独立笔记《线性代数快速复习总结》合并至本册末尾，便于在读完 Ch.12–16 后做全书串讲；涵盖 Ch1–Ch16。基于 3Blue1Brown「Essence of Linear Algebra」系列，核心思想：**几何直觉 ↔ 数值计算** 自由切换。

---

### 目录

1. [向量的本质](#ch1-向量的本质)
2. [线性组合、张成空间与基](#ch2-线性组合张成空间与基)
3. [线性变换与矩阵](#ch3-线性变换与矩阵)
4. [矩阵乘法 = 变换复合](#ch4-矩阵乘法--变换复合)
5. [三维线性变换](#ch5-三维线性变换)
6. [行列式](#ch6-行列式)
7. [逆矩阵、列空间与零空间](#ch7-逆矩阵列空间与零空间)
8. [非方阵：跨维度变换](#ch8-非方阵跨维度变换)
9. [点积与对偶](#ch9-点积与对偶)
10. [叉积](#ch10-叉积)
11. [叉积与线性变换（对偶视角）](#ch11-叉积与线性变换对偶视角)
12. [Cramer's rule（几何）](#ch12-cramers-rule几何)
13. [换基](#ch13-换基)
14. [特征向量与特征值](#ch14-特征向量与特征值)
15. [2×2 特征值速算](#ch15-22-特征值速算)
16. [抽象向量空间](#ch16-抽象向量空间)

---

### Ch1 向量的本质

**三种视角统一于两个运算：向量加法 + 标量乘法**

| 视角 | 向量是什么 | 关键特征 |
|------|-----------|---------|
| 物理 | 空间中的箭头 | 长度 + 方向确定，位置无关 |
| CS | 有序数字列表 | 顺序重要，维度 = 列表长度 |
| 数学 | 抽象对象 | 满足加法和数乘公理即可 |

**坐标系**：原点 + 坐标轴 → 向量与坐标一一对应

**两大运算**：
- **加法**（Tip-to-Tail）：$\begin{bmatrix} x_1 \\ y_1 \end{bmatrix} + \begin{bmatrix} x_2 \\ y_2 \end{bmatrix} = \begin{bmatrix} x_1+x_2 \\ y_1+y_2 \end{bmatrix}$
- **标量乘法**：$c \cdot \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} cx \\ cy \end{bmatrix}$（拉伸/压缩/翻转）

---

### Ch2 线性组合、张成空间与基

**核心链条**：坐标 = 缩放基向量 → 缩放求和 = 线性组合 → 所有可达结果 = Span → 无冗余张成全空间 = Basis

| 概念 | 定义 | 关键点 |
|------|------|--------|
| **Basis Vectors** | $\hat{\imath}$, $\hat{\jmath}$（坐标轴方向单位向量） | 坐标本质是对 basis 的缩放系数 |
| **Linear Combination** | $a\vec{v} + b\vec{w}$ | scalar 遍历所有实数 |
| **Span** | 一组向量所有线性组合的集合 | 2D 不共线 → 整个平面 |
| **Linearly Independent** | 没有向量可由其他向量线性组合得到 | 每个向量贡献新维度 |
| **Linearly Dependent** | 存在冗余向量 | 移除后 Span 不变 |
| **Basis** | Linearly Independent + Span the Space | 基的选择不唯一 |

---

### Ch3 线性变换与矩阵

**Linear Transformation 判定**：
1. 直线保持直线（网格平行等距）
2. 原点不动

**核心洞察**：只需记录 $\hat{\imath}$、$\hat{\jmath}$ 的落点，其他一切可推导！

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = x \begin{bmatrix} a \\ c \end{bmatrix} + y \begin{bmatrix} b \\ d \end{bmatrix} = \begin{bmatrix} ax+by \\ cx+dy \end{bmatrix}$$

- **第一列** = $\hat{\imath}$ 的落点
- **第二列** = $\hat{\jmath}$ 的落点
- **矩阵乘向量** = 对变换后的 basis vectors 做线性组合

**典型变换**：

| 变换 | 矩阵 |
|------|-------|
| 逆时针旋转 90° | $\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}$ |
| Shear（剪切） | $\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$ |
| 列线性相关 | 空间压缩到一条线（降维） |

---

### Ch4 矩阵乘法 = 变换复合

**核心**：$M_2 \cdot M_1$ 表示先应用 $M_1$，再应用 $M_2$（从右往左读）

**计算方法**：乘积每列 = 左矩阵 × 右矩阵对应列

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} e & f \\ g & h \end{bmatrix} = \begin{bmatrix} ae+bg & af+bh \\ ce+dg & cf+dh \end{bmatrix}$$

**性质**：

| 性质 | 结论 | 直觉 |
|------|------|------|
| 交换律 | ❌ $AB \neq BA$ | 先 shear 再旋转 ≠ 先旋转再 shear |
| 结合律 | ✅ $(AB)C = A(BC)$ | 同一操作序列，仅分组不同 |

---

### Ch5 三维线性变换

**2D → 3D 无缝推广**：

- 三个 basis vectors：$\hat{\imath}$, $\hat{\jmath}$, $\hat{k}$
- 3×3 矩阵：三列 = 三个 basis 的落点坐标
- 向量变换：$x \cdot \text{col}_1 + y \cdot \text{col}_2 + z \cdot \text{col}_3$

**组合变换**：复杂 3D 旋转可分解为多个简单变换的 composition

**矩阵乘法兼容条件**：$AB$ 有意义 ⟺ $A$ 的列数 = $B$ 的行数

---

### Ch6 行列式

**Determinant = 变换对空间的缩放因子**（2D 面积 / 3D 体积）

| det 值 | 含义 |
|--------|------|
| $\det > 0$ | 面积/体积放大，orientation 不变 |
| $\det = 0$ | 压缩到低维，列向量 linearly dependent |
| $\det < 0$ | $|\det|$ 为缩放因子，orientation 翻转 |

**计算公式**：
- 2D：$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$
- 3D：沿第一行 cofactor expansion

**乘积性质**：$\det(M_1 M_2) = \det(M_1) \cdot \det(M_2)$

---

### Ch7 逆矩阵、列空间与零空间

**线性方程组**：$A\vec{x} = \vec{v}$ → 寻找经变换 $A$ 后落在 $\vec{v}$ 上的向量

#### 求解策略

| 条件 | 结论 | 操作 |
|------|------|------|
| $\det(A) \neq 0$ | Inverse 存在 | 唯一解 $\vec{x} = A^{-1}\vec{v}$ |
| $\det(A) = 0$ | Inverse 不存在 | 看 $\vec{v}$ 是否在 Column Space 中 |

#### 三大概念

| 概念 | 定义 | 几何意义 |
|------|------|----------|
| **Rank** | Column Space 的维度 | Full rank ↔ $\det \neq 0$ |
| **Column Space** | 矩阵列向量的 Span | 所有可能输出的集合 |
| **Null Space (Kernel)** | 被映射到 $\vec{0}$ 的向量集合 | $A\vec{x}=\vec{0}$ 的解集 |

**Null Space 大小**：
- Full rank → 仅零向量
- 3D → 平面（rank 2）→ Null Space 是一条线
- 3D → 线（rank 1）→ Null Space 是一个平面

---

### Ch8 非方阵：跨维度变换

**$m \times n$ 矩阵**：$n$ 维 → $m$ 维的线性变换
- **列数** = 输入维度（有几个 basis vectors）
- **行数** = 输出维度（landing spot 坐标数）

| Matrix 尺寸 | 映射方向 | Column Space |
|---|---|---|
| 3×2 | 2D → 3D | 3D 中过原点的 2D 平面 |
| 2×3 | 3D → 2D | 2D 中的子空间 |
| 1×2 | 2D → 1D | 数轴（本质是 dot product） |

编码方式统一：columns = basis vectors 的 landing spots

---

### Ch9 点积与对偶

**数值**：$\mathbf{v}\cdot\mathbf{w}=\sum_i v_i w_i$（同维向量对应分量相乘再相加）。

**几何**：$\mathbf{v}\cdot\mathbf{w}=\|\mathbf{v}\|\times$（$\mathbf{w}$ 在 $\mathbf{v}$ 所在直线上的**有向投影长度**）；符号表示大致同向 / 反向 / 垂直（垂直时点积为 0）。

**顺序**：$\mathbf{v}\cdot\mathbf{w}=\mathbf{w}\cdot\mathbf{v}$（与“先投影哪一个”的几何叙述不同，但结果相同）。

**到一维的线性变换**：任意 $\mathbb{R}^n\to\mathbb{R}$ 的 **linear transformation** 都可写成 **$1\times n$** 矩阵；与某固定向量 $\mathbf{p}$ 做 **dot product** 在计算上完全等价（把矩阵“立起来”当向量）。

**Duality（对偶）**：「到数轴的线性映射」$\leftrightarrow$「空间中一个向量 $\mathbf{p}$」——施加变换等价于与 $\mathbf{p}$ **dot product**；点积可理解为把其中一个向量看成“线性泛函”的坐标表示。

**单位向量**：与单位向量 $\hat{\mathbf{u}}$ 点积 = 投影到 $\hat{\mathbf{u}}$ 方向再取有向长度；非单位向量时再多乘该向量长度。

---

### Ch10 叉积

**2D（有向面积）**：$\mathbf{v}\times\mathbf{w}$ 的绝对值为两向量张成**平行四边形面积**；符号由定向（$\mathbf{w}$ 相对 $\mathbf{v}$ 逆时针为正）决定；$\mathbf{v}\times\mathbf{w}=-\,\mathbf{w}\times\mathbf{v}$。

**计算（2D）**：把两向量坐标作为 $2\times 2$ 矩阵两列，取 **determinant**。

**3D（向量结果）**：$\mathbf{v}\times\mathbf{w}$ 为向量，**模长**=张成平行四边形面积，**方向**=与两向量都垂直且满足 **right-hand rule**。

**记忆公式**：可用含 $\hat{\imath},\hat{\jmath},\hat{k}$ 的 **$3\times 3$ determinant** 形式展开（下一章解释其必然性）。

---

### Ch11 叉积与线性变换（对偶视角）

**核心构造**：固定 $\mathbf{v},\mathbf{w}$，令 $L(\mathbf{u})=\det[\mathbf{u}\mid\mathbf{v}\mid\mathbf{w}]$（列向量拼成 $3\times 3$ 矩阵），$L$ 对 $\mathbf{u}$ **线性**，输出为数。

**对偶向量**：存在唯一 $\mathbf{p}$ 使得 $L(\mathbf{u})=\mathbf{p}\cdot\mathbf{u}$；该 $\mathbf{p}$ 即为 $\mathbf{v}\times\mathbf{w}$。

**几何与代数统一**：按第一列展开 **determinant** 得到叉积分量；几何上「体积 = 底面积 × 高」对应「点积 = 投影 × 长度」——故叉积向量垂直于 $\mathbf{v},\mathbf{w}$ 且模为面积。

---

### Ch12 Cramer's rule（几何）

**前提**：$\det(A)\neq 0$，$A\mathbf{x}=\mathbf{b}$ 有唯一解（本课几何叙述限于此情形）。

**思想**：坐标可用「与部分标准基张成的**有向面积/体积**」刻画；线性变换后，这些量统一乘以 $\det(A)$。

**2D 公式**：若 $A=[\mathbf{a}_1\mid\mathbf{a}_2]$，则  
$x=\dfrac{\det([\mathbf{b}\mid\mathbf{a}_2])}{\det(A)}$，$y=\dfrac{\det([\mathbf{a}_1\mid\mathbf{b}])}{\det(A)}$。

**评价**：大矩阵求解除法上多用 **Gaussian elimination**；Cramer 用于巩固 **determinant** 与方程组几何图像。

---

### Ch13 换基

**两套语言**：同一几何向量，在标准基与他人基 $\{\mathbf{b}_1,\mathbf{b}_2\}$ 下坐标不同。

**她 → 我**：若 $B=[\mathbf{b}_1\mid\mathbf{b}_2]$（列用**我方坐标**写出她的基），则 $\mathbf{x}_{\text{ours}}=B\,\mathbf{c}_{\text{hers}}$。

**我 → 她**：$\mathbf{c}_{\text{hers}}=B^{-1}\mathbf{x}_{\text{ours}}$。

**变换在不同基下的矩阵**：若我方矩阵为 $M$，她方下为 **$B^{-1}MB$**（先换到我方坐标 → 做 $M$ → 再换回她方）；记忆：**$A^{-1}MA$** 表示“换视角看同一变换”。

---

### Ch14 特征向量与特征值

**定义**：$A\mathbf{v}=\lambda\mathbf{v}$ 且 $\mathbf{v}\neq\mathbf{0}$，则 $\mathbf{v}$ 为 **eigenvector**，$\lambda$ 为 **eigenvalue**——变换后向量仍落在自身 **span** 上，效果仅为伸缩（可反向）。

**计算**：$(A-\lambda I)\mathbf{v}=\mathbf{0}$ 有非零解 ⟺ **$\det(A-\lambda I)=0$**；左边关于 $\lambda$ 为 **characteristic polynomial**。

**几何例**：3D **rotation** 的轴方向向量为 **eigenvector**，**eigenvalue** 常为 $1$（保长）。

**对角化直觉**：若存在由 **eigenvectors** 组成的 **eigenbasis**，则在该基下矩阵为**对角阵**（各坐标独立缩放），$A^k$ 等运算简单；并非所有变换都有足够特征向量张成全空间（如某些 **shear**）。

---

### Ch15 2×2 特征值速算

**恒等式**：$\operatorname{tr}(A)=\lambda_1+\lambda_2$，$\det(A)=\lambda_1\lambda_2$。

**技巧**：记 $m=\dfrac{\lambda_1+\lambda_2}{2}=\dfrac{a+d}{2}$，$p=\det(A)$，则  
$\lambda_{1,2}=m\pm\sqrt{m^2-p}$（由 $(m+d)(m-d)=m^2-d^2$ 得到）。

**用途**：小例子中快速读出特征根；与 **characteristic polynomial** 完全等价，但项更“有名字”。

---

### Ch16 抽象向量空间

**动机**：箭头与数字列表都是**具体模型**；换基后许多几何量与坐标无关，提示更一般的“向量”概念。

**函数像向量**：$(f+g)(x)=f(x)+g(x)$，$(cf)(x)=c\,f(x)$——可加、可缩放即可套线性代数语言。

**线性（形式定义）**：$L(\mathbf{u}+\mathbf{v})=L(\mathbf{u})+L(\mathbf{v})$，$L(c\mathbf{u})=cL(\mathbf{u})$；**derivative** 在函数空间上是 **linear transformation**（常称 **operator**）。

**例子**：多项式以 $1,x,x^2,\ldots$ 为基时，求导对应**无穷矩阵**（次对角线系数），与矩阵–向量乘法同一类结构。

**vector space**：集合 + 合理的加法与数乘，满足 **8 axioms**；定理只依赖公理，从而适用于箭头、元组、函数等所有实现。

---

### 全局知识图谱

```
向量 (坐标 = 缩放基向量)
  │
  ├── 线性组合 → Span → Basis (独立 + 张成)
  │
  ├── 线性变换 = 矩阵 (列 = 基的落点)
  │     │
  │     ├── 矩阵乘向量 = 对新基做线性组合
  │     ├── 矩阵乘矩阵 = 变换复合 (从右往左)
  │     └── 非方阵 = 跨维度变换
  │
  ├── 行列式 = 空间缩放因子
  │     ├── det ≠ 0 → 可逆
  │     └── det = 0 → 降维，不可逆
  │
  ├── 线性方程组 Ax = v
  │     ├── det ≠ 0 → x = A⁻¹v (唯一解)
  │     └── det = 0 → Column Space 判存在性
  │                   Null Space 描述解集结构
  │
  ├── 点积 / 叉积
  │     ├── Dot：投影、同向度；对偶 ↔ 到一维的线性泛函
  │     └── Cross：有向面积/法向；det([u|v|w]) 对 u 线性 → 对偶向量即 v×w
  │
  ├── Cramer：坐标 ↔ 替换列后的 det / det(A)
  │
  ├── 换基：B、B⁻¹、B⁻¹MB（同一变换不同坐标）
  │
  ├── 特征：Av = λv；det(A−λI)=0；eigenbasis → 对角化
  │
  └── 抽象向量空间：加法 + 数乘 + 公理 → 箭头/数组/函数统一框架
```

---

### 一句话记忆卡片

| # | 一句话 |
|---|--------|
| 1 | 向量 = 箭头 = 数字列表，统一于加法和数乘 |
| 2 | 坐标是对 basis 的缩放系数，basis 选择不唯一 |
| 3 | 矩阵 = 线性变换的数值描述，列 = 基的落点 |
| 4 | 矩阵乘法 = 变换复合，从右往左读 |
| 5 | 3D 与 2D 原理相同，3×3 矩阵三列记录三个基 |
| 6 | 行列式 = 面积/体积缩放因子，负值 = orientation 翻转 |
| 7 | det≠0 可逆有唯一解；det=0 看 Column Space 和 Null Space |
| 8 | m×n 矩阵：n 维输入 → m 维输出，列数=输入维，行数=输出维 |
| 9 | 点积 = 投影×长度；到一维线性变换 ↔ 与某向量点积（duality） |
| 10 | 2D 叉积 = 有向面积（det）；3D 叉积 = 垂直、模为面积、右手定则 |
| 11 | 以 $\mathbf{u},\mathbf{v},\mathbf{w}$ 为列的 $\det$ 对 $\mathbf{u}$ 线性 ⇒ 对偶向量即 $\mathbf{v}\times\mathbf{w}$ |
| 12 | Cramer：用替换列的 det 比 det(A) 读出各坐标（det≠0） |
| 13 | 换基 B、逆 B⁻¹；他方下同一变换矩阵为 B⁻¹MB |
| 14 | 特征：仍在自身 span 上；det(A−λI)=0；eigenbasis 下对角化 |
| 15 | 2×2：λ = m ± √(m²−p)，m 为对角均值，p 为 det |
| 16 | 向量空间 = 满足公理的加法与数乘；函数、导数也是线性代数对象 |

---

*来源：[3Blue1Brown - Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)*
