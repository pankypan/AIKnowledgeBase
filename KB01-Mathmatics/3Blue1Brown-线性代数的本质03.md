# 3Blue1Brown「线性代数的本质」— 合集（第 12–16 章及附录）

> 说明：本文档由 `ch12`–`ch16` 合并而成；**文末附录**为全书快速复习（Ch1–Ch16，原《线性代数快速复习总结》）。数学术语与公式保持英文原文为主。  
> 正文范围：Cramer's rule、换基、特征值与特征向量、2×2 快速技巧、抽象向量空间。


## Ch.12 — Cramer's rule, explained geometrically（Cramer's rule 的几何解释）

> 来源：[3Blue1Brown — Cramer's rule, explained geometrically](https://www.3blue1brown.com/lessons/cramers-rule)（Chapter 12）  
> 说明：数学符号与术语保留英文原文。


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

