# Probability and Distribution(概率与分布)

## 6.1 概率空间的构建

概率论旨在定义 **一个数学结构来描述实验结果的随机性**。利用这种概率的数学结构，目标是进行自动化推理，从这个意义上说，概率是对逻辑推理的泛化（Jaynes, 2003）。


### 哲学问题

概率的哲学基础以及它应该如何以某种方式（Jaynes, 2003）E. T. Jaynes（1922-1998）确定了三个数学标准，这些标准必须适用于所有可能性：

1. 可能性的程度由实数表示
2. 这些数字必须基于常识的规则
3. 所得推理必须是一致的，其中“一致”一词包含以下三层含义
   1. 一致性或无矛盾性：当可以通过不同方式达到相同结果时，在所有情况下都必须找到相同的可能性值。
   2. 诚实性：必须考虑所有可用数据。
   3. 可再现性：如果我们对两个问题的知识状态相同，那么我们必须为它们分配相同程度的可能性。

---

在机器学习和统计学中，概率有两种主要解释：
- **贝叶斯解释**使用概率来指定用户对某个事件发生的不确定性程度。它有时被称为“主观概率”或“信念程度”。
- **频率解释**则考虑感兴趣事件相对于发生事件总数的相对频率。当数据无限时，某事件的概率被定义为该事件的相对频率。



### 概率与随机变量

现代概率论基于Kolmogorov提出的一组公理（Grinstead and Snell, 1997; Jaynes, 2003），这些公理引入了**样本空间**、**事件空间**和**概率测度**这三个概念。概率空间模型用于模拟具有随机结果的现实世界过程（称为实验）。

> **定义 6.1（概率空间）**：概率空间是一个三元组 $(\Omega, \mathcal{A}, P)$，其中 $\Omega$ 是样本空间，$\mathcal{A}$ 是事件空间，$P$ 是概率测度。
>
> 1. **样本空间 $\Omega$** 是实验所有可能结果的集合，通常表示为 $\Omega$。
> 2. **事件空间 $\mathcal{A}$** 是实验潜在结果的集合。
> 3. **概率方程 $P$** 是一个函数，对于每个事件 $A\in\mathcal{A}$，我们关联一个数 $P(A)$，它衡量了事件发生的概率或信念程度，即 $P: \mathcal{A} \to [0, 1]$。
> 
> 对于一个事件 $A\in\mathcal{A}$，$P(A)$ 被称为 $A$ 的概率。
> 


> **定义 6.2（随机变量）**：对于任意实数 $x$，$\{ \omega: X(\omega) \leq x \} \in \mathcal{A}$ 的 **实值函数 $X$** 为随机变量。
>
> 1. **随机变量 $X$** 是一个函数 $X: \Omega \to \mathbb{R}$，对于每个样本点 $\omega \in \Omega$，我们关联一个实数 $X(\omega)$。
> 2. 由于 $\{\omega : X(\omega) \leq x\} \in \mathcal{A}$，我们可以通过概率测度得到 $0 \leq P(X \leq x) = P(\{\omega : X(\omega) \leq x\}) \leq 1$。
> 


$P$ 的输入**永远是事件（集合）**。随机变量 $X$ 的作用是提供了一种方便的方式，把实数上的条件（如 $X = x$ 或 $X \leq x$）**反向映射**回 $\Omega$ 中的事件，再交给 $P$ 来计算概率。

$$
P(X = x) \triangleq P\bigl(\{\omega \in \Omega : X(\omega) = x\}\bigr)
$$

| 表达式        | 实际含义                               | $P$ 的输入                                           |
| ------------- | -------------------------------------- | ---------------------------------------------------- |
| $P(A)$        | 事件 $A$ 的概率                        | 事件 $A \in \mathcal{A}$（$\Omega$ 的子集）          |
| $P(X = x)$    | $\{\omega : X(\omega) = x\}$ 的概率    | 事件 $\{\omega : X(\omega) = x\} \in \mathcal{A}$    |
| $P(X \leq x)$ | $\{\omega : X(\omega) \leq x\}$ 的概率 | 事件 $\{\omega : X(\omega) \leq x\} \in \mathcal{A}$ |


### 统计学

**概率论(Probability theory)** 和 **统计学(Statistics)** 经常被放在一起讨论，但它们关注的是 **不确定性的不同方面**。对比它们的一种方式是考虑所研究的问题类型。
- 使用**概率论**，我们可以考虑某个过程的模型，其中潜在的不确定性通过随机变量来捕捉，并利用概率规则来推导出所发生的事情。
- 在**统计学**中，我们观察到某件事情已经发生，并试图找出解释这些观察结果的潜在过程。

机器学习的目标更接近统计学，即构建一个能够充分表示数据生成过程的模型。我们可以利用概率规则来获得某些数据的“最佳拟合”模型。




## 6.2 离散概率与连续概率

### 离散概率

当目标空间是离散的时，我们可以将多个随机变量的概率分布想象为填充一个（多维）数字数组。图6.2给出了一个示例。

<div align="center">
   <img src="https://datawhalechina.github.io/math-for-ai/attachments/6.2.png" alt="图6.2"/>
   <center>图6.2 具有随机变量 $X$ 和 $Y$ 的离散二变量概率质量函数的可视化</center>
</div>


> **定义 6.3（联合概率）**：联合概率的目标空间是每个随机变量目标空间的笛卡尔积。对于两个随机变量 $X$ 和 $Y$，联合概率定义为两个值共同出现的条目：
> $$P(X=x_i,Y=y_j)=\frac{n_{ij}}{N} \tag{6.1}$$
> 
> 其中 $n_{ij}$ 是状态为 $x_i$ 和 $y_j$ 的事件数，$N$ 是事件的总数。联合概率是两个事件交集的概率，即 $P(X=x_i, Y=y_j) = P(X=x_i \cap Y=y_j)$。
> 

> **定义 6.4（概率质量函数）**：离散随机变量 $X$ 的 **概率质量函数（PMF）** 是一个函数 $p: \mathcal{T} \to [0,1]$，将每个可能的取值映射到其概率：
> $$p(x) = P(X = x) \tag{6.2}$$
> 
> 且满足 $\forall x \in \mathcal{T}: p(x) \geq 0$ 以及 $\sum_{x \in \mathcal{T}} p(x) = 1$。
>

在多变量情形下，常用以下简略记法（如图6.2所示）：
1. $p(x, y)$：**联合概率**，即 $P(X=x \cap Y=y)$。
2. $p(x)$：**边缘概率**，即不论 $Y$ 取何值时 $X=x$ 的概率。
3. $p(y \mid x)$：**条件概率**，即在 $X=x$ 条件下 $Y=y$ 的概率。
4. $X \sim p(x)$ 表示随机变量 $X$ 服从分布 $p(x)$。
 

> **例 6.1**　设随机变量 $X$ 表示一次掷骰子的点数，则 $X$ 的目标空间为 $\mathcal{T}=\{1,2,3,4,5,6\}$。对于均匀骰子，PMF 为
> $$p(x)=P(X=x)=\frac{1}{6}\,,\quad x\in\{1,2,3,4,5,6\}\,.$$
> 
> 容易验证 $p(x)\geq 0$ 且 $\sum_{x=1}^{6}p(x)=1$，满足定义 6.4 的条件。
> 
> 现在考虑联合概率。设 $Y$ 表示第二次掷骰子的点数（两次独立），联合 PMF 为
> 
> $$p(x,y)=P(X=x,Y=y)=P(X=x)\,P(Y=y)=\frac{1}{6}\times\frac{1}{6}=\frac{1}{36}\,.$$
> 
> 容易验证 $p(x,y)\geq 0$ 且 $\sum_{x=1}^{6}\sum_{y=1}^{6}p(x,y)=1$，满足定义 6.1 的条件。
> 
> $$p(x,y)=P(X=x,Y=y)=P(X=x)\,P(Y=y)=\frac{1}{6}\times\frac{1}{6}=\frac{1}{36}$$
> 
> 在此基础上可以进一步得到：
> 
> | 记号 | 含义 | 示例 |
> |:---:|:---:|:---:|
> | $p(x,y)$ | 联合概率 | $p(1,3)=\frac{1}{36}$ |
> | $p(x)$ | 边缘概率 | $p(1)=\sum_{y=1}^{6}p(1,y)=\frac{6}{36}=\frac{1}{6}$ |
> | $p(y\mid x)$ | 条件概率 | $p(3\mid 1)=\frac{p(1,3)}{p(1)}=\frac{1/36}{1/6}=\frac{1}{6}$ |
> 

### 连续概率

> **定义 6.5（概率密度函数）**：函数 $f:\mathbb{R}^D\to\mathbb{R}$ 称为 **概率密度函数（PDF）**，若满足：
>
> 1. $\forall \boldsymbol{x}\in \mathbb{R}^D$ 有
> $$f(\boldsymbol{x}) \geqslant 0 \tag{6.3}$$
> 
> 2. 其积分存在，且 
> $$\displaystyle\int_{\mathbb{R}^D}f(\boldsymbol{x})\,\mathrm{d}\boldsymbol{x}=1 \tag{6.4}$$
>
> 随机变量 $X$ 通过 PDF 与概率关联：
> $$P(a\leqslant X\leqslant b)=\int_{a}^{b}f(x)\,\mathrm{d}x \tag{6.5}$$
> 
> 其中 $a,b\in\mathbb{R}$。这种关联 (6.5) 称为随机变量 $X$ 的 **分布**。对于多元情形 $\boldsymbol{x}\in\mathbb{R}^D$，定义类似。
> 

**备注**：与离散随机变量不同，连续随机变量 $X$ 取特定值 $x$ 的概率 $P(X=x)=0$——这相当于在 (6.5) 中令 $a=b$，积分区间退化为一个点。

对于离散随机变量，(6.4) 中的积分替换为求和（定义 6.4），即 $\sum_{x \in \mathcal{T}} p(x) = 1$。
 


> **定义 6.6（累积分布函数）**：多元实值随机变量 $X=[X_1,\ldots,X_D]^\top$（状态 $\boldsymbol{x}=[x_1,\ldots,x_D]^\top \in \mathbb{R}^D$）的 **累积分布函数（CDF）** 定义为：
> $$F_X(\boldsymbol{x})=P(X_1\leqslant x_1,\ldots,X_D\leqslant x_D) \tag{6.6}$$
> 
> CDF 也可以表示为 PDF 的积分：
> $$F_{X}(\boldsymbol{x})=\int_{-\infty}^{x_{1}}\cdots\int_{-\infty}^{x_{D}}f(z_{1},\ldots,z_{D})\,\mathrm{d}z_{1}\cdots\mathrm{d}z_{D} \tag{6.7}$$
> 

**备注**：讨论"分布"时涉及两个不同概念：
1. PDF $f(x)$ 本身（非负且积分为 1 的函数）；
2. 随机变量 $X$ 的 **分布律**，即将 $X$ 与 $f(x)$ 关联起来的映射关系 (6.5)。 



### 离散分布与连续分布的对比

<div align="center">
   <img src="https://datawhalechina.github.io/math-for-ai/attachments/6.3.png" alt="图6.3" width="650"/>
   <center>图6.3 离散分布和连续均匀分布的例子</center>
</div>


> **例 6.2**　下面通过"均匀分布"（每个状态等可能）来对比离散与连续两种情形下"概率"的不同含义。
> 
> **离散情形（图6.3a）**。设 $Z$ 是一个离散均匀随机变量，取值集合为 $\{-1.1,\;0.3,\;1.5\}$。其 PMF 为
> 
> $$p(z)=P(Z=z)=\frac{1}{3}\,,\quad z\in\{-1.1,\;0.3,\;1.5\}$$
> 
> 在图6.3(a) 中，每根竖线的**高度直接就是概率**，三根竖线高度之和 $= 3\times\frac{1}{3}=1$。
> 
> **连续情形（图6.3b）**。设 $X$ 是在区间 $[0.9,\;1.6]$ 上均匀分布的连续随机变量，其 PDF 为
> 
> $$p(x)=\frac{1}{1.6-0.9}=\frac{10}{7}\approx 1.43\,,\quad x\in[0.9,\;1.6]\,.$$
> 
> 注意 PDF 的值**不是概率，而是概率密度**，因此可以大于 1。概率必须通过积分（面积）获得：
> 
> $$\int_{0.9}^{1.6}p(x)\,\mathrm{d}x=\frac{10}{7}\times 0.7=1$$
> 
> 
> $$p(x)=\frac{1}{1.6-0.9}=\frac{10}{7}\approx 1.43\,,\quad x\in[0.9,\;1.6]$$

**要点**：离散分布看"高度"——PMF 值之和为 1；连续分布看"面积"——PDF 的积分为 1。这正是定义 6.4（PMF）与定义 6.5（PDF）的核心区别。


### 条件概率

> **定义 6.7（条件概率）**：对于两个随机变量 $X$ 和 $Y$，在 $X=x$ 已发生（$p(x)>0$）的条件下，$Y=y$ 的 **条件概率** 定义为：
> $$p(y\mid x)=\frac{p(x,y)}{p(x)} \tag{6.8}$$
> 即联合概率除以边缘概率。

直觉上，条件概率回答的是："在已知 $X=x$ 的前提下，$Y=y$ 还有多大可能？" 它通过将联合概率 $p(x,y)$ 除以 $p(x)$ 来"归一化"——相当于把样本空间缩小到 $X=x$ 的那部分，再看 $Y=y$ 占多大比例。

由条件概率的定义 6.7 可以直接推出后续两条重要规则：
- **乘法规则**：将分母移到右边，得 $p(x,y)=p(y\mid x)\,p(x)$
- **贝叶斯定理**：交换 $x,y$ 的角色并代入乘法规则即可导出





## 6.3 基本运算法则与贝叶斯定理

### 基本法则

#### 加法规则

加法规则（也称 **边缘化**）将 **联合分布** 与 **边缘分布** 联系起来：对随机变量 $Y$ 的所有状态求和（离散）或积分（连续），即可从联合分布中"消去" $Y$，得到 $X$ 的边缘分布。

$$
p(\boldsymbol{x})=\left\{\begin{array}{ll}\displaystyle\sum_{\boldsymbol{y}\in\mathcal{Y}}p(\boldsymbol{x},\boldsymbol{y})&\quad\text{如果}\:\boldsymbol{y}\:\text{是离散的}\\\\\displaystyle\int_{\mathcal{Y}}p(\boldsymbol{x},\boldsymbol{y})\mathrm{d}\boldsymbol{y}&\quad\text{如果}\:\boldsymbol{y}\:\text{是连续的}\end{array}\right., \tag{6.9}
$$

其中 $\mathcal{Y}$ 是随机变量 $Y$ 的目标空间。当联合分布包含两个以上的随机变量时，加法规则可以应用于任意子集，得到仍可能包含多个随机变量的边缘分布。




#### 乘法规则

乘法规则将 **联合分布** 与 **条件分布** 联系起来：任何联合分布都可以分解为一个边缘分布与一个条件分布的乘积。

$$
p(\boldsymbol{x},\boldsymbol{y})=p(\boldsymbol{y}\mid\boldsymbol{x})\,p(\boldsymbol{x})\:, \tag{6.10}
$$

- $p(\boldsymbol{x})$：$X$ 的 **边缘分布**
- $p(\boldsymbol{y}\mid\boldsymbol{x})$：给定 $X$ 时 $Y$ 的 **条件分布**

由于联合分布中随机变量的顺序是任意的，因此同样有 $p(\boldsymbol{x},\boldsymbol{y})=p(\boldsymbol{x}\mid\boldsymbol{y})\,p(\boldsymbol{y})$。




### 贝叶斯定理

#### 概念与公式

**贝叶斯定理（也称为贝叶斯规则或贝叶斯定律）**: 假设我们对一个未观察到的随机变量 $\boldsymbol{x}$

1. 有一些先验知识 $p(\boldsymbol{x})$，
2. 以及 $\boldsymbol{x}$ 与我们可以观察到的第二个随机变量 $\boldsymbol{y}$ 之间的某种关系(似然函数) $p(\boldsymbol{y}\mid\boldsymbol{x})$,
3. 如果我们观察到了(证据) $p(\boldsymbol{y})$，

我们可以使用贝叶斯定理根据观察到的 $\boldsymbol{y}$ 的值来得出关于 $\boldsymbol{x}$ 的一些结论:

$$
\underbrace{p(\boldsymbol{x}\mid\boldsymbol{y})}_{\text{后验}}=\frac{\overbrace{p(\boldsymbol{y}\mid\boldsymbol{x})}^{\text{似然度}}\overbrace{p(\boldsymbol{x})}^{\text{先验}}}{\underbrace{p(\boldsymbol{y})}_{\text{证据}}}
\tag{6.11}
$$

(6.11) 中各项的含义：

| 符号 | 名称 | 含义 |
|:---:|:---:|:---|
| $p(\boldsymbol{x})$ | **先验** | 观察数据之前对 $\boldsymbol{x}$ 的主观信念。需确保对所有可能的 $\boldsymbol{x}$ 都有非零密度。 |
| $p(\boldsymbol{y}\mid\boldsymbol{x})$ | **似然度** | 已知 $\boldsymbol{x}$ 时观测到 $\boldsymbol{y}$ 的可能性。注意它通常被视为关于 $\boldsymbol{y}$ 的分布，而非关于 $\boldsymbol{x}$ 的分布（MacKay, 2003）。 |
| $p(\boldsymbol{x}\mid\boldsymbol{y})$ | **后验** | 观察到 $\boldsymbol{y}$ 之后对 $\boldsymbol{x}$ 的更新信念，是贝叶斯推断的目标量。 |
| $p(\boldsymbol{y})$ | **证据（边缘似然）** | 对分子关于 $\boldsymbol{x}$ 积分得到，起归一化作用，与 $\boldsymbol{x}$ 无关。 |

其中证据的计算方式为：

$$
p(\boldsymbol{y}):=\int p(\boldsymbol{y}\mid\boldsymbol{x})\,p(\boldsymbol{x})\,\mathrm{d}\boldsymbol{x}=\mathbb{E}_X[p(\boldsymbol{y}\mid\boldsymbol{x})]\,, \tag{6.12}
$$

即似然度在先验 $p(\boldsymbol{x})$ 下的期望（期望算子 $\mathbb{E}$ 将在第 6.4.1 节定义）。



#### 贝叶斯公式的推导

贝叶斯公式是乘法规则 (6.10) 的直接结果。对同一联合分布按不同顺序应用乘法规则，再令两式右端相等即可：

$$\begin{aligned}
p(\boldsymbol{x},\boldsymbol{y}) &= p(\boldsymbol{x}\mid\boldsymbol{y})\,p(\boldsymbol{y})
                                   = p(\boldsymbol{y}\mid\boldsymbol{x})\,p(\boldsymbol{x}) \\[4pt]
\Longrightarrow\quad p(\boldsymbol{x}\mid\boldsymbol{y})
                     &= \frac{p(\boldsymbol{y}\mid\boldsymbol{x})\,p(\boldsymbol{x})}{p(\boldsymbol{y})}
\end{aligned}$$




#### 贝叶斯定理的应用案例

> **例 6.3（医学检测）**　某疾病的人群患病率为 1%，即 $P(D=1)=0.01$。现有一种检测手段：
> 
> - 患者检测为阳性的概率（灵敏度）：$P(T=1\mid D=1)=0.99$
> - 健康人检测为阴性的概率（特异度）：$P(T=0\mid D=0)=0.95$
> 
> **问题**：某人检测结果为阳性，他实际患病的概率 $P(D=1\mid T=1)$ 是多少？
> 
> **第一步：整理已知量**
> 
> | 符号 | 值 | 含义 |
> |:---:|:---:|:---|
> | $P(D=1)$ | $0.01$ | 先验——患病率 |
> | $P(D=0)$ | $0.99$ | 先验——健康率 |
> | $P(T=1\mid D=1)$ | $0.99$ | 似然度——患者阳性率 |
> | $P(T=1\mid D=0)$ | $0.05$ | 似然度——健康人误检阳性率（假阳性） |
> 
> **第二步：计算证据（边缘似然）**
> 
> 先由加法规则对 $D$ 求和，再由乘法规则展开每项联合概率：
> 
> $$\begin{aligned} P(T=1) &= P(T=1,\,D=1) + P(T=1,\,D=0) \\ &= P(T=1\mid D=1)\,P(D=1) + P(T=1\mid D=0)\,P(D=0) \\ &= 0.99\times0.01 + 0.05\times0.99 \\ &= 0.0594 \end{aligned}$$
> 
> **第三步：代入贝叶斯定理**
> $$P(D=1\mid T=1)=\frac{P(T=1\mid D=1)\,P(D=1)}{P(T=1)}=\frac{0.99\times0.01}{0.0594}\approx0.167$$
>

**结论**：
1. 即使检测手段很准（灵敏度 99%、特异度 95%），阳性结果对应的实际患病概率也只有约 **16.7%**。
2. 这是因为先验患病率极低（1%），导致大量健康人产生的假阳性"稀释"了真阳性。
3. 这个违反直觉的结果正是贝叶斯定理的经典应用——**先验对后验有决定性影响**。





## 6.4 汇总统计量与独立性

### 期望值与均值

均值和（协）方差通常用于描述概率分布的性质（期望值和离散程度）。

> **定义 6.8（期望值）**：对于单变量连续随机变量 $X \sim p(x)$ 的函数 $g: \mathbb{R} \to \mathbb{R}$，其 **期望值** 定义为：
> $$\operatorname{E}_X[g(x)]=\int_{\mathcal{X}}g(x)p(x)\mathrm{d}x \tag{6.13}$$
> 
> 相应地，对于离散随机变量 $X\sim p(x)$ 的函数 $g$，其期望值定义为：
> $$\operatorname{E}_{X}[g(x)]=\sum_{x\in\mathcal{X}}g(x)p(x) \tag{6.14}$$
> 
> 其中 $\mathcal{X}$ 是随机变量 $X$ 所有可能结果（目标空间）的集合。
>

**备注**：
1. $X \sim p(x)$ 表示 $X$ 服从由密度函数（或质量函数）$p(x)$ 所刻画的分布。
2. 对于多元随机变量 $\boldsymbol{X} = [X_1,\ldots,X_D]^\top$，期望值逐元素定义。
3. 均值（定义 6.9）是期望值的特例，即取 $g$ 为恒等函数。
4. 期望值是**线性算子**：对 $f(\boldsymbol{x})=ag(\boldsymbol{x})+bh(\boldsymbol{x})$（$a,b\in\mathbb{R}$），有 $\operatorname{E}_{X}[f(\boldsymbol{x})]=a\operatorname{E}_{X}[g(\boldsymbol{x})]+b\operatorname{E}_{X}[h(\boldsymbol{x})]$。展开验证：
$$
\begin{aligned}\operatorname{E}_{X}[f(\boldsymbol{x})]&=\int f(\boldsymbol{x})p(\boldsymbol{x})\mathrm{d}\boldsymbol{x}\\&=\int[ag(\boldsymbol{x})+bh(\boldsymbol{x})]p(\boldsymbol{x})\mathrm{d}\boldsymbol{x}\\&=a\int g(\boldsymbol{x})p(\boldsymbol{x})\mathrm{d}\boldsymbol{x}+b\int h(\boldsymbol{x})p(\boldsymbol{x})\mathrm{d}\boldsymbol{x}\\&=a\operatorname{E}_{X}[g(\boldsymbol{x})]+b\operatorname{E}_{X}[h(\boldsymbol{x})]\:.\end{aligned}
$$

对于两个随机变量，我们可能希望描述它们之间的对应关系。协方差直观地表示了随机变量之间依赖性的概念。


> **定义 6.9（均值）**：随机变量 $X$，其状态 $x\in\mathbb{R}^D$，的 **均值** 定义为：
> $$\operatorname{E}_X[\boldsymbol{x}]=\begin{bmatrix}\operatorname{E}_{X_1}[x_1]\\\vdots\\\operatorname{E}_{X_D}[x_D]\end{bmatrix}\in\mathbb{R}^D \tag{6.15}$$
> 其中
> $$\operatorname{E}_{X_d}[x_d]:=\left\{\begin{array}{ll}\int_{\mathcal{X}}x_dp(x_d)\mathrm dx_d&\text{如果}X\text{是连续随机变量}\\\sum_{x_i\in\mathcal{X}}x_ip(x_d=x_i)&\text{如果}X\text{是离散随机变量}\end{array}\right. \tag{6.16}$$
> 对于 $d=1,\ldots,D$，下标 $d$ 表示 $x$ 的相应维度。积分和求和是针对随机变量 $X$ 的目标空间状态 $\mathcal{X}$ 进行的。


---

> **例 6.4**，考虑一个二维随机变量 $\boldsymbol{X} = [X_1, X_2]^\top \in \mathbb{R}^2$，其概率密度函数为两个高斯分布的混合（高斯分布 $\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$ 将在第 6.6 节中正式定义）：
> 
> $$p(\boldsymbol{x})=0.4\mathcal{N}\left(\boldsymbol{x}\:\bigg|\begin{bmatrix}10\\2\end{bmatrix},\begin{bmatrix}1&0\\0&1\end{bmatrix}\right)+0.6\mathcal{N}\left(\boldsymbol{x}\:\bigg|\begin{bmatrix}0\\0\end{bmatrix},\begin{bmatrix}8.4&2.0\\2.0&1.7\end{bmatrix}\right)$$
> 
> 该分布及其在每个维度上的边缘分布如图 6.4 所示。
> 
> <div align="center">
>    <img src="https://datawhalechina.github.io/math-for-ai/attachments/6.4.png" alt="图6.4" width="450">
>    <center>图6.4 一个二维数据集的均值、众数和中位数及其边缘密度的说明。</center>
>    <br>
> </div>
> 
> 从图中可以观察到：
> - **双峰性**：联合分布是双峰的（有两个众数），但垂直方向的边缘分布却是单峰的（仅一个众数）。
> - **均值 $\neq$ 中位数**：水平方向的双峰边缘分布说明，均值和中位数可能彼此不同。
> - **多维中位数的困难**：直觉上我们可能想将二维中位数定义为各维度中位数的拼接，但这需要对二维点定义全序关系。然而高维点的全序并不唯一——例如 $\begin{bmatrix}3\\0\end{bmatrix}$ 与 $\begin{bmatrix}2\\3\end{bmatrix}$ 之间的大小关系取决于排序规则的选取，因此这种定义方式存在根本性困难。



### 协方差与方差

> **定义 6.10（协方差（单变量））**：两个单变量随机变量 $X$ 和 $Y$，其状态分别为 $x,y\in\mathbb{R}$，它们之间的协方差由各自偏离各自均值的乘积的期望值给出，即
> $$\mathrm{Cov}_{X,Y}[x,y]:=\mathrm{E}_{X,Y}\big[(x-\mathrm{E}_{X}[x])(y-\mathrm{E}_{Y}[y])\big] \tag{6.17}$$
> 
> **注**：当与期望值或多变量随机协方差相关的随机变量通过其参数明确时，下标通常会被省略（例如，E$_X[x]$ 通常简写为 E$[x])$。
> 

通过使用期望的线性性质，定义 6.10 中的表达式可以重写为乘积的期望值减去期望值的乘积，即

$$
\mathrm{Cov}[x,y]=\mathrm{E}[xy]-\mathrm{E}[x]\mathrm{E}[y] \tag{6.18}
$$

1. **方差**：变量与其自身的协方差 Cov$[x,x]$，记作 $\mathcal{V}_X[x]$。
2. **标准差**：方差的平方根，通常记作 $\sigma(x)$。

---

协方差的概念可以推广到多变量随机变量。

> **定义 6.11（协方差[多变量]）**：如果我们考虑两个多变量随机变量 $X$ 和 $Y$，其状态分别为 $x\in\mathbb{R}^D$ 和 $y\in\mathbb{R}^E$，则 $X$ 和 $Y$ 之间的协方差定义为
> $$\mathrm{Cov}[\boldsymbol{x},\boldsymbol{y}]=\mathrm{E}[\boldsymbol{x}\boldsymbol{y}^{\top}]-\mathrm{E}[\boldsymbol{x}]\mathrm{E}[\boldsymbol{y}]^{\top}=\mathrm{Cov}[\boldsymbol{y},\boldsymbol{x}]^{\top}\in\mathbb{R}^{D\times E} \tag{6.19}$$
> 

定义 6.11 可以应用于两个参数中的相同多变量随机变量，这导致了一个有用的概念，它直观地捕获了随机变量的“散布”。

---

对于多变量随机变量，方差描述了随机变量各个维度之间的关系。

> **定义 6.12（方差[多变量]）**：随机变量 $X$ 的方差，其状态为 $x\in\mathbb{R}^D$，均值向量为 $\mu\in\mathbb{R}^D$，定义为
> $$\begin{aligned}\mathbb{V}_{X}[\boldsymbol{x}]&=\mathrm{Cov}_{X}[\boldsymbol{x},\boldsymbol{x}]\\&=\mathbb{E}_{X}[(\boldsymbol{x}-\boldsymbol{\mu})(\boldsymbol{x}-\boldsymbol{\mu})^{\top}]=\mathbb{E}_{X}[\boldsymbol{x}\boldsymbol{x}^{\top}]-\mathbb{E}_{X}[\boldsymbol{x}]\mathbb{E}_{X}[\boldsymbol{x}]^{\top}\\&=\begin{bmatrix}\mathrm{Cov}[x_1,x_1]&\mathrm{Cov}[x_1,x_2]&\ldots&\mathrm{Cov}[x_1,x_D]\\\mathrm{Cov}[x_2,x_1]&\mathrm{Cov}[x_2,x_2]&\ldots&\mathrm{Cov}[x_2,x_D]\\\vdots&\vdots&\ddots&\vdots\\\mathrm{Cov}[x_D,x_1]&\ldots&\ldots&\mathrm{Cov}[x_D,x_D]\end{bmatrix} \end{aligned} \tag{6.20}$$
>
> $(6.20)$ 中的 $D\times D$ 矩阵被称为多元随机变量 $X$ 的**协方差矩阵**。
> 

协方差矩阵是对称的且是半正定的，它向我们揭示了数据的分布情况。在其对角线上，协方差矩阵包含了边缘分布的方差

$$
p(x_i)=\int p(x_1,\ldots,x_D)\mathrm{d}x_{\setminus i} \tag{6.21}
$$

其中“$\setminus i$”表示“除了变量$i$之外的所有变量”。非对角线上的元素是$i,j=1,\ldots,D,i\neq j$时的交叉协方差项$\text{Cov}[x_i,x_j]$。

**备注**。在本书中，我们通常假设协方差矩阵是正定的，以便更好地理解。因此，我们不讨论导致半正定（低秩）协方差矩阵的特殊情况。



### 相关系数

当我们想要比较不同随机变量对之间的协方差时，发现每个随机变量的方差都会影响协方差的值。协方差的归一化版本被称为相关系数。

> **定义 6.13（相关系数）**：两个随机变量 $X,Y$ 之间的相关系数由
> $$\text{corr}[x,y]=\frac{\text{Cov}[x,y]}{\sqrt{\text{V}[x]\text{V}[y]}}\in[-1,1]\:. \tag{6.22}$$
> 

相关系数矩阵是标准化随机变量$x/\sigma(x)$的协方差矩阵。换句话说，在相关系数矩阵中，每个随机变量都被其标准差（方差的平方根）除。

协方差（和相关系数）表明了两个随机变量之间的关系；见图6.5。正相关$\text{corr}[x,y]$意味着当$x$增长时，$y$也预期会增长。负相关则意味着当$x$增加时，$y$会减小。

![](https://datawhalechina.github.io/math-for-ai/attachments/6.5.png)

<center>图6.5二维数据集沿每个轴（彩色线）具有相同的均值和方差，但具有不同的协方差。</center>



### 经验均值和经验协方差

> **定义 6.14（经验均值和协方差）**：经验均值向量是每个变量观测值的算术平均值，定义为
> $$\bar{\boldsymbol{x}}:=\frac{1}{N}\sum_{n=1}^{N}\boldsymbol{x}_{n}\:, \tag{6.23}$$
> 其中 $x_n\in\mathbb{R}^D$。经验协方差矩阵是一个 $D\times D$ 矩阵
> $$\boldsymbol{\Sigma}:=\frac{1}{N}\sum_{n=1}^{N}(\boldsymbol{x}_{n}-\bar{\boldsymbol{x}})(\boldsymbol{x}_{n}-\bar{\boldsymbol{x}})^{\top}. \tag{6.24}$$

为了计算特定数据集的统计量，我们将使用实现（观测值）$x_1,\ldots,x_N$，并使用(6.23)和(6.24)。经验协方差矩阵是对称的、半正定的（见第3.2.3节）。



### 理解方差的三种视角

**视角一：与均值的偏差**。方差衡量随机变量 $X$ 偏离其期望值 $\mu$ 的程度，即平方偏差的期望值：

$$\mathrm{V}_X[x] := \mathrm{E}_X[(x-\mu)^2] \tag{6.25}$$

**视角二：原始分数公式**。将 (6.25) 展开可得等价形式，便于实际计算：

$$\mathrm{V}_X[x] = \mathrm{E}_X[x^2] - \left(\mathrm{E}_X[x]\right)^2 \tag{6.26}$$

**视角三：成对差异**。给定样本 $x_1,\ldots,x_N$，方差也可以理解为所有观测对之间平方差的平均。展开平方可证：

$$\dfrac{1}{N^2}\sum_{i,j=1}^N(x_i-x_j)^2 = 2\left[\dfrac{1}{N}\sum_{i=1}^Nx_i^2 - \left(\dfrac{1}{N}\sum_{i=1}^Nx_i\right)^2\right] \tag{6.27}$$




### 统计独立性

> **定义 6.15（独立性）**：两个随机变量 $X,Y$ 是统计独立的当且仅当
> $$p(x,y)=p(x)p(y) \tag{6.28}$$
> 

直观上，如果两个随机变量$X$和$Y$是独立的，那么知道 $y$ 的值并不会给 $x$ 提供任何额外的信息（反之亦然）。

如果$X,Y$是（统计）独立的，那么

- $p(y\mid x) = p(y)$
- $p(x\mid y) = p(x)$
- $\mathrm{V}_{X,Y}[x+y]=\mathrm{V}_X[x]+\mathrm{V}_Y[y]$
- $\mathrm{Cov}_{X, Y}[x, y] = 0$

最后一点可能不总是成立的逆命题，即两个随机变量可以有协方差为零但并非统计独立。为了理解这一点，需要回顾协方差只衡量线性依赖关系。因此，非线性依赖的随机变量可能具有零协方差。

---

机器学习中另一个重要的概念是条件独立性。

> **定义 6.16（条件独立性）**：两个随机变量 $X$ 和 $Y$ 在给定 $Z$ 的条件下是条件独立的当且仅当
> $$p(\boldsymbol{x},\boldsymbol{y}\mid\boldsymbol{z})=p(\boldsymbol{x}\mid\boldsymbol{z})p(\boldsymbol{y}\mid\boldsymbol{z})\quad\mathrm{for~all}\quad\boldsymbol{z}\in\mathcal{Z} \tag{6.29}$$
> 
> 其中 $\mathcal{Z}$ 是随机变量 $Z$ 的状态集。我们用 $X\perp Y\mid Z$ 来表示给定 $Z$ 时，$X$ 与 $Y$ 是条件独立的。
> 




### 随机变量的内积

如果两个随机变量 $X,Y$ 不相关，则它们的方差具有可加性：

$$\mathrm{V}[X+Y]=\mathrm{V}[X]+\mathrm{V}[Y]\:. \tag{6.30}$$

方差以平方单位衡量，这与勾股定理 $c^2=a^2+b^2$ 形式一致。这并非巧合——随机变量可以视为线性空间中的向量，协方差恰好为其提供了内积结构（Eaton, 2007）。

对于均值为零的随机变量 $X$ 和 $Y$，定义内积为：

$$\langle X,Y\rangle := \mathrm{Cov}[X,Y] \tag{6.31}$$

协方差满足对称性、正定性和双线性，因此构成合法的内积。在此内积下，随机变量的"长度"为：

$$\|X\| = \sqrt{\mathrm{Cov}[X,X]} = \sqrt{\mathrm{V}[X]} = \sigma[X]\:, \tag{6.32}$$

即标准差。随机变量"越长"，不确定性越大；长度为 0 的随机变量是确定性的。

两个随机变量 $X,Y$ 之间的夹角 $\theta$ 满足：

$$\cos\theta = \frac{\langle X,Y\rangle}{\|X\|\,\|Y\|} = \frac{\mathrm{Cov}[X,Y]}{\sqrt{\mathrm{V}[X]\mathrm{V}[Y]}}\:, \tag{6.33}$$

这正是定义 6.13 中的相关系数。因此，**相关系数的几何意义就是两个随机变量之间夹角的余弦值**。根据定义 3.7，$X \perp Y \Longleftrightarrow \langle X,Y\rangle = 0$，即 $X$ 和 $Y$ 正交当且仅当 $\text{Cov}[X,Y]=0$（不相关）。图 6.6 展示了这一关系。

<div align="center">
   <img src="https://datawhalechina.github.io/math-for-ai/attachments/6.6.png" alt="图6.6" width="400">
   <center>图6.6 随机变量的几何形状。如果随机变量 X 和 Y 不相关，则它们是相应线性空间中的正交向量，勾股定理成立。</center>
   <br>
</div>

<div align="center">
   <img src="https://datawhalechina.github.io/math-for-ai/attachments/6.7.png" alt="图6.7" width="400">
   <center>图6.7两个随机变量x1和x2的高斯分布。</center>
   <br>
</div>




## 6.5 常见概率分布


### 6.5.1 伯努利族：试验、计数与共轭先验

#### 伯努利分布与二项分布

**伯努利分布** Ber$(\mu)$：单个二元随机变量 $x\in\{0,1\}$ 的分布，是最基本的离散分布

$$
p(x\mid\mu)=\mu^{x}(1-\mu)^{1-x},\quad \mathbb{E}[x]=\mu,\quad \mathbb{V}[x]=\mu(1-\mu) \tag{6.34}
$$

**二项分布** Bin$(N,\mu)$：$N$ 次独立伯努利试验中 $x=1$ 出现 $m$ 次的概率

$$
p(m\mid N,\mu)=\binom{N}{m}\mu^m(1-\mu)^{N-m},\quad \mathbb{E}[m]=N\mu,\quad \mathbb{V}[m]=N\mu(1-\mu) \tag{6.35}
$$

> **例 6.5（质量检测）**　某生产线的产品次品率为 $\mu=0.05$。从中随机抽取 $N=20$ 件产品，恰好有 $m=2$ 件次品的概率为：
>
> $$p(m=2\mid N=20,\mu=0.05)=\binom{20}{2}(0.05)^2(0.95)^{18}=190\times 0.0025\times 0.397\approx 0.189$$
>
> 期望次品数 $\mathbb{E}[m]=20\times 0.05=1$，方差 $\mathbb{V}[m]=20\times 0.05\times 0.95=0.95$。


#### 贝塔分布（伯努利/二项的共轭先验）

**贝塔分布** Beta$(\alpha,\beta)$：定义在 $\mu\in[0,1]$ 上的连续分布，常用于表示概率参数的先验

$$
p(\mu\mid\alpha,\beta)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}\mu^{\alpha-1}(1-\mu)^{\beta-1} \tag{6.36}
$$

$$
\mathbb{E}[\mu]=\frac{\alpha}{\alpha+\beta},\quad \mathbb{V}[\mu]=\frac{\alpha\beta}{(\alpha+\beta)^{2}(\alpha+\beta+1)} \tag{6.37}
$$

|                       符号                        | 含义                                                         |
| :-----------------------------------------------: | :----------------------------------------------------------- |
|                      $\mu$                        | 随机变量，取值范围 $[0,1]$，通常表示某个概率参数（如硬币正面朝上的概率） |
|                    $\alpha$                       | 形状参数（$\alpha>0$），控制分布向 $\mu=1$ 方向的集中程度     |
|                     $\beta$                       | 形状参数（$\beta>0$），控制分布向 $\mu=0$ 方向的集中程度      |
|                   $\Gamma(\cdot)$                  | 伽马函数，$\Gamma(t)=\int_0^\infty x^{t-1}e^{-x}\mathrm{d}x$，阶乘在实数域的推广 |
| $\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}$ | 归一化常数（即 $\frac{1}{B(\alpha,\beta)}$），确保密度在 $[0,1]$ 上积分为 1 |
|              $\mu^{\alpha-1}$                     | $\mu$ 的幂项，$\alpha$ 越大使密度在接近 1 处越高             |
|           $(1-\mu)^{\beta-1}$                     | $(1-\mu)$ 的幂项，$\beta$ 越大使密度在接近 0 处越高          |
|       $p(\mu\mid\alpha,\beta)$                    | 在给定参数 $\alpha,\beta$ 下，$\mu$ 处的概率密度值           |


其中 $\Gamma(t)=\int_{0}^{\infty}x^{t-1}e^{-x}\mathrm{d}x$。直观上，$\alpha$ 将质量推向 1，$\beta$ 推向 0。特别地：$\alpha=\beta=1$ 时退化为均匀分布。

> **例 6.6（硬币公平性的先验信念）**　贝塔分布最常用于表达"我们对某个概率参数 $\mu$ 知道多少"。假设你要评估一枚硬币是否公平：
>
> | 参数 | 分布形状 | 直觉含义 |
> |:---:|:---:|:---|
> | $\alpha=1,\;\beta=1$ | 平坦（均匀） | 完全无知——$\mu$ 取 $[0,1]$ 中任何值都一样可能 |
> | $\alpha=5,\;\beta=5$ | 对称钟形，集中在 0.5 | 较相信硬币公平，但不太确定 |
> | $\alpha=50,\;\beta=50$ | 非常窄的尖峰在 0.5 | 强烈相信硬币公平 |
> | $\alpha=2,\;\beta=8$ | 偏左，集中在 0.2 | 相信正面概率较低（约 0.2） |
>
> 在贝叶斯推断中，观察到数据后，贝塔先验会更新为新的贝塔后验（见 6.7.1 节共轭性）。例如先验 Beta$(1,1)$ 观察到 7 次正面、3 次反面后，后验为 Beta$(8,4)$，均值从 0.5 移动到 $8/12\approx 0.67$。


#### 分类分布与多项分布

**分类分布** Cat$(\boldsymbol{\pi})$：伯努利分布到 $K$ 类的推广，描述单次试验中出现类别 $k$ 的概率

$$
p(x=k\mid\boldsymbol{\pi})=\pi_k,\quad k=1,\ldots,K,\quad \sum_{k=1}^{K}\pi_k=1 \tag{6.38}
$$

|             符号             | 含义                                                         |
| :--------------------------: | :----------------------------------------------------------- |
|             $x$              | 随机变量，表示一次试验的结果，取值为类别编号 $k\in\{1,2,\ldots,K\}$ |
|             $K$              | 类别总数（例如骰子有 6 面则 $K=6$）                          |
|      $\boldsymbol{\pi}$      | 参数向量 $(\pi_1,\pi_2,\ldots,\pi_K)$，包含每个类别的概率    |
|           $\pi_k$            | 第 $k$ 个类别被选中的概率，$\pi_k\in[0,1]$                   |
|   $\sum_{k=1}^{K}\pi_k=1$    | 约束条件——所有类别的概率之和为 1（互斥且穷尽）               |
| $p(x=k\mid\boldsymbol{\pi})$ | 在给定参数 $\boldsymbol{\pi}$ 下，结果恰好为类别 $k$ 的概率  |


等价地，用 one-hot 编码 $\boldsymbol{x}\in\{0,1\}^K$（$\sum_k x_k=1$）表示：

$$
p(\boldsymbol{x}\mid\boldsymbol{\pi})=\prod_{k=1}^{K}\pi_k^{x_k} \tag{6.39}
$$

$$
\mathbb{E}[x_k]=\pi_k,\quad \mathbb{V}[x_k]=\pi_k(1-\pi_k) \tag{6.40}
$$

|       符号        | 含义                                                         |
| :---------------: | :----------------------------------------------------------- |
| $\boldsymbol{x}$  | one-hot 向量 $\in\{0,1\}^K$，长度为 $K$，**有且仅有一个分量为 1**，其余为 0 |
|       $x_k$       | $\boldsymbol{x}$ 的第 $k$ 个分量，$x_k\in\{0,1\}$            |
|  $\sum_k x_k=1$   | 约束条件——恰好只有一个类别被选中                             |
|   $\pi_k^{x_k}$   | 当 $x_k=1$ 时为 $\pi_k$；当 $x_k=0$ 时为 $\pi_k^0=1$（对乘积无贡献） |
| $\prod_{k=1}^{K}$ | 将所有 $K$ 项相乘                                            |
| $\mathbb{E}[x_k]$ | $x_k$ 的期望值。因为 $x_k$ 只能取 0 或 1，且 $P(x_k=1)=\pi_k$，所以期望就是 $\pi_k$ |
| $\mathbb{V}[x_k]$ | $x_k$ 的方差                                                 |
| $\pi_k(1-\pi_k)$  | 这就是伯努利分布的方差公式——因为**单看第 $k$ 个分量**，$x_k$ 就是一个参数为 $\pi_k$ 的伯努利随机变量（选中为 1，未选中为 0） |


分类分布在分类任务、语言模型的 token 预测等场景中无处不在。


---

**多项分布** Mult$(N,\boldsymbol{\pi})$：二项分布到 $K$ 类的推广，$N$ 次独立分类试验中各类别出现次数 $\boldsymbol{m}=(m_1,\ldots,m_K)$ 的分布

$$
p(\boldsymbol{m}\mid N,\boldsymbol{\pi})=\frac{N!}{m_1!\cdots m_K!}\prod_{k=1}^{K}\pi_k^{m_k},\quad \sum_{k=1}^{K}m_k=N \tag{6.41}
$$

$$
\mathbb{E}[m_k]=N\pi_k,\quad \mathbb{V}[m_k]=N\pi_k(1-\pi_k),\quad \mathrm{Cov}[m_i,m_j]=-N\pi_i\pi_j\;(i\neq j) \tag{6.42}
$$

|                          符号                          | 含义                                                         |
| :----------------------------------------------------: | :----------------------------------------------------------- |
|                          $N$                           | 独立试验的总次数                                             |
|                          $K$                           | 类别总数（例如骰子有 6 面则 $K=6$）                          |
|                   $\boldsymbol{m}$                      | 计数向量 $(m_1,m_2,\ldots,m_K)$，记录各类别出现的次数        |
|                         $m_k$                          | 第 $k$ 个类别在 $N$ 次试验中出现的次数，$m_k\in\{0,1,\ldots,N\}$ |
|              $\sum_{k=1}^{K}m_k=N$                     | 约束条件——所有类别的计数之和等于总试验次数                   |
|                   $\boldsymbol{\pi}$                    | 参数向量 $(\pi_1,\pi_2,\ldots,\pi_K)$，包含每个类别的概率    |
|                        $\pi_k$                         | 单次试验中第 $k$ 个类别被选中的概率，$\pi_k\in[0,1]$，$\sum_k\pi_k=1$ |
| $\frac{N!}{m_1!\cdots m_K!}$                           | 多项式系数，表示将 $N$ 次试验分配到各类别的组合数            |
| $\prod_{k=1}^{K}\pi_k^{m_k}$                          | 各类别按其出现次数贡献的概率乘积                             |
| $p(\boldsymbol{m}\mid N,\boldsymbol{\pi})$             | 在给定 $N$ 和 $\boldsymbol{\pi}$ 下，观测到计数向量 $\boldsymbol{m}$ 的概率 |


#### 狄利克雷分布（分类/多项的共轭先验）

**狄利克雷分布** Dir$(\boldsymbol{\alpha})$：贝塔分布到 $K$ 维单纯形上的推广，是分类/多项似然的共轭先验

$$
p(\boldsymbol{\mu}\mid\boldsymbol{\alpha})=\frac{\Gamma(\sum_{k=1}^K\alpha_k)}{\prod_{k=1}^K\Gamma(\alpha_k)}\prod_{k=1}^{K}\mu_k^{\alpha_k-1},\quad \mu_k\geqslant 0,\;\sum_{k=1}^K\mu_k=1 \tag{6.43}
$$

$$
\mathbb{E}[\mu_k]=\frac{\alpha_k}{\sum_j\alpha_j},\quad \mathbb{V}[\mu_k]=\frac{\alpha_k(\alpha_0-\alpha_k)}{\alpha_0^2(\alpha_0+1)},\quad \alpha_0=\sum_j\alpha_j \tag{6.44}
$$


|                              符号                              | 含义                                                         |
| :------------------------------------------------------------: | :----------------------------------------------------------- |
|                      $\boldsymbol{\mu}$                        | 随机向量 $(\mu_1,\mu_2,\ldots,\mu_K)$，位于 $K$ 维单纯形上，通常表示一组概率参数 |
|                           $\mu_k$                              | 第 $k$ 个分量，$\mu_k\in[0,1]$，满足 $\sum_k\mu_k=1$        |
|                            $K$                                 | 维度/类别总数                                                |
|                     $\boldsymbol{\alpha}$                       | 集中度参数向量 $(\alpha_1,\alpha_2,\ldots,\alpha_K)$，每个 $\alpha_k>0$ |
|                          $\alpha_k$                            | 第 $k$ 个集中度参数，$\alpha_k$ 越大则 $\mu_k$ 的期望越高、分布越集中 |
| $\frac{\Gamma(\sum_k\alpha_k)}{\prod_k\Gamma(\alpha_k)}$      | 归一化常数（多元贝塔函数的倒数），确保密度在单纯形上积分为 1 |
|                 $\prod_{k=1}^K\mu_k^{\alpha_k-1}$              | 核函数，各分量按对应集中度参数贡献的幂项乘积                 |
|              $\sum_{k=1}^K\mu_k=1$                             | 约束条件——所有分量之和为 1，向量位于单纯形上                 |
|         $p(\boldsymbol{\mu}\mid\boldsymbol{\alpha})$           | 在给定参数 $\boldsymbol{\alpha}$ 下，$\boldsymbol{\mu}$ 处的概率密度值 |

当 $K=2$ 时退化为贝塔分布。狄利克雷分布在主题模型（LDA）、贝叶斯多项分类等场景中广泛使用。

> **例 6.7（文档主题建模）**　在 LDA 主题模型中，假设有 $K=3$ 个主题（体育、科技、政治）。每篇文档的主题比例 $\boldsymbol{\mu}=(\mu_1,\mu_2,\mu_3)$ 从 Dir$(\boldsymbol{\alpha})$ 中采样。$\boldsymbol{\alpha}$ 的取值决定了文档主题分配的特点：
>
> | 参数 | 效果 | 示例采样 |
> |:---:|:---|:---|
> | $\boldsymbol{\alpha}=(0.1,0.1,0.1)$ | 稀疏——文档倾向于只涉及一个主题 | $(0.95,0.03,0.02)$ |
> | $\boldsymbol{\alpha}=(1,1,1)$ | 均匀——单纯形上任何主题比例等可能 | $(0.31,0.47,0.22)$ |
> | $\boldsymbol{\alpha}=(10,10,10)$ | 集中——文档倾向于均匀涉及所有主题 | $(0.35,0.32,0.33)$ |
> | $\boldsymbol{\alpha}=(10,2,2)$ | 偏向第一个主题（体育） | $(0.72,0.15,0.13)$ |
>
> 直观上，$\alpha_k<1$ 产生稀疏（极端）的概率向量，$\alpha_k>1$ 产生平滑（均匀）的概率向量。这与贝塔分布中 $\alpha,\beta$ 对形状的影响完全类似。



### 6.5.2 泊松族：事件计数与等待时间

#### 泊松分布

**泊松分布** Pois$(\lambda)$：描述固定时间/空间内事件发生次数的离散分布

$$
p(x=k\mid\lambda)=\frac{\lambda^k e^{-\lambda}}{k!},\quad k=0,1,2,\ldots \tag{6.45}
$$

$$
\mathbb{E}[x]=\lambda,\quad \mathbb{V}[x]=\lambda \tag{6.46}
$$

泊松分布的均值和方差相等，这是其重要特征。当二项分布的 $N\to\infty$、$\mu\to 0$ 且 $N\mu=\lambda$ 有限时，二项分布趋近于泊松分布。泊松分布常用于建模用户访问次数、文档中单词出现频率等计数数据。

> **例 6.8（呼叫中心）**　某客服中心平均每小时接到 $\lambda=4$ 个电话。问：某小时内恰好接到 6 个电话的概率是多少？一个电话都没有的概率呢？
>
> $$p(x=6\mid\lambda=4)=\frac{4^6 e^{-4}}{6!}=\frac{4096\times 0.0183}{720}\approx 0.104$$
>
> $$p(x=0\mid\lambda=4)=\frac{4^0 e^{-4}}{0!}=e^{-4}\approx 0.018$$
>
> 直观上，$\lambda=4$ 意味着平均 4 个/小时，出现 6 个的概率约 10.4%，一个都没有的概率仅 1.8%。


#### 指数分布与伽马分布

**指数分布** Exp$(\lambda)$：描述事件之间等待时间的连续分布，是几何分布的连续对应

$$
p(x\mid\lambda)=\lambda e^{-\lambda x},\quad x\geqslant 0 \tag{6.47}
$$

$$
\mathbb{E}[x]=\frac{1}{\lambda},\quad \mathbb{V}[x]=\frac{1}{\lambda^2} \tag{6.48}
$$

指数分布具有**无记忆性**：$P(X>s+t\mid X>s)=P(X>t)$，即已等待的时间不影响剩余等待时间的分布。

> **例 6.9（设备故障）**　某服务器平均 $1/\lambda=200$ 小时发生一次故障（$\lambda=0.005$）。问：超过 300 小时不出故障的概率？
>
> $$P(X>300)=e^{-0.005\times 300}=e^{-1.5}\approx 0.223$$
>
> 无记忆性意味着：如果服务器已经运行了 100 小时没出故障，再运行 300 小时不出故障的概率仍然是 $e^{-1.5}\approx 0.223$——之前的 100 小时"不算数"。


---

**伽马分布** Gamma$(\alpha,\beta)$：指数分布的推广，$\alpha$ 个独立同分布指数随机变量之和服从伽马分布

$$
p(x\mid\alpha,\beta)=\frac{\beta^{\alpha}}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x},\quad x>0 \tag{6.49}
$$

$$
\mathbb{E}[x]=\frac{\alpha}{\beta},\quad \mathbb{V}[x]=\frac{\alpha}{\beta^2} \tag{6.50}
$$


|                   符号                   | 含义                                                         |
| :--------------------------------------: | :----------------------------------------------------------- |
|                   $x$                    | 随机变量，取值范围 $(0,+\infty)$                             |
|                $\alpha$                  | 形状参数（$\alpha>0$），控制分布的形态；$\alpha$ 越大分布越接近对称钟形 |
|                 $\beta$                  | 速率参数（$\beta>0$），控制分布的尺度；$\beta$ 越大分布越集中于 0 附近 |
|            $\Gamma(\alpha)$              | 伽马函数，起归一化作用                                       |
| $\frac{\beta^{\alpha}}{\Gamma(\alpha)}$  | 归一化常数，确保密度在 $(0,+\infty)$ 上积分为 1              |
|            $x^{\alpha-1}$                | 幂项，决定密度在 $x\to 0$ 附近的行为                         |
|             $e^{-\beta x}$               | 指数衰减项，控制右尾的衰减速度                               |
|        $p(x\mid\alpha,\beta)$            | 在给定参数 $\alpha,\beta$ 下，$x$ 处的概率密度值             |


其中 $\alpha>0$ 为形状参数，$\beta>0$ 为速率参数。特别地：$\alpha=1$ 时退化为指数分布 Exp$(\beta)$；$\alpha=\nu/2,\;\beta=1/2$ 时为自由度 $\nu$ 的**卡方分布** $\chi^2(\nu)$。伽马分布是泊松似然的共轭先验，也常用于对正值参数（如精度、方差的倒数）建模。

> **例 6.10（等待第 $\alpha$ 个事件）**　延续例 6.8 的呼叫中心场景（$\beta=4$ 个/小时）。问：等到第 $\alpha=3$ 个电话打进来需要多长时间？
>
> 这正是 Gamma$(3,4)$ 分布。期望等待时间 $\mathbb{E}[x]=3/4=0.75$ 小时（45 分钟），方差 $\mathbb{V}[x]=3/16\approx 0.19$。
>
> 伽马分布的形状随 $\alpha$ 变化：$\alpha=1$ 时从原点单调递减（即指数分布）；$\alpha>1$ 时呈钟形，峰值在 $(\alpha-1)/\beta$ 处；$\alpha$ 越大分布越对称，趋近于正态分布（中心极限定理的体现）。



### 6.5.3 其他重要分布

#### 均匀分布

**均匀分布** $\mathcal{U}(a,b)$：在区间 $[a,b]$ 上等概率取值的连续分布

$$
p(x\mid a,b)=\frac{1}{b-a},\quad a\leqslant x\leqslant b \tag{6.51}
$$

$$
\mathbb{E}[x]=\frac{a+b}{2},\quad \mathbb{V}[x]=\frac{(b-a)^2}{12} \tag{6.52}
$$

均匀分布是"无信息"分布的典型代表——在有限区间上不偏好任何取值。它在蒙特卡洛采样和概率积分变换中扮演基础角色。


#### 学生 $t$ 分布

**学生 $t$ 分布** $t_\nu$：比高斯分布具有更重尾部的对称分布

$$
p(x\mid\nu)=\frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\;\Gamma\!\left(\frac{\nu}{2}\right)}\left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}} \tag{6.53}
$$

$$
\mathbb{E}[x]=0\;(\nu>1),\quad \mathbb{V}[x]=\frac{\nu}{\nu-2}\;(\nu>2) \tag{6.54}
$$


|                            符号                            | 含义                                                         |
| :--------------------------------------------------------: | :----------------------------------------------------------- |
|                           $x$                              | 随机变量，取值范围 $(-\infty,+\infty)$                       |
|                          $\nu$                             | 自由度参数（$\nu>0$），控制尾部厚度；$\nu$ 越小尾部越重，$\nu\to\infty$ 时趋向标准正态分布 |
|        $\Gamma\!\left(\frac{\nu+1}{2}\right)$              | 伽马函数，分子归一化因子的一部分                             |
|            $\sqrt{\nu\pi}\;\Gamma\!\left(\frac{\nu}{2}\right)$ | 分母归一化因子，与分子共同确保密度积分为 1                   |
| $\frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\;\Gamma\!\left(\frac{\nu}{2}\right)}$ | 完整归一化常数                                               |
|                   $\frac{x^2}{\nu}$                        | 标准化的平方偏差，衡量 $x$ 相对于自由度的偏离程度            |
|    $\left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$    | 核函数，产生幂律衰减的重尾（区别于高斯的指数衰减）           |
|                   $p(x\mid\nu)$                            | 在给定自由度 $\nu$ 下，$x$ 处的概率密度值                    |

其中 $\nu>0$ 为自由度参数。当 $\nu\to\infty$ 时趋近于标准正态分布；当 $\nu=1$ 时为**柯西分布**（均值和方差均不存在）。$t$ 分布因重尾特性而对离群值更加鲁棒，在小样本推断和鲁棒回归中非常重要。

> **例 6.11（离群值的鲁棒性）**　假设我们用高斯分布和 $t$ 分布（$\nu=3$）分别拟合一组含离群值的数据 $\{1.2,\;0.8,\;1.1,\;0.9,\;1.0,\;8.5\}$：
>
> - **高斯分布**：均值 $\bar{x}=2.25$，被离群值 8.5 严重拉偏（真实中心应在 1.0 附近）
> - **$t_3$ 分布**：由于重尾赋予离群值更高的概率密度，极端观测对参数估计的影响被"降权"，估计出的位置参数更接近 1.0
>
> 核心差异在于尾部衰减速度：高斯密度按 $e^{-x^2/2}$（指数级）衰减，而 $t_\nu$ 密度按 $x^{-(\nu+1)}$（多项式级）衰减。因此 $t$ 分布认为极端值"不那么意外"，不会为了解释它们而大幅调整参数。




## 6.6 高斯分布

高斯分布（正态分布）是连续型分布中最重要的分布，广泛应用于线性回归（第9章）、高斯过程、变分推理、卡尔曼滤波等领域。其核心优势在于**边际分布和条件分布具有闭式表达式**，且完全由均值和协方差确定。

**单变量高斯分布**的密度函数：

$$
p(x\mid\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \tag{6.55}
$$

**多元高斯分布**由均值向量 $\boldsymbol{\mu}$ 和协方差矩阵 $\boldsymbol{\Sigma}$ 完全描述：

$$
p(\boldsymbol{x}\mid\boldsymbol{\mu},\boldsymbol{\Sigma})=(2\pi)^{-\frac{D}{2}}|\boldsymbol{\Sigma}|^{-\frac{1}{2}}\exp\left(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu})^{\top}\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})\right) \tag{6.56}
$$

记作 $X\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$。当 $\mu=0$，$\Sigma=I$ 时称为**标准正态分布**。


### 6.6.1 边际分布和条件分布仍然是高斯分布

设联合高斯分布为

$$
p(\boldsymbol{x},\boldsymbol{y})=\mathcal{N}\left(\begin{bmatrix}\boldsymbol{\mu}_x\\\boldsymbol{\mu}_y\end{bmatrix},\:\begin{bmatrix}\boldsymbol{\Sigma}_{xx}&\boldsymbol{\Sigma}_{xy}\\\boldsymbol{\Sigma}_{yx}&\boldsymbol{\Sigma}_{yy}\end{bmatrix}\right) \tag{6.57}
$$

**条件分布** $p(\boldsymbol{x}\mid\boldsymbol{y})$ 也是高斯分布：

$$
p(\boldsymbol{x}\mid\boldsymbol{y})=\mathcal{N}(\boldsymbol{\mu}_{x\mid y},\:\boldsymbol{\Sigma}_{x\mid y}) \tag{6.58}
$$

$$
\boldsymbol{\mu}_{x\mid y}=\boldsymbol{\mu}_{x}+\boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1}(\boldsymbol{y}-\boldsymbol{\mu}_{y}) \tag{6.59}
$$

$$
\boldsymbol{\Sigma}_{x\mid y}=\boldsymbol{\Sigma}_{xx}-\boldsymbol{\Sigma}_{xy}\boldsymbol{\Sigma}_{yy}^{-1}\boldsymbol{\Sigma}_{yx} \tag{6.60}
$$

**备注**：条件高斯在卡尔曼滤波、高斯过程、潜在线性高斯模型（如PPCA）中广泛出现。

**边际分布** $p(\boldsymbol{x})$ 同样是高斯分布：

$$
p(\boldsymbol{x})=\int p(\boldsymbol{x},\boldsymbol{y})\mathrm{d}\boldsymbol{y}=\mathcal{N}\big(\boldsymbol{x}\mid\boldsymbol{\mu}_{x},\:\boldsymbol{\Sigma}_{xx}\big) \tag{6.61}
$$

直观上，边际化就是忽略（积分掉）不关心的维度，保留的分布参数就是对应的均值子向量和协方差子矩阵。


### 6.6.2 高斯密度的乘积

两个高斯密度的乘积 $\mathcal{N}(\boldsymbol{x}\mid\boldsymbol{a},\boldsymbol{A})\mathcal{N}(\boldsymbol{x}\mid\boldsymbol{b},\boldsymbol{B})$ 是一个缩放的高斯 $c\,\mathcal{N}(\boldsymbol{x}\mid\boldsymbol{c},\boldsymbol{C})$，其中：

$$
\boldsymbol{C}=(\boldsymbol{A}^{-1}+\boldsymbol{B}^{-1})^{-1} \tag{6.62}
$$

$$
\boldsymbol{c}=\boldsymbol{C}(\boldsymbol{A}^{-1}\boldsymbol{a}+\boldsymbol{B}^{-1}\boldsymbol{b}) \tag{6.63}
$$

缩放常数 $c=\mathcal{N}(\boldsymbol{a}\mid\boldsymbol{b},\boldsymbol{A}+\boldsymbol{B})$。这一结果在贝叶斯线性回归中计算后验时至关重要（似然 × 先验 = 后验）。


### 6.6.3 和与线性变换

若 $X,Y$ 是独立高斯随机变量，则其和仍是高斯分布：

$$
p(\boldsymbol{x}+\boldsymbol{y})=\mathcal{N}(\boldsymbol{\mu}_{x}+\boldsymbol{\mu}_{y},\:\boldsymbol{\Sigma}_{x}+\boldsymbol{\Sigma}_{y}) \tag{6.64}
$$

**高斯随机变量的线性/仿射变换仍为高斯分布**。设 $X\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$，$\boldsymbol{y}=\boldsymbol{A}\boldsymbol{x}$，则：

$$
p(\boldsymbol{y})=\mathcal{N}(\boldsymbol{y}\mid\boldsymbol{A}\boldsymbol{\mu},\:\boldsymbol{A}\boldsymbol{\Sigma}\boldsymbol{A}^{\top}) \tag{6.65}
$$

**备注**：高斯密度的加权和（混合模型）与高斯随机变量的加权和是不同的概念。对于混合密度 $p(x)=\alpha p_1(x)+(1-\alpha)p_2(x)$，其均值和方差为：

$$
\mathbb{E}[x]=\alpha\mu_1+(1-\alpha)\mu_2 \tag{6.66}
$$

$$
\mathrm{V}[x]=\underbrace{[\alpha\sigma_1^2+(1-\alpha)\sigma_2^2]}_{\text{条件方差的期望}}+\underbrace{[\alpha\mu_1^2+(1-\alpha)\mu_2^2]-[\alpha\mu_1+(1-\alpha)\mu_2]^2}_{\text{条件均值的方差}} \tag{6.67}
$$

这是**全方差定律**的一个实例：$\mathrm{V}_X[x]=\mathbb{E}_Y[\mathrm{V}_X[x\mid y]]+\mathrm{V}_Y[\mathbb{E}_X[x\mid y]]$。


### 6.6.4 从多元高斯分布中采样

从 $\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$ 中采样的步骤：

1. 从标准正态 $\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$ 中采样得到 $\boldsymbol{x}$
2. 对协方差矩阵做 Cholesky 分解 $\boldsymbol{\Sigma}=\boldsymbol{A}\boldsymbol{A}^{\top}$
3. 计算 $\boldsymbol{y}=\boldsymbol{A}\boldsymbol{x}+\boldsymbol{\mu}$，则 $\boldsymbol{y}\sim\mathcal{N}(\boldsymbol{\mu},\boldsymbol{\Sigma})$







## 6.7 共轭性与指数族分布

本节介绍在机器学习中操作概率分布时期望具备的三个性质：(1) 概率运算的**封闭性**（如贝叶斯推断后类型不变）；(2) 参数数量不随数据增多而膨胀；(3) 参数估计表现良好。指数族分布在保持一般性的同时满足这些性质。

### 6.7.1 共轭性

**定义 6.17（共轭先验）**：如果后验与先验具有相同的分布形式，则该先验是似然函数的**共轭先验**。

共轭性的好处在于可以通过**代数更新参数**来计算后验，无需数值积分。

> **贝塔-二项共轭**：对 $x\sim\mathrm{Bin}(N,\mu)$，若先验 $\mu\sim\mathrm{Beta}(\alpha,\beta)$，观察到 $x=h$ 次正面后：
>
> $$p(\mu\mid x=h)\propto\mu^{h+\alpha-1}(1-\mu)^{(N-h)+\beta-1}\propto\mathrm{Beta}(h+\alpha,\:N-h+\beta) \tag{6.68}$$
>
> 后验仍是贝塔分布——$\alpha$ 加上正面次数，$\beta$ 加上反面次数。

> **贝塔-伯努利共轭**类似：观察 $x\in\{0,1\}$ 后，后验为 $\mathrm{Beta}(\alpha+x,\:\beta+(1-x))$。

**备注**：常见共轭对包括：高斯似然→高斯先验（均值）、高斯似然→逆伽马/逆Wishart先验（方差/协方差）、多项式似然→狄利克雷先验、泊松似然→伽马先验。


### 6.7.2 充分统计量

直觉：**充分统计量**是从数据中能提取的、关于分布参数的全部信息。

**定理 6.18（Fisher-Neyman）**：统计量 $\phi(\boldsymbol{x})$ 是参数 $\theta$ 的充分统计量，当且仅当

$$
p(\boldsymbol{x}\mid\theta)=h(\boldsymbol{x})\,g_\theta(\phi(\boldsymbol{x})) \tag{6.69}
$$

其中 $h(\boldsymbol{x})$ 与 $\theta$ 无关，$g_\theta$ 仅通过 $\phi(\boldsymbol{x})$ 依赖于 $\theta$。

例如，对于高斯分布，样本均值和样本方差就是充分统计量。关键问题：哪类分布具有**有限维**充分统计量？答案是指数族分布。


### 6.7.3 指数族分布

**指数族分布**是参数化为如下形式的概率分布族：

$$
p(\boldsymbol{x}\mid\boldsymbol{\theta})=h(\boldsymbol{x})\exp\left(\boldsymbol{\theta}^{\top}\boldsymbol{\phi}(\boldsymbol{x})-A(\boldsymbol{\theta})\right) \tag{6.70}
$$

其中 $\boldsymbol{\phi}(\boldsymbol{x})$ 是充分统计量向量，$\boldsymbol{\theta}$ 是**自然参数**，$A(\boldsymbol{\theta})$ 是对数配分函数（归一化常数的对数）。忽略 $h(\boldsymbol{x})$ 和 $A(\boldsymbol{\theta})$ 后，核心结构为 $p(\boldsymbol{x}\mid\boldsymbol{\theta})\propto\exp(\boldsymbol{\theta}^{\top}\boldsymbol{\phi}(\boldsymbol{x}))$。

> **高斯分布属于指数族**：令 $\phi(x)=[x,\:x^2]^{\top}$，自然参数 $\boldsymbol{\theta}=[\mu/\sigma^2,\:-1/(2\sigma^2)]^{\top}$，则
>
> $$p(x\mid\boldsymbol{\theta})\propto\exp\left(\frac{\mu x}{\sigma^2}-\frac{x^2}{2\sigma^2}\right)\propto\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) \tag{6.71}$$

> **伯努利分布属于指数族**：自然参数 $\theta=\log\frac{\mu}{1-\mu}$，充分统计量 $\phi(x)=x$，反解得
>
> $$\mu=\frac{1}{1+\exp(-\theta)} \tag{6.72}$$
>
> 这就是 **sigmoid/logistic 函数**，将实数映射到 $(0,1)$，广泛用于逻辑回归和神经网络。

**指数族的共轭先验**具有统一形式：

$$
p(\boldsymbol{\theta}\mid\boldsymbol{\gamma})=h_c(\boldsymbol{\theta})\exp\left(\left\langle\begin{bmatrix}\gamma_1\\\gamma_2\end{bmatrix},\begin{bmatrix}\boldsymbol{\theta}\\-A(\boldsymbol{\theta})\end{bmatrix}\right\rangle-A_c(\boldsymbol{\gamma})\right) \tag{6.73}
$$

**备注**：指数族的核心优势——(1) 有限维充分统计量；(2) 共轭先验易于构造且也属于指数族；(3) 对数似然函数是凹函数，利于优化。






## 6.8 变量变换/逆变换

核心问题：已知随机变量 $X$ 的分布，经过变换 $Y=U(X)$ 后，$Y$ 的分布是什么？

对于**离散随机变量**，若 $U$ 可逆，则直接代入：

$$
P(Y=y)=P(X=U^{-1}(y)) \tag{6.74}
$$

对于**连续随机变量**，需要额外处理体积变化因子。下面介绍两种方法。


### 6.8.1 分布函数技术

基于累积分布函数（CDF）的定义进行推导：

1. 求 $Y$ 的 CDF：$F_Y(y)=P(Y\leqslant y)$
2. 对 CDF 求导得 PDF：$f(y)=\frac{\mathrm{d}}{\mathrm{d}y}F_Y(y)$

**定理 6.19（概率积分变换）**：设 $X$ 是连续随机变量，CDF 为严格单调的 $F_X(x)$，则

$$
Y:=F_X(X) \tag{6.75}
$$

服从均匀分布。该定理是从均匀分布采样转换为任意分布采样的理论基础：先从均匀分布采样，再通过逆 CDF 变换即可得到目标分布的样本。


### 6.8.2 变量替换

对于单变量，设 $Y=U(X)$，$U$ 可逆，则 $Y$ 的 PDF 为：

$$
f(y)=f_x(U^{-1}(y))\cdot\left|\frac{\mathrm{d}}{\mathrm{d}y}U^{-1}(y)\right| \tag{6.76}
$$

其中 $\left|\frac{\mathrm{d}}{\mathrm{d}y}U^{-1}(y)\right|$ 衡量变换 $U$ 引起的**单位体积变化量**。绝对值保证了无论 $U$ 是递增还是递减，结果一致。

**备注**：与离散情况 (6.74) 相比，连续情况多了微分因子——因为连续变量取特定值的概率为零，必须考虑密度在变换下的"拉伸/压缩"效应。

**定理 6.20（多元变量替换）**：设 $\boldsymbol{y}=U(\boldsymbol{x})$ 为可微且可逆的向量值函数，则 $Y$ 的 PDF 为

$$
f(\boldsymbol{y})=f_{\boldsymbol{x}}(U^{-1}(\boldsymbol{y}))\cdot\left|\det\left(\frac{\partial}{\partial\boldsymbol{y}}U^{-1}(\boldsymbol{y})\right)\right| \tag{6.77}
$$

多元情况下，单变量的导数绝对值被替换为 **Jacobian 矩阵行列式的绝对值**，它度量了变换在局部的体积缩放因子。

> **线性变换的例子**：设 $X\sim\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$，$\boldsymbol{y}=\boldsymbol{A}\boldsymbol{x}$，则逆变换为 $\boldsymbol{x}=\boldsymbol{A}^{-1}\boldsymbol{y}$，Jacobian 行列式为 $|\det(\boldsymbol{A})|^{-1}$，变换后的密度为
>
> $$f(\boldsymbol{y})=\frac{1}{(2\pi)^{D/2}|\det(\boldsymbol{A})|}\exp\left(-\frac{1}{2}\boldsymbol{y}^{\top}\boldsymbol{A}^{-\top}\boldsymbol{A}^{-1}\boldsymbol{y}\right)$$
>
> 这正是协方差为 $\boldsymbol{\Sigma}=\boldsymbol{A}\boldsymbol{A}^{\top}$ 的多元高斯分布。
















