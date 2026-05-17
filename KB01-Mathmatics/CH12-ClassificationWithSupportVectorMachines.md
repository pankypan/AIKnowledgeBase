# Classification with Support Vector Machines(支持向量机与分类)

本章讨论**二分类**问题：预测器输出为离散二值标签 $\{+1, -1\}$，形式为 $f: \mathbb{R}^D \to \{+1, -1\}$。核心方法是**支持向量机（SVM）**——一种基于几何直觉（间隔最大化）而非概率模型的分类方法。

SVM 的几何观点与第 9 章的最大似然观点不同：它从设计损失函数出发，利用内积、投影等几何工具推导优化问题，且该优化问题无解析解，需借助数值优化方法求解。



## 12.1 分隔超平面

给定样本 $\boldsymbol{x} \in \mathbb{R}^D$，定义线性函数

$$
f(\boldsymbol{x}) = \langle \boldsymbol{w}, \boldsymbol{x} \rangle + b \tag{12.2}
$$

其中 $\boldsymbol{w} \in \mathbb{R}^D$ 为权重向量，$b \in \mathbb{R}$ 为截距。**分隔超平面**定义为

$$
\{\boldsymbol{x} \in \mathbb{R}^D : f(\boldsymbol{x}) = 0\} \tag{12.3}
$$

$\boldsymbol{w}$ 是超平面的**法向量**（与超平面上任意向量正交），$b$ 控制超平面的位移。

**分类规则**：对测试样本 $\boldsymbol{x}_{\text{test}}$，当 $f(\boldsymbol{x}_{\text{test}}) \geq 0$ 时分类为 $+1$，否则分类为 $-1$。

训练时要求所有样本被正确分隔，即

$$
y_n (\langle \boldsymbol{w}, \boldsymbol{x}_n \rangle + b) \geq 0 \tag{12.7}
$$

此式将正样本 $\langle \boldsymbol{w}, \boldsymbol{x}_n \rangle + b \geq 0$ 和负样本 $\langle \boldsymbol{w}, \boldsymbol{x}_n \rangle + b < 0$ 的要求统一为一个不等式。



## 12.2 原始支持向量机

对于线性可分数据集，存在无数个分隔超平面。SVM 的核心思想是选择**间隔最大**的那一个。

### 12.2.1 间隔的概念

样本 $\boldsymbol{x}_a$ 到超平面的距离通过正交投影获得。设 $\boldsymbol{x}_a'$ 为投影点，则

$$
\boldsymbol{x}_a = \boldsymbol{x}_a' + r \frac{\boldsymbol{w}}{\|\boldsymbol{w}\|} \tag{12.8}
$$

其中 $r > 0$ 即为距离。选择最近样本的距离作为**间隔**，要求所有样本至少距超平面 $r$：

$$
y_n (\langle \boldsymbol{w}, \boldsymbol{x}_n \rangle + b) \geq r \tag{12.9}
$$

假设 $\|\boldsymbol{w}\| = 1$，则最大间隔优化问题为

$$
\max_{\boldsymbol{w}, b, r} \; r \quad \text{s.t.} \; y_n(\langle \boldsymbol{w}, \boldsymbol{x}_n \rangle + b) \geq r, \; \|\boldsymbol{w}\| = 1, \; r > 0 \tag{12.10}
$$

### 12.2.2 间隔的传统推导

另一种等价做法：不归一化 $\boldsymbol{w}$，而是缩放数据使最近样本上 $\langle \boldsymbol{w}, \boldsymbol{x}_a \rangle + b = 1$。此时间隔为

$$
r = \frac{1}{\|\boldsymbol{w}\|} \tag{12.14}
$$

约束变为 $y_n(\langle \boldsymbol{w}, \boldsymbol{x}_n \rangle + b) \geq 1$，最大化间隔等价于：

$$
\min_{\boldsymbol{w}, b} \; \frac{1}{2} \|\boldsymbol{w}\|^2 \quad \text{s.t.} \; y_n(\langle \boldsymbol{w}, \boldsymbol{x}_n \rangle + b) \geq 1 \quad \forall n \tag{12.18}
$$

这就是**硬间隔 SVM**——不允许任何样本违反间隔条件。

> **定理 12.1**：归一化权重的间隔最大化 (12.10) 与缩放数据的间隔等于 1 的最小化 (12.18) 是等价的。

### 12.2.4 软间隔 SVM：几何视角

当数据非线性可分时，引入**松弛变量** $\xi_n \geq 0$ 允许样本违反间隔：

$$
\min_{\boldsymbol{w}, b, \boldsymbol{\xi}} \; \frac{1}{2} \|\boldsymbol{w}\|^2 + C \sum_{n=1}^N \xi_n \quad \text{s.t.} \; y_n(\langle \boldsymbol{w}, \boldsymbol{x}_n \rangle + b) \geq 1 - \xi_n, \; \xi_n \geq 0 \tag{12.26}
$$

- **$C > 0$**：正则化参数，权衡间隔大小与分类错误容忍度。$C$ 越大，越不容忍错误（低正则化）
- $\|\boldsymbol{w}\|^2$ 项来源于间隔最大化，起正则化器的作用

### 12.2.5 软间隔 SVM：损失函数视角

等价地，可从经验风险最小化角度推导。SVM 使用**合页损失（Hinge Loss）**：

$$
\ell(t) = \max\{0, 1 - t\}, \quad t = y(\langle \boldsymbol{w}, \boldsymbol{x} \rangle + b) \tag{12.28}
$$

- $t \geq 1$：正确分类且在间隔外，损失为 0
- $0 < t < 1$：正确分类但在间隔内，损失线性增加
- $t < 0$：错误分类，损失更大且线性增加

合页损失是零一损失的**凸上界**，使得优化问题可解。加上 $\ell_2$ 正则化得到无约束问题：

$$
\min_{\boldsymbol{w}, b} \; \underbrace{\frac{1}{2} \|\boldsymbol{w}\|^2}_{\text{正则化项}} + \underbrace{C \sum_{n=1}^N \max\{0, 1 - y_n(\langle \boldsymbol{w}, \boldsymbol{x}_n \rangle + b)\}}_{\text{损失项}} \tag{12.31}
$$

此无约束形式 (12.31) 与带约束的软间隔 SVM (12.26) 完全等价。



## 12.3 对偶支持向量机

### 12.3.1 通过 Lagrange 乘数法的凸对偶

对软间隔 SVM (12.26) 引入 Lagrange 乘数 $\alpha_n \geq 0$（对应分类约束）和 $\gamma_n \geq 0$（对应松弛非负约束），对原始变量求导并令其为零，得到关键结果：

**表示定理**：最优权重向量是训练样本的线性组合

$$
\boldsymbol{w} = \sum_{n=1}^N \alpha_n y_n \boldsymbol{x}_n \tag{12.38}
$$

> 只有 $\alpha_n > 0$ 的样本对 $\boldsymbol{w}$ 有贡献，这些样本称为**支持向量**——这也是"支持向量机"名称的由来。

消去原始变量后得到**对偶 SVM**：

$$
\min_{\boldsymbol{\alpha}} \; \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N y_i y_j \alpha_i \alpha_j \langle \boldsymbol{x}_i, \boldsymbol{x}_j \rangle - \sum_{i=1}^N \alpha_i \quad \text{s.t.} \; \sum_{i=1}^N y_i \alpha_i = 0, \; 0 \leq \alpha_i \leq C \tag{12.41}
$$

对偶形式的优点：
- 优化变量维度为样本数 $N$（而非特征维度 $D$），适合高维小样本问题
- 目标函数中仅涉及样本间的**内积** $\langle \boldsymbol{x}_i, \boldsymbol{x}_j \rangle$，可方便地引入核函数
- 约束为**盒约束** $0 \leq \alpha_i \leq C$，数值求解高效

### 12.3.2 对偶 SVM 的凸包视角

另一种几何解释：分别对正、负类样本构建**凸包**，寻找两个凸包间距离最短的点对 $(c, d)$，令 $\boldsymbol{w} = c - d$，最小化 $\|\boldsymbol{w}\|^2$。

正类凸包中的点 $c$ 为正样本的凸组合：$c = \sum_{n:y_n=+1} \alpha_n^+ \boldsymbol{x}_n$，类似地定义负类点 $d$。要求凸组合系数之和为 1 导出约束 $\sum_n y_n \alpha_n = 0$，由此得到的优化问题与对偶硬间隔 SVM 等价。

> 软间隔对应**缩减凸包**：对系数 $\alpha_n$ 加上上界约束 $\alpha_n \leq C$，限制凸包大小。



## 12.4 核函数

对偶 SVM 目标函数中仅出现样本间的内积 $\langle \boldsymbol{x}_i, \boldsymbol{x}_j \rangle$。若使用非线性特征映射 $\phi(\boldsymbol{x})$，只需将内积替换为 $\langle \phi(\boldsymbol{x}_i), \phi(\boldsymbol{x}_j) \rangle$。

**核函数**定义为 $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$，满足

$$
k(\boldsymbol{x}_i, \boldsymbol{x}_j) = \langle \phi(\boldsymbol{x}_i), \phi(\boldsymbol{x}_j) \rangle_{\mathcal{H}} \tag{12.52}
$$

其中 $\phi: \mathcal{X} \to \mathcal{H}$ 为到希尔伯特空间 $\mathcal{H}$ 的特征映射。这一从内积到核函数的推广称为**核技巧（Kernel Trick）**。

核矩阵（Gram 矩阵）$K \in \mathbb{R}^{N \times N}$，$K_{ij} = k(\boldsymbol{x}_i, \boldsymbol{x}_j)$，必须是**对称正半定**的：

$$
\forall \boldsymbol{z} \in \mathbb{R}^N: \boldsymbol{z}^\top K \boldsymbol{z} \geq 0 \tag{12.53}
$$

常用核函数包括：
- **多项式核**：高维特征空间的高效替代
- **高斯径向基函数核（RBF）**：对应无限维特征空间，无法显式表示 $\phi$
- **有理二次核**

核技巧的价值：
- 无需显式计算高维甚至无限维的 $\phi(\boldsymbol{x})$，直接通过核函数计算相似度
- 输入不限于实数向量，可以是集合、序列、字符串、图等任意对象
- SVM 仍在求解线性分隔超平面，非线性决策边界由核函数引起



## 12.5 数值解

SVM 的优化问题无解析解，需要数值方法求解。

**次梯度方法**：合页损失在 $t = 1$ 处不可微，其次梯度为

$$
g(t) = \begin{cases} -1 & t < 1 \\ [-1, 0] & t = 1 \\ 0 & t > 1 \end{cases} \tag{12.54}
$$

可直接对无约束形式 (12.31) 使用次梯度下降。

**二次规划（QP）方法**：原始和对偶 SVM 均可转化为标准凸二次规划形式。

- **原始 SVM**：优化变量维度 $D + 1 + N$（$\boldsymbol{w}, b, \boldsymbol{\xi}$），适合低维问题
- **对偶 SVM**：优化变量维度 $N$（$\boldsymbol{\alpha}$），使用核矩阵 $K$ 表达为

$$
\min_{\boldsymbol{\alpha}} \; \frac{1}{2} \boldsymbol{\alpha}^\top Y K Y \boldsymbol{\alpha} - \mathbf{1}^\top \boldsymbol{\alpha} \quad \text{s.t. 盒约束与等式约束} \tag{12.57}
$$

其中 $Y = \text{diag}(\boldsymbol{y})$。实际中常用的 SVM 求解器（如 LIBSVM、SVMlight）基于专门的分解算法，比通用 QP 求解器更高效。



## 12.6 拓展阅读

SVM 之外的二分类方法：感知机、逻辑回归、Fisher 判别分析、最近邻、朴素贝叶斯、随机森林等。

关键联系：
- **合页损失的三种等价形式**：函数形式 (12.28)、分段线性形式 (12.29)、带约束优化形式 (12.33)
- **SVM 与概率方法**：SVM 本身非概率模型，但可通过 Platt 缩放等方法将输出转换为校准概率。将损失函数换为对数损失即得逻辑回归（属于广义线性模型）
- **核方法的推广**：核的概念基于再生核希尔伯特空间（RKHS），可推广到巴拿赫空间和 Kreĭn 空间
- **间隔与泛化**：Vapnik-Chervonenkis 理论表明间隔越大，函数类复杂度越低，泛化能力越强
