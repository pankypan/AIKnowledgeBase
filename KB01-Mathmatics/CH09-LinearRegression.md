# Linear Regression(线性回归)

线性回归是机器学习中最基础的回归问题。目标是找到一个函数 $f$，将输入 $\boldsymbol{x} \in \mathbb{R}^D$ 映射到函数值 $f(\boldsymbol{x}) \in \mathbb{R}$，使其不仅拟合训练数据，还能泛化到未见数据。

核心挑战包括：**模型选择与参数化**、**参数估计**、**过拟合与正则化**、**损失函数与先验的关系**、**不确定性建模**。



## 9.1 问题形式化

给定训练输入 $\boldsymbol{x}_n$ 和带噪声的观测值 $y_n$，假设观测噪声为零均值高斯噪声，**似然函数**为：

$$
p(y | \boldsymbol{x}) = \mathcal{N}(y | f(\boldsymbol{x}), \sigma^2) \tag{9.1}
$$

等价地，$\boldsymbol{x}$ 和 $y$ 之间的关系可以表示为：

$$
y = f(\boldsymbol{x}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2) \tag{9.2}
$$

在**线性回归**中，参数 $\boldsymbol{\theta}$ 线性出现在模型中：

$$
p(y | \boldsymbol{x}, \boldsymbol{\theta}) = \mathcal{N}(y | \boldsymbol{x}^\top \boldsymbol{\theta}, \sigma^2) \tag{9.3}
$$

$$
\Longleftrightarrow \quad y = \boldsymbol{x}^\top \boldsymbol{\theta} + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2) \tag{9.4}
$$

> **关键概念**："线性回归"指的是**参数线性**，而非输入线性。通过对输入 $\boldsymbol{x}$ 做非线性变换 $\boldsymbol{\phi}(\boldsymbol{x})$，再线性组合变换后的各分量，仍然属于线性回归框架：$y = \boldsymbol{\phi}(\boldsymbol{x})^\top \boldsymbol{\theta} + \epsilon$。



## 9.2 参数估计

给定训练集 $\mathcal{D} = \{(\boldsymbol{x}_1, y_1), \ldots, (\boldsymbol{x}_N, y_N)\}$，由于 $y_i$ 和 $y_j$ 给定各自输入后**条件独立**，似然函数可分解为：

$$
p(\mathcal{Y} | \mathcal{X}, \boldsymbol{\theta}) = \prod_{n=1}^N p(y_n | \boldsymbol{x}_n, \boldsymbol{\theta}) = \prod_{n=1}^N \mathcal{N}(y_n | \boldsymbol{x}_n^\top \boldsymbol{\theta}, \sigma^2) \tag{9.5}
$$

### 9.2.1 最大似然估计（MLE）

**最大似然估计**寻找最大化似然的参数：$\boldsymbol{\theta}_{\text{ML}} = \arg \max_{\boldsymbol{\theta}} p(\mathcal{Y} | \mathcal{X}, \boldsymbol{\theta})$。

实际操作中，对似然取对数并最小化**负对数似然**。利用高斯噪声，负对数似然（忽略常数）为：

$$
\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{2\sigma^2} \sum_{n=1}^N (y_n - \boldsymbol{x}_n^\top \boldsymbol{\theta})^2 = \frac{1}{2\sigma^2} \| \boldsymbol{y} - X \boldsymbol{\theta} \|^2 \tag{9.10}
$$

其中**设计矩阵** $X = [\boldsymbol{x}_1, \ldots, \boldsymbol{x}_N]^\top \in \mathbb{R}^{N \times D}$，$\boldsymbol{y} = [y_1, \ldots, y_N]^\top$。

$\mathcal{L}$ 关于 $\boldsymbol{\theta}$ 是**二次**的，存在唯一全局最优解。令梯度为零：

$$
\frac{d\mathcal{L}}{d\boldsymbol{\theta}} = \frac{1}{\sigma^2} (-\boldsymbol{y}^\top X + \boldsymbol{\theta}^\top X^\top X) = \boldsymbol{0}^\top \tag{9.11}
$$

得到**正规方程**的闭式解：

$$
\boxed{\boldsymbol{\theta}_{\text{ML}} = (X^\top X)^{-1} X^\top \boldsymbol{y}} \tag{9.12}
$$

该解要求 $\text{rank}(X) = D$（使得 $X^\top X$ 正定可逆）。

#### 带特征的最大似然估计

引入非线性特征变换 $\boldsymbol{\phi}: \mathbb{R}^D \to \mathbb{R}^K$ 后，模型变为：

$$
y = \boldsymbol{\phi}(\boldsymbol{x})^\top \boldsymbol{\theta} + \epsilon = \sum_{k=0}^{K-1} \theta_k \phi_k(\boldsymbol{x}) + \epsilon \tag{9.13}
$$

例如**多项式回归**中，$\boldsymbol{\phi}(x) = [1, x, x^2, \ldots, x^{K-1}]^\top$，对应 $K-1$ 阶多项式。

定义**特征矩阵（设计矩阵）** $\Phi \in \mathbb{R}^{N \times K}$，其中 $\Phi_{ij} = \phi_j(\boldsymbol{x}_i)$，最大似然解为：

$$
\boldsymbol{\theta}_{\text{ML}} = (\Phi^\top \Phi)^{-1} \Phi^\top \boldsymbol{y} \tag{9.19}
$$

#### 噪声方差的最大似然估计

对 $\sigma^2$ 也可做最大似然估计，令对数似然关于 $\sigma^2$ 的导数为零，得到：

$$
\sigma_{\text{ML}}^2 = \frac{1}{N} \sum_{n=1}^N (y_n - \boldsymbol{\phi}(\boldsymbol{x}_n)^\top \boldsymbol{\theta})^2 \tag{9.22}
$$

即残差平方的均值。

### 9.2.2 过拟合

模型质量通常用**均方根误差（RMSE）**衡量：

$$
\text{RMSE} = \sqrt{\frac{1}{N} \| \boldsymbol{y} - \Phi \boldsymbol{\theta} \|^2} = \sqrt{\frac{1}{N} \sum_{n=1}^N (y_n - \boldsymbol{\phi}(\boldsymbol{x}_n)^\top \boldsymbol{\theta})^2} \tag{9.23}
$$

**过拟合现象**：随多项式阶数 $M$ 增大，训练误差单调下降，但当 $M$ 过高时测试误差显著上升。在极端情况 $M = N-1$ 时，函数通过每个数据点但剧烈振荡，泛化性能很差。

### 9.2.3 最大后验估计（MAP）

为抑制过拟合，可在参数上放置**先验分布** $p(\boldsymbol{\theta})$。选择高斯先验 $p(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{0}, b^2 I)$ 后，通过贝叶斯定理得后验：

$$
p(\boldsymbol{\theta} | \mathcal{X}, \mathcal{Y}) = \frac{p(\mathcal{Y} | \mathcal{X}, \boldsymbol{\theta}) p(\boldsymbol{\theta})}{p(\mathcal{Y} | \mathcal{X})} \tag{9.24}
$$

MAP 估计最小化**负对数后验**：

$$
-\log p(\boldsymbol{\theta} | \mathcal{X}, \mathcal{Y}) = \frac{1}{2\sigma^2} \| \boldsymbol{y} - \Phi \boldsymbol{\theta} \|^2 + \frac{1}{2b^2} \boldsymbol{\theta}^\top \boldsymbol{\theta} + \text{const} \tag{9.28}
$$

令梯度为零，得到 MAP 估计的闭式解：

$$
\boxed{\boldsymbol{\theta}_{\text{MAP}} = \left( \Phi^\top \Phi + \frac{\sigma^2}{b^2} I \right)^{-1} \Phi^\top \boldsymbol{y}} \tag{9.31}
$$

与 MLE 解相比，唯一区别是逆矩阵中多了 $\frac{\sigma^2}{b^2} I$ 项，它保证逆总存在，并起到**正则化**作用。

### 9.2.4 MAP 估计作为正则化

MAP 估计等价于**正则化最小二乘法**，最小化：

$$
\| \boldsymbol{y} - \Phi \boldsymbol{\theta} \|^2 + \lambda \| \boldsymbol{\theta} \|_2^2 \tag{9.32}
$$

其中正则化参数 $\lambda = \sigma^2 / b^2$。

- 第一项为**数据拟合项**（负对数似然）
- 第二项为**正则化项**（负对数先验），等价于高斯先验 $p(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{0}, b^2 I)$

正则化解：

$$
\boldsymbol{\theta}_{\text{RLS}} = (\Phi^\top \Phi + \lambda I)^{-1} \Phi^\top \boldsymbol{y} \tag{9.34}
$$

> **注释**：选择 $\|\cdot\|_1$ 范数（LASSO）可得到**稀疏解**，即许多参数 $\theta_d = 0$，适用于变量选择。



## 9.3 贝叶斯线性回归（Bayesian Linear Regression）

贝叶斯线性回归不寻求参数的**点估计**，而是考虑参数的**完整后验分布**，在预测时对所有合理参数设置进行平均。

### 9.3.1 模型

$$
\text{先验：} \quad p(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta} | \boldsymbol{m}_0, S_0) \tag{9.35a}
$$

$$
\text{似然：} \quad p(y | \boldsymbol{x}, \boldsymbol{\theta}) = \mathcal{N}(y | \boldsymbol{\phi}(\boldsymbol{x})^\top \boldsymbol{\theta}, \sigma^2) \tag{9.35b}
$$

选择高斯先验使得参数向量 $\boldsymbol{\theta}$ 成为随机变量，构成**共轭模型**，后验也是高斯分布。

### 9.3.2 先验预测

在观测数据之前，通过对 $\boldsymbol{\theta}$ 积分得到先验预测分布：

$$
p(y^* | \boldsymbol{x}^*) = \int p(y^* | \boldsymbol{x}^*, \boldsymbol{\theta}) p(\boldsymbol{\theta}) d\boldsymbol{\theta} = \mathbb{E}_{\boldsymbol{\theta}}[p(y^* | \boldsymbol{x}^*, \boldsymbol{\theta})] \tag{9.37}
$$

由于共轭性，先验预测也是高斯分布：

$$
p(y^* | \boldsymbol{x}^*) = \mathcal{N}(y^* | \boldsymbol{\phi}(\boldsymbol{x}^*)^\top \boldsymbol{m}_0,\; \boldsymbol{\phi}(\boldsymbol{x}^*)^\top S_0 \boldsymbol{\phi}(\boldsymbol{x}^*) + \sigma^2) \tag{9.38}
$$

预测方差中 $\boldsymbol{\phi}(\boldsymbol{x}^*)^\top S_0 \boldsymbol{\phi}(\boldsymbol{x}^*)$ 来自**参数不确定性**，$\sigma^2$ 来自**测量噪声**。

参数分布 $p(\boldsymbol{\theta})$ 诱导了**函数分布** $p(f(\cdot))$：每个参数样本 $\boldsymbol{\theta}_i$ 对应一个函数 $f_i(\cdot) = \boldsymbol{\theta}_i^\top \boldsymbol{\phi}(\cdot)$。

### 9.3.3 后验分布

**定理 9.1（参数后验）**：给定训练数据，参数后验为高斯分布：

$$
p(\boldsymbol{\theta} | \mathcal{X}, \mathcal{Y}) = \mathcal{N}(\boldsymbol{\theta} | \boldsymbol{m}_N, S_N) \tag{9.43a}
$$

其中

$$
S_N = (S_0^{-1} + \sigma^{-2} \Phi^\top \Phi)^{-1} \tag{9.43b}
$$

$$
\boldsymbol{m}_N = S_N (S_0^{-1} \boldsymbol{m}_0 + \sigma^{-2} \Phi^\top \boldsymbol{y}) \tag{9.43c}
$$

推导利用**补全平方**（completing the square）方法：将对数似然和对数先验之和的关于 $\boldsymbol{\theta}$ 的二次项和线性项分别与目标高斯分布的对应项匹配，即可识别出后验的均值和协方差。

> **注释**：后验均值 $\boldsymbol{m}_N$ 就是 MAP 估计 $\boldsymbol{\theta}_{\text{MAP}}$，但贝叶斯方法额外提供了**不确定性量化**（通过 $S_N$）。

### 9.3.4 后验预测

将后验 $p(\boldsymbol{\theta} | \mathcal{X}, \mathcal{Y})$ 代替先验进行积分，得到**后验预测分布**：

$$
p(y^* | \mathcal{X}, \mathcal{Y}, \boldsymbol{x}^*) = \int p(y^* | \boldsymbol{x}^*, \boldsymbol{\theta}) p(\boldsymbol{\theta} | \mathcal{X}, \mathcal{Y}) d\boldsymbol{\theta} \tag{9.57a}
$$

$$
= \mathcal{N}(y^* | \boldsymbol{\phi}(\boldsymbol{x}^*)^\top \boldsymbol{m}_N,\; \boldsymbol{\phi}(\boldsymbol{x}^*)^\top S_N \boldsymbol{\phi}(\boldsymbol{x}^*) + \sigma^2) \tag{9.57c}
$$

- **预测均值**：$\boldsymbol{\phi}(\boldsymbol{x}^*)^\top \boldsymbol{m}_N$（与 MAP 预测一致）
- **预测方差**：$\boldsymbol{\phi}(\boldsymbol{x}^*)^\top S_N \boldsymbol{\phi}(\boldsymbol{x}^*) + \sigma^2$（参数不确定性 + 噪声不确定性）

对于**无噪声**函数值 $f(\boldsymbol{x}^*) = \boldsymbol{\phi}(\boldsymbol{x}^*)^\top \boldsymbol{\theta}$ 的预测：

$$
\mathbb{E}[f(\boldsymbol{x}^*) | \mathcal{X}, \mathcal{Y}] = \boldsymbol{\phi}(\boldsymbol{x}^*)^\top \boldsymbol{m}_N \tag{9.58}
$$

$$
\text{Var}[f(\boldsymbol{x}^*) | \mathcal{X}, \mathcal{Y}] = \boldsymbol{\phi}(\boldsymbol{x}^*)^\top S_N \boldsymbol{\phi}(\boldsymbol{x}^*) \tag{9.59}
$$

### 9.3.5 边际似然（Evidence）

**边际似然**用于贝叶斯模型选择，通过对参数积分得到：

$$
p(\mathcal{Y} | \mathcal{X}) = \int p(\mathcal{Y} | \mathcal{X}, \boldsymbol{\theta}) p(\boldsymbol{\theta}) d\boldsymbol{\theta} \tag{9.61}
$$

由于先验和似然都是高斯分布，边际似然也是高斯分布，其均值和协方差为：

$$
\mathbb{E}[\mathcal{Y} | \mathcal{X}] = X \boldsymbol{m}_0 \tag{9.62}
$$

$$
\text{Cov}[\mathcal{Y} | \mathcal{X}] = X S_0 X^\top + \sigma^2 I \tag{9.63}
$$

因此：

$$
p(\mathcal{Y} | \mathcal{X}) = \mathcal{N}(\boldsymbol{y} | X \boldsymbol{m}_0,\; X S_0 X^\top + \sigma^2 I) \tag{9.64}
$$

边际似然可看作在**先验下对训练目标的预测**，用于比较不同模型（如不同阶数的多项式）的优劣。



## 9.4 最大似然作为正交投影

最大似然估计具有直观的**几何解释**。考虑简单线性回归 $y = x\theta + \epsilon$，最大似然解为：

$$
\boldsymbol{\theta}_{\text{ML}} = \frac{X^\top \boldsymbol{y}}{X^\top X} \tag{9.66}
$$

对应的拟合值：

$$
X \boldsymbol{\theta}_{\text{ML}} = \frac{X X^\top}{X^\top X} \boldsymbol{y} \tag{9.67}
$$

其中 $\frac{X X^\top}{X^\top X}$ 是**投影矩阵**，$\boldsymbol{\theta}_{\text{ML}}$ 是投影坐标，$X\boldsymbol{\theta}_{\text{ML}}$ 是 $\boldsymbol{y}$ 到由 $X$ 张成的子空间的**正交投影**。

一般情况下，带特征 $\boldsymbol{\phi}(\boldsymbol{x}) \in \mathbb{R}^K$ 的最大似然解：

$$
\boldsymbol{y} \approx \Phi \boldsymbol{\theta}_{\text{ML}}, \quad \boldsymbol{\theta}_{\text{ML}} = (\Phi^\top \Phi)^{-1} \Phi^\top \boldsymbol{y} \tag{9.70}
$$

是将 $\boldsymbol{y}$ 正交投影到 $\Phi$ 的列空间（$\mathbb{R}^N$ 中的 $K$ 维子空间）上。若特征函数正交（$\Phi^\top \Phi = I$），投影简化为向各基向量的投影之和，特征间耦合消失。



## 9.5 拓展阅读

- **广义线性模型（GLM）**：通过非线性激活函数 $\sigma(\cdot)$ 将线性模型与观测联系起来，$y = \sigma(f(\boldsymbol{x}))$，其中 $f(\boldsymbol{x}) = \boldsymbol{\theta}^\top \boldsymbol{\phi}(\boldsymbol{x})$。逻辑回归使用 logistic 函数 $\sigma(f) = 1/(1+e^{-f})$，普通线性回归的激活函数是恒等函数
- **深度神经网络**：GLM 是其构建块。单层 $\boldsymbol{x}_{k+1} = \sigma_k(A_k \boldsymbol{x}_k + \boldsymbol{b}_k)$，递归组合 $K$ 层即为深度网络 $f_{K-1} \circ \cdots \circ f_0$
- **高斯过程（GP）**：直接在函数空间上放置分布，无需经过参数，与贝叶斯线性回归和核方法密切相关
- **稀疏先验**：拉普拉斯先验等价于 L1 正则化（LASSO），适用于 $N \ll D$ 的欠定问题，促进变量选择










