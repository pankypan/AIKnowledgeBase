# Density Estimation with Gaussian Mixture Models(高斯混合模型与密度估计)

## 11.1 高斯混合模型

**密度估计**的核心思想：用参数族中的密度函数（如高斯分布）紧凑地表示数据。单个高斯分布建模能力有限，无法捕捉多峰数据结构。

**混合模型**通过 $K$ 个基础分布的凸组合来描述更复杂的分布：

$$
p(\boldsymbol{x}) = \sum_{k=1}^{K} \pi_k \, p_k(\boldsymbol{x}), \quad 0 \leqslant \pi_k \leqslant 1, \quad \sum_{k=1}^{K} \pi_k = 1 \tag{11.1-11.2}
$$

其中 $\pi_k$ 为**混合权重**，$p_k$ 为基础分布（如高斯、伯努利、伽马分布）。

**定义（高斯混合模型，GMM）**：将 $K$ 个高斯分布组合：

$$
p(\boldsymbol{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \tag{11.3}
$$

其中参数集合 $\boldsymbol{\theta} := \{\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k : k = 1, \dots, K\}$。当 $K=1$ 时退化为单个高斯分布。



## 11.2 通过最大似然估计学习参数

给定独立同分布数据集 $\mathcal{X} = \{\boldsymbol{x}_1, \dots, \boldsymbol{x}_N\}$，对数似然函数为：

$$
\mathcal{L} = \log p(\mathcal{X} \mid \boldsymbol{\theta}) = \sum_{n=1}^{N} \log \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \tag{11.10}
$$

由于 $\log$ 内含求和，无法像单高斯情形那样得到封闭解，只能通过迭代求解。

### 11.2.1 责任度（Responsibilities）

对 $\mathcal{L}$ 关于各参数求导时，会反复出现以下关键量：

$$
r_{n,k} := \frac{\pi_k \, \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)} \tag{11.17}
$$

$r_{n,k}$ 称为第 $k$ 个分量对数据点 $\boldsymbol{x}_n$ 的**责任度**，表示 $\boldsymbol{x}_n$ 由第 $k$ 个分量生成的归一化概率。$r_{n,k} \in [0,1]$ 且 $\sum_k r_{n,k} = 1$。

定义第 $k$ 个分量的**有效样本数**：

$$
N_k := \sum_{n=1}^{N} r_{n,k} \tag{11.24}
$$

### 11.2.2 更新均值向量

令 $\partial \mathcal{L} / \partial \boldsymbol{\mu}_k = \boldsymbol{0}$，可得：

$$
\boldsymbol{\mu}_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^{N} r_{n,k} \, \boldsymbol{x}_n \tag{11.20}
$$

即各分量的均值更新为以责任度为权重的**加权平均**。

### 11.2.3 更新协方差矩阵

令 $\partial \mathcal{L} / \partial \boldsymbol{\Sigma}_k = \boldsymbol{0}$，可得：

$$
\boldsymbol{\Sigma}_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^{N} r_{n,k} (\boldsymbol{x}_n - \boldsymbol{\mu}_k)(\boldsymbol{x}_n - \boldsymbol{\mu}_k)^\top \tag{11.30}
$$

即以责任度加权的**经验协方差**。

### 11.2.4 更新混合权重

引入 Lagrange 乘子处理约束 $\sum_k \pi_k = 1$，求解得：

$$
\pi_k^{\text{new}} = \frac{N_k}{N} \tag{11.42}
$$

即第 $k$ 个分量的混合权重等于其有效样本数占总样本数的比例。

> **关键观察**：三组更新公式都依赖于责任度 $r_{n,k}$，而 $r_{n,k}$ 本身又依赖于当前的参数 $\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k, \pi_k$。因此参数之间相互耦合，无法一步求解，必须采用迭代方法。♦



## 11.3 EM 算法

**期望最大化（Expectation-Maximization, EM）**算法是求解 GMM 参数的标准迭代方法，交替执行两步：

1. **E 步（Expectation）**：用当前参数计算责任度（后验概率）：

$$
r_{n,k} = \frac{\pi_k \, \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j} \pi_j \, \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)} \tag{11.53}
$$

2. **M 步（Maximization）**：用责任度更新所有参数：

$$
\begin{align}
\boldsymbol{\mu}_k &\leftarrow \frac{1}{N_k} \sum_{n=1}^{N} r_{n,k} \, \boldsymbol{x}_n \tag{11.54} \\
\boldsymbol{\Sigma}_k &\leftarrow \frac{1}{N_k} \sum_{n=1}^{N} r_{n,k} (\boldsymbol{x}_n - \boldsymbol{\mu}_k)(\boldsymbol{x}_n - \boldsymbol{\mu}_k)^\top \tag{11.55} \\
\pi_k &\leftarrow \frac{N_k}{N} \tag{11.56}
\end{align}
$$

EM 算法的核心性质：
- 每次迭代**保证对数似然不减**
- 收敛至**局部最优**（不保证全局最优）
- 对**初始化敏感**：不同初始参数可能收敛到不同结果，实践中常多次随机初始化取最优



## 11.4 隐变量的视角

从**隐变量（latent variable）**的角度重新诠释 GMM，可以更深入地理解模型结构和 EM 算法的本质。

### 11.4.1 生成过程与概率模型

引入离散隐变量 $z \in \{1, \dots, K\}$（one-hot 编码 $\boldsymbol{z} \in \{0,1\}^K$），GMM 的数据生成过程为：

1. **选择分量**：按先验分布采样 $z_k = 1$，概率为 $p(z_k = 1) = \pi_k$
2. **生成数据**：从对应分量采样 $\boldsymbol{x} \sim \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$

**联合分布**：

$$
p(\boldsymbol{x}, z_k = 1) = p(\boldsymbol{x} \mid z_k = 1) \, p(z_k = 1) = \pi_k \, \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \tag{11.61}
$$

### 11.4.2 似然函数

对隐变量求边缘化，恢复混合模型：

$$
p(\boldsymbol{x} \mid \boldsymbol{\theta}) = \sum_{k=1}^{K} p(\boldsymbol{x} \mid \boldsymbol{\theta}, z_k = 1) \, p(z_k = 1 \mid \boldsymbol{\theta}) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \tag{11.66}
$$

这表明 GMM 可以看作一个**边缘似然**：对隐变量的所有可能取值求和，消除了隐变量。

完整数据集的似然：

$$
p(\mathcal{X} \mid \boldsymbol{\theta}) = \prod_{n=1}^{N} \sum_{k=1}^{K} \pi_k \, \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \tag{11.67}
$$

### 11.4.3 后验分布

利用 Bayes 定理，给定观测 $\boldsymbol{x}$ 计算隐变量的后验：

$$
p(z_k = 1 \mid \boldsymbol{x}) = \frac{\pi_k \, \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(\boldsymbol{x} \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)} = r_{n,k} \tag{11.69}
$$

这正是 E 步中计算的**责任度** $r_{n,k}$——它是隐变量的后验概率，表示"给定数据点 $\boldsymbol{x}_n$，它属于第 $k$ 个分量的概率"。

### 11.4.4 推广至整个数据集

为每个数据点 $\boldsymbol{x}_n$ 引入独立的隐变量 $\boldsymbol{z}_n = [z_{n,1}, \dots, z_{n,K}]^\top$。假设数据点在给定各自隐变量后条件独立：

$$
p(\boldsymbol{x}_1, \dots, \boldsymbol{x}_N \mid \boldsymbol{z}_1, \dots, \boldsymbol{z}_N) = \prod_{n=1}^{N} p(\boldsymbol{x}_n \mid \boldsymbol{z}_n) \tag{11.71}
$$

### 11.4.5 从隐变量视角重访 EM 算法

在隐变量框架下，EM 算法的目标函数为**期望完整数据对数似然**：

$$
Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)}) = \mathbb{E}_{\boldsymbol{z} \mid \boldsymbol{x}, \boldsymbol{\theta}^{(t)}} [\log p(\boldsymbol{x}, \boldsymbol{z} \mid \boldsymbol{\theta})] \tag{11.73}
$$

- **E 步**：固定参数 $\boldsymbol{\theta}^{(t)}$，计算隐变量后验 $p(\boldsymbol{z} \mid \boldsymbol{x}, \boldsymbol{\theta}^{(t)})$，即构造 $Q$ 函数
- **M 步**：最大化 $Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)})$，更新参数 $\boldsymbol{\theta}^{(t+1)} = \arg\max_{\boldsymbol{\theta}} Q(\boldsymbol{\theta} \mid \boldsymbol{\theta}^{(t)})$

> **隐变量视角的意义**：将"log 里有 sum"的难优化问题转化为"对完整数据 $(x,z)$ 的 log 求期望"，后者通常可以解析求解。EM 算法是处理含隐变量模型的通用框架，不限于 GMM。♦



## 11.5 拓展阅读

GMM 之外的密度估计方法包括：

- **直方图法**：将数据空间分区为 bins，用频率估计密度。简单直观但受 bin 宽度和位置选择影响，且在高维空间遭遇维数灾难。

- **核密度估计（KDE）**：以每个数据点为中心放置核函数，密度估计为：

$$
p(\boldsymbol{x}) = \frac{1}{Nh} \sum_{n=1}^{N} k\left(\frac{\boldsymbol{x} - \boldsymbol{x}_n}{h}\right) \tag{11.74}
$$

其中 $k(\cdot)$ 为核函数（如高斯核），$h$ 为带宽参数。KDE 是一种**非参数**方法，不假设数据服从特定分布族。

- GMM 可视为介于参数方法（单高斯）和非参数方法（KDE）之间的半参数方法：分量数 $K$ 固定时是参数模型，$K$ 增大时逼近非参数估计。
