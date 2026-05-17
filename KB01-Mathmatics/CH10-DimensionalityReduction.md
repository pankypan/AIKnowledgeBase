# Dimensionality Reduction(降维)

高维数据往往具有冗余性和内在的低维结构。降维利用这种结构，用更紧凑的表示替代原始数据，理想情况下不丢失关键信息——类似 JPEG/MP3 的有损压缩思想。

本章核心算法是**主成分分析（PCA）**，由 Pearson (1901) 和 Hotelling (1933) 提出，是最经典的线性降维方法，也称为 Karhunen-Loève 变换。PCA 的推导综合运用了基变换（§2.6-2.7）、投影（§3.8）、特征值分解（§4.2）、高斯分布（§6.5）和约束优化（§7.2）。



## 10.1 问题设定

给定独立同分布数据集 $\mathcal{X} = \{x_1, \ldots, x_N\}$，$x_n \in \mathbb{R}^D$，均值为 $\boldsymbol{0}$（预先中心化），数据协方差矩阵为

$$
S = \frac{1}{N} \sum_{n=1}^{N} x_n x_n^\top \tag{10.1}
$$

目标：找到低维压缩表示（编码）

$$
z_n = B^\top x_n \in \mathbb{R}^M \tag{10.2}
$$

其中投影矩阵 $B = [b_1, \ldots, b_M] \in \mathbb{R}^{D \times M}$，$B$ 的列标准正交（$b_i^\top b_j = \delta_{ij}$）。

PCA 的编码-解码框架：
- **编码器**：$z = B^\top x$（从 $\mathbb{R}^D$ 压缩到 $\mathbb{R}^M$）
- **解码器**：$\tilde{x} = Bz$（从 $\mathbb{R}^M$ 重构回 $\mathbb{R}^D$）

目标是找到最优的 $B$，使得重构 $\tilde{x}_n$ 尽可能接近原始 $x_n$。



## 10.2 最大方差视角

核心思想：将"保留信息"等价为"保留最大方差"。方差越大，数据的散布越充分，携带的信息越多。

### 10.2.1 第一主成分

数据投影到方向 $b_1$ 上的方差为

$$
V_1 = b_1^\top S b_1 \tag{10.9b}
$$

约束 $\|b_1\|^2 = 1$ 下最大化 $V_1$，构造 Lagrange 函数

$$
\mathfrak{L}(b_1, \lambda_1) = b_1^\top S b_1 + \lambda_1(1 - b_1^\top b_1) \tag{10.11}
$$

令偏导为零，得到

$$
S b_1 = \lambda_1 b_1 \tag{10.13}
$$

即 $b_1$ 是数据协方差矩阵 $S$ 的**特征向量**，$\lambda_1$ 是对应的**特征值**。此时投影方差为

$$
V_1 = \lambda_1 \tag{10.15}
$$

因此应选择 $S$ 的**最大特征值**对应的特征向量作为第一主成分。

### 10.2.2 $M$ 维主子空间

依次递推：从数据中减去前 $m-1$ 个主成分的影响后，第 $m$ 个主成分仍是 $S$ 的特征向量，且与第 $m$ 大特征值关联。

关键结论：$S$ 和去除前 $m-1$ 个主成分后的协方差矩阵 $\hat{S}$ 具有**相同的特征向量集**。差别仅在于：属于前 $m-1$ 个主成分的特征向量在 $\hat{S}$ 中对应特征值为 0。

PCA 用前 $M$ 个主成分捕获的最大方差为

$$
V_M = \sum_{m=1}^{M} \lambda_m \tag{10.24}
$$

相对捕获方差 $\frac{V_M}{V_D}$ 和相对损失 $1 - \frac{V_M}{V_D}$ 可用于评估压缩质量。



## 10.3 投影视角

核心思想：直接最小化平均平方重构误差，从而将 PCA 解释为**最优线性自编码器**。

### 10.3.1 目标函数

$$
J_M = \frac{1}{N} \sum_{n=1}^{N} \|x_n - \tilde{x}_n\|^2 \tag{10.29}
$$

### 10.3.2 最优坐标

给定标准正交基 $(b_1, \ldots, b_M)$，对 $J_M$ 关于坐标 $z_{in}$ 求导并令其为零，得到

$$
z_{in} = b_i^\top x_n \tag{10.32}
$$

即最优坐标就是 $x_n$ 在各基向量上的**正交投影坐标**。投影结果为

$$
\tilde{x}_n = BB^\top x_n \tag{10.34}
$$

### 10.3.3 最优基

位移向量 $x_n - \tilde{x}_n$ 恰好是数据点在主子空间正交补上的投影：

$$
x_n - \tilde{x}_n = \sum_{j=M+1}^{D} (b_j^\top x_n) b_j \tag{10.38b}
$$

将损失函数化简为

$$
J_M = \sum_{j=M+1}^{D} b_j^\top S b_j = \sum_{j=M+1}^{D} \lambda_j \tag{10.44}
$$

最小化 $J_M$ 等价于选择最小的 $D - M$ 个特征值对应的特征向量作为正交补的基，即**主子空间由 $S$ 的前 $M$ 个最大特征值对应的特征向量张成**。

> **两种视角的等价性**：最大化投影方差 $V_M = \sum_{m=1}^M \lambda_m$ 与最小化重构误差 $J_M = \sum_{j=M+1}^D \lambda_j$ 完全等价（$V_M + J_M = \text{tr}(S) = \text{const}$）。



## 10.4 特征向量计算与低秩近似

数据协方差矩阵与数据矩阵的关系：

$$
S = \frac{1}{N} X X^\top, \quad X = [x_1, \ldots, x_N] \in \mathbb{R}^{D \times N} \tag{10.45-10.46}
$$

### SVD 与 PCA 的联系

对 $X$ 做奇异值分解 $X = U \Sigma V^\top$，则

$$
S = \frac{1}{N} U \Sigma \Sigma^\top U^\top \tag{10.48}
$$

- $U$ 的列即为 $S$ 的特征向量（主成分方向）
- $S$ 的特征值与 $X$ 的奇异值关系为 $\lambda_d = \sigma_d^2 / N$ \hspace{1em} (10.49)

### 截断 SVD 实现 PCA

由 Eckart-Young 定理，$X$ 的最佳秩 $M$ 近似为

$$
\tilde{X}_M = U_M \Sigma_M V_M^\top \in \mathbb{R}^{D \times N} \tag{10.51}
$$

截取前 $M$ 个最大奇异值及对应的左/右奇异向量即可。

### 实际计算

- 对于 $>4 \times 4$ 的矩阵，无法用特征多项式求根（Abel-Ruffini 定理），需使用**迭代方法**
- 若仅需少数特征向量，**幂迭代**方法非常高效：

$$
x_{k+1} = \frac{S x_k}{\|S x_k\|}, \quad k = 0, 1, \ldots \tag{10.52}
$$

该序列收敛到 $S$ 的最大特征值对应的特征向量（Google PageRank 即基于此思想）。



## 10.5 高维 PCA

当 $N \ll D$（数据点数远小于维度）时，直接计算 $D \times D$ 的 $S$ 代价为 $O(D^3)$，不可行。

**核心技巧**：转而计算 $\frac{1}{N} X^\top X \in \mathbb{R}^{N \times N}$ 的特征分解。

从特征方程 $\frac{1}{N} X X^\top b_m = \lambda_m b_m$ 出发，左乘 $X^\top$ 得

$$
\frac{1}{N} X^\top X \, c_m = \lambda_m c_m, \quad c_m := X^\top b_m \tag{10.56}
$$

- $\frac{1}{N} X^\top X$ 与 $S$ 具有相同的非零特征值
- 从 $c_m$ 恢复 $S$ 的特征向量：$b_m \propto X c_m$（需归一化至单位长度）

这样特征分解的规模从 $D \times D$ 降为 $N \times N$，在 $N \ll D$ 时大幅提升效率。



## 10.6 PCA 实践步骤

1. **均值归零**：计算均值 $\mu$，令 $x_n \leftarrow x_n - \mu$

2. **标准化**：对每个维度除以标准差 $\sigma_d$，使各轴方差为 1

3. **协方差矩阵的特征分解**：计算 $S$ 的特征值和正交归一的特征向量

4. **投影**：对新数据点 $x_*$ 先标准化

$$
x_*^{(d)} \leftarrow \frac{x_*^{(d)} - \mu_d}{\sigma_d} \tag{10.58}
$$

再投影到主子空间

$$
\tilde{x}_* = BB^\top x_*, \quad z_* = B^\top x_* \tag{10.59-10.60}
$$

PCA 返回的是低维坐标 $z_*$。若需回到原始数据空间，需撤销标准化：

$$
\tilde{x}_*^{(d)} \leftarrow \tilde{x}_*^{(d)} \sigma_d + \mu_d \tag{10.61}
$$



## 10.7 潜在变量视角（概率 PCA）

概率 PCA（PPCA）引入连续潜在变量 $z \in \mathbb{R}^M$ 建立概率生成模型，相比确定性 PCA 额外提供了：似然函数、贝叶斯模型选择、生成新数据、处理缺失数据等能力。

### 10.7.1 生成模型

$$
z \sim \mathcal{N}(\boldsymbol{0}, I), \quad x \mid z \sim \mathcal{N}(Bz + \mu, \, \sigma^2 I) \tag{10.63-10.66}
$$

即观测 $x$ 由低维潜在原因 $z$ 经线性映射加噪声生成。

### 10.7.2 似然与联合分布

边缘化潜在变量后，似然函数为

$$
p(x \mid B, \mu, \sigma^2) = \mathcal{N}\big(x \mid \mu, \, BB^\top + \sigma^2 I\big) \tag{10.68-10.70}
$$

联合分布为

$$
p(x, z) = \mathcal{N}\left(\begin{bmatrix} x \\ z \end{bmatrix} \bigg| \begin{bmatrix} \mu \\ \boldsymbol{0} \end{bmatrix}, \begin{bmatrix} BB^\top + \sigma^2 I & B \\ B^\top & I \end{bmatrix}\right) \tag{10.72}
$$

### 10.7.3 后验分布

给定观测 $x$，潜在变量的后验为

$$
p(z \mid x) = \mathcal{N}(z \mid m, C) \tag{10.73}
$$

$$
m = B^\top (BB^\top + \sigma^2 I)^{-1}(x - \mu) \tag{10.74}
$$

$$
C = I - B^\top (BB^\top + \sigma^2 I)^{-1} B \tag{10.75}
$$

后验协方差 $C$ 不依赖于观测数据 $x$。$C$ 的行列式越小，潜在嵌入越确定；方差大则可能是异常点。



## 10.8 PCA 的扩展与联系

**线性自编码器视角**：PCA 等价于最小化平方自编码损失 $\frac{1}{N}\sum_n \|x_n - BB^\top x_n\|^2$。若用非线性映射（如深度神经网络）替代线性映射，则得到**深度自编码器**。

**PPCA 的最大似然估计**（Tipping & Bishop, 1999）：

$$
\mu_\text{ML} = \frac{1}{N}\sum_{n=1}^N x_n, \quad B_\text{ML} = T(\Lambda - \sigma^2 I)^{1/2} R, \quad \sigma^2_\text{ML} = \frac{1}{D-M}\sum_{j=M+1}^D \lambda_j \tag{10.77-10.79}
$$

其中 $T$ 为前 $M$ 个特征向量矩阵，$\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_M)$，$R$ 为任意正交矩阵。$\sigma^2_\text{ML}$ 的含义是主子空间正交补中的平均剩余方差。

**无噪声极限**：$\sigma \to 0$ 时 PPCA 退化为标准 PCA。

**相关方法对比**：

| 方法 | 先验 $p(z)$ | 噪声模型 | 特点 |
|---|---|---|---|
| PCA | — | 无 | 确定性线性降维 |
| PPCA | $\mathcal{N}(0, I)$ | 各向同性 $\sigma^2 I$ | 概率生成模型 |
| 因子分析 (FA) | $\mathcal{N}(0, I)$ | 对角 $\text{diag}(\sigma_1^2, \ldots, \sigma_D^2)$ | 各维噪声不同，无封闭解 |
| ICA | 非高斯 | $\sigma^2 I$ | 用于盲源分离 |

**非线性扩展**：
- **核 PCA**：利用核技巧隐式处理无限维特征空间
- **GP-LVM**：用高斯过程替代线性映射，边缘化模型参数
- **深度自编码器**：用深度神经网络实现非线性编码/解码
