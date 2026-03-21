# XGBoost

**XGBoost（eXtreme Gradient Boosting）极致梯度提升，是一种基于GBDT的算法或者说工程实现。**
- XGBoost的基本思想和GBDT相同，但是做了一些优化，比如二阶导数使损失函数更精准；
- 正则项避免树过拟合；
- Block存储可以并行计算等。


## 目标函数

**目标函数(objective function)**: 损失函数 + 正则项
$$
J(\theta) = L(\theta) + \Omega(f, \theta)
$$

1. **损失函数 $L(\theta)$**:
1.1 回归问题
  $$
  L(\theta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$
1.2 分类问题
  $$
  L(\theta)=\sum_i[y_i\ln(1+e^{-\hat{y_i}})+(1-y_i)\ln(1+e^{\hat{y_i}})]
  $$
3. **正则项 $\Omega(f, \theta)$**:
2.1 控制模型的复杂度
2.2 常用的正则化就是 $L2$ 正则，也就是所有参数的平方和。我们希望这个和尽可能小的同时，模型对训练数据有尽可能好的预测。
  $$
  \Omega(f, \theta) = \sum_{i=1}^{n} \theta_i^2
  $$



## XGBoost

**XGBoost对应的模型是一堆CART树。**

CART树和一堆CART树的示例，用来判断一个人是否会喜欢计算机游戏：
- 单棵CART树的示例：<img src="https://upload-images.jianshu.io/upload_images/1371984-a90c565a27c9874d.PNG?imageMogr2/auto-orient/strip|imageView2/2/w/835/format/webp">
- 一堆CART树的示例：<img src="https://upload-images.jianshu.io/upload_images/1371984-bbe17b3b253a6d1a.PNG?imageMogr2/auto-orient/strip|imageView2/2/w/901/format/webp">



### XGBoost模型

**XGBoost的模型可以表示为：**
$$
\hat{y}_i = \sum_{k=1}^{K} f_k(x_i), f_k \in \mathcal{F}
$$

- $K$: 表示树的个数
- $f_k$: 表示第 $k$ 棵树
- $\mathcal{F}$: 表示所有可能的树的集合


### XGBoost的目标函数

**XGBoost模型的目标函数为：**
$$
J(\theta) = \sum_{i=1}^{n} l(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
$$

- $l$: 表示损失函数
- $\Omega$: 表示正则项


## 训练XGBoost
获取了xgboost模型和它的目标函数，那么训练的任务就是通过最小化目标函数来找到最佳的参数组。 XGBoost模型由CART树组成，参数自然存在于每棵CART树之中。

**每棵CART树的参数**包括：
- **树的结构**，这个结构负责将一个样本映射到一个确定的叶子节点上，其本质上就是一个函数
- **各个叶子节点上的分数**



## 加法训练

运用加法训练，我们的目标不再是直接优化整个目标函数，这已经被我们证明是行不通的。而是分步骤优化目标函数，首先优化第一棵树，完了之后再优化第二棵树，直至优化完K棵树。
$$
\begin{aligned}
 & \hat{y}_{i}^{(0)}=0 \\
 & \hat{y}_{i}^{(1)}=f_1(x_i)=\hat{y}_i^{(0)}+f_1(x_i) \\
 & \hat{y}_{i}^{(2)}=f_1(x_i)+f_2(x_i)=\hat{y}_i^{(1)}+f_2(x_i) \\
 & \mathrm{...} \\
 & \hat{y}_{i}^{(t)}=\sum_{k=1}^tf_k(x_i)=\hat{y}_i^{(t-1)}+f_t(x_i)
\end{aligned}
$$

在第 $t$ 步时，我们添加了一棵最优的CART树 $f_t$，这棵最优的CART树 $f_t$ 是怎么得来的呢？非常简单，就是在现有的 $t-1$ 棵树的基础上，使得目标函数最小的那棵CART树，如下所示：
$$
\begin{aligned}
\mathrm{obj}^{(t)} & =\sum_{i=1}^nl(y_i,\hat{y}_i^{(t)})+\sum_{i=1}^t\Omega(f_i) \\
 & =\sum_{i=1}^nl(y_i,\hat{y}_i^{(t-1)}+f_t(x_i))+\Omega(f_t)+constant
\end{aligned}
$$

- $constant$ 就是前 $t-1$ 棵树的复杂度

假如我们使用的损失函数是MSE，那么上述表达式会变成这个样子：
$$
\begin{aligned}
\mathrm{obj}^{(t)} & =\sum_{i=1}^n(y_i-\hat{y}_i^{(t-1)}-f_t(x_i))^2+\Omega(f_t)+constant \\
 & =\sum_{i=1}^n(y_i-\hat{y}_i^{(t-1)})^2-2\sum_{i=1}^n(y_i-\hat{y}_i^{(t-1)})f_t(x_i)+\sum_{i=1}^nf_t(x_i)^2+\Omega(f_t)+constant
\end{aligned}
$$

这个式子非常漂亮
- 含有 $f_t(x_i)$ 的一次式和二次式
- 而且一次式项的系数是残差
- **注意**：$f_t(x_i)$ 是什么？它其实就是 $f_t$ 的某个叶子节点的值。



但是对于其他的损失函数，我们未必能得出如此漂亮的式子，所以，**对于一般的损失函数，我们需要将其作泰勒二阶展开**，对于损失函数 $l(y_i, \hat{y}_i)$，在点 $\hat{y}_i^{(t-1)}$ 处的泰勒二阶展开为：
$$
l(y_i, \hat{y}_i) \approx l(y_i, \hat{y}_i^{(t-1)}) + g_i(\hat{y}_i - \hat{y}_i^{(t-1)}) + \frac{1}{2}h_i(\hat{y}_i - \hat{y}_i^{(t-1)})^2
$$

所以，整体目标函数在 $\hat{y}_i^{(t-1)}$ 处展开，如下所示：
$$
\mathrm{obj}^{(t)}=\sum_{i=1}^n[l(y_i,\hat{y}_i^{(t-1)})+g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)+constant
$$

其中：
- $g_i$ 是损失函数的一阶导数
  $$
  g_{i}=\partial_{\hat{y}_i^{(t-1)}}l(y_i,\hat{y}_i^{(t-1)})
  $$
- $h_i$ 是损失函数的二阶导数
  $$
  h_{i}=\partial_{\hat{y}_i^{(t-1)}}^2l(y_i,\hat{y}_i^{(t-1)})
  $$


现在我们来审视下这个式子，哪些是常量，哪些是变量
- 常量
  - $constant$ 是常量 (前 $t-1$ 棵树的复杂度)
  - $l(y_i,\hat{y}_i^{(t-1)})$ 是常量 (前 $t-1$ 棵树的损失函数值)
- 变量
  - 第 $t$ 棵CART树的一次式
  - 第 $t$ 棵CART树的二次式
  - 整棵树的正则化项

我们的目标是让这个目标函数最小化，常数项显然没有什么用，我们把它们去掉，就变成了下面这样：
$$
\mathrm{obj}^{(t)}=\sum_{i=1}^n[g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]+\Omega(f_t)
$$

1. 这些一次式和二次式的系数是 $g_i$ 和 $h_i$，$g_i$ 和 $h_i$ 可以并行地求出来。
2. 而且，$g_i$ 和 $h_i$ 是不依赖于损失函数的形式的，只要这个损失函数二次可微就可以了。



## 模型正则化项

先对CART树作另一番定义，如下所示：
$$
f_t(x)=w_{q(x)},w\in R^T,q:R^d\to\{1,2,\cdots,T\}.
$$

- 一棵树有 $T$ 个叶子节点，这 $T$ 个叶子节点的值组成了一个 $T$ 维向量 $w$
- $q(x)$ 是一个映射，用来将样本 $x$ 映射成 $1$ 到 $T$ 的某个值，也就是把它分到某个叶子节点，$q(x)$ 其实就代表了CART树的结构。
- $w_{q(x)}$ 表示样本 $x$ 落在第 $q(x)$ 个叶子节点上，代表这棵树对样本 $x$ 的预测值


有了这个定义，xgboost就使用了如下的**正则化项**：
$$
\Omega(f)=\gamma T+\frac{1}{2}\lambda\sum_{j=1}^Tw_j^2
$$

- $\gamma$ 越大，表示越希望获得结构简单的树，因为此时对较多叶子节点的树的惩罚越大。
- $\lambda$ 越大也是越希望获得结构简单的树。




## 见证奇迹的时刻

关于第 $t$ 棵树的优化目标已然很清晰，下面我们对它做如下变形:
$$
\begin{gathered}
Obj^{(t)}\approx\sum_{i=1}^{n}[g_{i}w_{q(x_{i})}+\frac{1}{2}h_{i}w_{q(x_{i})}^{2}]+\gamma T+\frac{1}{2}\lambda\sum_{j=1}^{T}w_{j}^{2} \\
=\sum_{j=1}^T[(\sum_{i\in I_j}g_i)w_j+\frac{1}{2}(\sum_{i\in I_j}h_i+\lambda)w_j^2]+\gamma T
\end{gathered}
$$

- $I_j$ 代表什么？它代表一个集合，集合中每个值代表一个训练样本的序号，整个集合就是被第 $t$ 棵CART树分到了第 $j$ 个叶子节点上的训练样本。

理解了这一点，再看这步转换，其实就是内外求和顺序的改变。
进一步，我们可以做如下简化：
$$
\begin{aligned}
G_j & =\sum_{i\in I_j}g_i \\
H_j & =\sum_{i\in I_j}h_i
\end{aligned}
$$

- $G_j$ 和 $H_j$ 分别表示第 $j$ 个叶子节点上所有样本的一阶和二阶导数和。
- 于是，目标函数可以简化为：
  $$
  \mathrm{obj}^{(t)}=\sum_{j=1}^T[G_jw_j+\frac{1}{2}(H_j+\lambda)w_j^2]+\gamma T
  $$

对于第 $t$ 棵CART树的某一个确定的结构（可用 $q(x)$ 表示），所有的 $G_j$ 和 $H_j$ 都是确定的。而且上式中各个叶子节点的值 $w_j$ 之间是互相独立的。上式其实就是一个简单的二次式，我们很容易求出各个叶子节点的最佳值以及此时目标函数的值。如下所示：
$$
\begin{aligned}
w_{j}^{*}=-\frac{G_{j}}{H_{j}+\lambda} \\
\mathrm{obj}^{*}=-\frac{1}{2}\sum_{j=1}^{T}\frac{G_{j}^{2}}{H_{j}+\lambda}+\gamma T
\end{aligned}
$$

- $obj^{*}$ 代表最优目标函数，它表示了这棵树的结构有多好，值越小，代表这样结构越好
- $obj^{*}$ 只和 $G_j$ 和 $H_j$ 和 $T$ 有关，而它们又只和树的结构 $q(x)$ 有关，与叶子节点的值可是半毛关系没有。

<img src="https://upload-images.jianshu.io/upload_images/1371984-c0a66c1b44b39725.PNG?imageMogr2/auto-orient/strip|imageView2/2/w/834/format/webp">



## 找出最优的树结构

问题是：树的结构近乎无限多，一个一个去测算它们的好坏程度，然后再取最好的显然是不现实的。所以，我们仍然需要采取一点策略，这就是逐步学习出最佳的树结构。

以上文提到过的判断一个人是否喜欢计算机游戏为例子。最简单的树结构就是一个节点的树。我们可以算出这棵单节点的树的好坏程度 $obj^{*}$。假设我们现在想按照**年龄**将这棵单节点树进行分叉，我们需要知道：
1. 按照年龄分是否有效，也就是是否减少了 $obj$ 的值
2. 如果可分，那么以哪个年龄值来分。

为了回答上面两个问题，我们可以将这一家五口人按照年龄做个排序。如下图所示：

<img src="https://upload-images.jianshu.io/upload_images/1371984-65a7904e5a7e5c74.PNG?imageMogr2/auto-orient/strip|imageView2/2/w/722/format/webp">

按照这个图从左至右扫描，我们就可以找出所有的切分点。对每一个确定的切分点，我们衡量切分好坏的标准如下：
$$
Gain=\frac{1}{2}\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma
$$

- $Gain$ 为切分前的 $obj_{before}^{*}$ 值 - 切分后的 $obj_{after}^{*}$ 值
- 如果 $Gain > 0$，并且值越大，表示切分后 $obj_{after}^{*}$ 越小于单节点的 $obj_{before}^{*}$，就越值得切分。
- $\gamma$ 在这里实际上是一个临界值，它的值越大，表示我们对切分后obj下降幅度要求越严。这个值也是可以在xgboost中设定的。

<br>


参考链接：
- [xgboost的原理详解](https://www.jianshu.com/p/7467e616f227)
- [XGBoost的原理、公式推导、Python实现和应用](https://zhuanlan.zhihu.com/p/162001079)


