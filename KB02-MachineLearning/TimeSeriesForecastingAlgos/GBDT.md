# GBDT

## GBDT基础

**Gradient Boosting Decision Tree = Gradient Descent + Boosting + Decision Tree**


### CART

CART树可以分为回归树和决策树。GBDT中的树是CART回归树。
- **CART分类树**：样本输出是离散值
- **CART回归树**：样本输出是连续值

<img src=".\assets\screenshot-20250526-174350.png">


**CART回归树模型可表示为:**
$$
f(x)=\sum_{m=1}^M c_m I(x\in R_m)
$$
- $M$: 数据集划分成M个单元
- $c_m$: 第M个单元的**输出值**
- $I(x \in R_m)$: 指示函数, 若 $x \in R_m$ 则 $I =1$; 否则 $I =0$;



### 集成学习

集成学习（Ensemble Learning）通过构建并结合多个学习器来完成学习任务。根据个体学习器的生成方式，可以分为两大类：Bagging和Boosting。
- **Bagging**
  - 个体学习器间不存在强依赖关系、可同时生成的并行化方法。
  - 代表方法:RF随机森林
- **Boosting**
  - 个体学习器间存在强依赖关系、必须串行生成的序列化方法。
  - 代表方法:AdaBoost、GBDT、XGBoost

---

**Boosting的工作机制:**

- 第一步:先从初始训练集训练出一个基学习器;
- 第二步:再根据基学习器的表现对训练样本分布进行调整, **使得先前基学习器做错的训练样本在后续受到更多关注**;
- 第三步:然后基于调整后的样本分布来训练下一个基学习器;
- 第四步:如此重复进行,直至基学习器数目达到事先指定的值T;
- 第五步:最终将这T个基学习器进行加权结合。


### 梯度下降

**梯度的定义:**

设函数f(x)具有一阶连续偏导数,则f(x)在xo处的一阶泰勒展开:
$$
f(x)\approx f(x_0)+f^{\prime}(x_0)(x-x_0)
$$
这里, $f'(x_0)$ 为 $f(x)$ 在 $x_0$ 处的梯度。
- 每个位置(点)都有梯度
- 负梯度方向是使函数值下降最快的方向

---

**梯度下降的定义:**

顺着函数 $y=f(x)$ 当前点对应的梯度的反方向,按照规定步长进行选代搜索

**梯度下降公式**:
$$
x_{t+1}=x_t-\eta \nabla f(x_t)
$$
- $x_{t+1}$: 第 $t+1$ 次迭代后的自变量
- $x_t$: 第 $t$ 次迭代后的自变量
- $\eta$: 步长, 也称为学习率
- $\nabla f(x_t)$: 函数 $f(x)$ 在 $x_t$ 处的梯度


## GBDT原理

### 提升树原理

**提升树**

以决策树(分类树或者回归树)为基函数的提升方法称为**提升决策树(Boosting Decision Tree)**, 简称**提升树(Boosting Tree)**。

**提升树模型**

提升树模型可以表示为决策树的加法模型:
$$
f_M(x)=\sum_{m=1}^MT_m(x)
$$
其中, $T_m(x)$ 表示第 $m$ 棵决策树, $M$ 为树的个数。

---


**回归问题的提升树算法如下：**

输入：训练数据集 $T=\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{n},y_{n})\},x_{i}\in X\subseteq R^{n},y_{i}\in Y\subseteq R;$

输出：提升树 $f_M(x)$

1. 初始化 $f_0(x)=0$
2. 对 $m=1,2,...,M$
   1. 计算残差 (残差 = 真实值 - 预测值):
   $$
   r_{mi}=y_{i}-f_{m-1}(x_{i}),i=1,2,...,N
   $$
   2. 拟合残差 $r_{mi}$ 学习一个回归树, 得到 $T_m(x)$
   3. 更新 $f_m(x)=f_{m-1}(x)+T_m(x)$ (**加法模型** + **向前分步算法** 实现训练)
3. 得到回归问题提升树
    $$
    f_M(x)=f_0(x)+\sum_{m=1}^M T_m(x)
    $$


---

**提升树的训练优化目标如下:**
$$
\min_{s}\left[\min_{c_{1}}\sum_{x_{j}\in R_{1}}(y_{i}-c_{1})^{2}+\min_{c_{2}}\sum_{x_{i}\in R_{2}}(y_{i}-c_{2})^{2}\right]
$$
其中：
- $s$ 表示切分点；$R_1$ 表示切分点左边的数据集；$R_2$ 表示切分点右边的数据集
  $$
  R_1(i,s)=\{x|x_{i}\leq s\},R_2(i,s)=\{x|x_{i}>s\}
  $$
- $c_1$ 表示切分点左边的输出值; $c_2$ 表示切分点右边的输出值
  $$
  c_1=\frac{1}{N_1}\sum_{x_{i}\in R_{1}}y_{i},c_2=\frac{1}{N_2}\sum_{x_{i}\in R_{2}}y_{i}
  $$ 



### 提升树例子

<img src=".\assets\提升树例子01.png">


**求训练数据的切分点：**

<img src="./assets/提升树求解00_updated.png" width="600">

求解提升树：

- 提升树训练第一步：求第1棵树
  
  <img src=".\assets\提升树求解01.png" width="600">
- 提升树训练第二步：求第2棵树
  
  <img src=".\assets\提升树求解02.png" width="600">
- 提升树训练第三步：求第3棵树
  
  <img src=".\assets\提升树求解03.png" width="600">
- 提升树训练第四步：求第4棵树
  
  <img src=".\assets\提升树求解04.png" width="600">
- 提升树训练第五步：求第5棵树
  
  <img src=".\assets\提升树求解05.png" width="600">
- 提升树训练第六步：求第6棵树
  
  <img src=".\assets\提升树求解06.png" width="600">
- 提升树训练第七步：确定提升树模型
  
  <img src=".\assets\提升树求解07.png" width="600">



### 提升树的Python实现

**具体代码详见 `./boosting_decision_tree.py`**


### GBDT残差

- **Boosting Tree的残差**: 残差(Residual)是真实值与预测值的差。
  - 残差公式：$r_{3i}=r_{2i}-T_{2}(x_{i})$
  - 上一轮的预测结果的残差作为当前的训练数据集
- **GBDT的残差**: 用**负梯度**近似模拟残差。(损失函数 $L(y_i,f(x_i))$ 对树 $f(x_i)$ 的梯度)
  - 损失函数是 **指数损失、平方损失**, 每一步容易优化。
  - **一般损失函数**, 每一步优化并不那么容易, 所以用负梯度近似逼近残差。


### GBDT原理

梯度提升决策树(Gradient Boosting Decision Tree，GBDT)，2001年，Jerome H. Friedman在论文《 Greedy function approximation : A gradient boosting machine》中提出。

**核心思想：用负梯度近似模拟残差**

---
**梯度提升树算法（GBDT）算法如下：**

输入：训练数据集 $T=\{(x_{1},y_{1}),(x_{2},y_{2}),\cdots,(x_{n},y_{n})\},x_{i}\in X\subseteq R^{n},y_{i}\in Y\subseteq R;$ 损失函数 $L(y,f(x))$;

输出：回归树 $\hat{f}(x)$

**第一步:初始化弱学习器**
$$
f_0(x)=\arg\min_c\sum_{i=1}^NL(y_i,c)
$$
假设取损失函数为平方损失。因为平方损失函数是一个凸函数,直接对时 $c$ 求导:
$$
\sum_{i=1}^N\frac{\partial L(y_i,c)}{\partial c}=\sum_{i=1}^N\frac{\partial(\frac{1}{2}(y_i-c)^2)}{\partial c}=\sum_{i=1}^N(c-y_i)
$$
令导数等于0,得:
$$
c=\sum_{i=1}^Ny_i/N
$$
所以初始化时,c取值为所有训练样本标签值的均值。此时得到奶始学习器:
$$
f_0(x)=c
$$


**第二步:迭代训练 $m=1,2,..M$ 裸树**
1. 对每个样本 $i=1,2,...N$ ,计算负梯度,即残差:
   $$
   r_{mi}=-\left[\frac{\partial L(y_{i},f(x_{i}))}{\partial f(x_{i})}\right]_{fx)=f_{m-1}(x)}
   $$
2. 将上步 (1) 得到的残差 $r_{mi}$ 作为样本新的真实值, **并将数据 $(x_i, r_{im}),i=1,2,...N$作为下棵树的训练数据**,  得到一颗新回归树, 其对应的叶子节点区域 $R_{mj}, j=1,2,...,J$。其中 $J$ 为回归树的叶子结点的个数。
3. 对叶子区域 $j=1,2,...,J$ 计算最佳拟合值: ($c_{mj}$ 是 $R_{mj}$ 的平方损失最小值)
   $$
   c_{mj}=\arg\min_{c}\sum_{x_{i}\in R_{mj}}L(y_{i},f_{m-1}(x_{i})+c)
   $$
4. 更新强学习器: ($I$ 指示函数，若 $x\in R_{mj}$ 则 $I=1$，否则 $I=0$)
   $$
   f_{m}(x)=f_{m-1}(x)+\sum_{j=1}^{J}c_{mj}I(x\in R_{mj})
   $$


**第三步：得到最终学习器GBDT**
$$
\hat{f}(x)=f_{M}(x)=f_{0}(x)+\sum_{m=1}^{M}\sum_{j=1}^{J}c_{mj}I(x\in R_{mj})
$$



[Gradient Boosting in ML](https://www.geeksforgeeks.org/ml-gradient-boosting/)

<img src="./assets/GradientBoostedTrees.png">


### GBDT例子

<img src=".\assets\GBDT例子01.png" width="600">


例1的GBDT训练第1棵树：

<img src=".\assets\GBDT例子02.png" width="600">

类似的，例1的GBDT训练第6棵树：

<img src=".\assets\GBDT例子03.png" width="600">


## GBDT应用

GBDT可用于特征组合、二分类、多分类等。
1. GBDT特征组合，更多内容可参考：Facebook的论文[《Practical Lessons from Predicting Clicks on Ads at Facebook》](https://quinonero.net/Publications/predicting-clicks-facebook.pdf)
2. GBDT二分类
   
   <img src="./assets/GBDT用于二分类.png" width="600">
3. GBDT多分类
   
   <img src="./assets/GBDT用于多分类.png" width="600">



## GBDT总结


**GBDT的优缺点**：
- 优点：
  - 灵活处理各种类型的数据(连续、离散)
  - 易于特征组合、特征选择
  - 相对少调参,预测精度高
- 缺点：
  - 串行生成,并行难
  - 数据维度高,计算复杂度高
<br>


**GBDT VS RF (Random Forest)**：
- **集成学习**: 都是集成学习方法,GBDT是Boosting思想,RF是Bagging思想
- **树的类型**: 都是由多棵树组成, GBDT是CART回归树, RF分类树、回归树都可以以
- **并行化**: GBDT的树只能顺序生成, RF的树可以并行生成
- **优化指标**: GBDT是偏差优化, RF是方差优化
- **训练样本**: GBDT是每次全样本训练, RF有放回抽样训练
- **最终结果**: GBDT是多棵树累加之和, RF是多棵树进行多数表决
<br>


**GBDT的演化**

**DT -> Boosting -> GBDT -> XGBoost**
| 模型 | 原始论文 | 年份 |
| --- | --- | --- |
| DT | Classification and Regression Trees | 1984 |
| Boosting | A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting | 1995 |
| GBDT | Greedy function approximation: A gradient boosting machine | 2001 |
| XGBoost | XGBoost: A Scalable Tree Boosting System | 2016 |

<br>
<br>

参考链接：[GBDT的原理](https://zhuanlan.zhihu.com/p/280222403)

