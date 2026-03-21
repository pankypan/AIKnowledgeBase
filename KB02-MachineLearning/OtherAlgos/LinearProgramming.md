# 线性规划(Linear Programming, LP)

## 什么是线性规划
如果一个函数 $L(x)$满足可加性和齐次性两个条件，则表明该函数是线性的。
1. **可加性:** $L(x_1 + x_2) = L(x_1) + L(x_2)$
2. **齐次性:** $\lambda L(x) = L(\lambda x)$
<br>


线性规划是目标函数和约束条件都为线性表达式的数学规划。
一个函数是线性函数应当是这样的：
$$
f(X)=A^TX+b \\
=\sum_{i=0}^na_ix_i+b
$$
- 其中 $A$ 为列向量
- $X$ 为自变量的列向量
- $b$ 为常量值

线性规划问题的标准型一般为：
$$
minz=\sum_{j=1}^nc_jx_j  \quad(1)
$$

$$
s.t.
\begin{cases}
 & f_i(X) & \leq0,i & =1,2,...,m_i \\
 & h_k(X) & =0,k & =1,2,...o_k \\
 & lb\leq & x_j & \leq ub,j & =1,2,...,n & 
\end{cases}(2)
$$
- 目标函数(1), `minz` 表示最小化目标函数, `maxz` 表示最大化目标函数
- 约束条件(2), subject to 简写 `s.t.` 构成的集合为可行域
- 可行域中使目标函数(1)达到最小值的点称为最优解



## 线性规划的几何含义

如下优化问题就是一个线性规划问题：
$$
\begin{array}{r}
\operatorname{maximize} x_{1}+2 x_{2} \\
\text { subject to } x_{1}+x_{2} \leq 3 \\
x_{2} \leq 2 \\
x_{1} \geq 0 \\
x_{2} \geq 0
\end{array}
$$

把如上的线性规划问题的图画出来

<img src="https://pica.zhimg.com/v2-5516e9ea0c3838fd203e470bd07f20a4_1440w.jpg" width="600">

- 每一条约束对应图中一条直线
- 四条约束构成四条直线，而这四条直线围起来的区域（图中蓝色部分）就表示线性规划问题的可行域
- 从图中来看线性规划问题的含义就是在可行域中找到最优点 $(x_1^*,x_2^*)$ 使得目标函数 $x_1 + 2x_2$ 最大
- 从图中观察可得，最优点为 $(1, 2)$ ，所对应的目标函数值为 $1 + 2 * 2 = 5$。




## 实际问题案例
### 匹配问题
相亲男女嘉宾匹配问题，如下图所示：

<img src="https://picx.zhimg.com/v2-67caad615fa3253b0b8540f85150cdcd_1440w.jpg" width="600">

用数学模型来描述上述的相亲问题
定义如下决策变量：
$$
x_{ij} = \begin{cases} 1, & \text{男性 } i \text{ 和女性 } j \text{ 匹配} \\ 0, & \text{男性 } i \text{ 和女性 } j \text{ 不匹配} \end{cases}
$$
假设 $N$ 位男性和 $M$ 位女性来相亲
$$
\begin{aligned}
 & \max\sum_{i=1}^{N}\sum_{j=1}^{M}c_{ij}x_{ij}\quad(1) \\
\mathrm{s.t.} & \sum_{i=1}^{N}x_{ij}=1,\forall j=1,2,\ldots,N\quad(2) \\
 & \sum_{j=1}^{M}x_{ij}=1,\forall i=1,2,\ldots,M\quad(3)
\end{aligned}
$$
- $c_{ij}$ 表示男性 $i$ 和女性 $j$ 的匹配度，$c_{ij}$ 越大表示男女生越合适
- 目标函数 $(1)$ 表示让总的匹配度最大。
- 约束条件 $(2)$ 表示每个男性只能匹配一个女性
- 约束条件 $(3)$ 表示每个女性只能匹配一个男性



### 运输问题
现在假设你是一家工厂，你在全国各地有四个大型的工厂可以生产某种产品，这四个大型工厂分别为：西安，成都，东莞，哈尔滨。每个工厂都有一个产能的上限。全国有三家主要的销售地分别为：北京，上海，深圳。每个销售地对该产品有不同的需求。如下图所示：

<img src="https://pic2.zhimg.com/v2-5c09039afe4124a5064d2567876491bf_1440w.jpg" width="600">

用数学模型来描述上述的运输问题:
- 设某种产品有 $N$ 个工厂，$M$ 个销售地
- 每个工厂的生产能力上限为 $p_i$，每个销售地对产品的需求为 $d_j$
- 设 $c_{ij}$ 为从第 $i$ 个工厂到第 $j$ 个销售地的单位运输成本
- 设 $x_{ij}$ 为从第 $i$ 个工厂到第 $j$ 个销售地的运输量

数学模型如下：
$$
\begin{aligned}
 & \operatorname*{min}\sum_{i=1}^{N}\sum_{j=1}^{M}c_{ij}x_{ij}\quad(1) \\
\mathrm{s.t.} & \sum_{i=1}^{N}x_{ij}=d_{j},\forall j=1,2,\ldots,M\quad(2) \\
 & \sum_{j=1}^{M}x_{ij}\leq p_{i},\forall i=1,2,\ldots,N\quad(3)
\end{aligned}
$$
- 目标函数 (1) 表示总的运输费用最小。
- 约束条件 (2) 表示每个销售地对产品的需求必须得到满足。
- 约束条件 (3) 表示每个工厂的生产能力不能超过其上限。


## 线性规划的求解

线性规划使用了一些优化算法和策略来高效地找到最优解。

### Simplex Method
单纯形法（Simplex Method）：这是最经典的线性规划求解算法。单纯形法通过在可行域的顶点之间移动来寻找最优解。它利用了线性规划问题的几何特性，即最优解总是在可行域的顶点上。

### Interior Point Method
内点法（Interior Point Method）：这是另一种常用的线性规划求解算法。内点法通过在可行域的内部移动来寻找最优解。与单纯形法不同，内点法不局限于可行域的边界。

1984年贝尔实验室 Karmarkar 提出内点法，其核心思想是将目标函数下降方向投影到约束的 null space + affine transformation + barrier 加速。1987年内点法被证明是多项式算法 ，其算法复杂度为 $O(N^3L)$。目前内点法的算法复杂度被进一步降低到。

### Branch and Bound Method
分支定界法（Branch and Bound）：主要用于整数线性规划问题。它通过分解问题和排除不可能的解来缩小搜索空间。



## Python求解线性规划案例

某机床厂生产甲、乙两种机床，每台销售后的利润分别为4000 元与3000 元。生产甲机床需用 A、B机器加工，加工时间分别为每台 2 小时和 1 小时；生产乙机床需用 A、B、C三种机器加工，加工时间为每台各一小时。若每天可用于加工的机器时数分别为A 机器10 小时、B 机器8 小时和C 机器7 小时，问该厂应生产甲、乙机床各几台，才能使总利润最大？
线性规划模型为：
$$
\begin{aligned}
 & \mathrm{max}z=4000x_{1}+3000x_{2} \\
 & \mathrm{s.t.} \\
 & 2x_1+x_2\leq10 \\
 & x_1+x_2\leq8 \\
 & x_{2}\leq7 \\
 & x_{1}\geq0,x_{2}\geq0
\end{aligned}
$$


用Python和OR-tools编程求解上述模型:
```python
from ortools.linear_solver import pywraplp


def main():
    # Select a solver according to the type of your problem.
    solver = pywraplp.Solver(name='SolveSimpleSystem', problem_type=pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    # Create the variables
    x1 = solver.NumVar(0, solver.infinity(), name='x1')
    x2 = solver.NumVar(0, 7, name='x2')
    # create the constraints
    constraint1 = solver.Constraint(-solver.infinity(), 10)
    constraint1.SetCoefficient(x1, 2)
    constraint1.SetCoefficient(x2, 1)

    constraint2 = solver.Constraint(-solver.infinity(), 8)
    constraint2.SetCoefficient(x1, 1)
    constraint2.SetCoefficient(x2, 1)

    # Create the objective function
    objective = solver.Objective()
    objective.SetCoefficient(x1, 4000)
    objective.SetCoefficient(x2, 3000)
    objective.SetMaximization()
    # Call the solver.
    solver.Solve()
    print('Number of variables =', solver.NumVariables())
    print('Number of constraints =', solver.NumConstraints())
    # The value of each variable in the solution.
    print('Solution:')
    print('x1 = ', x1.solution_value())
    print('x2 = ', x2.solution_value())
    # The objective value of the solution.
    opt_solution = 4 * x1.solution_value() + 3 * x2.solution_value()
    print('Optimal objective value =', opt_solution)


if __name__ == '__main__':
    main()
```

<br>
参考文献：

1. [线性规划简介](https://zhuanlan.zhihu.com/p/509030805)
2. [线性规划问题建模技巧与求解方法](https://blog.csdn.net/tyhj_sf/article/details/85863219)
3. [OR-tools 线性规划](https://developers.google.cn/optimization/introduction/python?hl=zh-cn)


