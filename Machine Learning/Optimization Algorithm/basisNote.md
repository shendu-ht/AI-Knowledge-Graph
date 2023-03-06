## 最优化的基础知识


### 1. 向量空间和矩阵

> * **子空间**：$a_1, a_2, \cdots, a_n$是$R^n$空间中的任意向量，其所有线性组合的集合称为子空间，记为
$$ \mathrm{span}\begin{bmatrix} \mathbf{a_1} & \mathbf{a_2} & \cdots & \mathbf{a_n} \end{bmatrix} = 
\begin{Bmatrix} \sum_{i=1}^{n} \alpha_i \mathbf{a_i} | \alpha_1, \cdots, \alpha_n \in R \end{Bmatrix} $$
> * **矩阵的秩**：考虑$m\times n$矩阵
$$ \mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\\\
a_{21} & a_{22} & \cdots & a_{2n} \\\\
\vdots & \vdots & \ddots & \vdots \\\\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix} $$
A的第$k$列用$\mathbf{a_k}$表示：
$$ \mathbf{a_k} = \begin{bmatrix}
a_{1k} \\\\ a_{2k} \\\\ \vdots \\\\ a_{mk}
\end{bmatrix} $$
矩阵A中线性无关列的最大数量称为$\mathbf{A}$的秩，记为$\mathrm{rank} \mathbf{A}$。$\mathrm{rank} \mathbf{A}$是子空间
$\mathrm{span}\begin{bmatrix} \mathbf{a_1} & \mathbf{a_2} & \cdots & \mathbf{a_n} \end{bmatrix}$的维数。
> * **内积**：对于$\mathbf{x}, \mathbf{y} \in R^{n}$，其欧式内积为 $\left\langle \mathbf{x}, \mathbf{y} \right\rangle = \sum_{i=1}^{n} x_i y_i = 
\mathbf{x}^{T} \mathbf{y}$
> * **范数**：对于$\mathbf{x} \in R^{n}$，其欧式范数为 $\left\langle \mathrm{x}, \mathrm{x} \right\rangle = \begin{Vmatrix} x \end{Vmatrix}^2$



### 2. 集合


> * **超平面**：对于$u_1, u_2, \cdots, u_n, v \in R$，满足方程$u_1 x_1 + u_2 x_2 + \cdots + u_n x_n = v$的点$\mathbf{x} = 
\begin{bmatrix} x_1 & x_2 & \cdots & x_n \end{bmatrix}^T$组成的集合称为空间$R^n$的超平面，记为$
\begin{Bmatrix} \mathbf{x} \in R^n : \mathbf{u}^T \mathbf{x} = v
\end{Bmatrix}$。
> * **凸集**：对于所有$\mathbf{u}, \mathbf{v} \in \Theta$，都有$\alpha \mathbf{u} + (1-\alpha) \mathbf{v} \in \Theta, \alpha\ \in (0, 1)$, 
则$\Theta$是一个凸集。
> * **邻域**：点$\mathbf{x} \in R^n$的邻域可以表示为$\begin{Bmatrix} 
\mathbf{y} \in R^{n}: \begin{Vmatrix} \mathbf{y} - \mathbf{x} \end{Vmatrix} < \epsilon
\end{Bmatrix}$。



### 3. 微积分


> * **收敛序列与极限**：对于任意正数$\varepsilon$，存在$K$，使得对于所有$k>K$，都有$\begin{vmatrix} x_k - x^{\*} \end{vmatrix} < \varepsilon$，
则称$x^{\*} \in R$为序列$\begin{Bmatrix} x_k \end{Bmatrix}$的极限，记为 $x^{*} = \lim_{k\rightarrow \infty} x_k$。如果序列存在极限，则该序列称为收敛序列。
> * **矩阵收敛**：给定$m\times m$矩阵队列$\mathbf{A_k}$和$m\times m$矩阵$\mathbf{A}$，如果$\lim_{k \rightarrow \infty}
\begin{Vmatrix} \mathbf{A_k} - \mathbf{A} \end{Vmatrix} = 0$，则称矩阵序列收敛于矩阵$\mathbf{A}$。
> * **矩阵收敛引理**：给定$\mathbf{A} \in R^{n\times n}$, 当且晋档$\mathbf{A}$的所有特征值满足
$\begin{vmatrix} \lambda_i(\mathrm{A}) \end{vmatrix} < 1 (i=1,\cdots, n) $时，存在$\lim_{k\rightarrow \infty} \mathbf{A}^{k} = \mathbf{0}$。
> * **矩阵收敛引理**：$n\times n$的矩阵序列
$$ \mathbf{I_n} + \mathbf{A} + \mathbf{A^2} + \cdots + \mathbf{A^k}  + \cdots$$
是收敛序列，当且仅当$\lim_{k\rightarrow \infty} \mathbf{A^{k}} = \mathbf{0}$. 此时序列和为$(\mathbf{I_n} - \mathbf{A})^{-1}$。
> * **矩阵连续**：给定矩阵值函数$\mathbf{A}: R^{r} \rightarrow R^{n\times n}$和点$\mathbf{\xi_0} \in R^{r}$，如果
$$ \lim_{\begin{Vmatrix} \mathbf{\xi} - \mathbf{\xi_0}  \end{Vmatrix} \rightarrow 0}  \begin{Vmatrix} \mathbf{A}(\mathbf{\xi}) - \mathbf{A}(\mathbf{\xi_0})  \end{Vmatrix} = 0$$
那么$\mathbf{A}$在点$\mathbf{\xi_0}$处连续。
> * **可微**：给定函数$\mathbf{f}: \Omega \rightarrow R^{m}, \Omega \subset R^{n}$，如果存在仿射函数$\mathcal{L}: R^{n} \rightarrow R^{m}$，使得
$$ \lim_{x\rightarrow x_0, x\in \Omega} \frac{\begin{Vmatrix} f(x) - (f(x_0) + \mathcal{L}(x-x_0)) \end{Vmatrix}}{\begin{Vmatrix} x - x_0 \end{Vmatrix}} = 0 $$
则函数$\mathbf{f}$在点$x_0\in \Omega$处是可微的，$\mathcal{L}$称为$\mathbf{f}$在点$x_0$的导数。
> * **导数矩阵**：给定任意函数$\mathbf{f}: R^{n} \rightarrow R^{m}$，其导数$\mathcal{L}$可表示为$m\times n$矩阵
$$ \begin{bmatrix} \frac{\partial f}{\partial x_1}(x_0) & \cdots  & \frac{\partial f}{\partial x_n}(x_0) \end{bmatrix} =  
\begin{bmatrix} \frac{\partial f_1}{\partial x_1}(x_0) & \cdots  & \frac{\partial f_1}{\partial x_n}(x_0) \\\\
\vdots & \ddots & \vdots \\\\
\frac{\partial f_m}{\partial x_1}(x_0) & \cdots  & \frac{\partial f_m}{\partial x_n}(x_0) 
\end{bmatrix}$$
> * **梯度**：如果$f: R^n \rightarrow R$是可微的，如下函数称为$f$的梯度
$$ \nabla f(x) = \begin{bmatrix} \frac{\partial f}{\partial x_1} (x) \\\\ \vdots \\\\ \frac{\partial f}{\partial x_n} (x) \end{bmatrix} $$
> * **Hessian矩阵**：给定函数$f: R^n \rightarrow R$，如果梯度$\nabla f(x)$，则$f$是二次可微的，Hessian矩阵如下：
$$ D^2 f = \begin{bmatrix} 
\frac{\partial f^2}{\partial x_1^2} & \frac{\partial f^2}{\partial x_2 \partial x_1} & \cdots & \frac{\partial f^2}{\partial x_n \partial x_1}  \\\\ 
\frac{\partial f^2}{\partial x_1 \partial x_2} & \frac{\partial f^2}{\partial x_2^2} & \cdots & \frac{\partial f^2}{\partial x_n \partial x_2}  \\\\ 
\vdots & \vdots & \ddots & \vdots  \\\\ 
\frac{\partial f^2}{\partial x_1 \partial x_n} & \frac{\partial f^2}{\partial x_2 \partial x_n} & \cdots & \frac{\partial f^2}{\partial x_n^2} 
\end{bmatrix}  $$
> * **微分链式法则**：如果$g: \mathcal{D} \rightarrow \mathcal{R}$在$D\subset R^n$上是可微的，且$f: (a. b) \rightarrow \mathcal{D}$在$(a,b)$上可微。那么其复合函数$h: (a, b)\rightarrow R$，$h(t) = g(f(t))$在$(a,b)$上是可微的，且导数为
$$ h^{'}(t) = Dg(f(t))Df(t) = \nabla g(f(t))^{T} \begin{bmatrix} f^{'}_{1}(t) \\\\ \vdots \\\\ f^{'}_{n}(t) \end{bmatrix}$$
> * **泰勒定理**：假定函数$f: R\rightarrow R$在区间$[a, b]$上是$m$阶连续可微的。令$h=b-a$，有
$$ f(b) = f(a) + \frac{h}{1!}f^{(1)}(a) + \frac{h^2}{2!}f^{(2)}(a) + \cdots + \frac{h^{m-1}}{(m-1)!}f^{(m-1)}(a) + R_m$$
$$ R_m = \frac{h^m}{m!} f^{(m)} (a + \theta h) $$

    

### 4. 无约束优化基础知识

> * **无约束优化问题**：$\min f(x)$ subject to $x \in \Omega$
> * **局部极小值的一阶必要条件**：多元实值函数$f$在约束集$\Omega \subset R^{n}$上一阶连续可微，即$f\in C^{1}$。如果$x^{\*}$是函数$f$在约束集$\Omega$上的局部极小值，且是$Omega$的内点，则有
$$ \nabla f(x^{\*}) = 0$$
> * **局部极小值的二阶必要条件**：Hessian矩阵是半正定矩阵，即$d^{T} \cdot D^2 f \cdot d \geq 0$。
> * **一维优化搜索方法**：$f: R \rightarrow R$时的搜索方法有 黄金分割法、斐波那契数列法、二分法、牛顿法、割线法、划界法等。
> * **全局搜索算法**：Nelder-Mead单纯形法、模拟退火法、粒子群优化算法、遗传算法
> * **梯度方法**：随机梯度下降、牛顿法等

