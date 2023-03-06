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


> * **收敛序列与极限**：对于任意正数$\varepsilon$，存在$K$，使得对于所有$k>K$，都有$\begin{vmatrix} x_k - x^{*} \end{vmatrix} < \varepsilon$，
则称$x^{*} \in R$为序列$\begin{Bmatrix} x_k \end{Bmatrix}$的极限，记为 $x^{*} = \lim_{k\rightarrow \infty} x_k$。如果序列存在极限，则该序列称为收敛序列。
> * **矩阵收敛**：给定$m\times m$矩阵队列$\mathbf{A_k}$和$m\times m$矩阵$\mathbf{A}$，如果$\lim_{k \rightarrow \infty}
\begin{Vmatrix} \mathbf{A_k} - \mathbf{A} \end{Vmatrix} = 0$，则称矩阵序列收敛于矩阵$\mathbf{A}$。
> * **矩阵收敛引理**：给定$$
