## 最优化的基础知识


#### 1. 向量空间和矩阵

> * 子空间：$a_1, a_2, \cdots, a_n$是$R^n$空间中的任意向量，其所有线性组合的集合称为子空间，记为
$$ \mathrm{span}\begin{bmatrix} \mathbf{a_1} & \mathbf{a_2} & \cdots & \mathbf{a_n} \end{bmatrix} = 
\begin{Bmatrix} \sum_{i=1}^{n} \alpha_i \mathbf{a_i} | \alpha_1, \cdots, \alpha_n \in R \end{Bmatrix} $$
> * 矩阵的秩：考虑$m\times n$矩阵
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
> * 内积：对于$\mathbf{x}, \mathbf{y} \in R^{n}$，其欧式内积为 $\left\langle \mathbf{x}, \mathbf{y} \right\rangle = \sum_{i=1}^{n} x_i y_i = 
\mathbf{x}^{T} \mathbf{y}$
> * 范数：对于$\mathbf{x} \in R^{n}$，其欧式范数为 $\left\langle \mathrm{x}, \mathrm{x} \right\rangle = \begin{Vmatrix} x \end{Vmatrix}^2$