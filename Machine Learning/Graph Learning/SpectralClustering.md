## A tutorial on spectral clustering

### 0. 概括

> 

### 1. 相似图 [Similarity Graphs]

> 定义：
> * 无向图$G = (V, E)$的节点集$V = \begin{Bmatrix} v_1, \cdots, v_ n \end{Bmatrix}$，邻接矩阵$W=(w_{i, j})_{i, j = 1,\cdots, n}$，其中$w_{i, j}$是节点$v_i$和$v_j$间的权重。
> * 节点$v_i \in V$的度为$d_i = \sum_{j=1}^{n} w_{i.j}$，度矩阵是对角元素为$d_1, \cdots, d_n$的对角矩阵。
> * 节点子集$A\subset V$，通过向量$(f_1, f_2, \cdots, f_n)^{\prime} \in R^{n}$来表示节点$v_i$是否在子集A中。
> * 如果$A_i \cap A_j = \emptyset$且$A_1 \cup A_2 \cup \cdots \cup A_k = V$，则非空集合$A_1, A_2, \cdots, A_k$是图$G$的分割。
> 
> 相似图：目的是构建数据的局部拓扑关系。
> * $\varepsilon$-邻接图：距离低于$\varepsilon$的节点所构成的子图。
> * $k$-邻接图：距离节点$v_i$最近的$k$的节点所构成的子图。
> * 全连接图：连接所有节点，并通过相似度函数衡量节点间的相似性，如高斯函数 $s(x_i, x_j) = e^{-\frac{\|x_i - x_j \|^2}{2\sigma^2}}$。


### 2. 图拉普拉斯


