## LINE: Large-scale Information Network Embedding


### 0. 概括




### 1. 问题定义

> **信息网络 [Information Network]**：$G=(V, E)$，边$e\in E$是有序对$e=(u, v)$，
> 其权重为$w_{uv} > 0$，用来表征网络节点$u$和$v$的关系强度。如果是有向图，则$w(u, v) \leq w(v, u)$，
> 如果是无向图，则有$w(u, v) = w(v, u)$。
>
> **一阶邻近度 [First-order Proximity]**：两个节点间的边权，权重越高，两个节点间的相似度也越高，
> 可由$p_u=(w_{u1}, w_{u2}, \cdots, w_{u|V|})$表示。
> 
> **二阶邻近度 [Second-order Proximity]**：两个节点的邻域网络的相似度，由$p_u$和$p_v$间的相似度决定。
> 
> **信息网络嵌入 [Information Network Embedding]**：学习$f_G: V \rightarrow R^{d}$。


### 2. LINE方法详情

> **LINE表征一阶邻近度**：
> 
> **LINE表征二阶邻近度**：
> 
> **结合一阶和二阶邻近度**：

