## LINE: Large-scale Information Network Embedding


### 0. 概括

 无向边通过联合概率密度，有向边通过条件概率密度进行建模。


### 1. 问题定义

 **信息网络 [Information Network]**：$G=(V, E)$，边$e\in E$是有序对$e=(u, v)$，
 其权重为$w_{uv}  0$，用来表征网络节点$u$和$v$的关系强度。如果是有向图，则$w(u, v) \leq w(v, u)$，
 如果是无向图，则有$w(u, v) = w(v, u)$。

 **一阶邻近度 [First-order Proximity]**：两个节点间的边权，权重越高，两个节点间的相似度也越高，
 可由$p_u=(w_{u1}, w_{u2}, \cdots, w_{u|V|})$表示。
 
 **二阶邻近度 [Second-order Proximity]**：两个节点的邻域网络的相似度，由$p_u$和$p_v$间的相似度决定。
 
 **信息网络嵌入 [Information Network Embedding]**：学习$f_G: V \rightarrow R^{d}$。


### 2. LINE方法详情

 **LINE表征一阶邻近度**：对于无向边$(i, j)$，节点$v_i$和$v_j$间的联合概率可表示为
 $$ p_{1} (v_i, v_j) = \frac{1}{1 + e^{- \vec{u_i}^{T}\cdot \vec{v_i}}} $$
 其中$\vec{u_i} \in R^{d}$是节点$v_i$的低维向量表征，$p_1 (\cdot, \cdot)$衡量嵌入向量空间的距离。节点$v_i$和$v_j$间的经验概率可表示为 
 $$\hat{p_1} (i, j) = \frac{w_{i,j}}{W}, W = \sum_{(i,j)\in E} w_{i, j}$$
 通过优化 $\hat{p_1} (i, j)$和 $p_{1} (v_i, v_j)$间的KL散度来学习一阶邻近度：
 $$ O_1 = - \sum_{(i, j)\in E} w_{i, j} \log p_1 (v_i, v_j) $$
 
 **LINE表征二阶邻近度**：通过引入两个向量来实现节点的有向关系，$\vec{u_i}$表示节点出度的嵌入，$\vec{u_i}^{\prime}$表示节点入度的嵌入。
 对于有向边，其条件概率可以表示为： 
 $$ p_{2} (v_j | v_i) = \frac{e^{\vec{u_{j}}^{T} \cdot \vec{u_i}}}{\sum_{k=1}^{|V|} e^{\vec{u_{k}}^{T} \cdot \vec{u_i}}} $$
 其中$|V|$是节点$v_i$的下游节点数量。最终通过如下公式来学习二阶邻近度：
 $$ O_2 = - \sum_{(i, j)\in E} w_{i, j} \log p_2 (v_j | v_i) $$
 
 **结合一阶和二阶邻近度**：在实际计算过程中，二阶邻近度$p_2 (v_j | v_i)$难以实现，因此采用负采样对其进行优化。最终对于边(i, j)，其优化函数如下：
 $$ \log \sigma(\vec{u_{j}}^{\prime} {}^{T} \cdot \vec{u_i})  + \sum_{i=1}^{K} E_{v_n \sim P_n(v)} [\log \sigma (-\vec{u_{n}}^{\prime} {}^{T} \cdot \vec{u_i})] $$
 其中 $\sigma(x) = 1 / (1+e^{-x})$，第一部分对既有边进行建模，第二部分进行负采样，$P_n \propto d_{v}^{3/4}$，$d_v$是节点$v$的出度。
 与此同时，由于直接使用边权$w_{i,j}$会导致梯度爆炸，会对边进行采样，并使用采样后的结果作为权重来进行模型训练。


### 3. 评估

<div align="center"
<img src=./Figure/LINEval.png width=40% /
</div
