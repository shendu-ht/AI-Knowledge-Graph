## Efficient graphlet kernels for large graph comparison

### 0. 概括

图的核函数包含随机游走核、最短路径核、子树核和循环核等，本文提出Graphlet核，通过计算Graphlet的分布差异来衡量图间距离。


### 1. Graphlet核 

大小为$k$的graphlet可表示为$\mathcal{G} = \begin{Bmatrix} graphlet(1), \cdots, graphlet(N_k) \end{Bmatrix}$，大小为4的所有Graphlet如下图所示。

<div align="center">
<img src=./Figure/Graphlet.png width=40% />
</div>

图G的k谱$f_G$是图中大小为$k$的graphlet的子图数量：$f_G (i) = \mathrm{number}(graphlet(i) \subseteq G)$，归一化后的向量为：
$$ D_G = \frac{1}{\mathrm{number\ of\ all\ graphlets\ in\ G}} f_{G} $$

**Graphlet核**：给定两张图$G$和$G^{\prime}$，graphlet核$k_g$可定义为
$$ k_g (G, G^{\prime}) := D_{G}^{T} D_{G^{\prime}} $$


**递归Graphlet核**：考虑图节点数量较多时，Graphlet核的计算方法。

* **$k$-minors**：图G的邻接矩阵记为$M\in R^{n\times n}$，随机剔除掉矩阵$M$的$n-k$行和对应的列，所得到的子矩阵集合称为矩阵$M$的$k$-minors。
* **递归Graphlet核**：给定两个大小为$n$的图$G$和$G^{\prime}$，记$M$和$M^{\prime}$是$1$-minors的子矩阵集合，那么图$G$和$G^{\prime}$的大小为k的Graphlet核可由如下公式计算，其中如果$G$和$G^{\prime}$同质，则$\delta(G \cong G^{\prime})=1$，否则为0。
$$ k_n (G, G^{\prime}) = \left\lbrace \begin{matrix} \frac{1}{(n-k)^2} \sum_{S\in M, S^{\prime} \in M^{\prime}} k_{n-1} (S, S^{\prime}), & n > k \\\\  \delta(G \cong G^{\prime}) , & n=k  \end{matrix} \right. $$
* 更进一步，记$M_k$和$M_k^{\prime}$分别为图$G$和$G^{\prime}$的$k$-minors，那么Graphlet核由如下公式计算：
$$k_g(G, G^{\prime}) = k_n (G, G^{\prime}) = \sum_{S\in M_k} \sum_{S^{\prime} \in M_k^{\prime}} \delta(S \cong S^{\prime}) $$

$$ \Rightarrow  k_g(G, G^{\prime}) = \sum_{S, S^{\prime} \in \mathcal{G}} \mathrm{number\ of}(S \sqsubseteq G) \cdot \mathrm{number\ of}(S^{\prime} \sqsubseteq G^{\prime}) \cdot \delta(S \cong S^{\prime}) $$


### 2. 图采样

**概率分布距离**：记$\mathcal{A} = \left\lbrace 1, 2, \cdots, a  \right\rbrace$，对于$\mathcal{A}$上的两个概率分布$P$和$Q$，其$L1$距离可定义为
$$ \begin{Vmatrix} P-Q \end{Vmatrix}_1 := \sum_{i=1}^{a} \begin{vmatrix} P(i) - Q(i) \end{vmatrix} $$

给定集合$X := \left\lbrace X_j \right\rbrace_{j=1}^{m}$，$X_j$是从概率分布$D$中采样的结果，那么$D$的经验分布可定义为
$$ \hat{D}^{m} (i) = \frac{1}{m} \sum_{j=1}^{m} \delta(X_j = D(i)) $$

**采样样本量**：记$D$是集合$\mathcal{A} = \left\lbrace 1, 2, \cdots, a \right\rbrace$上的概率分布，$X := \left\lbrace X_j \right\rbrace_{j=1}^{m}$, $X_j \sim D$。给定$\epsilon > 0, \delta > 0$，满足条件$P\left\lbrace \begin{Vmatrix} D - \hat{D}^{m} \end{Vmatrix} \geq \epsilon \right\rbrace \leq \delta$所依赖的采样样本量为：
$$ m = \left\lceil \frac{2(\log (2\cdot a) + \log (\frac{1}{\delta}))}{\epsilon^2} \right\rceil $$


### 3. 有界度图

**枚举所有连接的Graphlet**：记图$G$的度是有界的，且最大度为$d$，那么计算图$G$中所有连接的大小为$k\in \left\lbrace 3, 4, 5 \right\rbrace$的Graphlets，其复杂度为$\mathcal{O}(n\cdot d^{k-1})$。


**计算图中所有的Graphlet**：对于一个固定的节点$v_1$，我们计算该节点的大小为3或4的子图分布，其时间复杂度分别为$\mathcal{O} (d^{2})$和$\mathcal{O} (d^{3})$。
