## A tutorial on spectral clustering

### 0. 概括

> 

### 1. 相似图 [Similarity Graphs]

> 定义：
> * 无向图$G = (V, E)$的节点集$V = \begin{Bmatrix} v_1, \cdots, v_ n \end{Bmatrix}$，邻接矩阵$W=(w_{i, j}), {i, j = 1,\cdots, n}$，其中$w_{i, j}$是节点$v_i$和$v_j$间的权重。
> * 节点$v_i \in V$的度为$d_i = \sum_{j=1}^{n} w_{i.j}$，度矩阵是对角元素为$d_1, \cdots, d_n$的对角矩阵。
> * 节点子集$A\subset V$，通过向量$\vec{1}_A = (f_1, f_2, \cdots, f_n)^{\prime} \in R^{n}$来表示节点$v_i$是否在子集A中。
> * 如果$A_i \cap A_j = \emptyset$且$A_1 \cup A_2 \cup \cdots \cup A_k = V$，则非空集合$A_1, A_2, \cdots, A_k$是图$G$的分割。
> 
> 相似图：目的是构建数据的局部拓扑关系。边权越大，代表节点间的相似度越高，越应该被聚到同一类。
> * $\varepsilon$-邻接图：距离低于$\varepsilon$的节点所构成的子图。
> * $k$-邻接图：距离节点$v_i$最近的$k$的节点所构成的子图。
> * 全连接图：连接所有节点，并通过相似度函数衡量节点间的相似性，如高斯函数 $s(x_i, x_j) = e^{-\frac{\|x_i - x_j \|^2}{2\sigma^2}}$。


### 2. 图拉普拉斯

> 非归一化的图拉普拉斯：$L = D - W$
> * 拉普拉斯矩阵$L$满足
\begin{align*} 
f^{\prime} L f & =  f^{\prime} D f -  f^{\prime} W f = \sum_{i=1}^{n} d_i f_i^2  - \sum_{i,j=1}^{n} f_i f_j w_{i,j} \\\\
& = \frac{1}{2} \left[ \sum_{i=1}^{n} \left( \sum_{j=1}^{n} w_{i, j} \right) f_i^2 \cdot - 2\sum_{i,j=1}^{n} f_i f_j w_{i,j} + \sum_{j=1}^{n} \left( \sum_{i=0}^{n} w_{i, j} \right) f_i^2  \right] = \frac{1}{2} \sum_{i, j=1}^{n} w_{i, j} (f_i - f_j)^2
\end{align*} 
> 对于任意$f\in R^n$，存在$f^{\prime} L f \geq 0$，即拉普拉斯矩阵是半正定的，且L有n个非负特征值，最小特征值为0，$0 \leq \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$。
> * 拉普拉斯矩阵$L$的特征值0所对应特征向量数量$k$等价于图$G$的连通分量的数量$A_1, \cdots, A_k$，特征值0对应的特征向量空间可表示为$1_{A_1}, \cdots, 1_{A_k}$。
> 
> 归一化的图拉普拉斯：$L_{sym} := D^{-\frac{1}{2}} L D^{-\frac{1}{2}}  = I - D^{-\frac{1}{2}} W D^{-\frac{1}{2}}$和$L_{rm} := D^{-1} L = I - D^{-1} W$
> * 对于任意$f\in R^n$，存在
$$ f^{\prime} L_{sym} f = \frac{1}{2} \sum_{i, j=1}^{n} w_{i, j} \left( \frac{f_i}{\sqrt{d_i}} - \frac{f_j}{\sqrt{d_j}} \right)^2 $$
> * 假设$\lambda$是$L_{rm}$的特征值，且对应的特征向量为$u$ $[Lu = \lambda Du]$，那么$\lambda$是$L_{sym}$的特征值，且对应的特征向量为$w = D^{\frac{1}{2}} u$。
> * $L_{rw}$特征值0对应的特征向量为$1$，$L_{sym}$特征值0对应的特征向量为$D^{\frac{1}{2}}\cdot 1$，$L_{sym}$和$L_{rw}$都是半正定的。

### 3. 谱聚类算法

> 非归一化的谱聚类方法如下。相似度计算可以用诸如高斯似然等方法。
> $$ s(x_i, x_j) = e^{-\frac{|x_i - x_j|^2}{2 \sigma^2}} $$

<div align="center">
<img src=./Figure/SpectralClusteringU.png width=40% />
</div>

> 归一化的谱聚类方法如下：
 
<div align="center">
<img src=./Figure/SpectralClusteringN.png width=40% />
</div>

<div align="center">
<img src=./Figure/SpectralClusteringN2.png width=40% />
</div>


### 4. 图切割视角

> * 图分割可直接转换为优化如下公式的mincut问题，
$$\mathrm{cut} (A_1, \cdots, A_k) := \frac{1}{2} \sum_{i=1}^{k} W(A_i, \bar{A_i}) $$
$$ \bar{A_i} = V \backslash A_i ,  W(A, B) := \sum_{i\in A, j\in B} w_{i,j} $$
> * mincut的缺点：容易出现单个节点被分为一类。优化后的目标函数包含RatioCut和归一化的Ncut：
$$ \mathrm{RatioCut} (A_1, \cdots, A_k) :=  \frac{1}{2} \sum_{i=1}^{k} \frac{W(A_i, \bar{A_i})}{|A_i|} = \sum_{i=1}^{k} \frac{\mathrm{cut} (A_i, \bar{A_i})}{|A_i|}$$
$$ \mathrm{Ncut} (A_1, \cdots, A_k) :=  \frac{1}{2} \sum_{i=1}^{k} \frac{W(A_i, \bar{A_i})}{\mathrm{vol} (A_i)} = \sum_{i=1}^{k} \frac{\mathrm{cut} (A_i, \bar{A_i})}{\mathrm{vol} (A_i)}$$
$$ |A| := \mathrm{number\ of\ vectices\ in\ }  A, \mathrm{vol}(A) := \sum_{i\in A} d_i  $$
> 1. **$k=2$场景的RatioCut**
> * * 目标：$\min_{A\subset V} \mathrm{RatioCut} (A, \bar{A})$
> * * 定义向量$f = (f_1, \cdots, f_n)^{\prime} \in R^n$，其中$f_i = \left\lbrace \begin{matrix} \sqrt{|\bar{A}| / |A|}, & v_i \in A \\\\ -\sqrt{|A|/|\bar{A}|}, & v_i \in \bar{A} \end{matrix} \right. $，则有
\begin{align*}
f^{\prime} L f & = \frac{1}{2} \sum_{i, j=1}^{n} w_{i,j}(f_i - f_j)^2 \\\\
& = \frac{1}{2} \sum_{i \in A, j \in \bar{A}} w_{i,j} \left(\sqrt{|\bar{A}| / |A|} + \sqrt{|A|/|\bar{A}|} \right)^2 + \frac{1}{2} \sum_{i \in \bar{A}, j \in A} w_{i,j} \left(-\sqrt{|\bar{A}| / |A|} - \sqrt{|A|/|\bar{A}|} \right)^2 \\\\
& = \mathrm{cut} (A, \bar{A}) \left(\frac{|\bar{A}|}{ |A|} + \frac{|A|}{|\bar{A}|} + 2  \right) = \mathrm{cut} (A, \bar{A})  \left( \frac{|A| + |\bar{A}|}{|A|} + \frac{|A| + |\bar{A}|}{|\bar{A}|} \right) \\\\
& = |V| \cdot \mathrm{RatioCut} (A, \bar{A})
\end{align*}

> 2. **$k\greater 2$场景的RatioCut**
> * * 
> 3. **Ncut**
> * * 


### 5. 随机游走视角

