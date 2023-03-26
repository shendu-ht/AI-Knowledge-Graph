## Translating Embeddings for Modeling Multi-relational Data

### 0. 概括


### 1. 多关系型数据建模

多关系型数据可通过$(head, label, tail)$或$(h, l, t)$来表示实体和多元关系。
多关系型数据建模可归结为提取实体间的局部或全局连接模式。局部关系可能包含结构信息和节点属性，
对多关系数据建模可能会涉及不同类型的关系和实体，因此需要更通用的方法来同时构建异质关系。


### 2. TransE方法

思想：将关系看作是实体在嵌入向量空间内的转换。如$(h, l, t)$存在，那么实体$t$的嵌入向量趋近于实体$h$的嵌入向量乘以某个与关系$l$相关的向量。

模型：给定三元组$(h, l, t)$的训练集$S$，其中实体$h, t \in E$，关系$l \in L$，TransE旨在学习实体和关系的嵌入向量。当$(h, l, t)$存在时，TransE期望$\vec{h} + \vec{l} \approx \vec{t}$，如果关系不存在，则$\vec{h} + \vec{l}$需要尽可能远离$\vec{t}$。$\vec{h} + \vec{l}$和$\vec{t}$的差异度量可表示为$d(\vec{h} + \vec{l}, \vec{l})$【TransE的差异度量为L1正则与L2正则】。

优化函数：TransE通过如下损失函数来学习节点和边的嵌入向量
$$ \mathcal{L} = \sum_{(h, l, t) \in S} \sum_{(h^{\prime}, l, t^{\prime}) \in S^{\prime}_{(h, l, t)}} \begin{bmatrix} \gamma + d(h+l, t) - d(h^{\prime} + l, t^{\prime}) \end{bmatrix} $$

$$  S^{\prime}_{(h, l, t)} = \left\lbrace (h^{\prime}, l, t) | h^{\prime} \in E \right\rbrace \cup \left\lbrace (h, l, t^{\prime}) | t^{\prime} \in E \right\rbrace $$

在优化过程中，实体嵌入向量的L2正则约束为1，关系嵌入向量没有约束。详细地优化过程如下图所示。

<div align="center">
<img src=./Figure/TransE.png width=40% />
</div>
