## Anonymous Walk Embeddings

### 0. 概括


### 1. 匿名游走 Anonymous Walks

**位置函数**：记$s = (u_1, u_2, \cdots, u_k)$是一组有序列表，定义位置函数$pos: (s, u_i) \mapsto q$，如给定列表$s = (a, b, c, b, c)$, $pos(s, a) = (1)$，$pos(s, b) = (2, 4)$。

**匿名游走**：记$w = (v_1, v_2, \cdots, v_k)$是一组随机游走结果，其对应的匿名游走结果是$a = (f(v_1), f(v_2), \cdots, f(v_k))$, 其中$f(v_i) = \min_{p_j \in pos(w, v_i)} pos(w, v_i)$。如给点游走结果$w = (a, b, c, b, c)$，其匿名游走结果是$(1, 2, 3, 2, 3)$。

**采用匿名游走的原因**：网络/图的全局拓扑难以被观测到，如在社交网络中，个人的关系网是不允许被外人观测到的，需要通过哈希等手段进行去隐私化处理，最终得到的是匿名节点。


### 2. AWE算法

#### _2.1 基于特征的AWE模型_

定义：
* 原始图$G = (V, E, \Omega)$，其中$V=\{ v_1, v_2, \cdots, v_n \}$是一组节点，$E\subseteq V \times V$是一组边，$\Omega \in R$是一组边权。
* 基于原始图，构造随机游走图$R = (V, E, P)$，其中连接节点$u$和节点$v$的边$e=(u, v)$的权重为$p_e = \frac{\omega_e}{\sum_{v^{\prime} \in N_{out}(u)} \omega(u, v^{\prime})}$，$N_{out}(u)$是节点u的出邻接节点。
* 在图$R$上游走次数$l$的随机游走$w$节点序列为$u_1, u_2, \cdots, u_{l+1}$，每次游走$(u_i, u_{i+1})$基于概率$p_{(u_i, u_{i+1})}$进行选择，最终随机游走的概率为$p(w) = \Pi_{e\in w} p_{e}$。




#### _2.2 数据驱动的AWE模型_
