## Anonymous Walk Embeddings

### 0. 概括


### 1. 匿名游走 Anonymous Walks

**位置函数**：记$s = (u_1, u_2, \cdots, u_k)$是一组有序列表，定义位置函数$pos: (s, u_i) \mapsto q$，如给定列表$s = (a, b, c, b, c)$, $pos(s, a) = (1)$，$pos(s, b) = (2, 4)$。

**匿名游走**：记$w = (v_1, v_2, \cdots, v_k)$是一组随机游走结果，其对应的匿名游走结果是$a = (f(v_1), f(v_2), \cdots, f(v_k))$, 其中$f(v_i) = \min_{p_j \in pos(w, v_i)} pos(w, v_i)$。如给点游走结果$w = (a, b, c, b, c)$，其匿名游走结果是$(1, 2, 3, 2, 3)$。

**采用匿名游走的原因**：网络/图的全局拓扑难以被观测到，如在社交网络中，个人的关系网是不允许被外人观测到的，需要通过哈希等手段进行去隐私化处理，最终得到的是匿名节点。


### 2. AWE算法

#### _2.1 基于特征的AWE模型_


**随机游走图**：原始图$G = (V, E, \Omega)$，其中$V=\{ v_1, v_2, \cdots, v_n \}$是一组节点，$E\subseteq V \times V$是一组边，$\Omega \in R$是一组边权。
基于原始图，构造随机游走图$R = (V, E, P)$，其中连接节点$u$和节点$v$的边$e=(u, v)$的权重为$p_e = \frac{\omega_e}{\sum_{v^{\prime} \in N_{out}(u)} \omega(u, v^{\prime})}$，$N_{out}(u)$是节点u的出邻接节点。

**匿名路径概率**：在图$R$上游走次数$l$的随机游走$w$节点序列为$u_1, u_2, \cdots, u_{l+1}$，每次游走$(u_i, u_{i+1})$基于概率$p_{(u_i, u_{i+1})}$进行选择，最终随机游走的概率为$p(w) = \Pi_{e\in w} p_{e}$。

从初始节点$u$出发的一组游走长度为$l$的$\eta$个不同随机游走结果可表示为$W_{l}^{u} = (a_1^u, a_2^u, \cdots, a_{\eta}^u)$，从节点$u$出发的匿名游走结果$a_{i}^{u}$的概率为$p(a_{i}^{u}) = \sum_{w \in W_{l}^{u}, w \mapsto a_i} p(w)$。在整张图$G$上，匿名游走结果$a_i$的概率为
$$p(a_i) = \frac{1}{N} \sum_{u\in G} p(a_i^u) = \frac{1}{N} \sum_{u\in G} \sum_{w \in W_{l}^{u}, w \mapsto a_i} p(w) $$

**基于特征的AWE**：记游走路径为$l$的所有可能匿名游走的结果为$A_l=(a_1, a_2, \cdots, \cdots, a_{\eta})$，那么图$G$的匿名游走嵌入为
$$ f_{G} = (p(a_1), p(a_2), \cdots, p(a_{\eta})) $$。

**采样**：获取游走长度为$l$的全部结果对应的时间复杂度是$\mathcal{O} (n(d_{in}^{max}(v) \cdot d_{out}^{max} (v))^{l/2}\times l)$，全量游走复杂度高，因此需要对匿名游走进行采样。

记$A_l = (a_1, a_2, \cdots, a_{\eta})$是长度为$l$的所有可能匿名游走结果，对于$A_l$上的两个独立概率分布$P$和$Q$，定义距离函数
$$ \| P- Q \|_1 =  \sum_{a_i \in A} | P(a_i) - Q(a_i) | $$

对于图$G$，记原始匿名游走结果的分布为$\mathcal{D}_l$，$X^{m} = (X_1, X_2, \cdots, X_m)$是从$\mathcal{D}_l$中的随机采样结果，则计算出来的经验分布为$\mathcal{D}^{m}$，计算公式如下，其中如果$x$为真，则$[x] = 1$。
$$ \mathcal{D}^{m} (i) = \frac{1}{m} \sum_{X_j \in X^{m}} [X_j = a_i] $$

那么对于所有$\epsilon > 0$且$\delta \in [0, 1]$，样本量$m$需满足$P\{ \| \mathcal(D)^{m} - \mathcal{D}  \|_1 \geq \epsilon \} \leq \delta$，这等价于：
$$ m = \left\lceil  \frac{2}{\epsilon^2} (\log (2^{\eta} - 2) - \log (\delta)) \right\rceil $$

#### _2.2 数据驱动的AWE模型_
