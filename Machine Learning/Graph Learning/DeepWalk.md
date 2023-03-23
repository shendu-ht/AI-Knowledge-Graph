## DeepWalk: Online Learning of Social Representations


### 0. 概括

 文章提出了一种学习图中节点表示的新方法DeepWalk，它基于使用随机游走生成节点序列，然后使用这些序列训练神经网络来学习节点嵌入。


### 1. 问题定义

 目标场景：社交网络等网络场景的节点分类问题。数学表示为： $G=(V, E)$ ，其中 $V$ 为网络节点，
 $E\subseteq (V\times V)$为网络节点间的边。 $G_L = (V, E, X, Y)$为部分标注的网络，其中 $X\in \mathbb{R}^{|V|\times S}$为网络节点属性，
 $S$是属性向量的维度， $Y\in \mathbb{R}^{|V|\times |\mathcal{Y}|}$， $\mathcal{Y}$为节点标签。

 DeepWalk目标：学习 $X_E \in \mathbb{R}^{|V|\times d}$，其中 $d$为潜在表征的维度。


### 2. 构建网络节点表征需考虑的特性

 1. **适应性 [Adaptability]**：社交网络等是在不断变化的，新社交关系的出现不应该重新学习整个社交网络。
 2. **社区表征 [Community aware]**：网络节点的表征需要能够描述节点间的相似性，相似节点表征应具有更小的距离。
 3. **低维度 [Low dimensional]**：当社交网络稀疏时，低维表征可以保证更好地模型泛化，同时加速收敛和推理。
 4. **连续性 [Continuous]**：在连续空间上对节点进行表征，以确保模型能产出更鲁棒和平滑的决策边界。


### 3. DeepWalk的基础支撑

 1. **随机游走 [Random Walk]**：① 局部游走可以并行化处理；② 从局部随机游走获得的信息，不需要在全局重新计算的情况下适应图结构的微小变化。
 我们可以通过在图的变化区域进行新的游走来更新学习到的模型。

 2. **语言建模 [Language Modeling]**：语言建模的目标是估计 **单词序列 [sentence of words]** 出现在语料库中的可能性，即最大化语料库中的单词序列 $Pr(w_n | w_0, w_1, \cdots, w_{n-1})$。
节点表征的目标是构建 $\Phi: v\in V \Rightarrow \mathbb{R}^{|V|\times d}$，将随机游走的节点序列看成单词序列，则优化目标可抽象成 $Pr(v_i | (\Phi(v_1), \Phi(v_2), \cdots, \Phi(v_{i-1})))$。
实际的训练任务：给定节点 $v_i$，最大化前序 $w$和后序 $w$节点的预测概率， 
$$\mathrm{minimize} - \log Pr(\begin{Bmatrix} v_{i-w}, \cdots, v_{i-1}, v_{i+1}, \cdots, v_{i+w} \end{Bmatrix} | \Phi(v_i))$$
如此具有相似邻域节点的网络节点将具备相似的表征。

     
### 4. DeepWalk方法详情

<div align="center"
<img src=./Figure/OverviewDeepWalk.png width=60% /
</div

 DeepWalk的整体流程如上图所示，其伪代码如下图所示。
 
 计算$Pr(u_k | \Phi(v_j))$时，采用分层Softmax分解概率。它将随机游走到的网络节点分配到二叉树的叶子节点，从二叉树根节点到$u_k$的路径可表示为$b_0, \cdots, b_{\left\lceil \log |V| \right\rceil}$
 其中$b_0 = root, b_{\left\lceil \log |V| \right\rceil} = u_k$，如此则有
 $$ Pr(u_k | \Phi(v_j)) = \Pi_{l=1}^{\left\lceil \log |V| \right\rceil} Pr(b_l | \Phi(v_j)) $$
 $$ Pr(b_l | \Phi(v_j)) = \frac{1}{1 + e^{-\Phi(v_j)\cdot \Psi(b_l)}} $$
 其中$\Psi(b_l)\in \mathbb{R}^{d}$是节点$b_l$的父节点表征，$\Phi$和$\Psi$为待优化的模型参数。

<div align="center"
<img src=./Figure/DeepWalk.png width=40% /
</div

<div align="center"
<img src=./Figure/SkipGram.png width=40% /
</div


### 5. 评估


 基线对比方法：
 * SpectralClustering
 * Modularity
 * EdgeCluster
 * wvRN
 * Majority
 
 评估任务
 * 样本集：BlogCatalog、Flickr、YouTube
 * 任务类型：多分类任务
 * 评价指标：Macro-F1 and Micro-F1
 
 评估结果如下图所示
 
<div align="center"
<img src=./Figure/DeepWalkEval.png width=40% /
</div