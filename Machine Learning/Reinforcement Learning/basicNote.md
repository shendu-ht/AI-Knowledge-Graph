## 强化学习的基础知识


### 1. Markov决策过程

> 马尔科夫决策过程（Markov Decision Process, MDP）是强化学习的理想数学形式。
> * **Agent**：learner 与 decision maker。
> * **Environment**：与Agent交互的一切外在事物。
> * **Agent-Environment Interface**：agent和environment在离散时间步长下进行交互。
在step t，agent接受来自环境的state输入$S_t \in \mathcal{S}$。Agent基于状态输入进行决策，并选择action $A_t \in \mathcal{A}(s)$。
随后，基于采取action的影响，agent会从环境获取reward $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$。
> * **Dynamics function**：
$$ p(s^{'}, r | s, a)  \doteq Pr\begin{Bmatrix} S_t = s^{'}, R_t = r | S_{t-1} = s, A_{t-1} = a  \end{Bmatrix} $$
其中对于任意$s\in S, a \in A(s)$，都存在
$$ \sum_{s^{'} \in S}  \sum_{r\in R} p(s^{'}, r | s, a) = 1 $$ 
> * **状态转移概率**：
$$ p(s^{'} | s, a ) \doteq Pr\begin{Bmatrix} S_t = s^{'} | S_{t-1} = s, A_{t-1} = a \end{Bmatrix} = \sum_{r\in R} p(s^{'}, r | s, a ) $$
$$ r(s, a) \doteq  \mathbb{E} \begin{bmatrix} R_t | S_{t-1} = s, A_{t-1} = a \end{bmatrix} = \sum_{s^{'} \in S} \sum_{r\in R} r\cdot p(s^{'}, r | s, a )  $$
$$ r(s, a, s^{'})  \doteq \mathbb{E} \begin{bmatrix} R_t | S_{t-1} = s, A_{t-1} = a, S_t = s^{'} \end{bmatrix} = \sum_{r\in R} r\cdot \frac{p(s^{'}, r | s, a )}{p(s^{'} | s, a )}$$
> * 