## Knowledge Graph Embedding by Translating on Hyperplanes

### 0. 概括

TransE难以处理自映射、一对多、多对多等不同属性的关系，本文提出TransH来解决这一问题。


### 1. 知识图谱表征

前人方法和TransH的Score函数和参数量级对比结果如下图所示。

<div align="center">
<img src=./Figure/TransHmodels.png width=40% />
</div>


### 2. TransH方法详情

#### 关系映射

记$\Delta$是存在关系的三元组集合，$(h, r, t)\in \Delta$表示三元组关系是存在的。

<div align="center">
<img src=./Figure/TransH-TransE.png width=40% />
</div>



#### TransH方法



#### TransH训练
