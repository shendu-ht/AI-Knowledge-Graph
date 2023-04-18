## Deep Learning for Click-Through Rate Estimation

### 0. 概括

本文是一篇关于深度学习在CTR预估中的综述，文章介绍了CTR预估的重要性以及深度学习模型如何提高CTR预估的性能。


### 1. CTR背景

CTR预估是指通过分析用户的历史行为数据，预测用户是否会点击某个广告或者某个推荐内容。CTR预估任务的数据案例如下图所示：

<div align="center">
<img src=Figure/ClickInstance.png width=75% />
</div>

CTR预估的模型训练任务可构建为二分类问题，且损失函数如下：
$$ \mathcal{L}(x, y, \theta) = -y\cdot \log \sigma(f_{\theta}(x)) - (1-y)\cdot \log (1-\sigma(f_{\theta} (x))) $$

