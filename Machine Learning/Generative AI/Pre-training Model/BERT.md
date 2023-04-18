## BERT

### 0. 概括

> 文章提出一种预训练双向Transformer模型
【Bidirectional Encoder Representations from Transformers, BERT】。
BERT是一种自监督的语言模型，通过利用大量未标注的语料库提供输入，可以预测句子中每个词汇的上下文。
它改进了单向语言模型，增加了双向概率，并采用先进的预训练技术，从而获得了比以往更好的性能。
此外，BERT模型可以应用于多个NLU任务，包括语义相似性和问答，并在相应的基准数据集上取得了最先进的结果。




### 1. 方法

> #### *模型架构*
> Bert模型架构是多层双向Transformer Encoder。

![Model Structure Comparison](Figure/Bert%20Structure.png)




> #### *预训练任务*
> **A. MLM [Masked Language Model]**
> 
> Masked LM预训练任务【完形填空任务】是Bert模型中的一个重要部分，帮助我们理解上下文关系。
> 该任务将文本中的一定比例单词（15%左右）进行Mask，然后使用Bert模型来预测被遮蔽的单词。
> 
> Bert的优化点：在Mask候选词时，80%概率替换为[MASK], 10%概率随机替换一词，10%概率保持不变。
> 因为随机替换仅发生在15%*10%=1.5%，因此并不会影响模型的语言理解能力
>
>  
> **B. NSP [Next Sentence Prediction]**
> 
> NSP（Next Sentence Prediction）是Bert模型中的另一个重要预训练任务。
> 该任务使用Bert模型来预测一对文本之间的关系。该任务旨在帮助模型学习文本衔接能力，并辅助它理解上下文信息。
> NSP任务可由随机生成的文本对进行定制，并使用了“[CLS]”标记作为样本之间关系的特征向量来进行预测。


> #### *微调*
> 
> 下图展示了Bert的四种微调任务，（a）和（b）是sentence-level的任务，（c）和（d）token-level的任务。
> 

![Bert Fine-tuning](Figure/Bert%20fine-tuning.png)


### 2. 评估任务

> #### *i. GLUE [General Language Understanding Evaluation]*
> 
>
> #### *ii. SQuAD [Stanford Question Answering]*
> 
>
> #### *iii. SWAG [Situations With Adversarial Generations]*
> 
> 