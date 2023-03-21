# 史上最全的AI知识体系

# 目标

> * **跟踪前沿技术方向。**
> * **构建全知的AI知识体系。**


# 机器学习

### 1. Generative AI

> **简要介绍**
> 
> 生成模型是一种机器学习技术，可以用来自动根据历史数据创造出新的、可能的输出内容。其工作原理是通过使用数据集中出现的特征
> （例如颜色、形状、位置和大小）来判断其他图片中应该存在何种对象。这些已有的特征帮助生成模型扫描图片并识别里面包含的对象。
> 此外，生成模型也可以用来生成声音信号、文字信息和合成图片。

> **Paper List**
> 
> **Generative Model**
> * **[VAE]** Auto-Encoding Variational Bayes
[[pdf]](https://arxiv.org/pdf/1312.6114.pdf) [[notes]](./Machine%20Learning/Generative%20Model/VAE.md)
> * * **[CVAE]** Learning Structured Output Representation using Deep Conditional Generative Models
[[pdf]](https://proceedings.neurips.cc/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf)
[[notes]](./Machine%20Learning/Generative%20Model/CVAE.md)
> * * **[$\beta-$VAE]** Understanding disentangling in $\beta-$VAE
[[pdf]](https://arxiv.org/pdf/1804.03599.pdf) [[notes]](./Machine%20Learning/Generative%20Model/)
> * * **[ControlVAE]** ControlVAE: Controllable Variational Autoencoder 
[[pdf]](http://proceedings.mlr.press/v119/shao20b/shao20b.pdf) [[notes]](./Machine%20Learning/Generative%20Model/ControlVAE.md)
> * * **[VQ-VAE]** Neural Discrete Representation Learning
[[pdf]](https://proceedings.neurips.cc/paper/2017/file/7a98af17e63a0ac09ce2e96d03992fbc-Paper.pdf)
[[notes]](./Machine%20Learning/Generative%20Model/VQ-VAE.md)
> * **[GAN]** Generative Adversarial Networks
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3422622) [[notes]](./Machine%20Learning/Generative%20Model/GAN.md)
> * * **[VQ-GAN]** Taming Transformers for High-Resolution Image Synthesis
[[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.pdf)
[[notes]](./Machine%20Learning/Generative%20Model/VQ-GAN.md)
> * **[DDPM]** Denoising Diffusion Probabilistic Models
[[pdf]](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)
[[notes]](./Machine%20Learning/Generative%20Model/DDPM.md)
> * * **[LDM]** High-Resolution Image Synthesis with Latent Diffusion Models
[[pdf]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf) 
[[notes]](./Machine%20Learning/Generative%20Model/LDM.md)
> * **[GPT-3]** Language Models are Few-Shot Learners
[[pdf]](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) [[notes]](./Machine%20Learning/Pre-training%20Model/GPT3.md)



### 2. Pre-training Model

> **简要介绍**
> 
> 预训练模型（Pre-training model）是指在机器学习算法中，使用大量训练数据，对模型进行预先训练，以获得更好的性能和更有效地学习结果。
> 预训练模型通常通过对模型参数进行训练来提高模型的性能，从而使模型在真实数据上有更好表现。
> 预训练模型的优点在于可以提高模型的性能，从而更好地满足实际应用的需求。但是，这种模型也存在一定的局限性，比如训练数据量较小时，可能无法获得有效的训练结果，或者训练模型可能会出现过拟合现象。

> **Paper List**
> 
> * BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. 
[[pdf]](https://arxiv.org/pdf/1810.04805.pdf&usg=ALkJrhhzxlCL6yTht2BRmH9atgvKFxHsxQ) [[notes]](./Machine%20Learning/Pre-training%20Model/BERT.md)
> * Improving Language Understanding by Generative Pre-Training
[[pdf]](https://www.cs.ubc.ca/~amuham01/LING530/papers/radford2018improving.pdf) [[notes]](./Machine%20Learning/Pre-training%20Model/GPT.md)
> * Language Models are Unsupervised Multitask Learners
[[pdf]](https://cs.brown.edu/courses/csci1460/assets/papers/language_models_are_unsupervised_multitask_learners.pdf) [[notes]](./Machine%20Learning/Pre-training%20Model/GPT2.md)
> * Language Models are Few-Shot Learners
[[pdf]](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf) [[notes]](./Machine%20Learning/Pre-training%20Model/GPT3.md)
> * Training language models to follow instructions with human feedback
[[pdf]](https://arxiv.org/pdf/2203.02155.pdf?fbclid=IwAR2nZdBpdZZzvxpwI6H_bRmP4RwGOyzke9Ud63lWBe1YlyI_1BRAFhnUMUg) [[notes]](./Machine%20Learning/Pre-training%20Model/InstructGPT.md)
> * WebGPT: Browser-assisted question-answering with human feedback
[[pdf]](https://arxiv.org/pdf/2112.09332.pdf) [[notes]](./Machine%20Learning/Pre-training%20Model/WebGPT.md)
> * LaMDA: Language Models for Dialog Applications
[[pdf]](https://arxiv.org/pdf/2201.08239.pdf) [[notes]](./Machine%20Learning/Pre-training%20Model/LaMDA.md)



### 3. Reinforcement Learning

> **简要介绍**
> 
> 强化学习（Reinforcement Learning）是一种机器学习方法，其目标是让一个智能体在经历不断的尝试和错误之后，学习到如何在一个环境中取得最优表现。
> 它是一种在环境中的自主学习，通过在环境中采取行动来获得反馈，然后根据反馈不断学习。
> 
> 强化学习的过程可以描述为：智能体（Agent）从环境（Environment）中获得观察（Observation），然后根据观察选择行动（Action），
> 行动导致环境发生变化，变化产生新的状态和奖励（Reward），奖励可以让智能体学习到哪些行动有利于达到最终目标。

> **Books**
> 
> * Reinforcement Learning: an Introduction [[pdf]](http://incompleteideas.net/book/RLbook2018.pdf)
> * Algorithms for Reinforcement Learning [[pdf]](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)

> **Lecture**
> 
> * UCL $\times$ DeepMind Reinforcement Learning [[slides]](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021)

> **Paper List**
> 
> * **[BasicNotes]** [[notes]]()
> * **[DQN]** Human-level control through deep reinforcement learning
[[pdf]](https://daiwk.github.io/assets/dqn.pdf) [[notes]](./Machine%20Learning/Reinforcement%20Learning/DQN.md)
> * **[PPO]** Proximal Policy Optimization Algorithms
[[pdf]](https://arxiv.org/pdf/1707.06347.pdf) [[notes]](./Machine%20Learning/Reinforcement%20Learning/PPO.md)
> * 


### 4. Graph Learning

> **简要介绍**
> 
> 图学习（Graph Learning）是一种机器学习技术，它使用图表来表示数据，并利用图表结构中的信息来进行分析和预测。它既可以用于分类任务，
> 也可以用于非监督学习，因为它可以找出数据之间的相互关系，可以帮助我们看出不同的关系，从而发现有价值的信息。
> 它可以模拟复杂的关系，可以对大规模数据集进行有效的分析，也可以查找节点间的关系，从而更好地理解数据。
> 
> 图学习可以用来构建复杂的模型，以发现数据之间的关系，从而进行精确的分类和预测。它也可以用于探查数据中的模式，有助于提高算法的准确性和效率。
> 图学习的应用非常广泛，可以用于计算机视觉，自然语言处理，社交网络分析，推荐系统，机器人控制等等。

> **Books**
> 
> * Graph Representation Learning
[[pdf]](https://www.cs.mcgill.ca/~wlh/grl_book/files/GRL_Book.pdf)

> **Lecture**
> 
> * CS224W: Machine Learning with Graphs
[[slides]](http://web.stanford.edu/class/cs224w/)
> * CS228: Probabilistic Graphical Models
[[syllabus]](https://ermongroup.github.io/cs228/) [[contents]](https://ermongroup.github.io/cs228-notes/)

> **Paper List**
> 
> **Node/Edge/Graph/KnowledgeGraph Embedding**
> 1. A tutorial on spectral clustering
[[pdf]](https://arxiv.org/pdf/0711.0189.pdf)
[[notes]](./Machine%20Learning/Graph%20Learning/SpectralClustering.md)
> 2. **[TransE]** Translating Embeddings for Modeling Multi-relational Data
[[pdf]](https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)
[[notes]](./Machine%20Learning/Graph%20Learning/TransE.md)
> 3. **[TransH]** Knowledge Graph Embedding by Translating on Hyperplanes
[[pdf]](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=2a3f862199883ceff5e3c74126f0c80770653e05)
[[notes]](./Machine%20Learning/Graph%20Learning/TransH.md)
> 4. **[TransR]** Learning Entity and Relation Embeddings for Knowledge Graph Completion
[[pdf]](https://ojs.aaai.org/index.php/AAAI/article/view/9491/9350)
[[notes]](./Machine%20Learning/Graph%20Learning/TransR.md)
> 5. **[TransD]** Knowledge Graph Embedding via Dynamic Mapping Matrix
[[pdf]](https://aclanthology.org/P15-1067.pdf)
[[notes]](./Machine%20Learning/Graph%20Learning/TransD.md)
> 6. **[DeepWalk]** DeepWalk: Online Learning of Social Representations
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/2623330.2623732)
[[notes]](./Machine%20Learning/Graph%20Learning/DeepWalk.md)
> 7. **[LINE]** LINE: Large-scale Information Network Embedding
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/2736277.2741093)
[[notes]](./Machine%20Learning/Graph%20Learning/LINE.md)
> 8. **[AWE]** Anonymous Walk Embeddings
[[pdf]](http://proceedings.mlr.press/v80/ivanov18a/ivanov18a.pdf) 
[[notes]](./Machine%20Learning/Graph%20Learning/AnonymousWalk.md)
> 
> **Graph Neural Network**
> 1. **[GCN]** Semi-Supervised Classification with Graph Convolutional Networks
[[pdf]](https://arxiv.org/pdf/1609.02907.pdf) [[notes]](./Machine%20Learning/Graph%20Learning/GCN.md)
> 2. **[GraphSage]** Inductive Representation Learning on Large Graphs
[[pdf]](https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf) [[notes]](./Machine%20Learning/Graph%20Learning/GraphSage.md)
> 3. **[GAT]** Graph Attention Networks
[[pdf]](https://arxiv.org/pdf/1710.10903.pdf) [[notes]](./Machine%20Learning/Graph%20Learning/GAT.md)
> 4. **[HAN]** Heterogeneous Graph Attention Network
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3308558.3313562) [[notes]](./Machine%20Learning/Graph%20Learning/HAN.md)
> 5. **[GTN]** Graph Transformer Networks
[[pdf]](https://proceedings.neurips.cc/paper/2019/file/9d63484abb477c97640154d40595a3bb-Paper.pdf) [[notes]](./Machine%20Learning/Graph%20Learning/GTN.md)



### 5. Bayesian Learning

> **简要介绍**
> 
> 贝叶斯学习（Bayesian Learning）是一种基于概率的机器学习方法，它以贝叶斯定理为基础，利用收集的数据对模型进行训练并作出预测。
> 它可以帮助AI在遇到新问题时根据已有信息来产生合理的预测。在Bayesian Learning中，使用不同形式的贝叶斯公式来衡量各个特征之间相关性，
> 从而准确估计参数。此外，Bayesian Learning还能够通过对数据所包含信念或先前证明所得到结论之间相关性进行评估来帮助AI作出决定。

> **Paper List**
> 
> * Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
[[pdf]](http://proceedings.mlr.press/v48/gal16.pdf) [[notes]](./Machine%20Learning/Bayesian%20Learning/Dropout.md)
> * 

### 6. Active Learning

> **简要介绍**
> 

> **Paper List**
> 
> * Active Learning with Label Comparisons
[[pdf]](https://proceedings.mlr.press/v180/yona22a/yona22a.pdf) [[notes]](./Machine%20Learning/Active%20Learning/NbrGraphSGD.md)
> * 

### 7. Semi-Supervised Learning

> **简要介绍**
> 

> **Paper List**
> 
> * AdaMatch: A Unified Approach to Semi-Supervised Learning and Domain Adaptation
[[pdf]](https://arxiv.org/pdf/2106.04732.pdf) [[notes]](./Machine%20Learning/Semi-Supervised%20Learning/AdaMatch.md)
> * 



### 8. In-Context Learning

> **简要介绍**
> 
> 情景式学习（In-Context Learning）是一种学习范式，它主要目的是使模型能够在实际情境中学习新知识。它强调以实际情境为基础，
> 通过反复练习，使模型能够有效地融入新知识。

> **Paper List**
> 
> * An Explanation of In-context Learning as Implicit Bayesian Inference
> * 


### 9. Optimization Algorithm

> **简要介绍**
> 

> **Optimization Approaches**
> * BasicNotes [[notes]](./Machine%20Learning/Optimization%20Algorithm/basisNote.md)
> * SGD [[notes]](./Machine%20Learning/Optimization%20Algorithm/SGD.md)
> * Momentum [[notes]](./Machine%20Learning/Optimization%20Algorithm/momentum.md)
> * RMSProp [[notes]](./Machine%20Learning/Optimization%20Algorithm/RMSProp.md)
> * AdaGrad [[pdf]](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) [[notes]](./Machine%20Learning/Optimization%20Algorithm/adaGrad.md)
> * Adam: [[pdf]](https://arxiv.org/pdf/1412.6980.pdf) [[notes]](./Machine%20Learning/Optimization%20Algorithm/adam.md)



# 应用场景

## *一、智能机器人领域*

### 1. SLAM

> **简要介绍**
> 
> SLAM（Simultaneous Localization and Mapping）指的是同时完成机器人的定位和地图构建的技术。
> SLAM的核心思想是利用传感器（如激光雷达，视觉等）搜集的数据来实时构建、更新和维护机器人的位置和环境地图，从而实现机器人的自主定位、自主导航等。
> SLAM的目的在于实现机器人的自主理解和规划，并且能够面对环境变化快速响应。 
> SLAM的技术结构上分为前端传感器数据处理和后端建模优化两个部分。
> 前端传感器处理主要包括数据采集、特征提取、路径追踪、数据融合等，以实现机器人的定位；后端建模优化主要包括非线性优化、地图构建等，实现机器人的环境建模和地图更新等功能。 
> SLAM技术对机器人导航技术具有重要意义，能够有效提高机器人的自主性，为机器人导航赋予智能。
> SLAM技术在机器人自动化领域也有着广泛应用，如自动驾驶、服务机器人等。


> **Paper List**
> 
> * a



## *二、计算机视觉领域*

### 1. Object Detection

> **简要介绍**
> 


> **Paper List**
> 
> * a



## *三、自然语言处理领域*

### 1. Natural Language Generation

> **简要介绍**


> **Paper List**
> 
> * a





## *四、程序语言领域*

### 1. Program Debugging and Fault Localization

> **简要介绍**
> 
> 

> **Paper List**
> 
> * A Survey of Algorithmic Debugging
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3106740) [[notes]](./Programming%20Language/Program%20Debugging/csur_algorithmic_debug.md)
> * Survey of Source Code Bug Detection Based on Deep Learning
[[pdf]](http://www.jos.org.cn/josen/article/abstract/6696) [[notes]](./Programming%20Language/Program%20Debugging/js_survey.md)
> * 



### 2. Automatic Program Repair

> **简要介绍**
> 
> 自动程序修复 (APR) 旨在使用来自软件工程、计算机编程和机器学习的算法和技术来自动检测和修复计算机程序中的错误。
> APR可以用来修复软件bug，优化代码，提高软件的整体性能。 它还可用于检测和修复代码中的安全漏洞。

> **Paper List**
> 
> * Automatic Patch Generation by Learning Correct Code
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/2837614.2837617) [[notes]](./Programming%20Language/Automatic%20Program%20Repair/Prophet.md)
> * TBar: Revisiting Template-Based Automated Program Repair
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3293882.3330577) [[notes]](./Programming%20Language/Automatic%20Program%20Repair/TBar.md)
> * DEAR: A Novel Deep Learning-based Approach for Automated Program Repair
[[pdf]](https://dl.acm.org/doi/pdf/10.1145/3510003.3510177) [notes](./Programming%20Language/Automatic%20Program%20Repair/DEAR.md)


### 3. Code Generation

> **简要介绍**
> 
> 代码生成是一种自动生成软件代码的技术。它可以根据用户提供的设计或模型自动生成软件代码，从而大大简化编码工作，提高软件开发的效率。 
> 代码生成可以分为四个基本步骤：分析、转换、生成和调试。分析阶段，用户需要提供模型或设计，以便系统能够识别要实现的功能。
> 在转换阶段，系统将模型或设计转换成代码，这一步的转换可以是从模型到代码的直接转换或者从模型到另一个中间模型的转换，然后再从中间模型到代码的转换。
> 在生成阶段，系统根据转换的结果生成代码。最后，在调试阶段，用户可以通过查看代码，检查并修正生成的代码，以确保生成的代码满足用户的要求。
> 代码生成技术可以用来自动生成各种软件，包括桌面应用程序、移动应用程序、Web应用程序、Web服务、数据库应用程序等。
> 它可以帮助软件开发者以快速、高效的方式实现软件的开发，进而提高软件开发的效率。


> **Paper List**
> 
> * Competition-Level Code Generation with AlphaCode [[pdf]](https://arxiv.org/pdf/2203.07814.pdf) [[notes]](./Programming%20Language/Code%20Generation/AlphaCode.md)

