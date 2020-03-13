# NOTE

### Word Vection

- word2vec：将词语转化成数值形式，或者嵌入到数值空间。
- one-hot representation：常见的词向量表达形式，用很长的向量表示一个词，向量的长度为为词典大小。向量的分量只有一个 1，其他全为 0， 1 的位置对应该词在词典中的位置，如00001,00010,00100,01000,10000。杜热编码（One-Hot encoding）也是一种方式，但对任意词的余弦相似度都为0，都是正交，无法表达不同词之间的相似度。
- 余弦相似度：通过测量两个向量的夹角的余弦值来度量其相似性，范围-1~1,但通常只用正空间，即0-1.

### 评估矩阵Envaluation Matrics

- 评估算法可以直接用训练数据来评估，容易出现过度拟合，难以发现算法的不足。

- 避免过度拟合是分类器设计中的一个核心任务，通常采用增大数据量和评估数据集的方法对分类器进行评估。 

1. 分离数据集方法：分离训练数据集和评估数据集、K折交叉验证分离、弃一交叉验证分离、重复随机评估，训练数据集分离
2. 分类算法评估矩阵：分类准确度、对数损失函数、AUC矩阵、混淆矩阵、分类报告
3. 回归算法评估矩阵：1.平方绝对误差（MAE） 2.均方误差（MSE） 3.决定系数(R2)。

   **AUC矩阵**
   ROC和AUC是评价分类器的指标。ROC是受试者工作特征曲线的简写，又称为感受性曲线。得此名在于曲线上各点反映相同的感受性，它们都是对同一信号刺激的反应，只不过是在几种不同的判定标准下所得的结果而已。ROC纵轴是“真正例率”，横轴是“假正例率”。AUC是处于ROC下方的那部分面积大小。通常，AUC的值介于0.5和1之间，AUC的值越大，诊断准确性越高。在ROC曲线上，靠近坐标图左上方的点为敏感性和特异性均较高的临界值。

###  Baseline & Benchmark model
简单的说：benchmark 是一个过程，baseline 是benckmark 这个过程中的一次实例。
baseline：a standard measurement or fact against which other measurements or facts are compared, especially in medicine or science. baseline可以理解成在比较中作为“参照物”的存在，强调比较，在比较中作为参照物，基线；
benchmark：something that is used as a standard by which other things can be judged or measured. benchmark本身是一种标准、规则。

### Keras on Deep Learning

https://machinelearningmastery.com/prepare-text-data-deep-learning-keras/

### CNN自然语言处理介绍

https://zhuanlan.zhihu.com/p/28087321






## 参考
- 代码参考：https://github.com/Kirupakaran/Toxic-comments-classification
- word2vec：https://zhuanlan.zhihu.com/p/26306795
- 词向量：http://wiki.jikexueyuan.com/project/deep-learning/word-vector.html
- 算法评估&评估矩阵：https://blog.csdn.net/Heloiselt/article/details/80870794

- keras 词向量的介绍：https://keras-cn-docs.readthedocs.io/zh_CN/latest/blog/word_embedding/

