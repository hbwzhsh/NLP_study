# NLP_study

## 介绍

本项目主要为本人对于 NLP 学习 过程中所积累下来的项目经验。

## 目录内容介绍

### eda_study 目录

#### commom_eda_study 项目

##### 项目介绍

常用的数据增强策列（增删改）

#### EDA_Markov_Chain 项目

##### 项目介绍

基于Markov Chain的极简数据增强方法

### algorithm_study 目录

#### self-attention_study 目录

##### 项目jeis

self-attention 学习


### tensorflow_study 目录

该目录下的所有 项目 都是 由  tensorflow 编写，具体 工程如下：

#### ner_study 目录

该项目主要是针对命名实体识别 方向。

##### bilstm_idcnn 目录

###### 项目介绍

该项目主要 采用  BiLSTM-CRF 和 IDCNN-CRF 两种 方法 进行命名实体识别。

###### 项目要求

* Python (>=3.5)

* TensorFlow (>=r1.0)

* jieba (>=0.37)

###### 项目使用

1. 数据准备。创建 data/ 目录，用于存放 训练数据 和 词向量(vec.txt) 
2. 手动创建 ckpt/ 、log/、result/ 目录 用于 保持 相关训练结果
3. 调参。修改 main.py FLAGS_Config 类 

```python
    self.clean = True   # clean train folder
    self.train = True   # Whether train the model
    train_data_path = "../data/"
    self.emb_file = os.path.join(train_data_path, "vec.txt")   
    self.train_file = os.path.join(train_data_path, "example.train")   
    self.dev_file = os.path.join(train_data_path, "example.dev")   
    self.test_file =  os.path.join(train_data_path, "example.test")
    self.model_type =  "idcnn"  # Model type, can be idcnn or bilstm
```

4. 训练。在命令行里面输入命令
```
    python main.py
```

5. 测试。修改 main.py FLAGS_Config 类 

```python
    self.clean = False   # clean train folder
    self.train = False   # Whether train the model
```
然后
```
    python main.py
```

###### 结果

1. IDCNN-CRF 效果评估

1.1 测试 一

> 参数
> seg_dim：30
> char_dim 300
> lstm_dim 100
> dropout 0.5
> batch_size 64
> decay_rate 0.9
> lr 0.001
> optimizer "adam"
> IDCNN dilation:  1 1 2

```
ALL: accuracy:  97.97%; precision:  85.31%; recall:  85.18%; FB1:  85.24

LOC: accuracy:  99.57%; precision:  87.59%; recall:  87.37%; FB1:  87.48  4989

LOC: accuracy:  99.57%; precision:  87.59%; recall:  87.37%; FB1:  87.48  4989

ORG: accuracy:  99.62%; precision:  79.55%; recall:  79.53%; FB1:  79.54  2739

ORG: accuracy:  99.62%; precision:  79.55%; recall:  79.53%; FB1:  79.54  2739

PER: accuracy:  99.78%; precision:  87.08%; recall:  87.05%; FB1:  87.07  2439

```

1.1 测试 二

> 参数
> seg_dim：30
> char_dim 300
> lstm_dim 100
> dropout 0.5
> batch_size 64
> decay_rate 0.9
> lr 0.001
> optimizer "adam"
> IDCNN dilation:  1 1 1 2

```
ALL: accuracy:  97.87%; precision:  85.29%; recall:  83.39%; FB1:  84.33

LOC: accuracy:  99.56%; precision:  85.93%; recall:  87.58%; FB1:  86.75  4939

LOC: accuracy:  99.56%; precision:  85.93%; recall:  87.58%; FB1:  86.75  4939

ORG: accuracy:  99.62%; precision:  81.73%; recall:  78.32%; FB1:  79.99  2709

ORG: accuracy:  99.62%; precision:  81.73%; recall:  78.32%; FB1:  79.99  2709

PER: accuracy:  99.74%; precision:  88.09%; recall:  81.05%; FB1:  84.43  2326

PER: accuracy:  99.74%; precision:  88.09%; recall:  81.05%; FB1:  84.43  2326

```




### keras_study 目录

该目录下的所有 项目 都是 由  tensorflow 编写，具体 工程如下：

#### classifier_study

##### 项目介绍

文本分类常用方法汇总：    
    - FastText
    - TextCNN
    - charCNN
    - TextRNN
    - TextRCNN
    - TextDCNN
    - TextDPCNN
    - TextVDCNN
    - TextCRNN
    - DeepMoji
    - SelfAttention
    - HAN
    - CapsuleNet
    - Transformer-encode

##### 项目要求

    - gensim>=3.7.1
    - jieba>=0.39
    - numpy>=1.16.2
    - pandas>=0.23.4
    - scikit-learn>=0.19.1
    - tflearn>=0.3.2
    - tqdm>=4.31.1
    - passlib>=1.7.1
    - keras==2.2.4
    - tensorflow-gpu==1.12.0
    - keras-bert>=0.80.0
    - keras-xlnet>=0.16.0
    - keras-adaptive-softmax>=0.6.0

##### 项目使用

看 具体介绍

#### muli_label_classifier_study

##### 项目介绍

多标签文本分类常用方法汇总：    
    - FastText
    - TextCNN
    - TextRNN
    - TextRCNN
    - SelfAttention
    - HAN
    - Transformer-encode

##### 项目要求

    - gensim>=3.7.1
    - jieba>=0.39
    - numpy>=1.16.2
    - pandas>=0.23.4
    - scikit-learn>=0.19.1
    - tflearn>=0.3.2
    - tqdm>=4.31.1
    - passlib>=1.7.1
    - keras==2.2.4
    - tensorflow-gpu==1.12.0
    - keras-bert>=0.80.0
    - keras-xlnet>=0.16.0
    - keras-adaptive-softmax>=0.6.0

##### 项目使用

看 具体介绍

### pytorch_study 目录

#### Chinese_Text_Classification

##### 项目介绍

中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer, 基于pytorch，开箱即用。

##### 项目要求

python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX

#### NER_study

##### 项目介绍

命名实体识别项目

#### QA_study

##### 项目介绍

自动问答项目