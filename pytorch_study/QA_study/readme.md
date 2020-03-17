# 使用PyTorch实现基于百度的中文问答数据集WebQA 数据 Chatbot

## 一、环境

- pytorch = 0.4.1
- python = 3.6 +
- jieba

## 二、文件介绍

1. config.py 配置文件
2. load_data.py 数据预处理
3. create_voc.py 创建词典文件
4. seq2seq_model.py 模型文件
5. main.py 主文件

## 三、运行

### 1、训练

1. 设置 config.py 配置文件
```
    # 模式 train or test
    self.mode = 'train'
```

2. 训练
```
    python main.py
```

### 2、测试
1. 设置 config.py 配置文件
```
    # 模式 train or test
    self.mode = 'test'
```

2. 测试  
```
    python main.py
```