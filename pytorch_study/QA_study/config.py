# -*- coding: utf-8 -*-
'''
    @Author: King
    @Date: 2019.03.20
    @Purpose: 使用PyTorch实现Chatbot
    @Introduction:  下面介绍
    @Datasets: 百度的中文问答数据集WebQA 数据
    @Link : 
    @Reference : 
'''
import torch
import os
class Config():
    def __init__(self):

        # 数据加载
        self.data_path_name="F:/document/datasets/nlpData/conversation_dataset/baidu_cn_WebQA/me_validation.ann.json"
        self.stopword_path="F:/document/datasets/jieba停用词and词典/stopwords1.txt"

        # 模式 train or test
        self.mode = 'test'

        self.MAX_LENGTH = 10  # 句子最大长度是10个词(包括EOS等特殊词)
        # 预定义的token
        self.PAD_token = 0  # 表示padding 
        self.SOS_token = 1  # 句子的开始 
        self.EOS_token = 2  # 句子的结束 
        self.MIN_COUNT = 1    # 阈值为1

        # 训练和测试模型
        self.USE_CUDA = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.USE_CUDA else "cpu")

        # 配置模型
        self.model_name = 'cb_model'
        self.attn_model = 'dot'
        #attn_model = 'general'
        #attn_model = 'concat'
        self.hidden_size = 500
        self.encoder_n_layers = 2
        self.decoder_n_layers = 2
        self.dropout = 0.1
        self.batch_size = 64

        # 从哪个checkpoint恢复，如果是None，那么从头开始训练。
        self.loadFilename = None
        self.checkpoint_iter = 100

        # 配置训练的超参数和优化器 
        self.save_dir = os.path.join("data", "save")
        self.corpus_name = "baidu_qa"
        self.clip = 50.0
        self.teacher_forcing_ratio = 1.0
        self.learning_rate = 0.0001
        self.decoder_learning_ratio = 5.0
        self.n_iteration = 4000
        self.print_every = 100
        self.save_every = 100