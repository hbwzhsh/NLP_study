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
import re

from config import Config
config = Config()

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {config.PAD_token: "PAD", config.SOS_token: "SOS", config.EOS_token: "EOS"}
        self.num_words = 3  # 目前有SOS, EOS, PAD这3个token。

    # 将句子中每个词添加到 Voc 类中
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # 删除频次小于min_count的token 
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # 重新构造词典 
        self.word2index = {}
        self.word2count = {}
        self.index2word = {config.PAD_token: "PAD", config.SOS_token: "SOS", config.EOS_token: "EOS"}
        self.num_words = 3 # Count default tokens
        
        # 重新构造后词频就没有意义了(都是1)
        for word in keep_words:
            self.addWord(word)

# 把Unicode字符串变成ASCII
# 参考https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 读取问答句对并且返回Voc词典对象 
def readVocs(question_cut_list,answer_list):
    print("Reading lines...")
    pairs = [[question,answer] for (question,answer) in zip(question_cut_list,answer_list)]

    voc = Voc("baidu_qa")
    return voc, pairs

def filterPair(p,max_length): 
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length

# 过滤太长的句对 
def filterPairs(pairs,max_length):
    return [pair for pair in pairs if filterPair(pair,max_length)]

# 使用上面的函数进行处理，返回Voc对象和句对的list 
def loadPrepareData(question_cut_list,answer_list,max_length):
    print("Start preparing training data ...")
    voc, pairs = readVocs(question_cut_list,answer_list)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filterPairs(pairs,max_length)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])
    print("Counted words:", voc.num_words)
    return voc, pairs

'''
    我们可以看到，原来共有221282个句对，经过处理后我们只保留了64271个句对。
    另外为了收敛更快，我们可以去除掉一些低频词。这可以分为两步：
    1) 使用voc.trim函数去掉频次低于MIN_COUNT 的词。
    2) 去掉包含低频词的句子(只保留这样的句子——每一个词都是高频的，也就是在voc中出现的)
'''
def trimRareWords(voc, pairs, MIN_COUNT):
    # 去掉voc中频次小于1的词 
    voc.trim(MIN_COUNT)
    # 保留的句对 
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # 检查问题
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # 检查答案
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # 如果问题和答案都只包含高频词，我们才保留这个句对
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), 
		len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs
