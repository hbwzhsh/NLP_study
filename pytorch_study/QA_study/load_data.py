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

'''
    工具包 begin
'''
import sys
if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False

def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word

def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content

def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)

'''
    工具包 end
'''
'''
    3  百度的中文问答数据集WebQA 数据预处理
'''
import json 
# 加载数据集
def load_data(data_path_name):
    with open(data_path_name,'r') as load_f:
        load_dict = json.load(load_f)

    question_list=[]
    answer_list=[]
    for (key,value) in load_dict.items():
        
        #print("key:{0};value:{1}".format(key,value))
        #print("question:{0}".format(value['question']))
        question_list.append(value['question'])
        for (key,value) in value['evidences'].items():
            #print("value:{0}".format(value['answer'][0]))
            answer_list.append(value['answer'][0])
        
    # for (question,answer) in zip(question_list,answer_list):
    #     print("question:{0}\tanswer:{1}".format(question,answer))
    return question_list,answer_list

# 分词
# 利用jieba分词对 question 句子进行分词，并储存到 qa_df['question_cut'] 中
import jieba
def stopwords_file_load(stopword_path=""):
    # 1.读取停用词文件
    with open_file(stopword_path) as f_stop:
        try:
            f_stop_text = f_stop.read()
        finally:
            f_stop.close()
    # 停用词清除
    f_stop_seg_list = f_stop_text.split('\n')
    return f_stop_seg_list

from config import Config
config = Config()
# 停用词加载
f_stop_seg_list = stopwords_file_load(stopword_path=config.stopword_path)
print("f_stop_seg_list:{0}".format(f_stop_seg_list))
def jieba_cut_word(subject,f_stop_seg_list=f_stop_seg_list):
    seg_list = jieba.cut(subject, cut_all=False)
    word_list = list(seg_list)
    mywordlist = []
    for myword in word_list:
        if not (myword in f_stop_seg_list):
            mywordlist.append(myword)

    word_list = " ".join(mywordlist)
    return word_list



# question_cut_list = qa_df['question'].apply(jieba_cut_word)

# if __name__ == "__main__":
#     # 数据加载
#     data_path_name="F:/document/datasets/nlpData/conversation_dataset/baidu_cn_WebQA/me_validation.ann.json"
#     question_list,answer_list=load_data(data_path_name)
#     print("question_list[0:2]:{0}".format(question_list[0:2]))
#     print("answer_list[0:2]:{0}".format(answer_list[0:2]))

#     # 停用词加载
#     f_stop_seg_list = stopwords_file_load(stopword_path="F:/document/datasets/jieba停用词and词典/stopwords1.txt")
#     print("f_stop_seg_list:{0}".format(f_stop_seg_list))

#     # 问题分词
#     question_cut_list = [jieba_cut_word(question,f_stop_seg_list) for question in question_list]
#     print("question_cut_list[0:2]:{0}".format(question_cut_list[0:2]))



