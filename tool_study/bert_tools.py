# encoding utf8
# author : King
# time : 2019.11.07
# 参考1：快速使用 BERT 生成词向量：bert-as-service【https://blog.csdn.net/qq_34832393/article/details/90414293】
# 参考2：bert-as-service.readthedocs 【https://bert-as-service.readthedocs.io/en/latest/】
''' 函数目录

    1. Bert_Class(object): bert_as_service 工具包


'''

from bert_serving.client import BertClient
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# bert_as_service 工具包
class Bert_Class(object):
    """docstring for ClassName"""
    def __init__(self):
        self.__bc = BertClient()

    # 功能：Dataframe 获取 Bert 句向量
    def get_vec_to_df(self,df,title_name):
        '''
        # 功能：Dataframe 获取 Bert 句向量
        :param df:           DataFrame 
        :param title_name:   String        需要编码的标题
        :return:
            vec:            arrry    bert 句向量
        '''
        return self.__bc.encode([df[title_name]])[0]


    # 功能： Dict 获取 Bert 句向量
    def get_vec_to_dict(self,text_list):
        '''
        # 功能：Dict 获取 Bert 句向量
        :param text_list:        String        文本向量
        :return:
            vec_dict:            Dict          bert 句向量字典
        '''
        vec_dict = {}
        for text in text_list:
            vec_dict[text] = self.__bc.encode([text])[0]

        return vec_dict


    # 功能： 计算两文本之间的相关性系数
    def cul_vec_cosine(self,text1,text2):
        '''
        功能： 计算两文本之间的相关性系数
        :param text1:        String        文本1
        :param text2:        String        文本2
        :return:
            np.corrcoef(z)[0][1]:            float          相关系数
        '''
        text1_vec,text2_vec = self.__bc.encode([text1,text2])
        z=np.vstack([text1_vec,text2_vec])
        return np.corrcoef(z)[0][1]




if __name__ == '__main__':
    bert_class = Bert_Class()
    print(bert_class.cul_vec_cosine('abcd','defg'))

