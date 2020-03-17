# encoding utf8
# author : King
# time : 2019.11.12
''' 函数目录

    1. data_pre(df,index_name)：pandas 行遍历

'''

import pandas as pd


# pandas 行遍历
def df_index_print(df,index_name):
    '''
    功能：pandas 行遍历
    :param df:           DataFrame 
    :param index_name:   Str 
    :return:
        Str or None : 如果不重复则返回该值，否则返回 None
    '''
    for index,row in df.T.iteritems():
        print(row[index_name])

# 


# 获取列名 list(df.columns.values)

if __name__ == '__main__':
    df_test = pd.DataFrame()

    df_test['测试'] = ['aaa[真的]bbbb{确定}','aaa','我是一个(a)人','我是一个人','我是一个人(中国人)']
    df_test['new'] =''
    data_pre(df_test,'测试')

