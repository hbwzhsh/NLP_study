# encoding utf8
# author : King
# time : 2019.11.12
# 参考1：
''' 函数目录

    1. read_xlsx_data(filename,index_list):功能：加载 xlsx 数据
    2. read_xlsx_data_v2(filename,index_list):功能：加载 xlsx 数据；版本：2；功能升级：将 DataFrame Dict 数据转化为  List Dict
    3. read_xlsx_data_v3(filename,index_list,header=None): 、功能：加载 xlsx 数据；版本：3；功能升级：在版本 1 的基础上，加一个 header 输入项，以判断是否以 header 作为头部
    4. write_dict_to_xlsx(filename,df_dict): 功能：数据写入 xlsx

'''

import tools
import pandas as pd 

''' xlsx 数据操作 begin '''
# 功能：加载 xlsx 数据
def read_xlsx_data(filename,index_list):
    '''
    功能：加载 xlsx 数据
    :param filename:       Str   文件名称
    :param index_list:     List  Sheet 栏列表
    :return:
        df_dict       Dict    DataFrame 数据字典
    数据格式：
            df_dict['检查项目']:     检查项目
                                0    生长发育
                                1      精神
                                2  自主呼吸节律
                                3    呼吸节律
                                4      尿量
                                    ...
    '''
    df_dict = {}
    for index in index_list:
        df = pd.read_excel(filename,sheetname= index,header=None)
        print(index)
        df.columns = [index]
        print("len(df[{0}]):{1}".format(index,len(df)))
        df_dict[index] = df
    return df_dict

# 功能：加载 xlsx 数据
# 版本：2
# 功能升级：将 DataFrame Dict 数据转化为  List Dict
def read_xlsx_data_v2(filename,index_list):
    '''
    功能：加载 xlsx 数据
    :param filename:       Str   文件名称
    :param index_list:     List  Sheet 栏列表
    :return:
        list_dict       Dict    List 数据字典
    数据格式：
            list_dict['检查项目']:    ['生长发育','精神','自主呼吸节律',...]
            ...
    '''
    list_dict = {}
    for index in index_list:
        df = pd.read_excel(filename,sheetname= index,header=None)
        df.columns = [index]
        print("len(df[{0}]):{1}".format(index,len(df)))
        list_dict[index] = list(df[index])
    return list_dict

# 功能：加载 xlsx 数据
# 版本：3
# 功能升级：在版本 1 的基础上，加一个 header 输入项，以判断是否以 header 作为头部
def read_xlsx_data_v3(filename,index_list,header=None):
    '''
    功能：加载 xlsx 数据
    :param filename:        Str   文件名称
    :param index_list:      List  Sheet 栏列表
    :param header:          bool  判断是否以 header 作为头部
    :return:
        df_dict         DataFrame Dict   数据字典
    数据格式：
            数据格式：
            df_dict['检查项目']:     检查项目
                                0    生长发育
                                1      精神
                                2  自主呼吸节律
                                3    呼吸节律
                                4      尿量
                                    ...
    '''
    df_dict = {}
    for index in index_list:
        df = pd.read_excel(filename,sheetname= index,header=header)
        if header is None:
            df.columns = [index]
            print("len(df[{0}]):{1}".format(index,len(df)))
            df_dict[index] = df[index]
        else:
            df_dict[index] = df
    return df_dict

# 功能：数据写入 xlsx
def write_dict_to_xlsx(filename,df_dict):
    '''
    功能：加载 xlsx 数据
    :param filename:       Str   文件名称
    :param df_dict:       DataFrame 数据字典
    :return:
        
    数据格式：
            df_dict['检查项目']:     检查项目
                                0    生长发育
                                1      精神
                                2  自主呼吸节律
                                3    呼吸节律
                                4      尿量
                                    ...
    '''
    writer = pd.ExcelWriter(filename)
    for (index,val) in df_dict.items():
        val.to_excel(writer, sheet_name = index, index = False,header=None)
        
    writer.save()
    writer.close()
''' xlsx 数据操作 end '''
