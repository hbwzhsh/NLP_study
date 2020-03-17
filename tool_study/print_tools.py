# encoding utf8
# author : King
# time : 2019.11.12
# 参考1：
''' 函数目录

    1. print_list(lists,num=0):功能：列表打印
    2. print_dict(data_dict,limit=0):功能：字典打印
    3. print_dict_list(data_dict,end=5,limit=0): 功能：字典列表打印

'''
import tools
import pandas as pd 

''' 数据打印模块 begin '''
# 功能：列表打印
def print_list(lists,num=0):
    '''
    功能：列表打印
    :param lists:     list  需要打印的列表
    :return:
            
    '''
    if num == 0:
        num = len(lists)

    print("--------------start--------------")
    for i in range(0,num):
        print("{0}: {1}".format(i+1,lists[i]))

# 功能：字典打印
def print_dict(data_dict,limit=0):
    '''
    功能：字典打印
    :param data_dict:     Dict  需要打印的字典
    :return:
            
    '''
    if limit == 0:
        limit = len(data_dict)
    num = 0
    print("--------------start--------------")
    for (index,val) in data_dict.items():
        print("data_dict['{0}']:{1}".format(index,val))
        num = num + 1
        if num > limit:
            break

# 功能：字典列表打印
def print_dict_list(data_dict,end=5,limit=0):
    '''
    功能：字典列表打印
    :param data_dict:     Dict  需要打印的字典列表
    :return:
            
    '''
    if limit == 0:
        limit = len(data_dict)
    num = 0
    print("--------------start--------------")
    for (index,val) in data_dict.items():
        print("data_dict['{0}']:{1}".format(index,val[0:end]))
        num = num + 1
        if num > limit:
            break

''' 数据打印模块 end '''