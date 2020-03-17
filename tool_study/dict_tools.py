# encoding utf8
# encoding utf8
# author : King
# time : 2019.11.22
# 参考1：
''' 函数目录

    1. dict_key_zh2en(zh2en_dict,list_dict): 功能：字典键值 中英翻译


'''


# 功能：字典键值 中英翻译
def dict_key_zh2en(zh2en_dict,list_dict):
    '''
        功能：字典键值 中英翻译
        :param zh2en_dict:     Dict  中英字典
        :param list_dict:        List Dict   词库字典
        :return:
            temp_List_dict:      List Dict   转化后的 词库字典         
    '''
    temp_list_dict = list_dict.copy()
    for (index,val) in zh2en_dict.items():
        if index in temp_list_dict:
            temp_list_dict[val]=temp_list_dict.pop(index)
    return temp_list_dict