# encoding utf8
# encoding utf8
# author : King
# time : 2019.11.22
# 参考1：
''' 函数目录

    1. : 功能：


'''
from functools import reduce

# 功能：列表项为dict的去重,只能处理包含一对键值对的词典列表
def deleteDuplicate(li):
    ''' 功能：列表项为dict的去重,只能处理包含一对键值对的词典列表
        :param li:        List       待去重列表
        :return 
            li    List  已去重列表
        :url: https://www.jianshu.com/p/980e44949a84
        :use
            li = [{'a': 1}, {'b': 2}, {'a': 1}]
            deleteDuplicate(li)
            [{'a': 1}, {'b': 2}]
    '''
    func = lambda x, y: x if y in x else x + [y]
    li = reduce(func, [[], ] + li)
    return li

# 功能：列表项为dict的去重,只能处理包含多对键值对的词典列表
def deleteDuplicate_2(li):
    ''' 功能：列表项为dict的去重,只能处理包含多对键值对的词典列表
        :param li:        List       待去重列表
        :return 
            li    List  已去重列表
        :url: https://www.jianshu.com/p/980e44949a84
        :use
            li = [{'a': 1,'b': 2}, {'b': 2}, {'a': 1,'b': 2}]
            deleteDuplicate(li)
            [{'a': 1,'b': 2}, {'b': 2}]
    '''
    li = li = [dict(t) for t in set([tuple(d.items()) for d in li])]
    return li

# 功能：去除二维数组/二维列表中的重复行
def deleteDuplicate_two_dim_list(li):
    ''' 功能：列表项为dict的去重
        :param li:        List       待去重列表
        :return 
            li    List  已去重列表
        :url: https://blog.csdn.net/u012991043/article/details/81067207
        :use
            li = [[1, 2],[3, 4],[5, 6],[7, 8],[3, 4],[1, 2]]
            deleteDuplicate_two_dim_list(li)
            [[3, 4, 4], [5, 6, 4], [7, 8, 4], [1, 2, 4]]
    '''
    li = [list(l) for l in list(set([tuple(t) for t in li]))]
    return li

if __name__ == "__main__":
    li = [{'a': 1,'b': 2}, {'b': 2}, {'a': 1,'b': 2}]
    li = deleteDuplicate(li)
    print(li)

    li = [[1, 2, 4],[3, 4, 4],[5, 6, 4],[7, 8, 4],[3, 4, 4],[1, 2, 4]]
    li = deleteDuplicate_two_dim_list(li)
    print(li)