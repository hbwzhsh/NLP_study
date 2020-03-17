# encoding utf8
import json
import jieba
import numpy as np
import pandas as pd
import os
''' 目录
    1. read_ner_labeled_data(filename,demo_flag=False): 功能：数据读取
    2. read_sent(filename,demo_flag=False,demo_num=50)： 功能：数据读取
    3. read_sent_min_len(filename,min_len = 3, demo_flag=False,demo_num=50): 功能：数据读取，限制句子最短长短
    4. write_sent(filename,sent_list,mode='w'): 功能：列表数据写入
    5. write_ner_labeled_data(filename,char_lists,tags_lists): 功能：命名实体识别标注数据写入
    6. listdir(path,file_type,origin_file_list,result_file_list): 功能：或者指定目录下的所有所有文件
    7. read_and_write_data(origin_file_list,result_file_list,write_file_name,mode='w'): 功能：将原文数据和标注数据汇总并写入文件
    8. tag_sent_pre(tag_sent_list): 功能：标注列表符号替换处理
    9. merge_sent_tag(sent_list,tag_sent_lists,is_cut = False): 功能：生成训练数据类别,处理后可采用 write_ner_labeled_data() 写入文件
    10. to_change(readfile,writefile): 功能：数据格式转换，生成训练数据类别,处理后可采用 write_ner_labeled_data() 写入文件
    11. build_jieba_dict_list(tag_sent_lists,class_list,word_frequency): 功能：构建 jieba 分词词典向量 
    12. read_origin_file(filename,demo_flag = False) 功能：读取原文 txt 文件数据
    13. read_result_file(filename,demo_flag = False) 功能：读取结果 txt 文件数据
    14. get_tag_set(result_list,sep='|') 功能： 获取标注类型
    15. transform_dict_key2val(key2val_dict) 功能：字典键值倒换  key-val  ->  val-key
    16. get_file_lists_data(file_list,read_file) 功能：批量获取文件数据
    17. tag_zh2en_fun(tag_lists,tag_zh2en,sep='-') 功能：对 标注数据进行中翻英或中翻英转化
    18. write_dict(filename,data_dict,sep='\t') 功能：字典数据写入
'''


'''工具包 begin'''
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

# 功能：停用词加载
def load_stopword(stopword_path):
    '''
    功能：停用词加载
    :param data_list:     List  需要处理的数据列表
    :return:
        f_stop_seg_list   List    停用词数据列表
    '''
    # 1.读取停用词文件
    with open_file(stopword_path) as f_stop:
        try:
            f_stop_text = f_stop.read()
        finally:
            f_stop.close()
    # 停用词清除
    f_stop_seg_list = f_stop_text.split('\n')
    return f_stop_seg_list

'''
    工具包 end
'''

'''数据读取 begin'''
# 功能：数据读取
def read_ner_labeled_data(filename,demo_flag=False):
    '''
        功能：数据读取
        :param filename:     Str  文件名称
        :return:
            sent_list       List    句子列表
            seg_str_lists   List    句子标注数据列表
        数据格式：
                sent1\n
                entity1(tag1) entity2(tag2) entity3(tag3)...\n
                \n
                sent2\n
                entity1(tag1) entity2(tag2) entity3(tag3)...\n
                \n
                ...
    '''
    contents_num = 0
    sent_list = []
    tag_sent_list = []
    with open_file(filename) as f:
        for line in f:
            line = line.strip().replace("\n","")
            if contents_num%3 == 0:
                sent_list.append(line)
            elif contents_num%3 == 1:
                tag_sent_list.append(line)
            contents_num = contents_num + 1
            if demo_flag and contents_num == 11:
                break
    return sent_list,tag_sent_list

# 功能：数据读取
def read_sent(filename,demo_flag=False,demo_num=50):
    '''
        功能：数据读取
        :param filename:     Str  文件名称
        :return:
            sent_list       List    句子列表
        数据格式：
                sent1\n
                sent2\n
                ...
    '''
    if not os.path.exists(filename):
        raise FileNotFountError("{0} 没有被找到！！".format(filename))

    contents_num = 0
    sent_list = []
    with open_file(filename) as f:
        for line in f:
            line = line.strip().replace("\n","")
            sent_list.append(line)
            contents_num = contents_num + 1
            if demo_flag and contents_num == demo_num:
                break
    return sent_list

# 功能：数据读取，限制句子最短长短
def read_sent_min_len(filename,min_len = 3, demo_flag=False,demo_num=50):
    '''
    功能：数据读取，限制句子最短长短
    :param filename:      Str  文件名称
    :param min_len:       int   句子最短长短
    :param demo_flag:     bool  文件名称
    :param demo_num:      int  文件名称
    :return:
        sent_list       List    句子列表
    数据格式：
            sent1\n
            sent2\n
            ...
    '''
    if not os.path.exists(filename):
        raise FileNotFountError("{0} 没有被找到！！".format(filename))

    contents_num = 0
    sent_list = []
    with open_file(filename) as f:
        for line in f:
            line = line.strip().replace("\n","").replace("\t","").replace(" ","")
            if len(line) > min_len:
                sent_list.append(line)
            contents_num = contents_num + 1
            if demo_flag and contents_num == demo_num:
                break
    return sent_list

# 功能：数据读取，限制句子最短长短
def read_sent_min_len_v2(filename,min_len = 3,clear_space=False, demo_flag=False,demo_num=50):
    '''
    功能：数据读取，限制句子最短长短
    :param filename:      Str   文件名称
    :param min_len:       int   句子最短长短
    :param clear_space:   bool  是否清除空格
    :param demo_flag:     bool  文件名称
    :param demo_num:      int   文件名称
    :return:
        sent_list       List    句子列表
    数据格式：
            sent1\n
            sent2\n
            ...
    '''
    if not os.path.exists(filename):
        raise FileNotFountError("{0} 没有被找到！！".format(filename))

    contents_num = 0
    sent_list = []
    with open_file(filename) as f:
        for line in f:
            line = line.strip().replace("\n","").replace("\t","")
            if clear_space:
                line = line.replace(" ","")
            if len(line) > min_len:
                sent_list.append(line)
            contents_num = contents_num + 1
            if demo_flag and contents_num == demo_num:
                break
    return sent_list

''' 数据读取 end '''

'''数据写入 begin'''
# 功能：列表数据写入
def write_sent(filename,sent_list,mode='w'):
    '''
        功能：列表数据写入
        :param filename:     Str  写入数据文件
        :param sent_list:     List  需要处理的数据列表
        数据写入格式：
                sent1\n
                sent2\n
                ...
    '''
    on_write = []
    with open_file(filename,mode) as f:
        for sent in sent_list:
            on_write.append("{0}\n".format(sent))
        f.writelines(on_write)

# 功能：命名实体识别标注数据写入
def write_ner_labeled_data(filename,char_lists,tags_lists):
    '''
    功能：命名实体识别标注数据写入
    :param filename:     Str  写入数据文件
    :param char_lists:     List  需要处理的数据列表
    :param tags_lists:     List  需要处理的数据列表
    数据写入格式：
            主   O
            诉   O
            :   O
            彩   B-seg
            超   I-seg
            ...
    '''
    on_write = []
    with open_file(filename,'w') as f:
        for (char_list,tags_list) in zip(char_lists,tags_lists):
            for (char,tags) in zip(char_list,tags_list):
                on_write.append("{0} {1}\n".format(char,tags))
            on_write.append("\n")
        f.writelines(on_write)

# 功能：命名实体识别标注数据写入
def write_ner_labeled_data_v1(filename,char_tags_lists):
    '''
    功能：命名实体识别标注数据写入
    :param filename:     Str  写入数据文件
    :param char_tags_lists:     List  需要处理的数据列表
    数据写入格式：
            主   O
            诉   O
            :   O
            彩   B-seg
            超   I-seg
            ...
    '''
    on_write = []
    with open_file(filename,'w') as f:
        for char_tags_list in char_tags_lists:
            for char_tags in char_tags_list:
                on_write.append("{0}\n".format(char_tags))
            on_write.append("\n")
        f.writelines(on_write)

# 功能：字典数据写入
def write_dict(filename,data_dict,mode,sep='\t'):
    '''
        功能：字典数据写入
        :param filename:     Str  写入数据文件
        :param data_dict:     Dict  需要处理的数据字典
        :param data_dict:     Dict  需要处理的数据字典
        数据写入格式：
                sent1\n
                sent2\n
                ...
    '''
    on_write = []
    with open_file(filename,mode) as f:
        for (key,val) in data_dict.items():
            on_write.append("{0}{1}{2}\n".format(key,sep,val))
        f.writelines(on_write)

''' 数据写入 end '''


'''将原始数据中的原始数据和标注数据进行汇总并写入 begin'''

# 功能：获取指定目录下的所有指定后缀的文件名
def listdir(path,file_type,origin_file_list,result_file_list):  
    '''
    功能：获取指定目录下的所有指定后缀的文件名
    :param path:                 Str         root 路径
    :param file_type:            Str         文件类型 eg ".txt"
    :param origin_file_list:     List        原文文件路径
    :param result_file_list:     List        标注结果文件路径
    :return:
    
    文件目录布局：
    - origin 
    --- 癫痫数据标注1
    ----- 原文
    ------- 传染病史_创伤性硬膜下出血_神经外科病区_2419.txt
          ...
    ----- 结果
    ------- 传染病史_创伤性硬膜下出血_神经外科病区_2419.txt
          ...
    ...

    数据输出格式：
        origin_file_list 数据输出格式：
            [
                'origin_data/原文\\4476.txt', 
                'origin_data/原文\\4477.txt', 
                ...
            ]
        result_file_list 数据输出格式：
             [
                'origin_data/结果\\4476.txt', 
                'origin_data/结果\\4477.txt',
                ...
            ]
    使用方式：
        path = 'origin_data/'
        origin_file_list = []
        result_file_list = []
        tools.listdir(path,'.txt',origin_file_list,result_file_list)
        
    '''
    for file in os.listdir(path):  
        file_path = os.path.join(path, file)  
        if os.path.isdir(file_path):  
            listdir(file_path,file_type, origin_file_list,result_file_list)  
        elif os.path.splitext(file_path)[1]==file_type:
            if '原文' in os.path.splitext(file_path)[0]:
                origin_file_list.append(file_path)
            elif '结果' in os.path.splitext(file_path)[0]:
                result_file_list.append(file_path)

# 功能：批量获取文件数据
def get_file_lists_data(file_list,read_file):
    ''' 功能：批量获取文件数据
        :param file_list:    List  文件数据列表
        :param read_file:    Fun   文件读取函数，eg: read_origin_file,read_result_file
        :return:
            lists:   list    内容列表

    '''
    lists = []
    for file in file_list:
        temp_list = read_file(file)
        lists = lists + temp_list
    return lists

# 读原文数据文件读取
def read_origin_file(filename,demo_flag = False):
    ''' 功能：读取 txt 文件数据
        :param filename:    String 文件名称包含路径
        :param demo_flag:   String True 只读取 1000 样本数据，Fasle 读取全部数据
        :return:
            origin_list:   list    内容列表

        txt 文件数据格式： 
            sent1\n
            sent2\n
            ...
    '''
    sent_list = []
    with open_file(filename) as f:
        for line in f:
            line = line.strip().replace('\n','')
            sent_list.append(line)

    return sent_list

# 标注结果数据读取
def read_result_file(filename,demo_flag = False):
    '''
        标注结果数据读取
        :param filename:    String 文件名称包含路径
        :param demo_flag:   String True 只读取 1000 样本数据，Fasle 读取全部数据
        :return:
            result_list:   list    内容列表

        txt 文件数据格式： 
            反复|程度
            腹泻|症状

            2月|时长
            咳嗽|症状
            5天|时长
            ...
    '''
    sent_list = []
    with open_file(filename) as f:
        lines = f.read()
        new_sent_list = []
        sent_list = lines.split("\n\n")
        for sent in sent_list:
            new_sent_list.append([s  for s in sent.split("\n") if s!=''])
    return new_sent_list

# 获取标注类型
def get_tag_set(result_list,sep='|'):
    ''' 获取标注类型
        :param result_list:    List    结果数据
        :param sep:            String  分隔符 
        :return:
            tag_set:   set    集合

        tag_set 数据格式： 
            {'部位', '症状', '时长', '程度', '形容词'}
    '''
    tag_set = set()
    for result in result_list:
        for r in result:
            tag_set.add(r.split(sep)[1])
    return tag_set

# 键值倒换  key-val  ->  val-key
def transform_dict_key2val(key2val_dict):
    ''' 功能：键值倒换  key-val  ->  val-key
        :param key2val_dict:    Dict    键值转化字典
        :return:
            val2key_dict:   Dict    键值转化字典

        key2val_dict 数据格式： 
            {'time': '时长', 'level': '程度', 'adj': '形容词', 'symptom': '症状', 'part': '部位'}
    '''
    val2key_dict = {}
    for (key,val) in key2val_dict.items():
        val2key_dict[val] = key
    return val2key_dict

# 功能：对 标注数据进行中翻英或中翻英转化
def tag_zh2en_fun(tag_lists,tag_zh2en,sep='-'):
    ''' 功能：对 标注数据进行中翻英或中翻英转化
        :param tag_lists:       List    标注列表
        :param key2val_dict:    Dict    键值转化字典
        :return:
            new_tag_lists:   List    标注列表

        tag_lists 数据格式： 
            [
                ['B-症状', 'I-症状', 'O',...], 
                ['B-程度', 'I-程度', ...],
                ...
            ]
        new_tag_lists 数据格式： 
            [
                ['B-symptom', 'I-symptom', 'O',...], 
                ['B-level', 'I-level', 'B-symptom', ...],
                ...
            ]
    '''
    new_tag_lists = []
    for tag_list in tag_lists:
        new_tag_list = []
        for tag in tag_list:
            if len(tag.split(sep)) == 2:
                t = tag.split(sep)
                new_tag_list.append("{0}-{1}".format(t[0],tag_zh2en[t[1]]))
            else:
                new_tag_list.append(tag)
        new_tag_lists.append(new_tag_list)
    return new_tag_lists

# 功能：将原文数据和标注数据汇总并写入文件
def read_and_write_data(origin_file_list,result_file_list,write_file_name,mode='w'):
    '''
    功能：将原文数据和标注数据汇总并写入文件
    :param origin_file_list:     List        原文文件路径
    :param result_file_list:     List        标注结果文件路径
    :param write_file_name:      Str         写入文件名称
    :param mode:                 Str         写入模式，w or a
    :return:
    
    写入格式： 
    origin_sent1 \n 
    result_sent1 \n
    \n
    origin_sent2 \n 
    result_sent2 \n
    \n
    ...
        
    '''
    with open_file(write_file_name,mode) as f_w:
        on_write = []
        
        for (origin_file,result_file) in zip(origin_file_list,result_file_list):
            with open_file(origin_file) as f_origin, open_file(result_file) as f_result:
                temp_origin = []
                for line in f_origin:
                    temp_origin.append(line.strip().replace("\n",""))
                    
                temp_result = []
                for line in f_result:
                    temp_result.append(line.strip().replace("\n",""))
                
                try:
                    if len(temp_origin) != len(temp_result):
                        raise Exception("数据长度问题！！！")
                except Exception as e:
                    print(e)
                    
                for (origin,result) in zip(temp_origin,temp_result):
                    on_write.append("{0}\n{1}\n\n".format(origin,result))
                    
        f_w.writelines(on_write)

''' 将原始数据中的原始数据和标注数据进行汇总并写入 end '''



''' 异常处理 begin '''
class FileNotFountError(Exception):
    def __init__(self, arg):
        self.args = arg