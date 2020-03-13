# encoding utf8
import json
import jieba
import numpy as np

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

'''
    数据读取 begin
'''
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

def read_sent(filename,demo_flag=False):
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
    contents_num = 0
    sent_list = []
    with open_file(filename) as f:
        for line in f:
            line = line.strip().replace("\n","")
            sent_list.append(line)
            contents_num = contents_num + 1
            if demo_flag and contents_num == 11:
                break
    return sent_list

'''
    数据读取 end
'''

'''
    数据写入 begin
'''
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
    with open_file(filename,'w') as f:
        for sent in sent_list:
            on_write.append("{0}\n".format(sent))
        f.writelines(on_write)

# 功能：命名实体识别标注数据写入
def write_ner_labeled_data(filename,char_lists,tags_lists,mode='w'):
    '''
        功能：命名实体识别标注数据写入
        :param filename:     Str  写入数据文件
        :param char_lists:     List  需要处理的数据列表
        :param tags_lists:     List  需要处理的数据列表
        数据写入格式：
                主	O
                诉	O
                :	O
                彩	B-seg
                超	I-seg
                发	O
                现	O
                动	B-seg
                脉	I-seg
                导	I-seg
                管	I-seg
                未	I-seg
                ...
    '''
    on_write = []
    with open_file(filename,mode) as f:
        for (char_list,tags_list) in zip(char_lists,tags_lists):
            for (char,tags) in zip(char_list,tags_list):
                on_write.append("{0} {1}\n".format(char,tags))
            on_write.append("\n")
        f.writelines(on_write)

# 功能：命名实体识别未标注数据写入
def write_ner_unlabeled_data(filename,sent_lists,mode='w'):
    '''
        功能：命名实体识别未标注数据写入
        :param filename:     Str  写入数据文件
        :param sent_lists:     List  需要处理的数据列表
        数据写入格式：
                主   
                诉   
                :   
                彩   
                ...
    '''
    on_write = []
    with open_file(filename,mode) as f:
        for sent in sent_lists:
            for char in sent:
                on_write.append("{0}\n".format(char))
            on_write.append("\n")
        f.writelines(on_write)

'''
    数据写入 end
'''

'''
    数据处理 begin
'''
# 功能：标注列表符号替换处理
def tag_sent_pre(tag_sent_list):
    '''
    功能：标注列表符号替换处理
    :param tag_sent_list:     List  需要处理的数据列表
    :return:
        tag_sent_lists:     List  处理后的数据列表
    原始数据格式：
            彩超[检查项目] 动脉导管未闭[诊断] ...
            ...
    处理后数据格式：
            彩超|检查项目 动脉导管未闭|诊断 ...
            ...
    '''
    tag_sent_lists = []
    for tag_sent in tag_sent_list:
        tag_sent_list = tag_sent.replace("[","|").replace("]","").split(" ")
        tag_sent_lists.append(tag_sent_list)
    return tag_sent_lists

# 功能：生成训练数据类别,处理后可采用 write_ner_labeled_data() 写入文件
def merge_sent_tag(sent_list,tag_sent_lists,is_cut = False):
    '''
    功能：生成训练数据类别,处理后可采用 write_ner_labeled_data() 写入文件
    :param sent_list:     List  句子数据列表
    :param tag_sent_lists:     List  句子所对应的标注列表
    :param is_cut:     Bool  是否为分词任务
    :return:
        new_sent_lists:     List  处理后的数据列表
        new_tag_lists:     List  处理后的数据列表
        list(tags_type_set):     List  标注类别
    原始数据格式：
            sent_list：彩超发现动脉导管未闭...
            tag_sent_lists:彩超[检查项目] 动脉导管未闭[诊断]...
            ...
    处理后数据格式：
            new_sent_lists:[['彩','超',...],...]
            new_tag_lists:[['B-seg','I-seg',...],...]
    '''

    new_sent_lists,new_tag_lists = [],[]
    tags_type_set = set()            # 用于保存 标签类型
    for (sentence,tag_sent_list) in zip(sent_list,tag_sent_lists): 
        entity_list = []
        tag_list = []
        tag_sent_list=list(set(tag_sent_list))
        for entity2tag in tag_sent_list:
            if len(entity2tag.split('|')) == 2:
                entity,tag = entity2tag.split('|')
            else:
                print("error:{0}".format(entity2tag))
                continue
                
            entity_list.append(entity)
            tag_list.append(tag)
    
        new_sent_list = []
        new_tag_list = []
        last_entity_len = 0               # 用于记录目前实体是否正在遍历
        temp_tag = ''                     # 目前所遍历的实体的标志
        for i in range(len(sentence)):
            if last_entity_len == 0:
                max_entity_len = 0
                probable_tag = ''
                for j in range(len(entity_list)):
                    # 匹配实体过程
                    if sentence[i] == entity_list[j][0] and sentence[i:(i+len(entity_list[j]))] == entity_list[j] and len(entity_list[j])>max_entity_len:
                        max_entity_len = len(entity_list[j])
                        probable_tag = tag_list[j]

                if max_entity_len > 1:
                    last_entity_len = max_entity_len - 1
                    if is_cut:
                        tags_type_set.add(probable_tag)
                        temp_tag = 'seg'
                    else:
                        temp_tag = probable_tag
                    new_sent_list.append(sentence[i])
                    new_tag_list.append("B-{0}".format(temp_tag))
                else:
                    new_sent_list.append(sentence[i])
                    new_tag_list.append("O")
                    
            elif last_entity_len >= 1:
                last_entity_len = last_entity_len-1
                new_sent_list.append(sentence[i])
                new_tag_list.append("I-{0}".format(temp_tag))
                
        new_sent_lists.append(new_sent_list)
        new_tag_lists.append(new_tag_list)

    return new_sent_lists,new_tag_lists,list(tags_type_set)

# 功能：数据格式转换
import codecs
def to_change(readfile,writefile):
    '''
    功能：生成训练数据类别,处理后可采用 write_ner_labeled_data() 写入文件
    :param readfile:     Str  数据输入文件
    :param writefile:     Str  数据写入文件
    :return:

    原始数据格式：
            主 O
            诉 O
            : O
            彩 B-seg
            超 I-seg
            发 O
            现 O
            ...
    处理后数据格式：
            主诉:|O  彩超|seg  发现|O  动脉导管未闭|seg ...
    '''
    f_write = codecs.open(writefile, 'w', encoding='utf-8') 
    with tools.open_file(readfile, 'r') as f:
        lines = f.read().split('\n\n')
        print("len(lines):{0}".format(len(lines)))
        for line in lines:
            if line == '':
                continue
            tokens = line.split('\n')
            features = []
            tags = []
            for token in tokens:

                #print("token:{0}".format(token))
                feature_tag = token.split()
                if len(feature_tag):
                    features.append(feature_tag[0])
                    tags.append(feature_tag[-1])
            samples = []
            i = 0
            while i < len(features):
                sample = []
                if tags[i] == 'O':
                    sample.append(features[i])
                    j = i + 1
                    while j < len(features) and tags[j] == 'O':
                        sample.append(features[j])
                        j += 1
                    samples.append(''.join(sample) + '|O')
                else:
                    if tags[i][0] != 'B':
                        print(tags[i][0] + ' error start')
                        j = i + 1
                    else:
                        sample.append(features[i])
                        j = i + 1
                        while j < len(features) and tags[j][0] == 'I' and tags[j][-3:] == tags[i][-3:]:
                            sample.append(features[j])
                            j += 1
                        samples.append(''.join(sample) + '|' + tags[i][-3:])
                i = j
            f_write.write('  '.join(samples) + '\n')


# 功能：构建 jieba 分词词典向量 
def build_jieba_dict_list(tag_sent_lists,class_list,word_frequency):
    '''
    功能：构建 jieba 分词词典向量 
    :param tag_sent_lists:     Str  数据文件
    :param class_list:     Str  tag 类别
    :param word_frequency  int 词频
    :return:
            jieba_dict_list  List   jiaba 词典列表
    
    原始数据格式
            彩超|检查项目 动脉导管未闭|诊断 ...
            ...
            
    jiaba 词典列表格式     
           ['FSH（mIU/ml）', '眼球运动', '白细胞形态', '卵巢形态',...]  
    '''
    jieba_dict_set = set()
    for tag_sent_list in tag_sent_lists:
        for tag_sent in tag_sent_list:
            temp_list = tag_sent.split("|")
            if len(temp_list) == 2 and temp_list[1] in class_list:
                jieba_dict_set.add("{0} {1}".format(temp_list[0],word_frequency))
    return list(jieba_dict_set)
'''
    数据处理 begin
'''