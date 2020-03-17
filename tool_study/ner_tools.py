# encoding utf8
# author : King
# time : 2019.11.12
# 参考1：
''' 函数目录

    1. read_ner_labeled_data(filename,demo_flag=False)：功能：数据读取
    2. write_ner_labeled_data(filename,char_lists,tags_lists)：功能：命名实体识别标注数据写入
    3. read_and_write_data(origin_file_list,result_file_list,write_file_name,mode='w')：功能：将原文数据和标注数据汇总并写入文件
    4. tag_sent_pre(tag_sent_list)：功能：标注列表符号替换处理
    5. merge_sent_tag(sent_list,tag_sent_lists,is_cut = False)：功能：生成训练数据类别,处理后可采用 write_ner_labeled_data() 写入文件
    6. to_change(readfile,writefile)：功能：生成训练数据类别,处理后可采用 write_ner_labeled_data() 写入文件
    7. build_jieba_dict_list(tag_sent_lists,class_list,word_frequency)：功能：构建 jieba 分词词典向量 
    8. sent_ner_labeled(sent_list,df_dict): 功能：将预测句子数据转化为 列表形式
    9. get_entity_tag(sent_list,outer_sep=' ',inter_sep='|'):功能：获取数据集中的实体
    10. new_word_find(sent_list,lexicon_dict,entity_tag_dict): 功能：将预测句子数据转化为 列表形式
    11. merge_diff_tag_entity(entity_tag_dict): 功能：合并不同类型实体
    12. count_diff_tag_entity_num(merge_tag_entity_dict): 功能：统计不同类型实体数量

'''
import tools

'''数据读取 begin'''
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
    with tools.open_file(filename) as f:
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

''' 数据读取 end '''

'''数据写入 begin'''

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
            发   O
            现   O
            动   B-seg
            脉   I-seg
            导   I-seg
            管   I-seg
            未   I-seg
            ...
    '''
    on_write = []
    with tools.open_file(filename,'w') as f:
        for (char_list,tags_list) in zip(char_lists,tags_lists):
            for (char,tags) in zip(char_list,tags_list):
                on_write.append("{0} {1}\n".format(char,tags))
            on_write.append("\n")
        f.writelines(on_write)

''' 数据写入 end '''


'''将原始数据中的原始数据和标注数据进行汇总并写入 begin'''

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
    with tools.open_file(write_file_name,mode) as f_w:
        on_write = []
        
        for (origin_file,result_file) in zip(origin_file_list,result_file_list):
            with tools.open_file(origin_file) as f_origin, tools.open_file(result_file) as f_result:
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


'''数据处理 begin'''
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
                sent_list：['呕吐、腹痛2天多',...]
                tag_sent_lists:[['呕吐|症状', '腹痛|症状', '2天|时长'], ...]
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
        print("sentence:{0}".format(sentence))
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
                try:
                    feature_tag = token.split()
                    features.append(feature_tag[0])
                    tags.append(feature_tag[-1])
                except Exception:
                    print("token:{0}".format(token))
            # print(features[0:5])
            # print(tags[0:5])
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
                        try:
                            samples.append(features[i+1] + '|O')
                            #print(tags[i][0] + ' error start')
                            j = i + 1
                        except Exception:
                            print("features[i]:{0}".format(features[i]))
                            print("tags[i]:{0}".format(tags[i]))
                            j = i + 1
                    else:
                        sample.append(features[i])
                        j = i + 1
                        while j < len(features) and tags[j][0] == 'I' and tags[j][2:] == tags[i][2:]:
                            sample.append(features[j])
                            j += 1
                        samples.append(''.join(sample) + '|' + tags[i][2:])
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
''' 数据处理 end '''

# 功能：基于词库的句子打标签
def sent_ner_labeled(sent_list,df_dict):
    '''
    功能：基于词库的句子打标签
    :param sent_list:       List             句子列表
    :param df_dict:         DataFrame Dict   词典数据
    :return:
        labeled_sent_list       List[List]    句子列表
    数据格式：
            [
                [sent1,'word1|entity1,word2|entity2,...'],
                [sent2,'word1|entity1,word2|entity2,...'],
                ...
            ]
            ...
    '''
    labeled_sent_list = []
    for sent in sent_list:
        labeled_sent = []
        tag_list = []
        for (index,val) in df_dict.items():
            word_list = list(val[index])
            for word in word_list:
                if word in sent :
                    tag_list.append("{0}|{1}".format(word,index))
        labeled_sent.append(sent)
        labeled_sent.append(','.join(tag_list))
        labeled_sent_list.append(labeled_sent)
    return labeled_sent_list    

########################################
#         新词复现模块 begin
########################################

# 功能：将预测句子数据转化为 列表形式
def pred_sent_data_to_list(sent_list,outer_sep=' ',inter_sep='|'):
    '''
        # 功能：将预测句子数据转化为 列表形式
        :param sent_list:     List  预测句子数据列表
        :param outer_sep:     Str   实体间的分隔符
        :param inter_sep:     Str   实体与标注间的分隔符
        :return:
            new_pred_sent_list:   List   转化后的列表
        原始数据格式：
                主诉:|O  彩超|seg  发现|O  动脉导管未闭|seg ...
                ...
        处理后数据格式：
                [
                    [
                        ["主诉:':'O'], ["彩超:':'seg'],...
                    ],
                    ...
                ]
    '''
    new_pred_sent_list = []
    for sent in sent_list:
        new_pred_sent_list.append([entity_tag.split(inter_sep) for entity_tag in sent.split(outer_sep)])
    return  new_pred_sent_list

# 功能：获取训练数据集中的实体   
def get_entity_tag(sent_list,outer_sep=' ',inter_sep='|'):
    '''
        功能：获取数据集中的实体
        :param sent_list:     List  句子数据列表
        :param outer_sep:     Str   实体间的分隔符
        :param inter_sep:     Str   实体与标注间的分隔符
        :return:
            entity_tag_dict:   Dict   数据集所包含的实体
        原始数据格式：
                主诉:|O  彩超|seg  发现|O  动脉导管未闭|seg ...
                ...
        处理后数据格式：
                {彩超 : seg, 动脉导管未闭 : seg, ...}
    '''
    entity_tag_dict = {}
    for sent in sent_list:
        for entity_tag in sent.split(outer_sep):
            entity,tag =entity_tag.split(inter_sep)
            if tag!='O' and entity not in entity_tag_dict:
                entity_tag_dict[entity] = tag
    return  entity_tag_dict

# 功能：新词发现统计函数
def new_word_find(sent_list,lexicon_dict,entity_tag_dict):
    '''
        # 功能：将预测句子数据转化为 列表形式
        :param sent_list:           List        预测句子数据列表
        :param lexicon_dict:           List Dict   词库
        :param entity_tag_dict:     Dict        训练集词典
        :return:
            new_true_word_dict:   Dict   训练集中没有，词库中有的数据
            new_word_dict:        Dict   训练集中没有，词库中没有的数据
        sent_list 数据格式：
                [
                    [
                        ["主诉:':'O'], ["彩超:':'seg'],...
                    ],
                    ...
                ]
        lexicon_dict 数据格式：
            {
                'check_item':['生长发育', '精神',...],
                ...
            }
        entity_tag_dict 数据格式：
            {
                '头皮血肿':operation, '噩梦':diagnosis ,...
            }
        new_true_word_dict 数据格式：
            {
                'check_item':['生长发育', '精神',...],
                ...
            }
        new_word_dict 数据格式：
            {
                '头皮血肿':operation, '噩梦':diagnosis ,...
            }      
    '''
    new_true_word_dict = {}
    new_word_dict = {}
    for sent in sent_list:
        for entity_tag in sent:
            if entity_tag[1] != 'O' and entity_tag[0] not in entity_tag_dict:
                if entity_tag[0] in lexicon_dict[entity_tag[1]]:                 # 如果新词在词典对应类型的词库中
                    new_true_word_dict[entity_tag[0]] = entity_tag[1]
                else:                                                            # 如果新词不在词典对应类型的词库中
                    new_word_dict[entity_tag[0]] = entity_tag[1]
    return new_true_word_dict,new_word_dict        


# 功能：合并不同类型实体
def merge_diff_tag_entity(entity_tag_dict):
    '''
        # 功能：合并不同类型实体
        :param entity_tag_dict:           Dict        预测句子数据字典
        :return
             tag_entity_num_dict    Dict    统计不同类型实体数量Dict
        输入数据格式：
                [
                    ["主诉:':'O'], ["彩超:':'seg'],...
                ]
        输出数据格式：
            {
                'check_item':['生长发育', '精神',...],
                ...
            }
    '''
    merge_tag_entity_dict = {}
    for (key,val) in entity_tag_dict.items():
        if val not in merge_tag_entity_dict:
            merge_tag_entity_dict[val] = []
        merge_tag_entity_dict[val].append(key)
    return merge_tag_entity_dict

# 功能：统计不同类型实体数量
def count_diff_tag_entity_num(merge_tag_entity_dict):
    '''
        # 功能：统计不同类型实体数量
        :param entity_tag_dict:           Dict        预测句子数据字典
        :return
             tag_entity_num_dict    Dict    统计不同类型实体数量Dict
        输入数据格式：
            {
                'check_item':['生长发育', '精神',...],
                ...
            }
        输出数据格式：
            {
                'check_item':15,
                ...
            }
    '''
    merge_tag_entity_num_dict = {}
    for (key,val) in merge_tag_entity_dict.items():
        merge_tag_entity_num_dict[key] = len(val)
    return merge_tag_entity_num_dict

########################################
#         新词复现模块 end
########################################

if __name__ == '__main__':
    pass

