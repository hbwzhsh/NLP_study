# encoding utf8
# author : King
# time : 2019.11.12
''' 介绍：NER 数据格式间的相互转化
    格式 1 ：
         [
            'sent1 \t word1|entity1,word2|entity2,...',
            'sent2 \t word1|entity1,word2|entity2,...',
            ...
        ]
    格式 2 ：
        new_sent_lists:[['彩','超',...],...]
        new_tag_lists:[['B-seg','I-seg',...],...]
    格式 3 ：
        [
            '主诉:|O  彩超|seg  发现|O  动脉导管未闭|seg ...',
            ...
        ]

    
'''
''' 函数目录

'''
import tools

##########################################
#               格式转化 begin
##########################################

# 功能：格式 1 ('sent1 \t word1|entity1,word2|entity2) 
#    ->  格式 2  ([['彩','超',...],...],[['B-seg','I-seg',...],...])
def ner_type1totype2(sent_tag_list, transform_dict=False,sep='\t',entity_sep=',',entity_tag_sep='|',merge_sep ='\t',is_cut = False):
    '''
        功能：合并训练所需的实体数据列表 sent_tag_lists 
        :param sent_tag_list:       List        句子数据列表,包含实体
        :param sep:                 String      句子和实体间的分隔符
        :param entity_sep:          String      实体和实体间的分隔符
        :param entity_tag_sep:      String      实体与标注间的分隔符
        :param is_cut:              Bool        是否为分词任务
        :return:
            new_sent_tag_lists:     List  处理后的数据列表
        原始数据格式：
            sent_tag_list:
                [
                    'sent1 \t word1|entity1,word2|entity2,...',
                    'sent2 \t word1|entity1,word2|entity2,...',
                    ...
                ]
                ...
        处理后数据格式：
            new_sent_tag_lists : ['2\tB-attack_time', '0\tI-attack_time', '1\tI-attack_time', '0\tI-attack_time', '年\tI-attack_time', '9\tI-attack_time', '月\tI-attack_time', '无\tO', ...]
    '''
    new_sent_tag_lists = []
    for sent_tag in sent_tag_list: 
        sentence,entity_tag_list = sent_tag.split(sep)
        entity_tag_list=list(set(entity_tag_list.split(entity_sep)))
        entity_list = []
        tag_list = []
        new_sent_tag_list = []
        if len(entity_tag_list) > 0 and entity_tag_list[0] != '':
            for entity2tag in entity_tag_list:
                if len(entity2tag.split(entity_tag_sep)) == 2:
                    entity,tag = entity2tag.split(entity_tag_sep)
                else:
                    print("error:{0}".format(entity2tag))
                    continue

                entity_list.append(entity)
                tag_list.append(tag)

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

                    if max_entity_len >= 1:
                        last_entity_len = max_entity_len - 1
                        if is_cut:
                            temp_tag = 'seg'
                        else:
                            temp_tag = probable_tag
                        if transform_dict:
                            new_sent_tag_list.append("{0}{1}{2}".format(sentence[i],merge_sep,"B-{0}".format(transform_dict[temp_tag])))
                        else:
                            new_sent_tag_list.append("{0}{1}{2}".format(sentence[i],merge_sep,"B-{0}".format(temp_tag)))
                    else:
                        new_sent_tag_list.append("{0}{1}{2}".format(sentence[i],merge_sep,'O'))

                elif last_entity_len >= 1:
                    last_entity_len = last_entity_len-1
                    if transform_dict:
                        new_sent_tag_list.append("{0}{1}{2}".format(sentence[i],merge_sep,"I-{0}".format(transform_dict[temp_tag])))
                    else:
                        new_sent_tag_list.append("{0}{1}{2}".format(sentence[i],merge_sep,"I-{0}".format(temp_tag)))
        else:
            for i in range(len(sentence)):
                new_sent_tag_list.append("{0}{1}{2}".format(sentence[i],merge_sep,"O"))

        new_sent_tag_lists.append(new_sent_tag_list)

    return new_sent_tag_lists

# 功能：格式 2 ([['神\tO', '清\tO', '，\tO', '精\tB-检查项目',...],...])
#      ->  格式 3 (['主诉:|O  彩超|seg  发现|O  动脉导管未闭|seg ...',...])
def ner_type2totype3(char_tag_lists):
    ''' 功能：格式 2 ->  格式 3 
        :param sent_tag_lists:     List  需要处理的数据列表
        :return:
            sent_tag_lists:        List   格式3 类似数据
        sent_tag_lists 原始数据格式：
                sent_tag_lists 数据样式：
                [
                    ['神\tO', '清\tO', '，\tO', '精\tB-检查项目',...],
                    ['神\tO', '清\tO', '，\tO', '精\tB-检查项目',...],
                    ...
                ]
        sent_tag_lists 处理后数据格式：
                主诉:|O  彩超|seg  发现|O  动脉导管未闭|seg ...
                ...
    '''
    sent_tag_lists = []
    for char_tag_list in char_tag_lists:
        features = []
        tags = []
        for char_tag in char_tag_list:
            #print("token:{0}".format(token))
            feature_tag = char_tag.split("\t")
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
                    while j < len(features) and tags[j][0] == 'I' and tags[j][2:] == tags[i][2:]:
                        sample.append(features[j])
                        j += 1
                    samples.append(''.join(sample) + '|' + tags[i][2:])
            i = j
        sent_tag_lists.append(" ".join(samples))
    return sent_tag_lists

# 功能：格式 1 () ->  格式 3 
def type1totype3(sent,entities_dict):
    ''' 功能：格式 1 ->  格式 3
        :param sent:              String  句子
        :param entities_dict:     Dict    实体词典
        :return:
            sent_tag_list   List         格式 3 数据
        原始数据格式：
                entities_dict 数据样式：
                {
                    '咳嗽': 'diagnosis',
                     '咳': 'diagnosis',
                     '色发红': 'diagnosis',
                     '发热': 'symptom',
                     '无': 'negative_words',
                     '寒战': 'diagnosis',
                     '1月余': 'time',
                     '左侧腹股沟可复性包块': 'symptom',
                     '11月余': 'time',
                     '5天前': 'time',
                     '2天': 'time',
                     '1天': 'time'
                 }
        处理后数据格式：
                主诉:|O  彩超|seg  发现|O  动脉导管未闭|seg ...
                ...
    '''
    sent_tag_list = []
    sent_len = len(sent)
    i = 0
    last_sent = ''
    while i < sent_len:
        max_len = 0
        temp_entity = ''
        temp_tag = ''
        for entities,tag in entities_dict.items():
            if i+len(entities) <= sent_len and sent[i:i+len(entities)] == entities:
                if max_len < len(entities):
                    max_len = len(entities)
                    temp_entity = entities
                    temp_tag = tag
        if temp_entity == '':
            last_sent = last_sent + sent[i]
            i = i + 1
        else:
            if last_sent != '':
                sent_tag_list.append("{0}|{1}".format(last_sent,'O'))
                last_sent = ''
            sent_tag_list.append("{0}|{1}".format(temp_entity,temp_tag))
            i = i + max_len
    return sent_tag_list


##########################################
#               格式转化 end
##########################################


##########################################
#               格式数据写入 begin
##########################################
# 功能：格式 2 数据写入
def write_type2_data(filename,sent_tag_lists):
    ''' 功能：格式 2 数据写入
        :param filename:     Str  写入数据文件
        :param sent_tag_lists:     List  需要处理的数据列表
        :return:
            None
        sent_tag_lists 数据样式：
            [
                ['神\tO', '清\tO', '，\tO', '精\tB-检查项目',...],
                ['神\tO', '清\tO', '，\tO', '精\tB-检查项目',...],
                ...
            ]
            
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
        for sent_tag_list in sent_tag_lists:
            for sent_tag in sent_tag_list:
                on_write.append("{0}\n".format(sent_tag))
            on_write.append("\n")
        f.writelines(on_write)

# 功能：格式 3 数据写入
def write_type3_data(filename,sent_tag_list):
    ''' 功能：格式 2 数据写入
        :param filename:     Str  写入数据文件
        :param sent_tag_list:     List  需要处理的数据列表
        :return:
            None
        sent_tag_list 数据样式：
            [
                '1.|O 咳嗽|symptom 1周余|time ，|O 加重|variety 伴|O 发热|symptom 2天|time',
                ...
            ]
    '''
    on_write = []
    with tools.open_file(filename,'w') as f:
        for sent_tag in sent_tag_list:
            on_write.append("{0}\n".format(sent_tag))
        f.writelines(on_write)

##########################################
#               格式数据写入 end
##########################################

##########################################
#               格式数据转化 begin
##########################################
#功能：格式 1 
# 转化前：{'患儿4月余前接种“手足口疫苗”后...': ['双眼向右凝视|diagnosis',...]}
# 转化后：('sent1 \t word1|entity1,word2|entity2) 
def ner_type1_tramsform(data_dict):
    data_list = []
    for key,val in data_dict.items():
        data_list.append("{0}\t{1}".format(key,",".join(val)))
    return data_list

##########################################
#               格式数据转化 end
##########################################