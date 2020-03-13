# encoding=utf8
import os
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import re
import codecs

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features


# 加载文件数据
def load_sentences(path, lower, zeros):
    """
        Load sentences. A line must contain at least a word and its tag.
        Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num+=1
        # print("num:{0}".format(num))
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        #print(list(line))
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word= line.split()
            # print("word:{0}".format(word))
            #assert len(word) >= 2, [word[0]]
            if len(word) >= 2:
                sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    '''   
        output:
            sentences[0:2]:
            [
                [['海', 'O'], ['钓', 'O'], ['比', 'O'], ['赛', 'O'], ['地', 'O'], ['点', 'O'], ['在', 'O'], ['厦', 'B-LOC'], ['门', 'I-LOC'], ['与', 'O'], ['金', 'B-LOC'], ['门', 'I-LOC'], ['之', 'O'], ['间', 'O'], ['的', 'O'], ['海', 'O'], ['域', 'O'], ['。', 'O']], 
                [['这', 'O'], ['座', 'O'], ['依', 'O'], ['山', 'O'], ['傍', 'O'], ['水', 'O'], ['的', 'O'], ['博', 'O'], ['物', 'O'], ['馆', 'O'], ['由', 'O'], ['国', 'O'], ['内', 'O'], ['一', 'O'], ['流', 'O'], ['的', 'O'], ['设', 'O'], ['计', 'O'], ['师', 'O'], ['主', 'O'], ['持', 'O'], ['设', 'O'], ['计', 'O'], ['，', 'O'], ['整', 'O'], ['个', 'O'], ['建', 'O'], ['筑', 'O'], ['群', 'O'], ['精', 'O'], ['美', 'O'], ['而', 'O'], ['恢', 'O'], ['宏', 'O'], ['。', 'O']]
            ]
    '''
    return sentences

# 加载文件数据
def load_test_sentences(path, lower, zeros):
    """
        Load sentences. A line must contain at least a word and its tag.
        Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        num+=1
        # print("num:{0}".format(num))
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        #print(list(line))
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                word = line.split()
                # word[0] = " "
            else:
                word= line.split()
            # print("word:{0}".format(word))
            #assert len(word) >= 2, [word[0]]
            if len(word) == 1:
                sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    '''   
        output:
            sentences[0:2]:
            [
                [['海', 'O'], ['钓', 'O'], ['比', 'O'], ['赛', 'O'], ['地', 'O'], ['点', 'O'], ['在', 'O'], ['厦', 'B-LOC'], ['门', 'I-LOC'], ['与', 'O'], ['金', 'B-LOC'], ['门', 'I-LOC'], ['之', 'O'], ['间', 'O'], ['的', 'O'], ['海', 'O'], ['域', 'O'], ['。', 'O']], 
                [['这', 'O'], ['座', 'O'], ['依', 'O'], ['山', 'O'], ['傍', 'O'], ['水', 'O'], ['的', 'O'], ['博', 'O'], ['物', 'O'], ['馆', 'O'], ['由', 'O'], ['国', 'O'], ['内', 'O'], ['一', 'O'], ['流', 'O'], ['的', 'O'], ['设', 'O'], ['计', 'O'], ['师', 'O'], ['主', 'O'], ['持', 'O'], ['设', 'O'], ['计', 'O'], ['，', 'O'], ['整', 'O'], ['个', 'O'], ['建', 'O'], ['筑', 'O'], ['群', 'O'], ['精', 'O'], ['美', 'O'], ['而', 'O'], ['恢', 'O'], ['宏', 'O'], ['。', 'O']]
            ]
    '''
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
        Check and update sentences tagging scheme to IOB2.
        Only IOB1 and IOB2 schemes are accepted.
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)          # IOB -> IOBES
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
        Create a dictionary and a mapping of words, sorted by frequency.
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)
    ''' print("dico:{0}".format(dico)) 
        output:
            dico:{'海': 1018, '钓': 45, '比': 1356, '赛': 1368,...,'在': 8262, '厦': 51, '门': 1000]
    '''
    
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    ''' char_to_id, id_to_char
        print("char_to_id:{0}".format(char_to_id))
        print("id_to_char:{0}".format(id_to_char))
        output:
            char_to_id:{'<PAD>': 0, '<UNK>': 1, '，': 2, '的': 3, '。': 4, ...}
            id_to_char:{0: '<PAD>', 1: '<UNK>', 2: '，', 3: '的', 4: '。', ...}
    '''

    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """

    none_index = tag_to_id["O"]

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        #data.append([string, chars, tags])
        data.append([string, chars, segs, tags])

    ''' data
        介绍：
            第一行：字
            第二行：字所对应词典的id
            第三行：分词：对于一个词语，第一个词为 1，中间的词为 2 ， 最后一个为 3， 单独一个的为 0 
            第四行：tags 

        print("data[0:2]:{0}".format(data[0:2]))
        output:
            data[0:2]:[
                [
                    ['海', '钓', '比', '赛', '地', '点', '在', '厦', '门', '与', '金', '门', '之', '间', '的', '海', '域', '。'], 
                    [235, 1574, 153, 152, 30, 236, 8, 1500, 238, 89, 182, 238, 112, 198, 3, 235, 658, 4], 
                    [1, 3, 1, 3, 1, 3, 0, 1, 3, 0, 1, 3, 1, 3, 0, 1, 3, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 2, 3, 0, 0, 0, 0, 0, 0]
                ], 
                [
                    ['这', '座', '依', '山', '傍', '水', '的', '博', '物', '馆', '由', '国', '内', '一', '流', '的', '设', '计', '师', '主', '持', '设', '计', '，', '整', '个', '建', '筑', '群', '精', '美', '而', '恢', '宏', '。'],
                    [22, 752, 693, 275, 2334, 219, 3, 763, 286, 579, 202, 5, 141, 6, 344, 3, 189, 266, 553, 63, 252, 189, 266, 2, 462, 27, 96, 1024, 443, 365, 79, 109, 1263, 1076, 4],
                    [1, 3, 1, 2, 2, 3, 0, 1, 2, 3, 0, 1, 3, 1, 3, 0, 1, 2, 3, 1, 3, 1, 3, 0, 1, 3, 1, 2, 3, 1, 3, 0, 1, 3, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                ]
            ]
    '''
    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which
    # we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)

