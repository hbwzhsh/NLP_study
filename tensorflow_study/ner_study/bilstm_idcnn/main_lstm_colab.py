# encoding=utf8
import os
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
import codecs
import pickle
import itertools
from collections import OrderedDict
import datetime
import random

import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme,load_test_sentences
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, test_ner,test_ner1
from data_utils import load_word2vec, create_input, input_from_line, BatchManager,TestBatchManager

class FLAGS_Config():
    def __init__(self):
        self.clean = True   # clean train folder
        self.train = True   # Whether train the model
        self.seg_dim = 30   # Embedding size for segmentation, 0 if not used
        self.char_dim = 300   # Embedding size for characters
        self.lstm_dim = 100   # Num of hidden units in LSTM, or num of filters in IDCNN
        self.tag_schema = "iob"   # tagging schema iobes or iob

        self.clip = 5   # Gradient clip
        self.dropout = 0.5   # Dropout rate
        self.batch_size = 64   # batch size
        self.decay_rate = 0.9   # decay_rate
        self.lr = 0.001   # Initial learning rate
        self.optimizer = "adam"   # Optimizer for training
        self.pre_emb = True   # Wither use pre-trained embedding
        self.zeros = False   # Wither replace digits with zero
        self.lower = True   # Wit#her lower case

        self.max_epoch = 100   # maximum training epochs
        self.steps_check = 50   # steps per checkpoint
        self.ckpt_path = "ckpt"   # Path to save model
        self.summary_path = "summary"   # Path to store summaries
        self.log_file = "train.log"   # file for log
        self.map_file = "maps.pkl"   # file for maps
        self.vocab_file = "vocab.json"   # file for maps vocab
        self.config_file = "config_file"   # file for config
        self.script = "conlleval"   # evaluation script
        self.result_path = "result"   # Path for results

        train_data_path = "data/"
        self.emb_file = os.path.join(train_data_path, "vec.txt")   
        self.train_file = os.path.join(train_data_path, "example.train")   
        self.dev_file = os.path.join(train_data_path, "example.dev")   
        self.test_file =  os.path.join(train_data_path, "example.test")

        self.model_type =  "bilstm"  # Model type, can be idcnn or bilstm
        self.g = tf.Graph()


        assert self.clip < 5.1, "gradient clip should't be too much"
        assert 0 <= self.dropout < 1, "dropout rate between 0 and 1"
        assert self.lr > 0, "learning rate must larger than zero"
        assert self.optimizer in ["adam", "sgd", "adagrad"]

FLAGS_config = FLAGS_Config()

# config for the model
def config_model(char_to_id, tag_to_id):
    config = OrderedDict()
    config["model_type"] = FLAGS_config.model_type
    config["num_chars"] = len(char_to_id)
    config["char_dim"] = FLAGS_config.char_dim
    config["num_tags"] = len(tag_to_id)
    config["max_epoch"] = FLAGS_config.max_epoch
    config["seg_dim"] = FLAGS_config.seg_dim
    config["lstm_dim"] = FLAGS_config.lstm_dim
    config["batch_size"] = FLAGS_config.batch_size
    config["decay_rate"] = FLAGS_config.decay_rate

    config["emb_file"] = FLAGS_config.emb_file
    config["train_file"] = FLAGS_config.train_file
    config["clip"] = FLAGS_config.clip
    config["dropout_keep"] = 1.0 - FLAGS_config.dropout
    config["optimizer"] = FLAGS_config.optimizer
    config["lr"] = FLAGS_config.lr
    config["tag_schema"] = FLAGS_config.tag_schema
    config["pre_emb"] = FLAGS_config.pre_emb
    config["zeros"] = FLAGS_config.zeros
    config["lower"] = FLAGS_config.lower
    return config


def evaluate(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.evaluate(sess, data, id_to_tag)
    eval_lines = test_ner(ner_results, FLAGS_config.result_path)
    for line in eval_lines:
        logger.info(line)
    f1 = float(eval_lines[1].strip().split()[-1])

    if name == "dev":
        best_test_f1 = model.best_dev_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_dev_f1, f1).eval()
            logger.info("new best dev f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1
    elif name == "test":
        best_test_f1 = model.best_test_f1.eval()
        if f1 > best_test_f1:
            tf.assign(model.best_test_f1, f1).eval()
            logger.info("new best test f1 score:{:>.3f}".format(f1))
        return f1 > best_test_f1


def evaluate_test(sess, model, name, data, id_to_tag, logger):
    logger.info("evaluate:{}".format(name))
    ner_results = model.test_evaluate(sess, data, id_to_tag)
    #print("ner_results[0:5]:{0}".format(ner_results[0:5]))
    test_ner1(ner_results, FLAGS_config.result_path)

    

def train():
    # load data sets
    all_sentences = load_sentences(FLAGS_config.train_file, FLAGS_config.lower, FLAGS_config.zeros)
    ''' train_sentences
        print("train_sentences[0:2]:{0}".format(train_sentences[0:2]))
        output:
            sentences[0:2]:
            [
                [['海', 'O'], ['钓', 'O'], ['比', 'O'], ['赛', 'O'], ['地', 'O'], ['点', 'O'], ['在', 'O'], ['厦', 'B-LOC'], ['门', 'I-LOC'], ['与', 'O'], ['金', 'B-LOC'], ['门', 'I-LOC'], ['之', 'O'], ['间', 'O'], ['的', 'O'], ['海', 'O'], ['域', 'O'], ['。', 'O']], 
                [['这', 'O'], ['座', 'O'], ['依', 'O'], ['山', 'O'], ['傍', 'O'], ['水', 'O'], ['的', 'O'], ['博', 'O'], ['物', 'O'], ['馆', 'O'], ['由', 'O'], ['国', 'O'], ['内', 'O'], ['一', 'O'], ['流', 'O'], ['的', 'O'], ['设', 'O'], ['计', 'O'], ['师', 'O'], ['主', 'O'], ['持', 'O'], ['设', 'O'], ['计', 'O'], ['，', 'O'], ['整', 'O'], ['个', 'O'], ['建', 'O'], ['筑', 'O'], ['群', 'O'], ['精', 'O'], ['美', 'O'], ['而', 'O'], ['恢', 'O'], ['宏', 'O'], ['。', 'O']]
            ]
    '''
    random.shuffle(all_sentences)
    split_rate = 0.7
    train_sentences = all_sentences[0:int(len(all_sentences)*split_rate)]
    dev_sentences = all_sentences[int(len(all_sentences)*split_rate):-1]
    test_sentences = load_sentences(FLAGS_config.test_file, FLAGS_config.lower, FLAGS_config.zeros)

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, FLAGS_config.tag_schema)
    update_tag_scheme(test_sentences, FLAGS_config.tag_schema)
    #print("train_sentences[0:2]:{0}".format(train_sentences[0:2]))
    ''' print("train_sentences[0:2]:{0}".format(train_sentences[0:2]))
        output:
            train_sentences[0:2]:
            [
                [['海', 'O'], ['钓', 'O'], ['比', 'O'], ['赛', 'O'], ['地', 'O'], ['点', 'O'], ['在', 'O'], ['厦', 'B-LOC'], ['门', 'E-LOC'], ['与', 'O'], ['金', 'B-LOC'], ['门', 'E-LOC'], ['之', 'O'], ['间', 'O'], ['的', 'O'], ['海', 'O'], ['域', 'O'], ['。', 'O']], 
                [['这', 'O'], ['座', 'O'], ['依', 'O'], ['山', 'O'], ['傍', 'O'], ['水', 'O'], ['的', 'O'], ['博', 'O'], ['物', 'O'], ['馆', 'O'], ['由', 'O'], ['国', 'O'], ['内', 'O'], ['一', 'O'], ['流', 'O'], ['的', 'O'], ['设', 'O'], ['计', 'O'], ['师', 'O'], ['主', 'O'], ['持', 'O'], ['设', 'O'], ['计', 'O'], ['，', 'O'], ['整', 'O'], ['个', 'O'], ['建', 'O'], ['筑', 'O'], ['群', 'O'], ['精', 'O'], ['美', 'O'], ['而', 'O'], ['恢', 'O'], ['宏', 'O'], ['。', 'O']]
            ]
    '''

    # create maps if not exist
    if not os.path.isfile(FLAGS_config.map_file):
        # create dictionary for word
        if FLAGS_config.pre_emb:
            dico_chars_train = char_mapping(train_sentences, FLAGS_config.lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                FLAGS_config.emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, FLAGS_config.lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(FLAGS_config.map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(FLAGS_config.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    #print("id_to_char:{0}".format(id_to_char))

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, FLAGS_config.lower
    )
    ''' train_data
        print("train_data[0:2]:{0}".format(train_data[0:2]))
        output:
            train_data[0:2]:[
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
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, FLAGS_config.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS_config.lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    # 批处理
    train_manager = BatchManager(train_data, FLAGS_config.batch_size)
    dev_manager = BatchManager(dev_data, 10)
    #test_manager = BatchManager(test_data, 25)

    # make path for store log and model if not exist
    make_path(FLAGS_config)
    if os.path.isfile(FLAGS_config.config_file):
        config = load_config(FLAGS_config.config_file)
    else:
        config = config_model(char_to_id, tag_to_id)
        save_config(config, FLAGS_config.config_file)
    make_path(FLAGS_config)

    log_path = os.path.join("log", FLAGS_config.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    steps_per_epoch = train_manager.len_data
    
    with tf.Session(config=tf_config,graph=FLAGS_config.g) as sess:
        model = create_model(sess, Model, FLAGS_config.ckpt_path, load_word2vec, config, id_to_char, logger)
        logger.info("start training")
        loss = []
        for i in range(config['max_epoch']):
            for batch in train_manager.iter_batch(shuffle=True):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % FLAGS_config.steps_check == 0:
                    iteration = step // steps_per_epoch + 1
                    logger.info("iteration:{} step:{}/{}, "
                                "NER loss:{:>9.6f}".format(
                        iteration, step%steps_per_epoch, steps_per_epoch, np.mean(loss)))
                    loss = []

            best = evaluate(sess, model, "dev", dev_manager, id_to_tag, logger)
            if best:
                save_model(sess, model, FLAGS_config.ckpt_path, logger)
            #evaluate(sess, model, "test", test_manager, id_to_tag, logger)


def test():
    # load data sets
    test_sentences = load_test_sentences(FLAGS_config.test_file, FLAGS_config.lower, FLAGS_config.zeros)
    #print("test_sentences:{0}".format(test_sentences))


    with open(FLAGS_config.map_file, "rb") as f:
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)


    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, FLAGS_config.lower, False
    )
    print(" %i sentences in  test." % (len(test_data)))

    # 批处理
    test_manager = TestBatchManager(test_data,1)

    # make path for store log and model if not exist
    config = load_config(FLAGS_config.config_file)
    
    make_path(FLAGS)

    log_path = os.path.join("log", FLAGS_config.log_file)
    logger = get_logger(log_path)
    print_config(config, logger)

    # limit GPU memory
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    
    with tf.Session(config=tf_config,graph=FLAGS_config.g) as sess:
        model = create_model(sess, Model, FLAGS_config.ckpt_path, load_word2vec, config, id_to_char, logger, False)
        
        best = evaluate_test(sess, model, "test", test_manager, id_to_tag, logger)



def main(_):

    if FLAGS_config.train:
        if FLAGS_config.clean:
            clean(FLAGS_config)

        starttime = datetime.datetime.now()
        train()
        endtime = datetime.datetime.now()

        logger = get_logger(FLAGS_config.log_file)
        logger.info("train_file : {0}".format(FLAGS_config.train_file))
        logger.info("speet time : {0}".format(endtime - starttime))
   
    else:
        test()


if __name__ == "__main__":
    tf.app.run(main)



