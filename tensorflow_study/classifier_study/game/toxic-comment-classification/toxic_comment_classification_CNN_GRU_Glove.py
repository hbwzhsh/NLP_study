#!/usr/bin/env python
# coding: utf-8

import nltk
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from keras.models import load_model
from keras.models import Model
from keras.layers import Bidirectional, GRU
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, SpatialDropout1D, concatenate
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Embedding
from keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import re
import os
import sys
import pickle
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入nltk预处理数据
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# file path
DIR_ROOT = os.path.dirname(os.path.abspath("__file__"))
# data path
DIR_DATA = os.path.join(DIR_ROOT, 'data')
TRAIN_DATA_FILE = os.path.join(DIR_DATA, 'train.csv')
# vords_vector path
DIR_WORDS_VECTOR = os.path.join(DIR_ROOT, 'words_vector')
EMBEDDED_FILE = os.path.join(DIR_WORDS_VECTOR, 'glove.840B.300d.txt')
# mode saving path
DIR_MODEL = os.path.join(DIR_ROOT, 'model')
PREPROCESSOR_FILE = os.path.join(DIR_MODEL, 'preprocessor.pkl')
ARCHITECTURE_FILE = os.path.join(DIR_MODEL, 'cnn_gru_architecture.json')
WEIGHTS_FILE = os.path.join(DIR_MODEL, 'cnn_gru_weights.h5')


# 数据预处理
MAX_NUM_WORDS = 100000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 300


class DataClean():
    # 数据清洗
    def lemmatize_all(self, sentence):
        wnl = WordNetLemmatizer()
        for word, tag in pos_tag(word_tokenize(sentence)):
            if tag.startswith("NN"):
                yield wnl.lemmatize(word, pos='n')
            elif tag.startswith('VB'):
                yield wnl.lemmatize(word, pos='v')
            elif tag.startswith('JJ'):
                yield wnl.lemmatize(word, pos='a')
            elif tag.startswith('R'):
                yield wnl.lemmatize(word, pos='r')
            else:
                yield word

    def text_cleaned_stopwords(self, text_raw):
        text_raw = str(text_raw)
        text_raw = str(text_raw.lower())
        text_raw = re.sub(r'[^a-zA-Z]', ' ', text_raw)

        words = text_raw.split()

        # 移除长度小于3的词语
        words2 = []
        for i in words:
            if len(i) >= 0:
                words2.append(i)
        # 去停止词
        stops = set(stopwords.words('english'))

        result_text = []
        result_text = " ".join([w for w in words2 if not w in stops])

        return(" ".join(self.lemmatize_all(result_text)))

    def text_cleaned_process(self, all_text):
        print("Cleaning texts begins...")
        # 去掉数字
        all_text.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

        # 输出清洗后的数据
        all_text_cleaned = []

        for i in range(0, all_text.size):
            all_text_cleaned.append(
                self.text_cleaned_stopwords(all_text[i]))

        # 构建pd形式
        all_text_cleaned = pd.Series(all_text_cleaned)
        # 原始数据可视化分析
        print("Cleaning texts end! Len of all_text_cleaned:", len(all_text_cleaned))
        # 数据清理前后对比
        print("Text[0] before cleaned: \n", all_text[0])
        print("Text[0] after cleaned: \n", all_text_cleaned[0])

        return all_text_cleaned


class WordToken():
    def __init__(self, text):
        self.text_untokened = text

    def fit_texts(self, num_words):
        # 分词
        print("Fit texts begins...")
        self.tokenizer = Tokenizer(num_words)
        self.tokenizer.fit_on_texts(self.text_untokened)
        print("Fit texts ends!")

    def transform_texts(self):
        print("Transform texts begins...")
        sequences = self.tokenizer.texts_to_sequences(self.text_untokened)
        # 分词完成
        # Pads sequences to the same length， return(len(sequence, maxlen))
        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        print("Transform texts ends!")
        return data


def get_embeddings(EMBEDDING_FILE, word_index):
    print("Embeddngs begins!")
    # 加入Glove预训练词
    embeddings_index = {}

    # 读取glove文件
    f = open(EMBEDDING_FILE, encoding='utf-8')
    for line in f:
        # 按空格分词
        values = line.split()
        word = ''.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs
    # 关闭glove文件
    f.close()

    print('Total %s word vectors in glove.840B.300d.' % len(embeddings_index))

    # 生成embedding matrix
    min_num_words = min(MAX_NUM_WORDS, len(word_index))
    embedding_matrix = np.zeros((min_num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print("Embeddngs ends!")
    return embedding_matrix, min_num_words


def get_model(embedding_matrix, num_words):
    print("Get model begins...")
    # 构建CNN模型
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
    # 载入预训练词向量作为Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedded_sequence = Embedding(num_words,
                                  EMBEDDING_DIM,
                                  embeddings_initializer=Constant(
                                      embedding_matrix),
                                  input_length=MAX_SEQUENCE_LENGTH,
                                  trainable=False)(sequence_input)

    x = SpatialDropout1D(0.2)(embedded_sequence)

    x = Bidirectional(GRU(128, return_sequences=True, unroll=True))(x)
    # model0
    conv_0 = Conv1D(128, 1, kernel_initializer='normal', activation='relu')(x)
    drop_0 = Dropout(0.4)(conv_0)
    max_pool0 = MaxPooling1D(pool_size=4)(drop_0)
    gru_0 = GRU(100, dropout=0.2, recurrent_dropout=0.2)(max_pool0)
    # model1
    conv_1 = Conv1D(128, 2, kernel_initializer='normal', activation='relu')(x)
    drop_1 = Dropout(0.45)(conv_1)
    max_pool1 = MaxPooling1D(pool_size=4)(drop_1)
    gru_1 = GRU(100, dropout=0.2, recurrent_dropout=0.2)(max_pool1)
    # model2
    conv_2 = Conv1D(128, 4, kernel_initializer='normal', activation='relu')(x)
    drop_2 = Dropout(0.5)(conv_2)
    max_pool2 = MaxPooling1D(pool_size=4)(drop_2)
    gru_2 = GRU(100, dropout=0.2, recurrent_dropout=0.2)(max_pool2)

    # 模型融合
    conv_sum = concatenate([gru_0, gru_1, gru_2], axis=1)

    # 压缩成对应6个标签
    #conv_sum = Flatten()(conv_sum)
    dense1 = Dense(50, activation='relu')(conv_sum)
    preds = Dense(6, activation="sigmoid")(dense1)

    # 生成模型
    model = Model(inputs=sequence_input, outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print("Get model ends...")
    return model


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print(
                "\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))


if __name__ == "__main__":
    # 数据读取
    class_names = ['toxic', 'severe_toxic', 'obscene',
                   'threat', 'insult', 'identity_hate']
    # read data
    train = pd.read_csv(TRAIN_DATA_FILE).fillna(' ')
    # test = pd.read_csv('data/test.csv').fillna(' ')
    # submission = pd.read_csv('data/sample_submission.csv')

    # 单独保存comment_text
    train_text = train['comment_text'].str.lower()
    # test_text = test['comment_text'].str.lower()
    # 获得y_train
    y_train = train[['toxic', 'severe_toxic',
                     'obscene', 'threat', 'insult', 'identity_hate']]

    # 连接所有文字用于分词
    # all_text = pd.concat([train_text, test_text], axis=0, ignore_index=True)
    all_text = train_text

    # 数据清洗
    data_clean_process = DataClean()
    all_text_cleaned = data_clean_process.text_cleaned_process(all_text)

    # 分词
    preprocessor = WordToken(all_text_cleaned)
    preprocessor.fit_texts(num_words=MAX_NUM_WORDS)
    data = preprocessor.transform_texts()
    # A dictionary of words and their uniquely assigned integers
    word_index = preprocessor.tokenizer.word_index
    print('Number of Unique Tokens(word_index)', len(word_index))
    # summarize what was learned
    # print(tokenizer.word_counts) # A dictionary of words and their counts
    # print(tokenizer.document_count) # A dictionary of words and how many documents each appeared in.
    # print(tokenizer.word_docs) # An integer count of the total number of documents that were used to fit the Tokenizer.

    # save preprocess file
    print(f"Saving the text transformer: {PREPROCESSOR_FILE}")
    with open(PREPROCESSOR_FILE, 'wb') as file:
        pickle.dump(preprocessor, file)
    del preprocessor

    # 重塑train与test数据
    x_train = data[:len(train_text)]
    # x_test = data[len(train_text):]
    print("Len of x_train: ", len(x_train))
    # print("Len of y_train: ", len(y_train))

    # 拆分train数据
    # 根据mentor意见提供训练集数据，但是效果基本不变。
    x_train_fit, x_val, y_train_fit, y_val = train_test_split(
        x_train, y_train, test_size=0.10, random_state=255)

    # creata embeddings.
    embedding_matrix, min_num_words = get_embeddings(EMBEDDED_FILE, word_index)

    # 模型构建
    model = get_model(embedding_matrix, min_num_words)

    # 开始训练
    # 回调函数查看训练得分
    ROC_AUC = RocAucEvaluation(validation_data=(x_val, y_val), interval=1)

    # fit
    history = model.fit(x_train_fit, y_train_fit, batch_size=256, epochs=2, validation_data=(x_val, y_val), verbose=2,
                        callbacks=[ROC_AUC])

    # save mode file
    print(f"Saving the architecture: {ARCHITECTURE_FILE}")
    with open(ARCHITECTURE_FILE, 'w') as file:
        architecture_json = model.to_json()
        file.write(architecture_json)

    print(f"Saving the weights: {WEIGHTS_FILE}")
    model.save_weights(WEIGHTS_FILE)
