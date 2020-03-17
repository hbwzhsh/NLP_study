# -*- coding: utf-8 -*-

import _pickle as pickle
import loader
import itertools
from utils import *
from loader import *


def preprocess_data(parameters, opts, mapping_file):
    lower = parameters['lower']
    tag_scheme = parameters['tag_scheme']
    train_sentences = loader.load_sentences(opts.train)
    dev_sentences = loader.load_sentences(opts.dev)
    test_sentences = loader.load_sentences(opts.test)

    update_tag_scheme(train_sentences, tag_scheme)
    update_tag_scheme(dev_sentences, tag_scheme)
    update_tag_scheme(test_sentences, tag_scheme)

    dico_words_train = word_mapping(train_sentences, lower)[0]
    # 将pretaind word2vec中的词加入词典中
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )

    dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

    train_data = prepare_dataset(train_sentences, word_to_id, tag_to_id, lower)
    dev_data = prepare_dataset(dev_sentences, word_to_id, tag_to_id, lower)
    test_data = prepare_dataset(test_sentences, word_to_id, tag_to_id, lower)

    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    all_word_embeds = {}
    for i, line in enumerate(codecs.open(opts.pre_emb, 'r', 'utf-8')):
        s = line.strip().split()
        if len(s) == parameters['word_dim'] + 1:
            all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), opts.word_dim))

    for w in word_to_id:
        if w in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w]
        elif w.lower() in all_word_embeds:
            word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

    print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

    with open(mapping_file, 'wb') as f:
        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'parameters': parameters,
            'word_embeds': word_embeds
        }
        pickle.dump(mappings, f)

    print('word_to_id: ', len(word_to_id))

    return word_to_id, tag_to_id, id_to_tag, word_embeds, train_data, dev_data, test_data