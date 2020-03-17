# -*- coding: utf-8 -*-

import optparse
from collections import OrderedDict
import torch
import time
from torch.autograd import Variable
from eval import return_report
from utils import *
from loader import *
from model import BiLSTM_CRF
from preprocess import preprocess_data

t = time.time()
models_path = "models/"

if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)


def train():
    word_to_id, tag_to_id, id_to_tag, word_embeds, train_data, dev_data, test_data =\
        preprocess_data(parameters, opts, mapping_file=mapping_file)

    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data)))

    model = BiLSTM_CRF(vocab_size=len(word_to_id),
                       tag_to_ix=tag_to_id,
                       embedding_dim=parameters['word_dim'],
                       hidden_dim=parameters['word_lstm_dim'],
                       use_gpu=use_gpu,
                       pre_word_embeds=word_embeds,
                       use_crf=parameters['crf'])
    if parameters['reload']:
        print("reload pre trained model %s" % model_name)
        model.load_state_dict(torch.load(model_name))
    if use_gpu:
        model.cuda()
    learning_rate = 0.0015
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    loss = 0.0
    best_dev_f = -1.0
    plot_every = 20
    eval_every = 40
    count = 0

    model.train(True)
    for epoch in range(1, parameters['epochs']):
        print("train epoch %i :" % epoch)
        for i, index in enumerate(np.random.permutation(len(train_data))):
            count += 1
            data = train_data[index]
            model.zero_grad()

            sentence_in = data['words']
            sentence_in = Variable(torch.LongTensor(sentence_in))
            tags = data['tags']

            targets = torch.LongTensor(tags)
            if use_gpu:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda())
            else:
                neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)
            loss += neg_log_likelihood.data[0] / len(data['words'])
            neg_log_likelihood.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optimizer.step()

            if count % plot_every == 0:
                loss /= plot_every
                print(count, ': ', loss)
                loss = 0.0

            if count % eval_every == 0 and count > (eval_every * 20) or \
               count % (eval_every * 4) == 0 and count < (eval_every * 20):
                model.train(False)
                best_dev_f, new_dev_f, save = evaluating(model, dev_data, best_dev_f, tag_to_id, id_to_tag)
                if save:
                    torch.save(model.state_dict(), model_name + str(count))
                model.train(True)

            if count % len(train_data) == 0:
                adjust_learning_rate(optimizer, lr=learning_rate / (1 + 0.05 * count / len(train_data)))

    print(time.time() - t)


def evaluating(model, datas, best_f, tag_to_id, id_to_tag):
    prediction = []
    save = False
    new_f = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        dwords = Variable(torch.LongTensor(data['words']))
        if use_gpu:
            val, out = model(dwords.cuda())
        else:
            val, out = model(dwords)
        predicted_id = out
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')
    predf = eval_temp + '/pred.' + name
    print("predf:{0}".format(predf))

    with open(file=predf,mode='w',encoding="utf8") as f:
        f.write('\n'.join(prediction))

    eval_lines = return_report(predf)

    for i, line in enumerate(eval_lines):
        print(line)
        if i == 1:
            new_f = float(line.strip().split()[-1])
            if new_f > best_f:
                best_f = new_f
                save = True
                print('the best F is ', new_f)

    print(("{: >2}{: >9}{: >9}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    ))
    for i in range(confusion_matrix.size(0)):
        print(("{: >2}{: >9}{: >9}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
            *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
              ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
        ))
    return best_f, new_f, save


if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option(
        "-T", "--train", default="../../../data/ner/example.train",
        help="Train set location"
    )
    optparser.add_option(
        "-d", "--dev", default="../../../data/ner/example.dev",
        help="Dev set location"
    )
    optparser.add_option(
        "-t", "--test", default="../../../data/ner/example.test",
        help="Test set location"
    )
    optparser.add_option(
        "-s", "--tag_scheme", default="iobes",
        help="Tagging scheme (IOB or IOBES)"
    )
    optparser.add_option(
        "-e", "--epochs", default="10001",
        type='int', help="The epochs of training"
    )
    optparser.add_option(
        "-l", "--lower", default="1",
        type='int', help="Lowercase words (this will not affect character inputs)"
    )
    optparser.add_option(
        "-w", "--word_dim", default="100",
        type='int', help="Token embedding dimension"
    )
    optparser.add_option(
        "-W", "--word_lstm_dim", default="200",
        type='int', help="Token LSTM hidden layer size"
    )
    optparser.add_option(
        "-B", "--word_bidirect", default="1",
        type='int', help="Use a bidirectional LSTM for words"
    )
    optparser.add_option(
        "-p", "--pre_emb", default="../../../data/pre_trained/vec.txt",
        help="Location of pretrained embeddings"
    )
    optparser.add_option(
        "-A", "--all_emb", default="1",
        type='int', help="Load all embeddings"
    )
    optparser.add_option(
        "-f", "--crf", default="1",
        type='int', help="Use CRF (0 to disable)"
    )
    optparser.add_option(
        "-D", "--dropout", default="0.5",
        type='float', help="Droupout on the input (0 = no dropout)"
    )
    optparser.add_option(
        "-r", "--reload", default="0",
        type='int', help="Reload the last saved model"
    )
    optparser.add_option(
        "-g", '--use_gpu', default='1',
        type='int', help='whether or not to ues gpu'
    )
    optparser.add_option(
        '--name', default='model',
        help='model name'
    )
    opts = optparser.parse_args()[0]

    parameters = OrderedDict()
    parameters['tag_scheme'] = opts.tag_scheme
    parameters['lower'] = opts.lower == 1
    parameters['word_dim'] = opts.word_dim
    parameters['word_lstm_dim'] = opts.word_lstm_dim
    parameters['word_bidirect'] = opts.word_bidirect == 1
    parameters['pre_emb'] = opts.pre_emb
    parameters['all_emb'] = opts.all_emb == 1
    parameters['crf'] = opts.crf == 1
    parameters['dropout'] = opts.dropout
    parameters['reload'] = opts.reload == 1
    parameters['name'] = opts.name
    parameters['epochs'] = opts.epochs

    parameters['use_gpu'] = opts.use_gpu == 1 and torch.cuda.is_available()
    use_gpu = parameters['use_gpu']

    mapping_file = 'models/mapping.pkl'

    name = parameters['name']
    model_name = models_path + name  # get_name(parameters)
    tmp_model = model_name + '.tmp'

    train()

