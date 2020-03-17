# -*- coding: utf-8 -*-
'''
    @Author: King
    @Date: 2019.03.20
    @Purpose: 使用PyTorch实现Chatbot
    @Introduction:  下面介绍
    @Datasets: 百度的中文问答数据集WebQA 数据
    @Link : 
    @Reference : 
'''
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
from io import open
import itertools
import math
import os

from config import Config
from load_data import load_data,jieba_cut_word
from create_voc import loadPrepareData,trimRareWords
from seq2seq_model import batch2TrainData,EncoderRNN,LuongAttnDecoderRNN,indexesFromSentence,GreedySearchDecoder

config = Config()

''' Masked损失
    # forward实现之后，我们就需要计算loss。seq2seq有两个RNN，Encoder RNN是没有直接定义损失函数的，它是通过影响Decoder从而影响最终的输出以及loss。Decoder输出一个序列，前面我们介绍的是Decoder在预测时的过程，它的长度是不固定的，只有遇到EOS才结束。给定一个问答句对，我们可以把问题输入Encoder，然后用Decoder得到一个输出序列，但是这个输出序列和”真实”的答案长度并不相同。

    # 而且即使长度相同并且语义相似，也很难直接知道预测的答案和真实的答案是否类似。
    # 那么我们怎么计算loss呢？比如输入是”What is your name?”，
    # 训练数据中的答案是”I am LiLi”。
    # 假设模型有两种预测：”I am fine”和”My name is LiLi”。
    # 从语义上显然第二种答案更好，但是如果字面上比较的话可能第一种更好。

    # 但是让机器知道”I am LiLi”和”My name is LiLi”的语义很接近这是非常困难的，
    # 所以实际上我们通常还是通过字面上里进行比较。
    # 我们会限制Decoder的输出，使得Decoder的输出长度和”真实”答案一样，
    # 然后逐个时刻比较。Decoder输出的是每个词的概率分布，
    # 因此可以使用交叉熵损失函数。
    # 但是这里还有一个问题，因为是一个batch的数据里有一些是padding的，
    # 因此这些位置的预测是没有必要计算loss的，
    # 因此我们需要使用前面的mask矩阵把对应位置的loss去掉，
    # 我们可以通过下面的函数来实现计算Masked的loss。
'''
def maskNLLLoss(inp, target, mask):
    # 计算实际的词的个数，因为padding是0，非padding是1，因此sum就可以得到词的个数
    nTotal = mask.sum()
    
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(config.device)
    return loss, nTotal.item()

''' 训练过程
    1) 把整个batch的输入传入encoder 
    2) 把decoder的输入设置为特殊的，初始隐状态设置为encoder最后时刻的隐状态 
    3) decoder每次处理一个时刻的forward计算 
    4) 如果是teacher forcing，把上个时刻的"正确的"词作为当前输入，否则用上一个时刻的输出作为当前时刻的输入 
    5) 计算loss 
    6) 反向计算梯度 
    7) 对梯度进行裁剪 
    8) 更新模型(包括encoder和decoder)参数
'''
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=config.MAX_LENGTH):

    # 梯度清空
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 设置device，从而支持GPU，当然如果没有GPU也能工作。
    input_variable = input_variable.to(config.device)
    lengths = lengths.to(config.device)
    target_variable = target_variable.to(config.device)
    mask = mask.to(config.device)

    # 初始化变量
    loss = 0
    print_losses = []
    n_totals = 0

    # encoder的Forward计算
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Decoder的初始输入是SOS，我们需要构造(1, batch)的输入，表示第一个时刻batch个输入。
    decoder_input = torch.LongTensor([[config.SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(config.device)

    # 注意：Encoder是双向的，而Decoder是单向的，因此从下往上取n_layers个
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # 确定是否teacher forcing
    use_teacher_forcing = True if random.random() < config.teacher_forcing_ratio else False

    # 一次处理一个时刻 
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: 下一个时刻的输入是当前正确答案
            decoder_input = target_variable[t].view(1, -1)
            # 计算累计的loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 不是teacher forcing: 下一个时刻的输入是当前模型预测概率最高的值
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(config.device)
            # 计算累计的loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # 反向计算 
    loss.backward()

    # 对encoder和decoder进行梯度裁剪
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # 更新参数
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

''' 训练迭代过程
    最后是把前面的代码组合起来进行训练。
    函数trainIters用于进行n_iterations次minibatch的训练。

    值得注意的是我们定期会保存模型，我们会保存一个tar包，
    包括encoder和decoder的state_dicts(参数),
    优化器(optimizers)的state_dicts, loss和迭代次数。
    这样保存模型的好处是从中恢复后我们既可以进行预测也可以进行训练(因为有优化器的参数和迭代的次数)。
'''
def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, 
              embedding, encoder_n_layers, decoder_n_layers,hidden_size, save_dir, n_iteration, batch_size, 
              print_every, save_every, clip, corpus_name, loadFilename):

    # 随机选择n_iteration个batch的数据(pair)
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    # 初始化
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # 训练
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # 训练一个batch的数据
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # 进度
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}"
			.format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # 保存checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'
		.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

# 测试对话函数
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=config.MAX_LENGTH):
    ### 把输入的一个batch句子变成id
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # 创建lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # 转置 
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # 放到合适的设备上(比如GPU)
    input_batch = input_batch.to(config.device)
    lengths = lengths.to(config.device)
    # 用searcher解码
    tokens, scores = searcher(input_batch, lengths, max_length)
    # ID变成词。
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words

# 对话模型
def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        #try:
        # 得到用户终端的输入
        input_sentence = input('> ')
        # 是否退出
        if input_sentence == 'q' or input_sentence == 'quit': break
        # 句子归一化
        #input_sentence = normalizeString(input_sentence)
        #print("input_sentence:{0}".format(input_sentence))
        input_sentence = jieba_cut_word(input_sentence)
        print("input_sentence:{0}".format(input_sentence))
        # 生成响应Evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        # 去掉EOS后面的内容
        words = []
        for word in output_words:
            if word == 'EOS':
                break
            elif word != 'PAD':
                words.append(word)
        print('Bot:', ' '.join(words))

        # except KeyError:
        # print("Error: Encountered unknown word.")


if __name__ == "__main__":
    question_list,answer_list=load_data(config.data_path_name)
    print("question_list[0:2]:{0}".format(question_list[0:2]))
    print("answer_list[0:2]:{0}".format(answer_list[0:2]))

    # 问题分词
    question_cut_list = [jieba_cut_word(question) for question in question_list]
    print("question_cut_list[0:2]:{0}".format(question_cut_list[0:2]))

    # Load/Assemble voc and pairs
    # save_dir = os.path.join("data", "save")
    voc, pairs = loadPrepareData(question_cut_list,answer_list,config.MAX_LENGTH)
    # 输出一些句对
    print("\npairs:")
    for pair in pairs[:10]:
        print(pair)

    # 实际进行处理
    pairs = trimRareWords(voc, pairs, config.MIN_COUNT)
    # print("voc:{0}".format(voc.word2index))

    if config.mode == 'train':
        # 如果loadFilename不空，则从中加载模型 
        if config.loadFilename:
            # 如果训练和加载是一条机器，那么直接加载 
            checkpoint = torch.load(config.loadFilename)
            # 否则比如checkpoint是在GPU上得到的，但是我们现在又用CPU来训练或者测试，那么注释掉下面的代码
            #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
            encoder_sd = checkpoint['en']
            decoder_sd = checkpoint['de']
            encoder_optimizer_sd = checkpoint['en_opt']
            decoder_optimizer_sd = checkpoint['de_opt']
            embedding_sd = checkpoint['embedding']
            voc.__dict__ = checkpoint['voc_dict']

        print('Building encoder and decoder ...')
        # 初始化word embedding
        embedding = nn.Embedding(voc.num_words, config.hidden_size)
        if config.loadFilename:
            embedding.load_state_dict(embedding_sd)

        # 初始化encoder和decoder模型
        encoder = EncoderRNN(config.hidden_size, embedding, config.encoder_n_layers, config.dropout)
        decoder = LuongAttnDecoderRNN(config.attn_model, embedding, config.hidden_size, voc.num_words, 
                        config.decoder_n_layers, config.dropout)
        if config.loadFilename:
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)

        # 使用合适的设备
        encoder = encoder.to(config.device)
        decoder = decoder.to(config.device)
        print('Models built and ready to go!')

        # 设置进入训练模式，从而开启dropout 
        encoder.train()
        decoder.train()

        # 初始化优化器 
        print('Building optimizers ...')
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.learning_rate)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.learning_rate * config.decoder_learning_ratio)
        if config.loadFilename:
            encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            decoder_optimizer.load_state_dict(decoder_optimizer_sd)

        # 开始训练
        print("Starting Training!")
        trainIters(config.model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
                embedding, config.encoder_n_layers, config.decoder_n_layers, config.hidden_size, config.save_dir, config.n_iteration, config.batch_size,
                config.print_every, config.save_every, config.clip, config.corpus_name, config.loadFilename)
    
    elif config.mode == 'test':
        config.loadFilename = 'data/save/cb_model/baidu_qa/2-2_500/4000_checkpoint.tar'
        if not config.loadFilename:
            raise Exception("loadFilename 不能为空!")
        
        if config.loadFilename:
            # 如果训练和加载是一条机器，那么直接加载 
            checkpoint = torch.load(config.loadFilename)
            # 否则比如checkpoint是在GPU上得到的，但是我们现在又用CPU来训练或者测试，那么注释掉下面的代码
            #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
            encoder_sd = checkpoint['en']
            decoder_sd = checkpoint['de']
            encoder_optimizer_sd = checkpoint['en_opt']
            decoder_optimizer_sd = checkpoint['de_opt']
            embedding_sd = checkpoint['embedding']
            voc.__dict__ = checkpoint['voc_dict']

        print('Building encoder and decoder ...')
        # 初始化word embedding
        embedding = nn.Embedding(voc.num_words, config.hidden_size)
        if config.loadFilename:
            embedding.load_state_dict(embedding_sd)
            
        # 初始化encoder和decoder模型
        encoder = EncoderRNN(config.hidden_size, embedding, config.encoder_n_layers, config.dropout)
        decoder = LuongAttnDecoderRNN(config.attn_model, embedding, config.hidden_size, voc.num_words, 
                        config.decoder_n_layers, config.dropout)

        if config.loadFilename:
            encoder.load_state_dict(encoder_sd)
            decoder.load_state_dict(decoder_sd)

        # 使用合适的设备
        encoder = encoder.to(config.device)
        decoder = decoder.to(config.device)
        print('Models built and ready to go!')


        # 进入eval模式，从而去掉dropout。 
        encoder.eval()
        decoder.eval()

        # 构造searcher对象 
        searcher = GreedySearchDecoder(encoder, decoder)

        # 测试
        evaluateInput(encoder, decoder, searcher, voc)
    else:
        raise Exception("mode is train or test !!!")


