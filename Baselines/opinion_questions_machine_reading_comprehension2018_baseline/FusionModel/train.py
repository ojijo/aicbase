# -*- coding: utf-8 -*-
import argparse
# import cPickle
import _pickle as cPickle
import torch

# from model import MwAN
from preprocess import process_data
from utils import *

import logging
from model import FusionNet_Model



parser = argparse.ArgumentParser(description='Fusion net')



parser.add_argument('--no_seq_dropout', dest='do_seq_dropout', action='store_false')
parser.add_argument('--my_dropout_p', type=float, default=0.3)
parser.add_argument('--dropout_emb', type=float, default=0.3)
parser.add_argument('--dropout_EM', type=float, default=0.6)

#数据路径
parser.add_argument('--data', type=str, default='../data/',
                    help='location directory of the data corpus')

#词典大小
parser.add_argument('--vocab_size', type=int, default=96972,
                    help='size of the word')
# parser.add_argument('--threshold', type=int, default=5,
#                     help='threshold count of the word')
# parser.add_argument('--epoch', type=int, default=5,
#                     help='training epochs')
# parser.add_argument('--emsize', type=int, default=300,
#                     help='size of word embeddings')
# parser.add_argument('--nhid', type=int, default=128,
#                     help='hidden size of the model')
# parser.add_argument('--batch_size', type=int, default=32, metavar='N',
#                     help='batch size')
# parser.add_argument('--log_interval', type=int, default=300,
#                     help='# of batches to see the training error')
# parser.add_argument('--dropout', type=float, default=0.2,
#                     help='dropout applied to layers (0 = no dropout)')
# parser.add_argument('--cuda', action='store_true', default=True,
#                     help='use CUDA')
# parser.add_argument('--save', type=str, default='model.pt',
#                     help='path to save the final model')

args = parser.parse_args()

# vocab_size = process_data(args.data, args.threshold)
vocab_size = 96972

with open(args.data + 'train.pickle', 'rb') as f:
    train_data = cPickle.load(f)
with open(args.data + 'dev.pickle', 'rb') as f:
    dev_data = cPickle.load(f)
dev_data = sorted(dev_data, key=lambda x: len(x[1]))

print('train data size {:d}, dev data size {:d}'.format(len(train_data), len(dev_data)))        

with open('../data/embedding.obj', 'rb') as f:
    pretrained_weight = cPickle.load(f)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)   
log.info('[program starts.]')
opt = vars(args) # changing opt will change args
model = FusionNet_Model(opt, pretrained_weight)
# model = MwAN(vocab_size=vocab_size, embedding_size=args.emsize, encoder_size=args.nhid, drop_out=args.dropout,pretrained_weight=pretrained_weight)
print('Model total parameters:', get_model_parameters(model))
print(model)
if args.cuda:
    model.cuda()
optimizer = torch.optim.Adamax(model.parameters())




def train(epoch):
    model.train()
    data = shuffle_data(train_data, 1)
    total_loss = 0.0
    for num, i in enumerate(range(0, len(data), args.batch_size)):
        one = data[i:i + args.batch_size]
        query, _ = padding([x[0] for x in one], max_len=50)
        passage, _ = padding([x[1] for x in one], max_len=350)
        answer = pad_answer([x[2] for x in one])
        query, passage, answer = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer)
        if args.cuda:
            query = query.cuda()
            passage = passage.cuda()
            answer = answer.cuda()
        optimizer.zero_grad()
        loss = model([query, passage, answer, True])
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        if (num + 1) % args.log_interval == 0:
            print( '|------epoch {:d} train error is {:f}  eclipse {:.2f}%------|'.format(epoch,
                                                                                         total_loss / args.log_interval,
                                                                                         i * 100.0 / len(data)))
            total_loss = 0


def test():
    model.eval()
    r, a = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(dev_data), args.batch_size):
            one = dev_data[i:i + args.batch_size]
            query, _ = padding([x[0] for x in one], max_len=50)
            passage, _ = padding([x[1] for x in one], max_len=500)
            answer = pad_answer([x[2] for x in one])
            query, passage, answer = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer)
            if args.cuda:
                query = query.cuda()
                passage = passage.cuda()
                answer = answer.cuda()
            output = model([query, passage, answer, False])
            r += torch.eq(output, 0).sum().item()
            a += len(one)
    return r * 100.0 / a


def main():
    best = 0.0
    for epoch in range(args.epoch):
        print("start epoch " + str(epoch))
        train(epoch)
        acc = test()
        if acc > best:
            best = acc
            with open(args.save, 'wb') as f:
                torch.save(model, f)
        print( 'epcoh {:d} dev acc is {:f}, best dev acc {:f}'.format(epoch, acc, best))


if __name__ == '__main__':
    main()


# import re
# import os
# import sys
# import random
# import string
# import logging
# import argparse
# import pickle
# from shutil import copyfile
# from datetime import datetime
# from collections import Counter, defaultdict
# import torch
# import msgpack
# import numpy as np
# from FusionModel.model import FusionNet_Model
# from general_utils import BatchGen, load_train_data, load_eval_data
# 
# parser = argparse.ArgumentParser(
#     description='Train FusionNet model for Natural Language Inference.'
# )
# # system
# parser.add_argument('--name', default='', help='additional name of the current run')
# parser.add_argument('--log_file', default='output.log',
#                     help='path for log file.')
# parser.add_argument('--log_per_updates', type=int, default=80,
#                     help='log model loss per x updates (mini-batches).')
# 
# parser.add_argument('--train_meta', default='multinli_1.0/train_meta.msgpack',
#                     help='path to preprocessed training meta file.')
# parser.add_argument('--train_data', default='multinli_1.0/train_data.msgpack',
#                     help='path to preprocessed training data file.')
# parser.add_argument('--dev_data', default='multinli_1.0/dev_mismatch_preprocessed.msgpack',
#                     help='path to preprocessed validation data file.')
# parser.add_argument('--test_data', default='multinli_1.0/dev_match_preprocessed.msgpack',
#                     help='path to preprocessed testing (dev set 2) data file.')
# 
# parser.add_argument('--MTLSTM_path', default='glove/MT-LSTM.pth')
# parser.add_argument('--model_dir', default='models',
#                     help='path to store saved models.')
# parser.add_argument('--save_all', dest="save_best_only", action='store_false',
#                     help='save all models in addition to the best.')
# parser.add_argument('--do_not_save', action='store_true', help='don\'t save any model')
# parser.add_argument('--seed', type=int, default=1023,
#                     help='random seed for data shuffling, dropout, etc.')
# parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available(),
#                     help='whether to use GPU acceleration.')
# # training
# parser.add_argument('-e', '--epoches', type=int, default=20)
# parser.add_argument('-bs', '--batch_size', type=int, default=32)
# parser.add_argument('-op', '--optimizer', default='adamax',
#                     help='supported optimizer: adamax, sgd, adadelta, adam')
# parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
# parser.add_argument('-tp', '--tune_partial', type=int, default=1000,
#                     help='finetune top-x embeddings (including <PAD>, <UNK>).')
# parser.add_argument('--fix_embeddings', action='store_true',
#                     help='if true, `tune_partial` will be ignored.')
# # model
# parser.add_argument('--number_of_class', type=int, default=3)
# parser.add_argument('--final_merge', default='linear_self_attn')
# 
# parser.add_argument('--hidden_size', type=int, default=125)
# parser.add_argument('--enc_rnn_layers', type=int, default=2, help="Encoding RNN layers")
# parser.add_argument('--inf_rnn_layers', type=int, default=2, help="Inference RNN layers")
# parser.add_argument('--full_att_type', type=int, default=2)
# 
# parser.add_argument('--pos_size', type=int, default=56,
#                     help='how many kinds of POS tags.')
# parser.add_argument('--pos_dim', type=int, default=12,
#                     help='the embedding dimension for POS tags.')
# parser.add_argument('--ner_size', type=int, default=19,
#                     help='how many kinds of named entity tags.')
# parser.add_argument('--ner_dim', type=int, default=8,
#                     help='the embedding dimension for named entity tags.')
# 
# parser.add_argument('--no_seq_dropout', dest='do_seq_dropout', action='store_false')
# parser.add_argument('--my_dropout_p', type=float, default=0.3)
# parser.add_argument('--dropout_emb', type=float, default=0.3)
# parser.add_argument('--dropout_EM', type=float, default=0.6)
# 
# args = parser.parse_args()
# 
# if args.name != '':
#     args.model_dir = args.model_dir + '_' + args.name
#     args.log_file = os.path.dirname(args.log_file) + 'output_' + args.name + '.log'
# 
# # set model dir
# model_dir = args.model_dir
# os.makedirs(model_dir, exist_ok=True)
# model_dir = os.path.abspath(model_dir)
# 
# # set random seed
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# if args.cuda:
#     torch.cuda.manual_seed_all(args.seed)
# 
# # setup logger
# log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
# 
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.INFO)
# formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
# ch.setFormatter(formatter)
# log.addHandler(ch)
# 
# def main():
#     log.info('[program starts.]')
#     opt = vars(args) # changing opt will change args
#     train, train_embedding, opt = load_train_data(opt, args.train_meta, args.train_data)
#     dev, dev_embedding, dev_ans = load_eval_data(opt, args.dev_data)
#     test, test_embedding, test_ans = load_eval_data(opt, args.test_data)
#     log.info('[Data loaded.]')
# 
#     model = FusionNet_Model(opt, train_embedding)
#     if args.cuda: model.cuda()
#     log.info("[dev] Total number of params: {}".format(model.total_param))
# 
#     best_acc = 0.0
# 
#     for epoch in range(1, 1 + args.epoches):
#         log.warning('Epoch {}'.format(epoch))
# 
#         # train
#         batches = BatchGen(train, batch_size=args.batch_size, gpu=args.cuda)
#         start = datetime.now()
#         for i, batch in enumerate(batches):
#             model.update(batch)
#             if i % args.log_per_updates == 0:
#                 log.info('updates[{0:6}] train loss[{1:.5f}] remaining[{2}]'.format(
#                     model.updates, model.train_loss.avg,
#                     str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))

