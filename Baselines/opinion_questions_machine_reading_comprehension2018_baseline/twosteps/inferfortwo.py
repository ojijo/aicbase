# -*- coding: utf-8 -*-
import argparse
# import cPickle
import _pickle as cPickle
import codecs

import torch
from utils import *

from prefortwo import seg_data_for_infer, transform_data_to_id

parser = argparse.ArgumentParser(description='inference procedure, note you should train the data at first')

parser.add_argument('--data', type=str,
                    default='../data/ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json',
                    help='location of the test data')

parser.add_argument('--word_path', type=str, default='data/word2id.obj',
                    help='location of the word2id.obj')

parser.add_argument('--output', type=str, default='data/prediction.a.txt',
                    help='prediction path')
parser.add_argument('--model', type=str, default='model_77.29.pt',
                    help='model path')
parser.add_argument('--modelNS', type=str, default='model_NS_90.15.pt',
                    help='modelNS path')

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_true',default=True,
                    help='use CUDA')

args = parser.parse_args()

with open(args.model, 'rb') as f:
    model = torch.load(f)
if args.cuda:
    model.cuda()

with open(args.modelNS, 'rb') as f:
    modelNS = torch.load(f)
if args.cuda:
    modelNS.cuda()

with open(args.word_path, 'rb') as f:
    word2id = cPickle.load(f)

raw_data, raw_data_NS = seg_data_for_infer(args.data)
transformed_data_NS = transform_data_to_id(raw_data_NS, word2id)
transformed_data = transform_data_to_id(raw_data, word2id)
data = [x + [y[2]] for x, y in zip(transformed_data, raw_data)] #y[2]是选项
data = sorted(data, key=lambda x: len(x[1]))
print( 'test data size {:d}'.format(len(data)))
dataNS = [x + [y[2]] for x, y in zip(transformed_data_NS, raw_data_NS)] #y[2]是选项
dataNS = sorted(dataNS, key=lambda x: len(x[1]))
print( 'test dataNS size {:d}'.format(len(dataNS)))

def inferenceNS():
    modelNS.eval()
    predictions = {}
    
    with torch.no_grad():
        for i in range(0, len(dataNS), args.batch_size):
#         for i in range(0, len(data), 3):
            try:
                one = dataNS[i:i + args.batch_size]
    #             print(one)
                query, _ = padding([x[0] for x in one], max_len=50)
                passage, _ = padding([x[1] for x in one], max_len=300)
                answer = pad_answer([x[2] for x in one])
                str_words = [x[-1] for x in one]
                ids = [x[3] for x in one]
                query, passage, answer = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer)
                if args.cuda:
                    query = query.cuda()
                    passage = passage.cuda()
                    answer = answer.cuda()
                output = modelNS([query, passage, answer, False])
                for q_id, prediction, candidates in zip(ids, output, str_words):
                    posMax=prediction.argmax().item()

                    prediction_answer = u''.join(candidates[posMax])                    
                    predictions[str(q_id)]=prediction_answer
                
    #             print(i)
            except Exception as e:
                print(e)
                print(i)
                print(one)   
                
    return predictions

def inference(predictionsNS):
    model.eval()
    predictions = []
    
    notSureCountMax = 0
    notSureCountFind = 0
    
    with torch.no_grad():
        for i in range(0, len(data), args.batch_size):
#         for i in range(0, len(data), 3):
            try:
                one = data[i:i + args.batch_size]
    #             print(one)
                query, _ = padding([x[0] for x in one], max_len=50)
                passage, _ = padding([x[1] for x in one], max_len=300)
                answer = pad_answer([x[2] for x in one])
                str_words = [x[-1] for x in one]
                ids = [x[3] for x in one]
                query, passage, answer = torch.LongTensor(query), torch.LongTensor(passage), torch.LongTensor(answer)
                if args.cuda:
                    query = query.cuda()
                    passage = passage.cuda()
                    answer = answer.cuda()
                output = model([query, passage, answer, False])
                for q_id, prediction, candidates in zip(ids, output, str_words):
                    posMax=prediction.argmax().item()
                    if predictionsNS[str(q_id)]== u'无法确定':
                        prediction_answer=u'无法确定'
                    else:
                        prediction_answer = u''.join(candidates[posMax])                    
                    predictions.append(str(q_id) + '\t' + prediction_answer)   
    #             print(i)
            except Exception as e:
                print(e)
                print(i)
                print(one)   
                
    outputs = u'\n'.join(predictions)
    with codecs.open(args.output, 'w',encoding='utf-8') as f:
        f.write(outputs)
    print( 'done!')

if __name__ == '__main__':
    predictionsNS = inferenceNS()
    inference(predictionsNS)
