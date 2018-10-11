# -*- coding: utf-8 -*-
import argparse
# import cPickle
import _pickle as cPickle
import torch

from model import MwAN
from preprocess import process_data
from utils import *
import codecs

parser = argparse.ArgumentParser(description='PyTorch implementation for Multiway Attention Networks for Modeling '
                                             'Sentence Pairs of the AI-Challenges')

parser.add_argument('--data', type=str, default='../data/',
                    help='location directory of the data corpus')
parser.add_argument('--threshold', type=int, default=5,
                    help='threshold count of the word')
parser.add_argument('--epoch', type=int, default=5,
                    help='training epochs')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size of the model')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--log_interval', type=int, default=300,
                    help='# of batches to see the training error')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')

parser.add_argument('--word_path', type=str, default='../data/word2id.obj',
                    help='location of the word2id.obj')

args = parser.parse_args()

# vocab_size = process_data(args.data, args.threshold)
vocab_size = 98745

from preprocess import seg_data, transform_data_to_id

parser = argparse.ArgumentParser(description='inference procedure, note you should train the data at first')

parser.add_argument('--data', type=str,
                    default='../data/ai_challenger_oqmrc_validationset_20180816/ai_challenger_oqmrc_validationset.json',
                    help='location of the test data')

parser.add_argument('--word_path', type=str, default='../data/word2id.obj',
                    help='location of the word2id.obj')

parser.add_argument('--output', type=str, default='../data/prediction.a.txt',
                    help='prediction path')
parser.add_argument('--model', type=str, default='model.pt',
                    help='model path')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_true',default=True,
                    help='use CUDA')

args = parser.parse_args()

with open(args.model, 'rb') as f:
    model = torch.load(f)
if args.cuda:
    model.cuda()

with open(args.word_path, 'rb') as f:
    word2id = cPickle.load(f)

raw_data = seg_data(args.data)
transformed_data = transform_data_to_id(raw_data, word2id)
data = [x + [y[2]] for x, y in zip(transformed_data, raw_data)]
dev_data = sorted(data, key=lambda x: len(x[1]))

print( 'test data size {:d}'.format(len(dev_data)))
notSureCode = 0

with open(args.word_path, 'rb') as f:
    word2id = cPickle.load(f)

    def map_word_to_id(word):
        output = []
        if word in word2id:
            output.append(word2id[word])
        else:
            chars = list(word)
            for char in chars:
                if char in word2id:
                    output.append(word2id[char])
                else:
                    output.append(1)
        return output
    notSureCode = map_word_to_id('无法确定')[0]
    
def test():
    countYesNoAnswer =0    
    countNotSureAnswer =0 
    countYesNoMax = 0
    countNotSureMax = 0
    countNotSureWin = 0
    countNotSureLose = 0
    countNotSureTotalMod = 0
 
    notSureResult = []
    
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
            for ind, itm in enumerate(output):
                posMax=itm.argmax().item()
                
                notSureItem = [0,[],"No"]
                if answer[ind][0][0]== notSureCode:  #如果对应答案是不确定
                    countNotSureAnswer+=1
                    notSureItem[0]=i + ind
                    notSureItem[1] =[itm[0].item(),itm[1].item(),itm[2].item()]
                else :
                    countYesNoAnswer += 1
                
                if posMax == 0:     #判断正确
                    if answer[ind][posMax][0]== notSureCode: #答案是不确定
                        countNotSureMax+=1
                        notSureItem[2] ="Yes"
                    else:
                        countYesNoMax+=1
                        
                if (notSureItem[0]>0):
                    notSureResult.append(notSureItem)
 
                for j in range(0,len(itm)):
                    if answer[ind][j][0]== notSureCode:
                        if itm[j] > 0.3:
                            countNotSureTotalMod += 1
                            #posMax=j
                            if j==0 and posMax!=0:
                                countNotSureWin+=1
                            else:       
                                if j!=0 and posMax==0:
                                    countNotSureLose+=1  
                        break

#             r += torch.eq(output, 0).sum().item()
                if posMax==0:
                    r += 1
#             a +=len(one)
                a += 1
    
    print("total cases " + str(a))              
    print("countYesNoAnswer " + str(countYesNoAnswer))                  
    print("countYesNoMax " + str(countYesNoMax))   
    print("countNotSureAnswer " + str(countNotSureAnswer))        
    print("countNotSureMax " + str(countNotSureMax)) 
    print("countNotSureTotalMod " + str(countNotSureTotalMod))               
    print("countNotSureWin " + str(countNotSureWin))      
    print("countNotSureLose " + str(countNotSureLose)) 
    
    rate = r * 100.0 / a
    modRate = (r + countNotSureWin - countNotSureLose)*100/a
    print("rate " + str(rate))
    print("modRate " + str(modRate) )
    
    predictions = ""
    for itm in notSureResult:
        predictions = predictions + str(itm[0]) + "," + str(itm[1][0])+ "," + str(itm[1][1])+ "," + str(itm[1][2]) + ","+ itm[2]+ "," + str(dev_data[itm[0]][3])  + '\n'
    with codecs.open("test_ouput.txt", 'w',encoding='utf-8') as f:
        f.write(predictions)
    
    return r * 100.0 / a


def main():
    best = 0.0
    acc = test()
    if acc > best:
        best = acc
    print( ' dev acc is {:f}, best dev acc {:f}'.format( acc, best))


if __name__ == '__main__':
    main()
