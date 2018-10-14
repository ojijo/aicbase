import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import argparse
import _pickle as cPickle
parser = argparse.ArgumentParser(description='inference procedure, note you should train the data at first')

parser.add_argument('--word_path', type=str, default='../data/word2id.obj',
                    help='location of the word2id.obj')

args = parser.parse_args()

with open(args.word_path, 'rb') as f:
    word2id = cPickle.load(f)


def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
    print(len(word2id))
    lines_num, dim = 2, 0
    vectors = [None for i in range(0,topn)]
    iw = []
    wi = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        #为提高效率先按词库文件遍历一边
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            
            tokens = line.rstrip().split(' ')
            if tokens[0] in word2id:
                rowNum = word2id[tokens[0]]
                vectors[rowNum] = np.asarray([float(x) for x in tokens[1:]])
            else:
                #词库里未能使用到的词向量
#                 print(tokens[0])
                continue
                
            lines_num += 1
#             vectors.append(np.asarray([float(x) for x in tokens[1:]]))
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
            
        print("****************")    
        print("找到 " + str(lines_num) + " 个匹配的词")
        print("****************")
    
    #    
    countNone = 0
    for i in range(0, topn):
#         print(type(vectors[i]))
        if type(vectors[i])== None.__class__ :
#             print( vectors[i])
            vectors[i] = np.asarray([float(0) for x in range(0,dim)])
#             vectors[i] = np.random.uniform(-0.1, 0.1, dim).round(6).tolist()
            countNone+=1        
    print(countNone)
    
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi, dim



def main():
    vectors_path = "/home/dl-ubuntu/Downloads/sgns.baidubaike.bigram-char"
    vocab_size = 96973  
#     vocab_size = 3  
    results = {}  # Records the results

    vectors, iw, wi, dim = read_vectors(vectors_path, vocab_size)  # Read top n word vectors. Read all vectors when topn is 0

    word_embeds = nn.Embedding(vocab_size, 300)
    pretrained_weight = np.array(vectors)
    word_embeds.weight.data.copy_(torch.from_numpy(pretrained_weight))
    with open('../data/embedding.obj', 'wb') as f:
        cPickle.dump(pretrained_weight, f)


if __name__ == '__main__':
    main()