import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = []
    iw = []
    wi = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors.append(np.asarray([float(x) for x in tokens[1:]]))
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi, dim



def main():
    vectors_path = "/Users/hongjie/Downloads/sgns.baidubaike.bigram-char"
    vocab_size = 1000
    results = {}  # Records the results

    vectors, iw, wi, dim = read_vectors(vectors_path, vocab_size)  # Read top n word vectors. Read all vectors when topn is 0

    word_embeds = nn.Embedding(vocab_size, 300)
    pretrained_weight = np.array(vectors)
    word_embeds.weight.data.copy_(torch.from_numpy(pretrained_weight))



if __name__ == '__main__':
    main()