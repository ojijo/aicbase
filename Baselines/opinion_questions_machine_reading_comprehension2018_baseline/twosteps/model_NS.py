# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F


class MwAN(nn.Module):
    def __init__(self, vocab_size, embedding_size, encoder_size, drop_out=0.2):
        super(MwAN, self).__init__()
        self.drop_out=drop_out
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim=embedding_size)
        self.q_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.p_encoder = nn.GRU(input_size=embedding_size, hidden_size=encoder_size, batch_first=True,
                                bidirectional=True)
        self.a_encoder = nn.GRU(input_size=embedding_size, hidden_size=int(embedding_size / 2), batch_first=True,
                                bidirectional=True)
        self.a_attention = nn.Linear(embedding_size, 1, bias=False)
        # Concat Attention
        self.Wc1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wc2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vc = nn.Linear(encoder_size, 1, bias=False)
        # Bilinear Attention
        self.Wb = nn.Linear(2 * encoder_size, 2 * encoder_size, bias=False)
        # Dot Attention :
        self.Wd = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vd = nn.Linear(encoder_size, 1, bias=False)
        # Minus Attention :
        self.Wm = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vm = nn.Linear(encoder_size, 1, bias=False)

        self.Ws = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vs = nn.Linear(encoder_size, 1, bias=False)

        self.gru_agg = nn.GRU(12 * encoder_size, encoder_size, batch_first=True, bidirectional=True)
        """
        prediction layer
        """
        self.Wq = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vq = nn.Linear(encoder_size, 1, bias=False)
        self.Wp1 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.Wp2 = nn.Linear(2 * encoder_size, encoder_size, bias=False)
        self.vp = nn.Linear(encoder_size, 1, bias=False)
        self.prediction = nn.Linear(2 * encoder_size, embedding_size, bias=False)
        self.initiation()

    def initiation(self):
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, 0.1)

    def forward(self, inputs):
        [query, passage, answer, is_train] = inputs
        #Step 0 - 词向量
        q_embedding = self.embedding(query)
        p_embedding = self.embedding(passage)
        a_embeddings = self.embedding(answer)
        
        #答案的计算
        #a_embeddings = a_embeddings.view(-1, a_embeddings.size(2), a_embeddings.size(3)) #合并前2维
        a_embedding, _ = self.a_encoder(a_embeddings.view(-1, a_embeddings.size(2), a_embeddings.size(3))) #加入上下文的词向量
        
        a_score = self.a_attention(a_embedding) #计算自身注意力向量
        a_score = F.softmax(a_score, 1)
        
        a_output = a_score.transpose(2, 1) #使用自身注意力
        a_output = a_output.bmm(a_embedding)
        a_output = a_output.squeeze()
        a_embedding = a_output.view(a_embeddings.size(0), 2, -1)
        
        #Step 1  - RNN上下文 hidden state       #p,q倒过来cuda报oom 
        hq, _ = self.q_encoder(p_embedding)
        hq=F.dropout(hq,self.drop_out)
        hp, _ = self.p_encoder(q_embedding)
        hp=F.dropout(hp,self.drop_out)

        #Step 2 - 4种注意力
        _s1 = self.Wc1(hq).unsqueeze(1)
        _s2 = self.Wc2(hp).unsqueeze(2)
        sjt = torch.tanh(_s1 + _s2)
        sjt = self.vc(sjt)
        sjt = sjt.squeeze()
        ait = F.softmax(sjt, 2)
        qtc = ait.bmm(hq)
        
        _s1 = self.Wb(hq).transpose(2, 1)
        sjt = hp.bmm(_s1)
        ait = F.softmax(sjt, 2)
        qtb = ait.bmm(hq)
        
        #
        _s1 = hq.unsqueeze(1)
        _s2 = hp.unsqueeze(2)
        sjt = self.Wd(_s1 * _s2)
        sjt = torch.tanh(sjt)
        sjt = self.vd(sjt)
        sjt = sjt.squeeze()
        ait = F.softmax(sjt, 2)
        qtd = ait.bmm(hq)
        
        #
        sjt = self.Wm(_s1 - _s2)
        sjt = torch.tanh(sjt)
        sjt = self.vm(sjt)        
        sjt = sjt.squeeze()
        ait = F.softmax(sjt, 2)
        qtm = ait.bmm(hq)
        
        #
        _s1 = hp.unsqueeze(1)
        _s2 = hp.unsqueeze(2)
        sjt = self.Ws(_s1 * _s2)
        sjt = torch.tanh(sjt)
        sjt = self.vs(sjt)
        sjt = sjt.squeeze()
        ait = F.softmax(sjt, 2)
        qts = ait.bmm(hp)
        
        # step 3 - 分两步汇总
        # step 3.1 先用RNN汇聚
        aggregation = torch.cat([hp, qts, qtc, qtd, qtb, qtm], 2)
        aggregation_representation, _ = self.gru_agg(aggregation)
        
        # step 3.2 再用attention
        sj = self.Wq(hq)
        sj = torch.tanh(sj)
        sj = self.vq(sj)
        sj = sj.transpose(2, 1)
        sj = F.softmax(sj, 2)
        rq = sj.bmm(hq)
        
        sj = self.Wp1(aggregation_representation) + self.Wp2(rq)
        sj = self.vp(sj)
        sj = sj.transpose(2, 1)
        sj = F.softmax(sj, 2)
        rp = sj.bmm(aggregation_representation)
        
        
        encoder_output = F.dropout(F.leaky_relu(self.prediction(rp)),self.drop_out)
        score = F.softmax(a_embedding.bmm(encoder_output.transpose(2, 1)).squeeze(), 1)
        if not is_train:
#             return score.argmax(1)
            return score
#         print(score[:, 0])
        loss = -torch.log(score[:, 0]).mean()
        return loss


