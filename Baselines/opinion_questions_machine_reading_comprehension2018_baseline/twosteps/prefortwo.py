# -*- coding: utf-8 -*-
# import cPickle
import _pickle as cPickle
import json

import jieba
from lib2to3.fixes.fix_imports import alternates

OptionList = [u'无法确认',u'不确定',u'不能确定',u'wfqd', u'无法比较',u'无法判断',u'无法选择',u'个人爱好',u'无关',u'不清楚',u'无法肯定']

countReplaceOption = 0
def format(o):
    o = o.strip()
    if o !=  u'无法确定':
        if (o in OptionList) or (u'无法确' in o):
            o = u'无法确定'
            global countReplaceOption
            countReplaceOption += 1
    return o

def seg_line(line):
    return list(jieba.cut(line))


def seg_data(path):
    countSure = 0
    countNotSure = 0
    countIllegalOption = 0
    global countReplaceOption
    countReplaceOption = 0
    print( 'start process ', path)
    data = []
    dataNS = []
    with open(path, 'r') as f:
        for line in f:
            dic = json.loads(line, encoding='utf-8')
            question = dic['query']
            doc = dic['passage']
            alternatives = dic['alternatives']
            query_id = dic['query_id']
            
            alternatives = alternatives.split('|')
            alternatives = [ format(o) for o in alternatives]
            
            if len(alternatives) != 3 :    #
                print("illegal data at "  + query_id + "" + str(alternatives))
                continue
            
            if (alternatives[0]==u'无法确定') :
                dataNS.append([seg_line(question), seg_line(doc), [u'无法确定',u'可以确定'], query_id])
                countNotSure += 1
            else:
                dataNS.append([seg_line(question), seg_line(doc), [u'可以确定',u'无法确定'], query_id])
                countSure += 1
                try:
                    alternatives.remove(u'无法确定')
                except Exception as e:
                    print('except:', e)
                    print(alternatives)
                    countIllegalOption+=1
                    del alternatives[2]
                    
                data.append([seg_line(question), seg_line(doc), alternatives, query_id])
            
        print("find " + str(countIllegalOption) + " new illegal option")
        print("replace " + str(countReplaceOption) + " known illegal option")
        print("find  not sure " + str(countNotSure))
        print("find sure " + str(countSure))
        
    return data, dataNS

def seg_data_for_infer(path):

    countIllegalOption = 0
    global countReplaceOption
    countReplaceOption = 0
    print( 'start process ', path)
    data = []
    dataNS = []
    with open(path, 'r') as f:
        for line in f:
            dic = json.loads(line, encoding='utf-8')
            question = dic['query']
            doc = dic['passage']
            alternatives = dic['alternatives']
            query_id = dic['query_id']
            
            alternatives = alternatives.split('|')
            alternatives = [ format(o) for o in alternatives]
            
            if len(alternatives) != 3 :    #
                print("illegal data at "  + query_id + "" + str(alternatives))
                continue
            
            dataNS.append([seg_line(question), seg_line(doc), [u'无法确定',u'可以确定'], query_id])

            try:
                alternatives.remove(u'无法确定')
            except Exception as e:
                print('except:', e)
                print(alternatives)
                countIllegalOption+=1
                del alternatives[2]
                
            data.append([seg_line(question), seg_line(doc), alternatives, query_id])
            
        print("find " + str(countIllegalOption) + " new illegal option")
        print("replace " + str(countReplaceOption) + " known illegal option")
        
    return data, dataNS
def build_word_count(data):
    wordCount = {}

    def add_count(lst):
        for word in lst:
            if word not in wordCount:
                wordCount[word] = 0
            wordCount[word] += 1

    for one in data:
        [add_count(x) for x in one[0:3]]
    print( 'word type size ', len(wordCount))
    return wordCount


def build_word2id(wordCount, threshold=10):
    word2id = {'<PAD>': 0, '<UNK>': 1}
    for word in wordCount:
        if wordCount[word] >= threshold:
            if word not in word2id:
                word2id[word] = len(word2id)
        else:
            chars = list(word)
            for char in chars:
                if char not in word2id:
                    word2id[char] = len(word2id)
    print( 'processed word size ', len(word2id))
    return word2id


def transform_data_to_id(raw_data, word2id):
    data = []

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

    def map_sent_to_id(sent):
        output = []
        for word in sent:
            output.extend(map_word_to_id(word))
        return output

    for one in raw_data:
        question = map_sent_to_id(one[0])
        doc = map_sent_to_id(one[1])
        candidates = [map_word_to_id(x) for x in one[2]]
        length = [len(x) for x in candidates]
        max_length = max(length)
        if max_length > 1:
            pad_len = [max_length - x for x in length]
            candidates = [x[0] + [0] * x[1] for x in zip(candidates, pad_len)]
        data.append([question, doc, candidates, one[-1]])
    return data


def process_data(data_path, word_min_count=5):
    train_file_path = data_path + 'ai_challenger_oqmrc_trainingset_20180816/ai_challenger_oqmrc_trainingset.json'
    dev_file_path = data_path + 'ai_challenger_oqmrc_validationset_20180816/ai_challenger_oqmrc_validationset.json'
    test_a_file_path = data_path + 'ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json'
    output_path = ['data/' + x for x in ['train.pickle','trainNS.pickle','dev.pickle','devNS.pickle']]
    
    raw_data = []
    dataTrain, dataTrainNS =  seg_data(train_file_path)
    dataDev, dataDevNS = seg_data(dev_file_path)
    dataTest, dataTestNS = seg_data(test_a_file_path)
    
    print('Length of dataTrain ' + str(len(dataTrain)))
    print('Length of dataTrainNS ' + str(len(dataTrainNS)))
    print('Length of dataDev ' + str(len(dataDev)))
    print('Length of dataDevNS ' + str(len(dataDevNS)))
    
    raw_data.append(dataTrainNS)
    raw_data.append(dataDevNS)
    raw_data.append(dataTestNS)
    
    word_count = build_word_count([y for x in raw_data for y in x])
    with open('data/word-count.obj', 'wb') as f:
        cPickle.dump(word_count, f)
    word2id = build_word2id(word_count, word_min_count)
    with open('data/word2id.obj', 'wb') as f:
        cPickle.dump(word2id, f)
    for one_raw_data, one_output_file_path in zip([dataTrain,dataTrainNS,dataDev,dataDevNS], output_path):
        with open(one_output_file_path, 'wb') as f:
            one_data = transform_data_to_id(one_raw_data, word2id)
            cPickle.dump(one_data, f)
    return len(word2id)



# process_data('../data/',5)