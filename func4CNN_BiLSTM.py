
class TagDic():
    def __init__(self):
        # special tags
        self.idx2tag = []
        self.idx2tag.append('<pad>')  # 0
        self.idx2tag.append('<sos>')  # 1
        self.idx2tag.append('<eos>')  # 2
        self.idx2tag.append('<unk>')  # 3
        
        # tags
        self.idx2tag.append('O')
        for entity_type in ('Disease', 'Drug', 'Test_items', 'Test_Value', 'Pathogenesis', 'Anatomy', 'Reason',
                            'ADE', 'Frequency', 'Duration', 'Duration', 'Level', 'Method', 'Test', 'Symptom',
                            'Treatment', 'Amount', 'Operation', 'Class'):
            self.idx2tag.append('B-' + entity_type)
            self.idx2tag.append('I-' + entity_type)
            self.idx2tag.append('E-' + entity_type)
   
        
        self.tag_indx = {}
        for i, tag in enumerate(self.idx2tag):
            self.tag_indx[tag] = i
            
    def __getitem__(self, idx):
        if type(idx) in (list, tuple):
            return [self.idx2tag[ID] for ID in idx]
        return [self.self.idx2tag[idx]]
    
    def __len__(self):
        return len(self.idx2tag)
    
    def tag2idx(self, tags):
        if type(tags) in (list, tuple):
            return [self.tag_indx[tag] for tag in tags]
        return [self.tag_indx[tags]]
    

class Accumulator():
    def __init__(self, c):
        self.data = [0] * c
        self.count = [0.1] * c
        self.c = c
        self.backup = []
    def add(self, *args):
        assert len(args) == self.c
        for i in range(self.c):
            self.data[i] += args[i]
            self.count[i] += 1
                
    def clear(self):
        self.backup.append([a/c for a, c in zip(self.data, self.count)])
        self.data = [0] * self.c
        self.count = [0.1] * self.c
    def __getitem__(self, idx):
        return self.data[idx] / self.count[idx]

import random
def seprate_data(source, rate, rseed=None):
    '''划分训练集，验证集，测试集'''
    assert len(rate) == 3
    s = sum(rate)  #归一化
    rate = [i/s for i in rate]

    if rseed is None: #加载随机种子
        rseed = random.randint(0, 10000)
    random.seed(rseed)

    f = open(source + '.csv', 'r', encoding='utf-8')
    datas = f.readlines()
    f.close()

    sep1 = rate[0]
    sep2 = rate[0] + rate[1]

    trainF = open(source + '_train.csv', 'w', encoding='utf-8')
    validF = open(source + '_valid.csv', 'w', encoding='utf-8')
    testF = open(source + '_test.csv', 'w', encoding='utf-8')

    import numpy as np
    for line in datas:
        r = np.random.uniform(0, 1)
        if r < sep1:
            trainF.write(line)
        elif sep1 <= r < sep2:
            validF.write(line)
        else:
            testF.write(line)

    trainF.close()
    validF.close()
    testF.close()
    
import torch
def caculate_F(yhat, y0, tagdic):
    '''计算各种指标了
    yhat [B, L]的list
    y0 [L, B]
    '''
    #assert yhat.shape == y0.shape
    
    #yhat = torch.transpose(yhat, 0, 1)
    y0 = torch.transpose(y0, 0, 1)
    #yhat = yhat.tolist()
    y0 = y0.tolist()
    
    batch = len(yhat)
    actu_c = 0  # 实际的实体数量
    actu_len = 0  # 实际的实体总长度
    
    pred_c = 0 # 预测的实体数量（包括不完整的，包括一个B开头的就算一个
    pred_len = 0 # 预测的实体长度
    pred_comp_c = 0 # 预测的完整的实体数量
    
    pred_comp_right_c = 0 # 完全正确的预测的数量（位置，类别完全正确）
    pred_half_right_c = 0 # 部分正确的预测的数量（和实际实体有重叠，并且类别正确
    for sentence in y0:
        sentence = tagdic[sentence]
        for item in sentence:
            if item[0] in ('B', 'I', 'E'):
                actu_len += 1
                if item[0] == 'B':
                    actu_c += 1

    
    curEntity = 'O'
    for _yhat, _y0 in zip(yhat, y0):
        _yhat = tagdic[_yhat]
        _y0 = tagdic[_y0]
        for tag_t, tag_pred in zip(_yhat, _y0):
            if tag_pred[0] == 'B':
                if tag_pred == tag_t:
                    curEntity = tag_pred[2:]
            elif tag_pred[0] == 'I':
                if curEntity != tag_pred[2:] or tag_pred != tag_t: # 检验实体是不是就是现在这个
                    curEntity = 'O'  # 如果不是就清除实体记录
            elif tag_pred[0] == 'E':
                if curEntity == tag_pred[2:] and tag_pred == tag_t: # 检验实体是不是就是现在这个
                    pred_comp_right_c += 1 # 验证完整
                curEntity = 'O'  # 如果不是就清除实体记录
            else:
                curEntity = 'O'
                
    # 计算部分正确的数量，嵌入到了计算长度的里面了
    curEntity = 'O'
    realEntity = []
    for _yhat, _y0 in zip(yhat, y0):
        _yhat = tagdic[_yhat]
        _y0 = tagdic[_y0]
        for tag_t, item in zip(_y0, _yhat):
            if item[0] == 'B':
                pred_len += 1
                pred_c += 1
                if curEntity == 'O':
                    curEntity = item[2:]
                    realEntity = []
            elif item[0] == 'I':
                pred_len += 1
                if curEntity != item[2:]: # 检验实体是不是就是现在这个
                    curEntity = 'O'  # 如果不是就清除实体记录
            elif item[0] == 'E':
                pred_len += 1
                if curEntity == item[2:]: # 检验实体是不是就是现在这个
                    pred_comp_c += 1 # 验证完整
                    if curEntity in realEntity:
                        pred_half_right_c += 1
                        realEntity = []
                curEntity = 'O'  # 如果不是就清除实体记录
            else:
                curEntity = 'O'
            if tag_t != 'O':
                realEntity.append(tag_t[2:])
            
    
    
    return actu_c/batch, actu_len/batch, pred_c/batch, pred_len/batch, pred_comp_c/batch, pred_comp_right_c/batch, pred_half_right_c/batch

