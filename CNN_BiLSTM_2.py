import torch
#import numpy as np
import torch.nn as nn
import torch.optim as optim
#import torch.nn.functional as F
#import matplotlib.pyplot as plt
import torch.utils.data as tud

import pickle
#import os
import random

from func4skipgram import Vocab, MyBytes2Int, change_None
from skipgram_model import EmbeddingModel

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


class DataReader(tud.Dataset):
    def __init__(self, mode, DataTranser):
        super().__init__()
        self.data_csv = 'NERSource/diabetes_washed'
        f = open(self.data_csv + "_" + mode + '.csv', 'r', encoding='utf-8')
        data_s = f.readlines()
        f.close()
        self.datas = []
        for line in data_s:
            line = line.strip().split('\t')
            if len(line) == 2:
                self.datas.append(line)
        self.total = len(self.datas)
        self.DataTranser = DataTranser
        
    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # get item方法，给dataloader调用
        line = self.datas[idx]
        return list(line[0]), line[1].split(';')  # return x, y

    
    def dataiter(self, batch_size):
        for i in range(0, len(self), batch_size):
            X = []
            y = []
            for j in range(batch_size):
                if i + j < len(self):
                    _x, _y = self[i+j]
                    X.append(_x)
                    y.append(_y)
            X, y, masks = self.DataTranser.forward(X, y)
            yield X, y, masks


class DataTransformer():
    folder = 'NERSource/'
    # unk作为padding不具有可行性，因为在拼音处有很多unk，难蚌所以作为替代，我选择
    def __init__(self, tagdic):
        '''将数据读入并转化为矩阵'''
        # 读取词典 和 embedding
        self.vocabs = []
        self.embedding = []
        for source in ['meaning', 'radical', 'pinyin']:
            with open(self.folder + source + '.bin', 'rb') as f:
                self.vocabs.append(pickle.load(f))
            if source == 'meaning': embedDim = 50
            else: embedDim = 25
            model = EmbeddingModel(vocab_size=len(self.vocabs[-1]), embed_size=embedDim)  # 创建模型
            model.load_state_dict(torch.load(self.folder + f'skipgram_{source}.pth'))  # 加载模型
            model.eval()
            self.embedding.append(model.in_embed)  # 改为只是把embed层拿出来

        # 生成radical和pinyin的Transfer
        from cnradical import Radical, RunOption
        self.radicalTransfer = Radical(RunOption.Radical)
        self.pinyinTransfer = Radical(RunOption.Pinyin)


        # tag_dic 
        self.tagdic = tagdic
        
    def forward(self, X, y):
        '''X是文字，清洗过的语料，并且未被padding过'''
        '''X是一个二维的文字列表，一行为一句话而列为batch，validLen是一个int列表'''
        if not y is None:
            for i, item in enumerate(X):  # 检验输入的数据和tag是否长度一致
                try:
                    assert len(item) == len(y[i])
                except AssertionError:
                    print(item)
                    print(y[i])

        # 计算padding
        batch = len(X)
        validLen = [len(line) for line in X]
        max_len = max(validLen) + 2 # 以最大的长度为基准，至少padding 2个字符
        paddings = [max_len - i for i in validLen]
        masks = torch.ones((batch, max_len), dtype=torch.float32) # [B, L]
        
        # 转化得到偏旁和拼音
        radical_X = [[change_None(self.radicalTransfer.trans_ch, ele) for ele in line] for line in X]  # size [batchSize, sentenceLen]
        pinyin_X = [[change_None(self.pinyinTransfer.trans_ch, ele) for ele in line] for line in X]


        # 导入词典得到词的ID
        X = [self.vocabs[0][line] for line in X]
        radical_X = [self.vocabs[1][line] for line in radical_X]
        pinyin_X = [self.vocabs[2][line] for line in pinyin_X]
    
        sentences = []
        yout = []
        for b in range(batch):
            # 转成tensor后传入embedding层
            _1 = self.embedding[0](torch.IntTensor(X[b]))           # shape [len(sentence), 50]
            _2 = self.embedding[1](torch.IntTensor(radical_X[b]))   # shape [len(sentence), 25]
            _3 = self.embedding[2](torch.IntTensor(pinyin_X[b]))    # shape [len(sentence), 25]
            sentence = torch.cat((_1, _2, _3), dim=1)               # 将词在意义偏旁拼音三个部分的特征向量concat起来 shape [len(sentence), 100]
            # Padding 
            sentence = torch.cat((sentence, torch.zeros(size=(paddings[b], 100))), dim=0) # shape [seriesLen, 100]
            sentence = sentence.unsqueeze(dim=1) # 在1维添加，方便叠加batch # shape [seriesLen, 1, 100]
            sentences.append(sentence)
            if not y is None:
                tags = y[b] + ['<eos>'] + ['<pad>' for i in range(paddings[b] - 1)]
                tags = self.tagdic.tag2idx(tags)  # change tag to index
                yout.append(tags)
            
            masks[b][max_len - paddings[b]:] = 0.
        
        
            
        Xout = torch.cat(sentences, dim=1) # 在batch维度上叠加 # shape [seriesLen, batchSize, 100]
        if not y is None:
            yout = torch.IntTensor(yout)  # [B, L]
            yout = torch.transpose(yout, 0, 1)  # [L, B]
        return Xout, yout, torch.transpose(masks, 0, 1)
# -------------------------------------------~------------------------------------------------------------------------------------


PAD, PAD_IDX = "<pad>", 0 # padding
SOS, SOS_IDX = "<sos>", 1 # start of sequence
EOS, EOS_IDX = "<eos>", 2 # end of sequence
UNK, UNK_IDX = "<unk>", 3 # unknown token

class CRF(nn.Module):
    def __init__(self, num_tags):
        self.debug_log = [[], []]
        super().__init__()
        self.num_tags = num_tags

        # transition scores from j to i
        self.trans = nn.Parameter(torch.randn(num_tags, num_tags))
        nn.init.normal_(self.trans.data, mean=0, std=0.01)  # 初始化
        self.trans.data[SOS_IDX, :] = -10000 # no transition to SOS
        self.trans.data[:, EOS_IDX] = -10000 # no transition from EOS except to PAD
        self.trans.data[:, PAD_IDX] = -10000 # no transition from PAD except to PAD
        self.trans.data[PAD_IDX, :] = -10000 # no transition to PAD except from EOS
        self.trans.data[PAD_IDX, EOS_IDX] = 0
        self.trans.data[PAD_IDX, PAD_IDX] = 0

    def score(self, h, y0, mask):
        # h [L, B, C]
        # y0 shape [L, B]
        # mask shape [L, B]
        S = torch.Tensor(h.size(1)).fill_(0.)  # shape [B]
        h = h.unsqueeze(3) # [L, B, C, 1]
        trans = self.trans.unsqueeze(2) # [C, C, 1]

        for t, (_h, _mask) in enumerate(zip(h[:-1], mask[:-1])):
            _emit = torch.cat([_h[_y0] for _h, _y0 in zip(_h, y0[t + 1])])  # shape [Batch]
            _trans = torch.cat([trans[x] for x in zip(y0[t + 1], y0[t])])  # shape [B]
            # _mask shape [B]
            S += (_emit + _trans) * _mask
        
        # maybe noneed
        last_tag = y0.gather(0, mask.sum(0).long().unsqueeze(0)).squeeze(0) # gather: 在0维度上，根据给定的坐标选择值
        #print(last_tag) # 都是 <eos> 所有取到的是，句子最后一个字符后面的那个那个tag
        #print(last_tag.shape) # shape [batch]
        #input()
        S += self.trans[PAD_IDX, last_tag] # 原来的代码 S += self.trans[EOS_IDX, last_tag]
        # print(S.shape) [B]
        return S

    def partition(self, h, mask):

        Z = torch.Tensor(h.size(1), self.num_tags).fill_(-10000)  # shape [B, C]
        Z[:, SOS_IDX] = 0.
        trans = self.trans.unsqueeze(0) # [1, C, C]

        for _h, _mask in zip(h, mask): # forward algorithm
            _mask = _mask.unsqueeze(1)
            _emit = _h.unsqueeze(2) # [B, C, 1]
            _Z = Z.unsqueeze(1) + _emit + trans # [B, 1, C] -> [B, C, C]
            _Z = torch.logsumexp(_Z, dim=2)  # 替代下面那行
            #_Z = log_sum_exp(_Z) # [B, C, C] -> [B, C]
            Z = _Z * _mask + Z * (1 - _mask)
        
        #    Z + self.trans[EOS_IDX] # shape [B, C]
        Z = torch.logsumexp(Z + self.trans[EOS_IDX], dim=1)
        #Z = log_sum_exp(Z + self.trans[EOS_IDX])

        return Z # partition function

    def forward(self, h, y0, mask): # for training

        S = self.score(h, y0, mask)
        #print(S) #[ -6.5933, -14.9031,  -3.2972,  -4.2958,  -0.7219]
        self.debug_log[0].append(S.mean().item())
        Z = self.partition(h, mask)  # 为了训练状态转移矩阵
        #print(Z)  #[165.5909, 129.8591, 133.0644, 123.8566, 213.5855]
        self.debug_log[1].append(Z.mean().item())
        loss = torch.mean(Z - S) # NLL loss

        return loss

    def decode(self, h, mask): # Viterbi decoding
        # h [t, batch, ]
        # 
        bptr = torch.LongTensor()
        score = torch.Tensor(h.size(1), self.num_tags).fill_(-10000)  # [batchSize, tags]
        score[:, SOS_IDX] = 0.
        for _h, _mask in zip(h, mask):  # iter per t, iter mask 
            _mask = _mask.unsqueeze(1)
            _score = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
            _score, _bptr = _score.max(2) # best previous scores and tags
            _score += _h # add emission scores
            bptr = torch.cat((bptr, _bptr.unsqueeze(1)), 1)
            score = _score * _mask + score * (1 - _mask)
        score += self.trans[EOS_IDX]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(h.size(1)):
            i = best_tag[b]
            j = mask[:, b].sum().int()
            for _bptr in reversed(bptr[b][:j]):
                i = _bptr[i]
                best_path[b].append(i)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path
    
    
class CNN_BiLSTM_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagdic):
        super(CNN_BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # tag转化的函数和方法
        self.tagset_size = len(tagdic)

        # 数据转化
        #self.dataTransformer = DataTransformer(tagdic)
        # CNN
        self.cnns = [nn.Conv2d(1, 34, (3, 100), padding=(3//2, 0), stride=1, bias=True),
                     nn.Conv2d(1, 33, (5, 100), padding=(5//2, 0), stride=1, bias=True),
                     nn.Conv2d(1, 33, (7, 100), padding=(7//2, 0), stride=1, bias=True)]
        for net in self.cnns:
            nn.init.normal_(net.weight, mean=0, std=0.01)
            nn.init.zeros_(net.bias)
            
        # RELU
        self.relu = nn.ReLU()
        # LSTM
        self.lstm = nn.LSTM(  # 定义LSTM层
            input_size=embedding_dim,
            hidden_size=hidden_dim,  # 两个方向
            num_layers=1,
            bias=True,
            batch_first=False,
            bidirectional=True
        )
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0.01)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)
        nn.init.xavier_uniform_(self.hidden2tag.weight)
        nn.init.zeros_(self.hidden2tag.bias)
        
        # crf layer
        self.crf = CRF(self.tagset_size)
        
        #self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(p=0.5)

    #def init_hidden(self):
    #    return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2)) # hidden dim 包含了两个方向
        #return torch.randn()
        
    def _get_lstm_features(self, embeds):
        ''' embeds [seriesLen, batchSize, 100]
        out  [seriesLen, batchSize, tagSize]
        '''
        embeds = self.dropout(embeds)  # 在输入前dropout
        ## 传入cnn层
        CNN_out = []
        a = torch.transpose(embeds, 0 , 1).unsqueeze(1)  # trans [batchSize, seriesLen, 100] unsqueeze [batchSize, 1, seriesLen, 100]
        #print(a.shape)
        for layer in self.cnns:
            CNN_out.append(self.relu(layer(a)))
        
        CNN_out = torch.cat(CNN_out, dim=1) ## [batchSize, 100, seriesLen, 1]
        #print(CNN_out.shape)
        CNN_out = torch.transpose(CNN_out.squeeze(3), 1, 2) ## [batchSize, seriesLen, 100]
        CNN_out = torch.transpose(CNN_out, 0, 1)  ## [seriesLen, batchSize, 100]
        CNN_out = self.relu(CNN_out)  # 激活函数
        #print(CNN_out.shape)
        
        # BiLSTM
        #self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(torch.cat((embeds, CNN_out), dim=2))  ## after cat [batchSize, seriesLen, 200]
        #print(lstm_out.shape)  # [seriesLen, batchSize, hiddensize * 2]
        #print(len(self.hidden)) # 2 
        #print(self.hidden[0].shape)  # [2, batchSize, hiddensize]  不知道到底是啥
        
        #lstm_out = torch.transpose(lstm_out, )
        lstm_out = self.dropout(lstm_out)  # 第二个dropout层
        # 下一步完成了放射特征
        lstm_feats = self.hidden2tag(lstm_out)  # [seriesLen, batchSize, tagSize]
        
        
        return self.relu(lstm_feats)  # 激活函数


    def neg_log_likelihood(self, embeds, tags, masks):
        # CNN-BiLSTM部分的函数
        feats = self._get_lstm_features(embeds)
        
        # 传入crf层
        loss = self.crf(feats, tags, masks)
        return loss

    def forward(self, embeds):  # dont confuse this with _forward_alg above.
    
        embeds, _, masks = self.dataTransformer.forward(embeds, None) #处理raw sentence 和 tags 包括padding的参数
        
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(embeds)

        # Find the best path, given the features.
        best_path = self.crf.decode(lstm_feats, masks)
        return best_path


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
        self.c = c
    def add(self, *args):
        assert len(args) == self.c
        for i in range(self.c):
            self.data[i] += args[i]
    def clear(self):
        self.data = [0] * self.c
    def __getitem__(self, idx):
        return self.data[idx]



if __name__ == '__main__':
    batch_size = 20
    lr = 0.02
    epoch = 10
    
    PRINT_EVERY = 40

    tagdic = TagDic()
    DataTranser = DataTransformer(tagdic)


    TrainSet = DataReader('train', DataTranser) # 读取数据集
    def collate_fn(batch):
        # 给dataloader用的，我会在下面填充padding
        X = [item[0] for item in batch]
        y = [item[1] for item in batch]
        X, y, masks = DataTranser.forward(X, y)
        return X, y, masks
    TrainDataloader = tud.DataLoader(TrainSet, batch_size, shuffle=True, drop_last=True, num_workers=0, pin_memory=False, collate_fn=collate_fn) #读取数据
    ValidSet = DataReader('valid', DataTranser) # 读取数据集

    # 创建模型
    def init_para(model):
        pass 
    model = CNN_BiLSTM_CRF(embedding_dim=200, hidden_dim=100, tagdic=tagdic)  
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    #optimizer = optim.SGD(model.parameters(), lr=lr)

    # train
    import time
    start = time.time()
    metrics = Accumulator(4)  # 暂时只存train 和 valid loss，后面还要存很多指标
    for e in range(epoch):
        steps = 0
        model.train()
        for ebeds, tagIDs, masks in TrainDataloader:
            # 将数据转到显卡上
            #input_words, target_words, nega_words = input_words.to(DEVICE), target_words.to(DEVICE), nega_words.to(DEVICE)
            steps += 1
            
            #计算损失
            loss = model.neg_log_likelihood(ebeds, tagIDs, masks)
            
            metrics.add(loss.item(), 1, 0, 0) # 累积损失
            #打印损失
            if steps%PRINT_EVERY == 0:
                print(f"<{e+1}, {steps//PRINT_EVERY}>: loss = {metrics[0]/metrics[1]:.4f}", end='. ')
                print('time use: ',round((time.time() - start)/60), 'min')
            
            #梯度回传
            optimizer.zero_grad()
            loss.backward()
            #print(loss)
            #print(1)
            optimizer.step()  
            #print(2)


        print(f'epoch {e+1} finished')
        #torch.save(model.state_dict(), f'models/skipgram_{source}_epoch{e+1}_{str(datetime.date.today())}.pth')  # 这里需要修改以下
        '''if e%2 == 0: # 每两个epoc发一次邮件，将log信息传回
            mailtitle = f'skipgram_{source}_epoch{e+1}_log'
            with open(f'{source}_train.log', 'r', encoding='utf-8') as f:
                content = f.readlines()
                content = ''.join(content)
            try:
                send_mail('211820025@smail.nju.edu.cn', content, mailtitle)
            except Exception as err:
                print(err)'''
        

        # 验证集
        model.eval()
        for ebeds, tagIDs, masks in ValidSet.dataiter(5):
            l = model.neg_log_likelihood(ebeds, tagIDs, masks)
            metrics.add(0, 0, l, 1)
        

        print(f'train loss={metrics[0]/metrics[1]:.3f}, valid loss={metrics[2]/metrics[3]:.3f}')
        debug_log = model.crf.debug_log
        trans = model.crf.trans






        


    
        

