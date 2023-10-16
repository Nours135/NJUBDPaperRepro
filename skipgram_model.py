import torch
#import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import matplotlib.pyplot as plt
import torch.utils.data as tud



from func4skipgram import Vocab, MyBytes2Int
import pickle
import os
import random

import smtplib
from email.mime.text import MIMEText
def send_mail(toaddres, content, title):
    # 发件人邮箱地址
    sendAddress = '1926507843@qq.com'
    # 发件人授权码
    password = 'mjsqkpwdizychica'
    # 连接服务器
    server = smtplib.SMTP_SSL('smtp.qq.com', 465)
    # 登录邮箱
    loginResult = server.login(sendAddress, password)
    print(loginResult)

    # 正文
    # 即content
    # 定义一个可以添加正文的邮件消息对象
    msg = MIMEText(content, 'plain', 'utf-8')

    # 发件人昵称和地址
    msg['From'] = 'myqqmail<1926507843@qq.com>'
    # 收件人昵称和地址
    msg['To'] = f'<{toaddres}>'
    # 抄送人昵称和地址
    #msg['Cc'] = 'xxx<xxx@qq.com>;xxx<xxx@qq.com>'
    # 邮件主题
    msg['Subject'] = title
    server.sendmail(sendAddress, [toaddres], msg.as_string())



class WordEmbeddingDataset(tud.Dataset):
    folder = 'skipgram_corpus\\'
    def __init__(self, source, C, K):
        ''' source 可能的取值是 meaning, radical, pinyin，表示三种不同的素材
        '''
        super(WordEmbeddingDataset, self).__init__()  # 通过父类初始化模型，然后重写两个方法
        self.source = source  # 这个dataset读取的语料是什么
        with open(self.folder + source + '.bin', 'rb') as f:
            self.voca = pickle.load(f)
        self.corpus_f = open(self.folder + self.source + 'test', 'rb') # 后面可能需要更改命名
        self.corpus_size = os.path.getsize(self.folder + source + 'test')
        assert self.corpus_size % 2 == 0
        self.corpus_size = self.corpus_size // 2 
        # 生成负采样的weight tensor
        self.weights = torch.Tensor([item[1] ** 0.75 for item in self.voca._token_freqs])
        
        self.count = 1
        self.STEP = 200000

        self.K = K
        self.C = C

        
    def __len__(self):
        return self.corpus_size - 2*C # 返回所有单词的总数，即item的总数
    
    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''
        K = self.K
        C = self.C
        self.count += 1
        if self.count == self.STEP:
            self.count = 0
            self.corpus_f.seek(random.randint(0, (self.corpus_size - (self.STEP+1) * (C*4+2))//2) * 2, 0) # 真随机取batch，避免频繁seek

        #self.corpus_f.seek(idx*2, 0)
        context_words = self.corpus_f.read(2*C)  #取得前面的语境词
        center_words = MyBytes2Int(self.corpus_f.read(2))  # 取得中心词
        context_words = context_words + self.corpus_f.read(2*C)  #取得后面的语境词
        context_words = MyBytes2Int(context_words)
    
        neg_words = torch.multinomial(self.weights, K * C * 2, True)
        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 每采样一个正确的单词(positive word)，就采样K个错误的单词(negative word)，pos_words.shape[0]是正确单词数量
        
        # while 循环是为了保证 neg_words中不能包含背景词
        # 这个方法粗暴且低效
        '''while len(set(pos_indices.numpy().tolist()) & set(neg_words.numpy().tolist())) > 0:
            neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)'''
        #self.corpus_f.close()
        return torch.IntTensor(center_words), torch.IntTensor(context_words), neg_words  # 这里的intTensor的输入都是列表，还好还好


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
         
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)  # 中心词权重矩阵
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)  # 周围词权重矩阵
        #词向量层参数初始化
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
        
    def forward_input(self, input_labels):
        '''
            input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]
            
            return: loss, [batch_size]
            '''
        input_embedding = self.in_embed(input_labels) # [batch_size, words_count, embed_size]
        return input_embedding
    
    def forward_target(self, pos_labels):
        '''
            input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]
            
            return: loss, [batch_size]
            '''
        pos_embedding = self.out_embed(pos_labels)# [batch_size, (window * 2), embed_size]
        return pos_embedding
    
    def forward_negative(self, neg_labels):
        '''
            input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]
            
            return: loss, [batch_size]
            '''
        neg_embedding = self.out_embed(neg_labels) # [batch_size, (window * 2 * K), embed_size]
        return neg_embedding

    def input_embedding(self):
        return self.in_embed.weight.detach().numpy()
    
    def forward(self, input_labels):
        '''
            input_labels: center words, [batch_size]
            return: predicts, [vocab_size]
            '''
        input_embedding = self.in_embed(input_labels) # [batch_size, embed_size]
        out = torch.matmul(input_embedding, torch.transpose(self.out_embed.weight.detach(), 0, 1))
        s = nn.Softmax(dim=1)  # 在第一个维度求
        return s(out)
        
    
        
class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_embedding, pos_embedding, neg_embedding):
        '''
            input_labels: center words, [batch_size, 1, embed_size]
            pos_labels: positive words, [batch_size, (window * 2), embed_size]
            neg_labels：negative words, [batch_size, (window * 2 * K), embed_size]
            
            return: loss, [batch_size]
            '''
        # squeeze是挤压的意思，所以squeeze方法是删除一个维度，反之，unsqueeze方法是增加一个维度
        
        
        # bmm方法是两个三维张量相乘，两个tensor的维度是，（b * m * n）, (b * n * k) 得到（b * m * k），相当于用矩阵乘法的形式代替了网络中神经元数量的变化
        # 矩阵的相乘相当于向量的点积，代表两个向量之间的相似度
       
        input_embedding = torch.transpose(input_embedding, 1, 2) # 将第一维和第二维换一下，变成 [batch_size, embed_size, 1]
        #print(input_embedding.shape)
        #print(pos_embedding.shape)
        
        pos_dot = torch.matmul(pos_embedding, input_embedding)  # [batch_size, (window * 2), 1]
        #print(pos_dot.shape)
        #input()
        pos_dot = pos_dot.squeeze(2)  # [batch_size, (window * 2)]

        neg_dot = torch.matmul(neg_embedding, -input_embedding)  # [batch_size, (window * 2 * K), 1]，这里之所以用减法是因为下面loss = log_pos + log_neg，log_neg越小越好
        neg_dot = neg_dot.squeeze(2)  # batch_size, (window * 2 * K)]

        log_pos = F.logsigmoid(pos_dot).sum(1)  # .sum()结果只为一个数，.sum(1)结果是一维的张量，在序号1的维度求和
        log_neg = F.logsigmoid(neg_dot).sum(1)  # 这两个loss都是[batch_size]的张量
 
        loss = log_pos + log_neg
        return -loss.mean()  # [1]
    
import datetime

def train(model, LossCriterion, optimizer, dataiter, epochs, source):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #DEVICE = torch.device("cpu")
    print('training on: ', DEVICE)
    print(f'train source: {source}')
    import time
    start = time.time()
    PRINT_EVERY = 5000
    
    
    model.to(DEVICE) # 忘了移动到显卡了
    LossCriterion.to(DEVICE) # woc我知道为啥报错了，因为我的模型实际上有两半，这个也得是在cuda上
    for e in range(epochs[0]-1, epochs[1]):
        #获取输入词以及目标词
        steps = 0
        for input_words, target_words, nega_words in dataiter:
            # 将数据转到显卡上
            input_words, target_words, nega_words = input_words.to(DEVICE), target_words.to(DEVICE), nega_words.to(DEVICE)
            steps += 1
            
            #输入、输出以及负样本向量
            input_vectors = model.forward_input(input_words)
            target_vectors = model.forward_target(target_words)
            nega_vectors = model.forward_negative(nega_words)
            
            
            #计算损失
            loss = LossCriterion(input_vectors, target_vectors, nega_vectors)
            
            #打印损失
            if steps%PRINT_EVERY == 0:
                print(f"<{e+1}, {steps//PRINT_EVERY}>: loss = {loss}", end=' . ')
                print('time use: ',round((time.time() - start)/60), 'min')
                
            #梯度回传
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'epoch {e+1} finished')
        torch.save(model.state_dict(), f'models\\skipgram_{source}_epoch{e+1}_{str(datetime.date.today())}.pth')  # 这里需要修改以下
        if e%2 == 0: # 每两个epoc发一次邮件，将log信息传回
            mailtitle = f'skipgram_{source}_epoch{e+1}_log'
            with open(f'{source}_train.log', 'r', encoding='utf-8') as f:
                content = f.readlines()
                content = ''.join(content)
            try:
                send_mail('211820025@smail.nju.edu.cn', content, mailtitle)
            except Exception as err:
                print(err)



def main():
        # 模型参数
        source = input('输入训练的source: ')
        if source in ("radical", "pinyin"):
            EMBEDDING_SIZE = 25  # 词向量维度
        elif source == 'meaning':
            EMBEDDING_SIZE = 50  # 词向量维度
        else:
            raise ValueError("source输入错误")
        
        epochs = input('输入训练的epochs：')
        epochs = [int(i.strip())for i in epochs.split(',')]

            

        batch_size = 7000  # 每次训练的样本数量
        lr = 0.005
        

        C = 4  # 背景词
        K = 15  # 负采样的噪声词
        K = int(input('选择负采样的噪声词数量，一般情况为15，radical尤其需要减少：'))
        dataset = WordEmbeddingDataset(source, C=C, K=K) # 读取数据集

        dataloader = tud.DataLoader(dataset, batch_size, drop_last=True, num_workers=0, pin_memory=False) #读取数据
        model = EmbeddingModel(vocab_size=len(dataset.voca), embed_size=EMBEDDING_SIZE)  # 创建skip gram model
        LossCri = NegativeSamplingLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        #optimizer = optim.SGD(model.parameters(), lr=lr)

        mod = input('输入载入继续训练的模型：')
        if mod != 'new':
            model.load_state_dict(torch.load(f'models\\{mod}'))
            model.eval()
            print(f'model {mod} loaded successful')

        train(model=model,
                LossCriterion=LossCri,
                optimizer=optimizer,
                dataiter=dataloader,
                epochs=epochs,
                source=source)
        
        # 训练结束，存储模型
        #torch.save(model.state_dict(), f'models\\skipgram_{str(datetime.date.today())}.pth')  # 这里需要修改以下
    
if __name__ == '__main__':
    main()
        
    

     
    # forward方法返回的就是loss，这里不需要再实例化loss了
    '''model = EmbeddingModel(MAX_VOCAB_SIZE, EMBEDDING_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    
    
    
    for e in range(1):
        for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
            input_labels = input_labels.long()
            pos_labels = pos_labels.long()
            neg_labels = neg_labels.long()
     
            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
     
            optimizer.step()
     
            if i % 100 == 0:
                print('epoch', e, 'iteration', i, loss.item())
     
    embedding_weights = model.input_embedding()
    torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_SIZE))'''
    
    

