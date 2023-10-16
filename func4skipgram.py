import collections
class Vocab:  
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            self.reserved_tokens = []
        # 按出现频率排序
        self.counter = count_corpus(tokens)
        self.min_freq = min_freq
        self.generated = False #表示是否结束追加语料，一旦该值为True，Vocab便不可更改了
        
    def generate(self):
        '''以迭代方法输入完所有语料后，调用这个方法生成词库'''
        self._token_freqs = sorted(self.counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + self.reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < self.min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
        self.generated = True
        
        self.lenth = len(self.idx_to_token)
        self._token_freqs = self._token_freqs[:self.lenth] # 只保留未过滤掉的词的词频
        del self.counter

    def append(self, tokens):
        '''追加语料'''
        if not self.generated:
            count_corpus(tokens, self.counter)
        
    def __len__(self):
        return self.lenth

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens, counter=None): 
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
        
    if counter == None: #如果没有输入counter，即为新建，否则调用counter的update方法
        return collections.Counter(tokens)
    counter.update(tokens)
    return counter




def tokenize(lines):
    '''将文本拆分为单个的字'''
    return [token for line in lines for token in line]



from opencc import OpenCC
import re
def wash_chinese(line):
    '''清洗中文语料，
    仅支持两种输入：字符串，或者字符串的列表，输出是清洗后的字符串'''
    '''统一一下输出格式，token的字符串，长度可长可短'''
    if isinstance(line, list) or isinstance(line, tuple): #支持嵌套列表
        #print(1)
        resu = []
        for a in line:
            resu += wash_chinese(a) # 这边result + list(字符串)
        return resu
    
    # 非中文外的非法字符，包括韩语，包括空格与换行，里面选中的就是保留的字符
    #pattern = r'[^0-9a-zA-Z\u4e00-\u9fff,.!?;:()（）-。，！？；：\u0370-\u03ff\u1f00-\u1fff]'
    # 上面的pattarn保留了能区分句子的标点符号
    pattern = r'[^0-9a-zA-Z\u4e00-\u9fff()\-（）~/.·><≥≤%\u0370-\u03ff\u1f00-\u1fff]'
    line = re.sub(pattern, '', line)
    
    # 繁体转为简体
    cc = OpenCC('t2s') 
    line = cc.convert(line)
    return list(line)



import json
import os
def read_wiki():
    '''读取wiki中文的语料库'''
    home_path = 'wiki_zh'
    sub_folder_l = os.listdir(home_path)
    for sub_folder in sub_folder_l:
        file_l = os.listdir(home_path + '/' + sub_folder)
        for json_f in file_l:
            data = [] #  以一个json为单位返回数据
            f = open(home_path + '/' + sub_folder + '/' + json_f, 'r', encoding='utf-8')
            json_l = f.readlines()
            f.close()
            for line in json_l:
                wiki_data = json.loads(line)
                data.append(wiki_data['text'])
                
            yield data
            
def read_dia():
    '''读取糖尿病的语料'''
    home_path = '0521_new_format'
    file_l = os.listdir(home_path)
    
    for json_f in file_l:
        data = [] #  以一个json为单位返回数据
        f = open(home_path + '/'  + json_f, 'r', encoding='utf-8')
        json_data = json.load(f)
        f.close()
        paras = json_data['paragraphs']
        sentences = []
        for para in paras:
            sentences.append(para['paragraph'])
        yield sentences
            
            

def MyInt2Bytes(num_l):
    '''将数字转化成字符序列'''
    b = b''
    for num in num_l:
        b += num.to_bytes(2, 'big', signed=False)
    return b

def MyBytes2Int(b):
    '''输入是一个bytesstring，解码为int'''
    l = len(b)
    assert l%2 == 0
    nums = []
    for i in range(l//2):
        nums.append(int.from_bytes(b[i*2: i*2+2], 'big', signed=False))
    return nums

import pickle
def gen_bytes_data(dataGenerators, Vocabs, outfiles):
    ''''生成一个供skip gram训练模型读取的二进制文件（方便快速读取数据）
    dataGenerators是获取序列数据的方法们，是存储生成器的列表（提供可扩展性啦），在本文件中直接调用read_wiki就好
    vocabs是将文本内容转化为数字所需要的函数
    outfiles是二进制文件的名字
    后两个在本文件中的应用，分别是meaning，radical，pinyin，顺序不能有错'''    
    folder = 'skipgram_corpus\\'
    from cnradical import Radical, RunOption
    radical = Radical(RunOption.Radical)
    pinyin = Radical(RunOption.Pinyin)
    # 生成存储文件的file
    f1 = open(folder + outfiles[0], 'wb')
    f2 = open(folder + outfiles[1], 'wb')
    f3 = open(folder + outfiles[2], 'wb')
    
    for dataGenerator in dataGenerators:
        for pieceodfdata in dataGenerator:
            tokens = wash_chinese(pieceodfdata)  # 输出的是语料的列表
            radical_out = [change_None(radical.trans_ch, ele) for ele in tokens]  # 如何处理None值是个问题，用装饰器
            pinyin_out = [change_None(pinyin.trans_ch, ele) for ele in tokens]    
            # 调用词典转化为数字
            tokensNum = Vocabs[0][tokens]
            radicalNum = Vocabs[1][radical_out]
            pinyinNum = Vocabs[2][pinyin_out]
            f1.write(MyInt2Bytes(tokensNum))
            f2.write(MyInt2Bytes(radicalNum))
            f3.write(MyInt2Bytes(pinyinNum))
    
    f1.close()
    f2.close()
    f3.close()
    # 存储词典的二进制文件
    with open(folder + "meaning.bin", "wb") as file0:
        pickle.dump(Vocabs[0], file0)
    with open(folder + "radical.bin", "wb") as file1:
        pickle.dump(Vocabs[1], file1)
    with open(folder + "pinyin.bin", "wb") as file2:
        pickle.dump(Vocabs[2], file2)

            
def change_None(func, ele):
    # 装饰cnradical库的两个代码，讲none转化为<unk>，避免后面的幺蛾子
    # 又因为列表循环会调用两遍func，所以用这个装饰器实现它
    # 实验证明会增加20%的开销，但比翻倍强一些
    res = func(ele)
    if res is None:
        return '<unk>'
    else:
        return res

def main():
    import time 
    
    start = time.time()
    
    test_voca = Vocab(min_freq=10)
    radical_voca = Vocab(min_freq=10)
    pinyin_voca = Vocab(min_freq=10)
        
    # 引入生成拼音和偏旁的函数
    from cnradical import Radical, RunOption
    radical = Radical(RunOption.Radical)
    pinyin = Radical(RunOption.Pinyin)
    

    
    t = 0
    for text in read_wiki():
        tokens = wash_chinese(text)  # 输出的是语料的列表

        radical_out = [change_None(radical.trans_ch, ele) for ele in tokens]  # 如何处理None值是个问题，用装饰器
        pinyin_out = [change_None(pinyin.trans_ch, ele) for ele in tokens]      
        test_voca.append(tokens)
        radical_voca.append(radical_out)
        pinyin_voca.append(pinyin_out)
        t += 1
        print(t, end=' ')
        
    for i in range(10):
        for text in read_dia():
            tokens = wash_chinese(text)  # 输出的是语料的列表

            radical_out = [change_None(radical.trans_ch, ele) for ele in tokens]  # 如何处理None值是个问题，用装饰器
            pinyin_out = [change_None(pinyin.trans_ch, ele) for ele in tokens]      
            test_voca.append(tokens)
            radical_voca.append(radical_out)
            pinyin_voca.append(pinyin_out)
            t += 1
            print(t, end=' ')
        
    test_voca.generate()
    radical_voca.generate()
    pinyin_voca.generate()

    print('word: ', len(test_voca))
    print('radical: ', len(radical_voca))
    print('pinyin: ', len(pinyin_voca))
    

    
    print(f'总耗时{time.time()-start}s')
    
    gen_bytes_data([read_wiki()],
                   [test_voca, radical_voca, pinyin_voca],
                   ['meaningtest', 'radicaltest', 'pinyintest'])
    

    # 取window
    # 
    print(f'总耗时{time.time()-start}s')
    
if __name__ == "__main__":
    main()
    
    
    