import os
import json

from opencc import OpenCC
import re


def read_dia():
    '''读取糖尿病的语料'''
    home_path = r'diabets'
    file_l = os.listdir(home_path)
    
    for json_f in file_l:
        f = open(home_path + '/' + json_f, 'r', encoding='utf-8')
        json_data = json.load(f)
        f.close()
        paras = json_data['paragraphs']
        sentences = []
        for para in paras:
            sentences += para['sentences']
        for item in sentences:  # 来
            marker = entity_tagging(item['sentence'], item['entities'])
            yield item['sentence'], marker



def wash_diabet(line, marker):  # 需要重构一下了
    '''清洗中文语料，
    仅支持两种输入：字符串，或者字符串的列表，输出是清洗后的字符串'''
    '''统一一下输出格式，token的字符串，长度可长可短'''
    if isinstance(line, list) or isinstance(line, tuple): #支持嵌套列表
        raise Exception('仅支持单个句子输入')
        
    try:
        assert len(line) == len(marker)
    except AssertionError:
        print(line)
        print(marker)
    
    # 非中文外的非法字符，包括韩语，包括空格与换行，里面选中的就是保留的字符
    #pattern = r'[^0-9a-zA-Z\u4e00-\u9fff,.!?;:()（）-。，！？；：\u0370-\u03ff\u1f00-\u1fff]'
    # 上面的pattarn保留了能区分句子的标点符号
    pattern = r'[^0-9a-zA-Z\u4e00-\u9fff()\-（）~/.·><≥≤%\u0370-\u03ff\u1f00-\u1fff]'
    line = re.sub(pattern, '★', line)  # 用一个特殊字符填充这个东西
    
    idx = 1
    while idx != -1:
        idx = line.find('★')
        if idx != -1:
            line = line[:idx] + line[idx+1:]
            marker = marker[:idx] + marker[idx+1:]
    
    assert len(line) == len(marker)
    # 繁体转为简体
    cc = OpenCC('t2s') 
    line = cc.convert(line)
    return line, marker


# 1. 文本分割
def segment_text(sentence, marker):
    '''输入是一个句子的字符串，
    和一个marker的列表
    
    递归切分句子，返回的是切分后的列表
    '''
    assert len(sentence) == len(marker)
    
    seg_s = '！？。…；：!?，'  #加不加逗号，意味着分句是否足够细
    flag = False
    for item in seg_s:
        idx = sentence.find(item)
        if idx != -1:
            flag = True  # 只找一个
            break
    if flag: # 如果找到了
        return segment_text(sentence[:idx], marker[:idx]) + segment_text(sentence[idx+1:], marker[idx+1:])
    else: # 递归调用，返回的就是
        if len(sentence) != 0:
            return [(sentence, marker)]
        else:
            return []




def entity_tagging(sentence, entity_list):
    '''
    sentence是一个句子
    entity_list是json文件中与句子对应的实体列表
    返回 entity_markers，是一个marker的列表
    '''
    # 初始化标记列表，全部为'O'
    entity_markers = ['O'] * len(sentence)
    
    if len(entity_list) == 0:  # 如果没有实体
        return entity_markers

    # 找到sentence中所有entity出现的位置及其长度
    entity_positions = []
    for entity_dict in entity_list:
        entity_name = entity_dict['entity']
        enetity_type = entity_dict['entity_type']
        start_idx = entity_dict['start_idx']
        end_idx = entity_dict['end_idx']
        entity_positions.append((start_idx, end_idx, end_idx-start_idx, entity_name, enetity_type))
        

    # 标记包含关系中较长的实体
    for i, (start1, end1, length1, entityname1, entitytype1) in enumerate(entity_positions):
        for j, (start2, end2, length2, entityname2, entitytype2) in enumerate(entity_positions):
            if i != j and start1 <= start2 and end1 >= end2:
                entity_positions[j] = (-1, -1, -1, entityname2, entitytype2)
            if i != j and start1 < start2 and end1 > start2 and end2 > end1:
                print(start1, start2 , end1)
                print(entity_positions[i])
                print(entity_positions[j])
                raise Exception('实体存在交叉') # 实际上不存在
                
    # 处理实体位置
    for start, end, length, entityname, entitytype in entity_positions:
        if start == -1:
            #print('存在一个相互重合')
            continue
        # 标记entity的开始和结束
        entity_markers[start] = 'B-' + entitytype
        entity_markers[end - 1] = 'E-' + entitytype

        # 标记entity中间部分为'i'
        for i in range(start + 1, end - 1):
            entity_markers[i] = 'I-' + entitytype
            
    return entity_markers



'''
# 将结果写入csv文件
def save_entities_to_csv(enti, csv_filename):
    with open(csv_filename, mode='w', encoding='utf-8-sig', newline='') as csv_file:  # 使用 'utf-8-sig' 编码
        fieldnames = ['sentence','entity', 'result']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # 写入 CSV 文件的标题行
        writer.writeheader()

        # 遍历 enti 中的实体信息并写入 CSV 文件
        for entities in enti:
            for entity in entities:
                sentence, entity_text, result = entity
                writer.writerow({"sentence":sentence, 'entity': entity_text, 'result':result})'''


if __name__ == "__main__":
    
    # 清洗流程：
    # 读取entity 和 sentence的列表，标记sentence
    # 然后切分句子
    # 然后 wash_diabet
    
    
    import collections
    marker_counter = collections.Counter() # marker的统计
    sentence_len_counter = collections.Counter() # 句子长度的统计
    
    diabetes_f = open('diabetes.csv', 'w', encoding='utf-8')
    diabetes_washed_f = open('diabetes_washed.csv', 'w', encoding='utf-8')
    
    for sentence, marker in read_dia():
        
        segged = segment_text(sentence, marker)
        for senten, mark in segged:
            
            diabetes_f.write(senten + '\t' + ';'.join(mark) + '\n')
            s, m = wash_diabet(senten, mark) # wash_diabet
            diabetes_washed_f.write(s + '\t' + ';'.join(m) + '\n')
            
            sentence_len_counter.update([len(s)])
            marker_counter.update(m)
        
        
    for key, v in marker_counter.items():
        print(key, v)
    
    #for key, v in sentence_len_counter.items():
     #   print(key, v)
            
            
    diabetes_washed_f.close()
    diabetes_washed_f.close()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            