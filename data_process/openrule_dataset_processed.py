import re
import os
import random
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_file_name, data_dir='.data/'):
        super().__init__()
        self.data_list = []
        rule = set()
        with open(data_file_name) as f:
            for line in f.readlines():
                data = []
                line = line.strip('\n').split('\t')
                root_re = line[0]
                rule.add(root_re)
                A_type, B_type = type_process(line[1])
                rule_chain = hyrela_process(line[2])
                data.append(root_re)
                data.append(rule_chain) #root_re + select_re(从规则链中随机按序抽取的生成规则)
                data.append(A_type)
                data.append(B_type)
                self.data_list.append(data)
        print(len(rule))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        return self.data_list[item]


def replace_fun(s):
    s = s.replace('<', '')
    s = s.replace('>', '')
    return s

def type_process(s):
    rel1_result = re.findall(r'\'(.*?)\'', s)
    A_type = rel1_result[0]
    B_type = rel1_result[1]
    return A_type,B_type

def hyrela_process(s):
    re = s.strip('[').strip(']').split(',')

    # re_result = []
    # for word in re:
    #     if word != '' and 'A' in word and 'B' in word:
    #         word = word.replace('<', '')
    #         word = word.replace('>', '')
    #         word = word.replace('\'', '')
    #         re_result.append(word)
    return re

#Ouput:type是string
def select_hy(hy_re):
    num = random.randint(-1,len(hy_re)-1)
    if num == -1:
        return ''
    else:
        res = random.sample(hy_re,num)
        sorted_res = [x for x in hy_re if x in res]
        res = ','.join(sorted_res)
        return res

def get_data_loader(data_file_name):
    dataset = MyDataset(data_file_name)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    return data_loader

LOADER = get_data_loader('C:\\Users\\86183\\Desktop\\reasoning_dataset\\csv_data\\openrule.txt')
# #txt[0]:root_re  txt[1]:rule_chain  txt[2]:A_type  txt[3]:B_type
# for idx, txt in enumerate(LOADER):
#
#     print(idx,txt[0][0],txt[1][0][0],txt[2][0],txt[3][0])   #type(txt):turple
#     # print(type(txt))
#     break