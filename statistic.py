from collections import Counter

from config import maxlen
from path import *

with open(val_file_path, 'r', encoding = 'utf-8') as f:
    lines = f.readlines()
    count = 0
    sentense_len = []
    for i in lines:
        count += 1
        if i.strip('\n') == '':
            sentense_len.append(count)
            count = 0
    print("句数:", len(sentense_len))
    print("最长单句样本长度:", max(sentense_len))
    freq = dict(Counter(sentense_len))
    count_large = 0
    for length in sentense_len:
        if length > maxlen:
            count_large += 1
    print("大于{}数量:".format(maxlen), count_large)
    print("被截断比例:", count_large / len(sentense_len))
    print("\n句子长度数量统计(按长度):(句长, 数量)")
    for index, data_dict in enumerate(sorted(freq.items(), key = lambda d: d[0], reverse = True)):
        print(str(index + 1) + ':', data_dict, '\t', end = "")
        if (index + 1) % 10 == 0:
            print()
    print('\n\n句子长度数量统计(按数量):(句长, 数量)')
    for index, data_dict in enumerate(sorted(freq.items(), key = lambda d: d[1], reverse = True)):
        print(str(index + 1) + ':', data_dict, '\t', end = "")
        if (index + 1) % 10 == 0:
            print()
