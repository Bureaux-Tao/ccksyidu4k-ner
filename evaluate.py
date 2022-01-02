#! -*- coding: utf-8 -*-
import os

# albert tiny
import pickle

import pandas as pd
from matplotlib import pyplot as plt

from model import ALBERT
from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, train_file_path, test_file_path, val_file_path, \
    weights_path, label_dict_path, categories_f1_path
from preprocess import load_data, NamedEntityRecognizer
from utils.plot import f1_plot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.backend import keras, K
from utils.tokenizers import Tokenizer
from tqdm import tqdm

# save_file_path = "./weights/yidu_albert_tiny_lstm_crf.h5"

# bert配置
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)


def get_score(data, NER, tqdm_verbose = False):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    if tqdm_verbose:
        loop = tqdm(data, ncols = 100)
        for d in loop:
            loop.set_description("Evaluating General F1")
            R = set(NER.recognize(d[0]))
            T = set([tuple(i) for i in d[1:]])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
    
    else:
        for d in data:
            R = set(NER.recognize(d[0]))
            T = set([tuple(i) for i in d[1:]])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


def get_catetories_score(data, NER, categories, tqdm_verbose = False):
    """评测函数
    """
    labeded_set = {}
    for i in categories:
        labeded_set[i] = {'TP': 1e-10, 'TP+FP': 1e-10, 'TP+FN': 1e-10}
    if tqdm_verbose:
        loop = tqdm(data, ncols = 100)
        for d in loop:
            loop.set_description("Evaluating F1 of each Categories")
            for i in categories:
                R = set(NER.recognize(d[0]))
                R_labeled = set()
                for s, r, label in R:
                    if label == i:
                        R_labeled.add((s, r, label))
                T = set([tuple(i) for i in d[1:]])
                T_labeled = set()
                for s, r, label in T:
                    if label == i:
                        T_labeled.add((s, r, label))
                
                labeded_set[i]["TP"] += len(R_labeled & T_labeled)
                labeded_set[i]["TP+FP"] += len(R_labeled)
                labeded_set[i]["TP+FN"] += len(T_labeled)
    # print(labeded_set)
    for i in labeded_set:
        labeded_set[i]["precision"] = round(labeded_set[i]["TP"] / labeded_set[i]["TP+FP"], 4)
        labeded_set[i]["recall"] = round(labeded_set[i]["TP"] / labeded_set[i]["TP+FN"], 4)
        labeded_set[i]["f1"] = round(2 * labeded_set[i]["TP"] / (labeded_set[i]["TP+FP"] + labeded_set[i]["TP+FN"]), 4)
        labeded_set[i]["TP"] = int(labeded_set[i]["TP"])
        labeded_set[i]["TP+FP"] = int(labeded_set[i]["TP+FP"])
        labeded_set[i]["TP+FN"] = int(labeded_set[i]["TP+FN"])
        # f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return labeded_set


def evaluate(title, data, CRF, NER):
    trans = K.eval(CRF.trans)
    NER.trans = trans
    f1, precision, recall = get_score(data, NER, tqdm_verbose = True)
    print(title + ':  f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall))
    return f1, precision, recall


def evaluate_categories(title, data, categories, CRF, NER):
    trans = K.eval(CRF.trans)
    NER.trans = trans
    result = get_catetories_score(data, NER, categories, tqdm_verbose = True)
    # for i in result:
    #     print(i, result[i])
    df = pd.DataFrame(result)
    df = df.T
    df[["TP", "TP+FP", "TP+FN"]] = df[["TP", "TP+FP", "TP+FN"]].astype(int)
    # 设置value的显示长度为200，默认为50
    pd.set_option('max_colwidth', 200)
    # 显示所有列，把行显示设置成最大
    pd.set_option('display.max_columns', None)
    # 显示所有行，把列显示设置成最大
    pd.set_option('display.max_rows', None)
    print(df)
    return df


def list_all_files(rootdir):
    _files = []
    # 列出文件夹下所有的目录与文件
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        # 构造路径
        path = os.path.join(rootdir, list[i])
        # 判断路径是否为文件目录或者文件
        # 如果是目录则继续递归
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files


def evaluate_all(dir):
    with open(label_dict_path, 'rb') as f:  # 打开文件
        categories = set(pickle.load(f))
    # 标注数据
    test_data = load_data(test_file_path, categories)
    val_data = load_data(val_file_path, categories)
    
    categories = list(sorted(categories))
    
    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case = True)
    
    albert = ALBERT(config_path,
                    checkpoint_path,
                    categories)
    model = albert.get_model()
    
    model_list = list_all_files(dir)
    
    model_name_list = []
    val_precision_list = []
    val_recall_list = []
    val_f1_list = []
    test_precision_list = []
    test_recall_list = []
    test_f1_list = []
    
    for model_path in model_list:
        model_name_list.append(model_path.split('/')[-1])
        save_file_path = model_path
        
        model.load_weights(save_file_path)
        CRF = albert.get_CRF()
        NER = NamedEntityRecognizer(tokenizer, model, categories, trans = K.eval(CRF.trans), starts = [0], ends = [0])
        
        print("\n" + save_file_path + ":")
        # evaluate("train", train_data, CRF, NER)
        val_f1, val_precision, val_recall = evaluate("validate", val_data, CRF, NER)
        test_f1, test_precision, test_recall = evaluate("test", test_data, CRF, NER)
        
        val_precision_list.append(val_precision)
        test_precision_list.append(test_precision)
        val_recall_list.append(val_recall)
        test_recall_list.append(test_recall)
        val_f1_list.append(val_f1)
        test_f1_list.append(test_f1)
    
    data = {
        'epoch': range(1, len(model_name_list) + 1),
        'path': model_name_list,
        'val_precision': val_precision_list,
        'val_recall': val_recall_list,
        'val_f1': val_f1_list,
        'test_precision': test_precision_list,
        'test_recall': test_recall_list,
        'test_f1': test_f1_list
    }
    
    f1_plot(data)


def evaluate_one(save_file_path, dataset_path, csv_path = categories_f1_path, evaluate_categories_f1 = False):
    with open(label_dict_path, 'rb') as f:  # 打开文件
        categories = set(pickle.load(f))
    
    albert = ALBERT(config_path,
                    checkpoint_path,
                    categories,
                    summary = False)
    model = albert.get_model()
    
    # 标注数据
    test_data = load_data(dataset_path, categories)
    categories = list(sorted(categories))
    
    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case = True)
    
    model.load_weights(save_file_path)
    CRF = albert.get_CRF()
    NER = NamedEntityRecognizer(tokenizer, model, categories, trans = K.eval(CRF.trans), starts = [0], ends = [0])
    
    print("\nweight path:" + save_file_path)
    print("evaluate dataset path:" + dataset_path)
    f1, precision, recall = evaluate("General", test_data, CRF, NER)
    if evaluate_categories_f1:
        df = evaluate_categories("Each Categories:", test_data, categories, CRF, NER)
        df.to_csv(csv_path, encoding = 'utf-8-sig')
    return f1, precision, recall


if __name__ == '__main__':
    # evaluate_all(weights_path)
    
    # evaluate_one(weights_path + '/yidu_albert_tiny_ep15.h5', "./data/yidu.submit", "finaltestset")
    # evaluate_one(weights_path + '/yidu_albert_tiny_ep16.h5', "./data/yidu.submit", "finaltestset")
    
    # evaluate_one(weights_path + '/yidu_albert_tiny_ep15.h5', "./data/yidu.validate", "validate")
    evaluate_one(save_file_path = weights_path + '/yidu_albert_tiny_ep15.h5',
                 dataset_path = "./data/yidu.submit",
                 csv_path = './report/yidu_albert_tiny_ep15.csv',
                 evaluate_categories_f1 = True)
    
    evaluate_one(save_file_path = weights_path + '/yidu_albert_tiny_ep16.h5',
                 dataset_path = "./data/yidu.submit",
                 csv_path = './report/yidu_albert_tiny_ep16.csv',
                 evaluate_categories_f1 = True)
