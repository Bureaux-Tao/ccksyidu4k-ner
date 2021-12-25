import os
import pickle

from model import ALBERT
from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, label_dict_path, weights_path
from preprocess import NamedEntityRecognizer
from utils.tokenizers import Tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils.backend import K

# bert配置
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)


def predict(txt, save_file_path):
    with open(label_dict_path, 'rb') as f:  # 打开文件
        categories = pickle.load(f)
    
    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case = True)
    
    albert = ALBERT(config_path,
                    checkpoint_path,
                    categories,
                    summary = False)
    model = albert.get_model()
    model.load_weights(save_file_path)
    CRF = albert.get_CRF()
    NER = NamedEntityRecognizer(tokenizer, model, categories, trans = K.eval(CRF.trans), starts = [0], ends = [0])
    entities = []
    for start, end, tag in set(NER.recognize(txt)):
        entities.append((txt[start:end + 1], tag, start, end))
    print(entities)


if __name__ == '__main__':
    txt = '考虑“消化道穿孔可能”，予禁食、胃肠减压、“生长抑素”抑制消化液分泌，“舒普深、奥硝唑”抗感染及制酸、补液等处理后，腹痛稍缓解。'
    predict(txt = txt,
            save_file_path = weights_path + '/yidu_albert_tiny_ep15.h5')
