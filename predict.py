#  -*-coding:utf8 -*-
import os
import pickle

from model import BERT
from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, label_dict_path, weights_path
from preprocess import NamedEntityRecognizer
from utils.tokenizers import Tokenizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils.backend import K

# bert配置
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)


def predict(txt, weights_path, label_dict_path, trans_path):
    with open(label_dict_path, 'rb') as f:  # 打开文件
        categories = pickle.load(f)
    
    # 建立分词器
    
    tokenizer = Tokenizer(dict_path, do_lower_case = True)
    
    bert = BERT(config_path,
                checkpoint_path,
                categories,
                summary = False)
    model = bert.get_model()
    model.load_weights(weights_path)
    # CRF = bert.get_CRF()
    NER = NamedEntityRecognizer(tokenizer, model, categories, trans = pickle.load(open(trans_path, 'rb')), starts = [0],
                                ends = [0])
    NER.trans = pickle.load(open(trans_path, 'rb'))
    entities = []
    for start, end, tag in set(NER.recognize(txt)):
        entities.append((txt[start:end + 1], tag, start, end))
    return sorted(entities, key = lambda d: d[2])


if __name__ == '__main__':
    # segment_ids后长于512的部分将被截断，无法预测
    txt = '1997-8-6行胃癌根治术，2010.11发现CA724 升高最高1295 ，复查PET-CT检查未见复发转移，之后多次复查CA724 波动在500-800之间，多次查胃镜提示吻合口炎，给予对症治疗，患者感左下腹隐痛下腹隐痛不适，2013.10.15复查血CA724 147 CA199 13.62 ,2013.10.23复查腹部CT检查提示胰腺占位，考虑恶性，胰头周围，肝门，腹膜后多发多发淋巴结转移。PET-CT提示：胰头区高代谢，考虑恶性病变。患者近10天出现午饭后左下腹部胀痛，持续2-3小时候可自行缓解。体重近1月上降2KG.患者胰腺穿刺取病理示低分化腺癌，免疫组化示CEA+,CGA+/-,CD56+/-,SYN+/-,对手术有顾虑，且手术风险较大，2013-11-26行放疗30次，2014-1-7放疗结束。2013-11-28始行单药吉西他滨化疗4周期。末次2014-1-7.放化疗中出现黄疸，对症治疗后好转。化疗后患者出现II度白细胞降低、II度血小板降低。2014-1-24复查胰头区病灶及腹腔淋巴结均较强缩小，胰腺穿刺病理中低分化腺癌，免疫组化CA19+,CK7+,CGA-,SYN-,CD56-,CA199+，符合胆、胰导管来源浸润性腺癌。CA72.4 明显上降。2014-1-27病理比对原胃切除标本报告与胰腺肿瘤存在较大形态差异。考虑患者明确胰腺癌，于2014-2-7行第5周期GEM化疗，2014-2复查后病灶缩小SD，于2014-2-21开始第六周期化疗，因第八天白细胞减少推迟到2014-3-3。2014-4-7第8周期化疗。末次给药2014-4-14.2014-4-21复查评效SD，略有缩小，CA72.4降低至11.12.2014-4-28继续单药GEM化疗，末次给药时间2014-9-1.GEM双周一次，2014-7-24复查胰腺病灶继续缩小,评效PR。现患者无明显不适，饮食、睡眠可，体重较前上降约4KG。'
    for i in predict(txt = txt,
                     weights_path = weights_path + '/yidu_roformer_v2_base.h5',
                     label_dict_path = label_dict_path,
                     trans_path = "./weights/yidu_roformer_v2_crf_trans.pkl"):
        print(i)
