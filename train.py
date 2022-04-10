# -*- coding:utf-8 -*-
import os
import pickle
from config import batch_size, maxlen, epochs
from evaluate import get_score
from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, train_file_path, test_file_path, val_file_path, \
    weights_path, event_type, MODEL_TYPE, label_dict_path
from plot import train_plot, f1_plot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from model import BERT
from preprocess import load_data, data_generator, NamedEntityRecognizer
from utils.backend import keras, K
from utils.adversarial import adversarial_training
from utils.tokenizers import Tokenizer

# bert配置
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)

# 标注数据
categories = set()
train_data = load_data(train_file_path, categories)
val_data = load_data(val_file_path, categories)

categories = list(sorted(categories))

with open(label_dict_path, 'wb') as f:
    pickle.dump(categories, f)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case = True)

bert = BERT(config_path,
            checkpoint_path,
            categories)

model = bert.get_model()
optimizer = bert.get_optimizer()
CRF = bert.get_CRF()
NER = NamedEntityRecognizer(tokenizer, model, categories, trans = K.eval(CRF.trans), starts = [0], ends = [0])

adversarial_training(model, 'Embedding-Token', 0.5)

f1_list = []
recall_list = []
precision_list = []
count_model_did_not_improve = 0


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    
    def __init__(self, patience = 5):
        super().__init__()
        self.best_val_f1 = 0
        self.patience = patience
    
    def on_epoch_end(self, epoch, logs = None):
        global count_model_did_not_improve
        save_file_path = ("{}/{}_{}_base".format(weights_path, event_type, MODEL_TYPE)) + ".h5"
        trans = K.eval(CRF.trans)
        NER.trans = trans
        # print(NER.trans)
        optimizer.apply_ema_weights()
        f1, precision, recall = get_score(val_data, NER)
        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(save_file_path)
            pickle.dump(K.eval(CRF.trans),
                        open(("{}/{}_{}_crf_trans.pkl".format(weights_path, event_type, MODEL_TYPE)), 'wb'))
            count_model_did_not_improve = 0
        else:
            count_model_did_not_improve += 1
            print("Early stop count " + str(count_model_did_not_improve) + "/" + str(self.patience))
            if count_model_did_not_improve >= self.patience:
                self.model.stop_training = True
                print("Epoch %05d: early stopping THR" % epoch)
        optimizer.reset_old_weights()
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )


train_generator = data_generator(train_data, batch_size, tokenizer, categories, maxlen)
valid_generator = data_generator(val_data, batch_size, tokenizer, categories, maxlen)
# test_generator = data_generator(test_data, batch_size, tokenizer, categories, maxlen)

for i, item in enumerate(train_generator):
    print("\nbatch_token_ids shape: shape:", item[0][0].shape)
    print("batch_segment_ids shape:", item[0][1].shape)
    print("batch_labels shape:", item[1].shape)
    if i == 4:
        break
# batch_token_ids: (32, maxlen) or (32, n), n <= maxlen
# batch_segment_ids: (32, maxlen) or (32, n), n <= maxlen
# batch_labels: (32, maxlen) or (32, n), n <= maxlen

evaluator = Evaluator(patience = 5)

print('\n\t\tTrain start!\t\t\n')

history = model.fit(
    train_generator.forfit(),
    steps_per_epoch = len(train_generator),
    epochs = epochs,
    verbose = 1,
    callbacks = [evaluator]
)

print('\n\tTrain end!\t\n')

train_plot(history.history, history.epoch)
data = {
    'epoch': range(1, len(f1_list) + 1),
    'f1': f1_list,
    'recall': recall_list,
    'precision': precision_list
}

f1_plot(data)
