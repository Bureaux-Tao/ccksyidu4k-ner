# -*- coding:utf-8 -*-
import os
import pickle
from config import batch_size, maxlen, epochs
from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, train_file_path, test_file_path, val_file_path, \
    weights_path, event_type, MODEL_TYPE, label_dict_path

from utils.plot import train_plot

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.callbacks import EarlyStopping, ModelCheckpoint

from model import ALBERT
from preprocess import load_data, data_generator
from utils.adversarial import adversarial_training
from utils.tokenizers import Tokenizer

# bert配置
config_path = BASE_CONFIG_NAME
checkpoint_path = BASE_CKPT_NAME
dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)

# 标注数据
categories = set()
train_data = load_data(train_file_path, categories)
test_data = load_data(test_file_path, categories)
val_data = load_data(val_file_path, categories)

categories = list(sorted(categories))

with open(label_dict_path, 'wb') as f:
    pickle.dump(categories, f)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case = True)

albert = ALBERT(config_path,
                checkpoint_path,
                categories)

model = albert.get_model()

adversarial_training(model, 'Embedding-Token', 0.5)

save_file_path = ("{}/{}_{}_tiny".format(weights_path, event_type, MODEL_TYPE)) + "_ep{epoch:02d}.h5"

train_generator = data_generator(train_data, batch_size, tokenizer, categories, maxlen)
valid_generator = data_generator(val_data, batch_size, tokenizer, categories, maxlen)
# test_generator = data_generator(test_data, batch_size, tokenizer, categories, maxlen)
# reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor = 0.5, patience = 3, verbose = 1)
early_stopping = EarlyStopping(monitor = 'val_sparse_accuracy', patience = 10, verbose = 1)  # 提前结束
save_model = ModelCheckpoint(save_file_path, monitor = 'val_sparse_accuracy', verbose = 0, period = 1,
                             mode = 'max', save_weights_only = True, save_best_only = False)

for i, item in enumerate(train_generator):
    print("\nbatch_token_ids shape: shape:", item[0][0].shape)
    print("batch_segment_ids shape:", item[0][1].shape)
    print("batch_labels shape:", item[1].shape)
    if i == 4:
        break
# batch_token_ids: (32, maxlen) or (32, n), n <= maxlen
# batch_segment_ids: (32, maxlen) or (32, n), n <= maxlen
# batch_labels: (32, maxlen) or (32, n), n <= maxlen

print('\n\t\tTrain start!\t\t\n')

history = model.fit(
    train_generator.forfit(),
    steps_per_epoch = len(train_generator),
    validation_data = valid_generator.forfit(),
    validation_steps = len(valid_generator),
    epochs = epochs,
    verbose = 1,
    callbacks = [save_model, early_stopping]
)

print('\n\tTrain end!\t\n')

train_plot(history.history, history.epoch)
