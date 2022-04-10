import pickle

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

from path import BASE_CONFIG_NAME, BASE_CKPT_NAME, BASE_MODEL_DIR, label_dict_path
from utils.adversarial import adversarial_training
from utils.backend import keras, K
from utils.models import build_transformer_model
from utils.tokenizers import Tokenizer
from utils.optimizers import Adam, extend_with_piecewise_linear_lr, extend_with_exponential_moving_average
from utils.snippets import sequence_padding, DataGenerator
from utils.snippets import ViterbiDecoder, to_array
from utils.layers import ConditionalRandomField
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.models import Model
from tqdm import tqdm
from config import *


class SetLearningRate:
    """层的一个包装，用来设置当前层的学习率
    """
    
    def __init__(self, layer, lamb, is_ada = False):
        self.layer = layer
        self.lamb = lamb  # 学习率比例
        self.is_ada = is_ada  # 是否自适应学习率优化器
    
    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma',
                    'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb  # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb ** 0.5  # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight) / lamb)  # 更改初始化
                setattr(self.layer, key, weight * lamb)  # 按比例替换
        return self.layer(inputs)


class BERT:
    def __init__(self, config_path,
                 checkpoint_path,
                 categories,
                 summary = True):
        model = build_transformer_model(
            config_path,
            checkpoint_path,
            model = model_type,
            load_pretrained_model = True
        )
        output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
        bert_output = model.get_layer(output_layer).output
        
        lstm = SetLearningRate(
            Bidirectional(LSTM(lstm_hidden_units,
                               return_sequences = True,
                               kernel_initializer = 'he_normal')),
            100, True
        )(bert_output)
        
        x = keras.layers.concatenate(
            [lstm, bert_output],
            axis = -1
        )  # [batch_size, seq_length, lstm_units * 2 + 768]
        
        x = keras.layers.TimeDistributed(
            keras.layers.Dropout(dropout_rate))(x)
        
        x = SetLearningRate(
            TimeDistributed(Dense(len(categories) * 2 + 1, activation = 'relu',
                                  kernel_initializer = 'he_normal')),
            100, True
        )(x)
        
        self.CRF = ConditionalRandomField(lr_multiplier = crf_lr_multiplier)
        final_output = self.CRF(x)
        
        model = Model(model.input, final_output)
        for layer in model.layers:
            layer.trainable = True
        
        if summary:
            model.summary()
        
        optimizer_name = "AdamEMA"
        AdamEMA = extend_with_exponential_moving_average(Adam, name = optimizer_name)
        self.optimizer = AdamEMA(lr = max_lr)
        
        model.compile(
            loss = self.CRF.sparse_loss,
            optimizer = self.optimizer,
            metrics = [self.CRF.sparse_accuracy]
        )
        
        self.bert_model = model
    
    def get_model(self):
        return self.bert_model
    
    def get_CRF(self):
        return self.CRF
    
    def get_optimizer(self):
        return self.optimizer

if __name__ == '__main__':
    config_path = BASE_CONFIG_NAME
    checkpoint_path = BASE_CKPT_NAME
    dict_path = '{}/vocab.txt'.format(BASE_MODEL_DIR)
    with open(label_dict_path, 'rb') as f:  # 打开文件
        categories = pickle.load(f)
    bert = BERT(config_path,
                checkpoint_path,
                categories)
    
    model = bert.get_model()
    plot_model(model, to_file = './images/model.jpg', show_shapes = True)