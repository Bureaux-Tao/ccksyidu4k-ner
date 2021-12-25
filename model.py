from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from utils.adversarial import adversarial_training
from utils.backend import keras, K
from utils.models import build_transformer_model
from utils.tokenizers import Tokenizer
from utils.optimizers import Adam, extend_with_piecewise_linear_lr
from utils.snippets import sequence_padding, DataGenerator
from utils.snippets import open, ViterbiDecoder, to_array
from utils.layers import ConditionalRandomField
from keras.layers import Dense, LSTM, Bidirectional, TimeDistributed, Dropout
from keras.models import Model
from tqdm import tqdm
from config import *


class ALBERT:
    def __init__(self, config_path,
                 checkpoint_path,
                 categories,
                 model_type = 'albert',
                 summary = True):
        model = build_transformer_model(
            config_path,
            checkpoint_path,
            model = model_type,
            load_pretrained_model = True
        )
        output_layer = 'Transformer-FeedForward-Norm'
        output = model.get_layer(output_layer).get_output_at(bert_layers - 1)
        
        output = Bidirectional(LSTM(lstm_hidden_units,
                                    return_sequences = True,
                                    dropout = dropout_rate,
                                    recurrent_dropout = dropout_rate))(output)
        output = TimeDistributed(Dense(len(categories) * 2 + 1))(output)
        output = Dropout(dropout_rate)(output)
        # output = Dense(len(categories) * 2 + 1)(output)
        self.CRF = ConditionalRandomField(lr_multiplier = crf_lr_multiplier)
        output = self.CRF(output)
        
        model = Model(model.input, output)
        for layer in model.layers:
            layer.trainable = True
        
        if summary:
            model.summary()
        
        AdamLR = extend_with_piecewise_linear_lr(Adam, name = 'AdamLR')
        
        model.compile(
            loss = self.CRF.sparse_loss,
            optimizer = AdamLR(lr = max_lr, lr_schedule = {
                1000: 1,
                2000: 0.1
            }),
            metrics = [self.CRF.sparse_accuracy]
        )
        
        self.albert_model = model
    
    def get_model(self):
        return self.albert_model
    
    def get_CRF(self):
        return self.CRF
