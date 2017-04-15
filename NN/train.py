from __future__ import print_function, unicode_literals
import numpy as np
np.random.seed(1337)
import os
from six.moves import zip
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Input, merge
from keras.layers import GRU, Dense, Embedding, ChainCRF, LSTM, Bidirectional, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Conv1D,ZeroPadding1D
from keras.optimizers import Adam, RMSprop
from keras.utils.data_utils import get_file
from keras.callbacks import Callback
from subprocess import Popen, PIPE, STDOUT
import data

X_test_cut,X_test_len,X_word=data.get_X('ner_dev')

class Callback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        
        Y_pred = model.predict(X_test_cut)
        data.write_result_to_file('ner_pre',Y_pred,X_test_len,X_word)
        print()
        os.system('python evaluate.py ner_pre ner_dev')

maxlen = 100 
word_embedding_dim = 50
lstm_dim = 50
batch_size = 64

print('Loading data...')

nb_word = len(data.index_word)
nb_tag = len(data.index_tag)

X_train = data.X_train
Y_train = data.Y_train
Y_train = np.expand_dims(Y_train, -1)

print('Unique words:', nb_word)
print('Unique tags:', nb_tag)
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)

print('Build model...')

word_input = Input(shape=(maxlen,), dtype='int32', name='word_input')
word_emb = Embedding(nb_word, word_embedding_dim, input_length=maxlen, dropout=0.2, name='word_emb')(word_input)
bilstm = Bidirectional(LSTM(lstm_dim, dropout_W=0.1, dropout_U=0.1, return_sequences=True))(word_emb)
bilstm_d = Dropout(0.1)(bilstm)

half_window_size=2

paddinglayer=ZeroPadding1D(padding=half_window_size)(word_emb)
conv=Conv1D(nb_filter=50,filter_length=(2*half_window_size+1),border_mode='valid')(paddinglayer)
conv_d = Dropout(0.1)(conv)
dense_conv = TimeDistributed(Dense(50))(conv_d)
rnn_cnn_merge=merge([bilstm_d,dense_conv], mode='concat', concat_axis=2)


dense = TimeDistributed(Dense(nb_tag))(rnn_cnn_merge)
crf = ChainCRF()
crf_output = crf(dense)

model = Model(input=[word_input], output=[crf_output])

model.compile(loss=crf.sparse_loss,
              optimizer=RMSprop(0.00001),
              metrics=['sparse_categorical_accuracy'])

model.summary()

model.load_weights('model.weights')
mCallBack = Callback()
model.fit(X_train, Y_train,
          batch_size=batch_size, nb_epoch=10, callbacks=[mCallBack])

model.save_weights('model.weights')
