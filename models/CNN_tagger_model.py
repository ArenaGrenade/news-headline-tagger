import codecs

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Conv1D, MaxPooling1D, Flatten, SpatialDropout1D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

from base.abstract_model import AbstractModel


def load_glove_model(glove_file):
    f = codecs.open(glove_file, 'r', encoding='utf-8')
    model = {}
    index = 1
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        model[word] = (index, embedding)
        index += 1
    return model, index-1


def embeddingLayerBuild(model, train_data, test_data, MAX_SEQ_LEN, WV_DIM, NB_WORDS):
    train_sequences = pad_sequences([[model.get(word, [0])[0] for word in title] for title in train_data],
                                    maxlen=MAX_SEQ_LEN, padding="pre", truncating="post")

    test_sequences = pad_sequences([[model.get(word, [0])[0] for word in title] for title in test_data],
                                   maxlen=MAX_SEQ_LEN, padding="pre", truncating="post")

    wv_matrix = (np.random.rand(NB_WORDS, WV_DIM) - 0.5) / 5.0

    for word, (ind, embedding) in model.items():
        wv_matrix[ind-1] = embedding

    return train_sequences, test_data, wv_matrix


class ConvTaggerModel(AbstractModel):
    def __init__(self, config, MAX_SEQ_LEN, nb_words, wv_dim, wv_mat):
        super(ConvTaggerModel, self).__init__(config)
        self.seq_len = MAX_SEQ_LEN
        self.nb_words = nb_words
        self.WV_DIM = wv_dim
        self.wv_matrix = wv_mat

    def buildModel(self):

        wv_layer = Embedding(
            self.nb_words,
            self.WV_DIM,
            mask_zero=False,
            weights=[self.wv_matrix],
            input_length=self.seq_len,
            trainable=False
        )

        title_input = Input(shape=(self.seq_len, ), dtype='int64')
        word_embeddings = wv_layer(title_input)
#        print(word_embeddings.shape)

        conv1 = Conv1D(1024, 5, activation='relu')(word_embeddings)
#        print(conv1.shape)
        pool1 = MaxPooling1D(3)(conv1)
#        print(pool1.shape)

        conv2 = Conv1D(1024, 5, activation='relu')(pool1)
#        print(conv2.shape)
        pool2 = MaxPooling1D(3)(conv2)
#        print(pool2.shape)

        conv3 = Conv1D(1024, 3, activation='relu')(pool2)
#        print(conv3.shape)
        pool3 = MaxPooling1D(3)(conv3)
#        print(pool3.shape)

        flat = Flatten()(pool3)

        dense = Dense(2048, activation='relu')(flat)

        cat = Dense(7, activation='softmax')(dense)

        self.model = Model(inputs=[title_input], outputs=cat)

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer=SGD(learning_rate=0.001, momentum=0.6, nesterov=False),
                           metrics=['SparseCategoricalAccuracy']
                           )

        return self.model
