import math, os, json, sys, re, numpy as np, pickle, PIL, scipy
from glob import glob
from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict
import itertools
from itertools import chain

import pandas as pd
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
import bcolz


import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer


def get_LSTM_model(units=100, embeding_size=50, emb_reg=1e-5, dropout=0.2,
                   seq_len=500, vocab_size=5000):
    
    # weights=[emb]
    inputs = Input(shape=(seq_len,), dtype='int32')
    x = Embedding(vocab_size, embeding_size, input_length=seq_len,
                  mask_zero=True,
                  embeddings_regularizer=regularizers.l2(emb_reg))(inputs)
    x = Dropout(dropout)(x)
    x = LSTM(units, implementation=2, dropout=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_glove_dataset(dataset="6B.50d", glove_path='data/glove/results/'):
    """Download the requested glove dataset from files.fast.ai
    and return a location that can be passed to load_vectors.
    """
    # see wordvectors.ipynb for info on how these files were
    # generated from the original glove data.
    md5sums = {'6B.50d': '8e1557d1228decbda7db6dfd81cd9909',
               '6B.100d': 'c92dbbeacde2b0384a43014885a60b2c',
               '6B.200d': 'af271b46c04b0b2e41a84d8cd806178d',
               '6B.300d': '30290210376887dcc6d0a5a6374d8255'}
    glove_path = os.path.abspath('.glove/')
    if not os.path.exists(glove_path):
        os.makedirs(glove_path)
    datset_path = glove_path + "/" + dataset
    if not os.path.exists(datset_path + ".dat"):
        get_file(dataset,
            'http://files.fast.ai/models/glove/' + dataset + '.tgz',
            cache_subdir=glove_path,
            md5_hash=md5sums.get(dataset, None),
            untar=True)
    return datset_path


def load_array(fname):
    return bcolz.open(fname)[:]


def load_vectors(loc):
    return (load_array(loc+'.dat'),
        pickle.load(open(loc+'_words.pkl','rb')),
        pickle.load(open(loc+'_idx.pkl','rb')))


def create_emb(vocab_size, glove_path, idx2word):
    vecs, words, wordidx = load_vectors(glove_path)
    n_fact = vecs.shape[1]
    emb = np.zeros((vocab_size, n_fact))

    for i in range(1,len(emb)):
        word = idx2word[i]
        if word and re.match(r"^[a-zA-Z0-9\-]*$", word) and word in wordidx:
            src_idx = wordidx[word]
            emb[i] = vecs[src_idx]
        else:
            # If we can't find the word in glove, randomly initialize
            emb[i] = normal(scale=0.6, size=(n_fact,))

    # This is our "rare word" id - we want to randomly initialize
    emb[-1] = normal(scale=0.6, size=(n_fact,))
    emb/=3
    return emb

