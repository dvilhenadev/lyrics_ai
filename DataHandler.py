from os import listdir
from os.path import isfile, join
import sys
import collections, functools, operator
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
from functions import *

class DataHandler():

    def __init__(self):
        dataset_path = "./dataset"
        files_in_dataset = [f for f in listdir(dataset_path) if isfile(join(dataset_path, f))]
        char_indices_list = []
        indices_char_list = []

        total_lyrics = ""
        for filename in files_in_dataset:
            filepath = dataset_path +"/"+filename
            with open(filepath, 'r',encoding='utf8') as file:
                text = file.read().lower()
                total_lyrics += "\n\n"+text 

        chars = sorted(list(set(total_lyrics))) # getting all unique chars

        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))

        max_len = 40
        step = 3
        sentences = []
        next_chars = []

        for i in range(0, len(total_lyrics) - max_len, step):
            sentences.append(total_lyrics[i: i + max_len])
            next_chars.append(total_lyrics[i + max_len])

        x = np.zeros((len(sentences), max_len, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        self.text = text
        self.chars = chars
        self.max_len = max_len
        self.x = x
        self.y = y
        self.char_indices = char_indices
        self.indices_char = indices_char

    def get_text(self):
        return self.text
    
    def get_chars(self):
        return self.chars
    
    def get_max_len(self):
        return self.max_len

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_char_indices(self):
        return self.char_indices
    
    def get_indices_char(self):
        return self.indices_char
