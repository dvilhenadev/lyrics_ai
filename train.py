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
from LyricsModel import *
from DataHandler import DataHandler

datahandler = DataHandler()

text = datahandler.get_text()
chars = datahandler.get_chars()
max_len = datahandler.get_max_len()
x = datahandler.get_x()
y = datahandler.get_y()

model = LyricsModel(max_len,chars).getModel()

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2,patience=1, min_lr=0.001)

callbacks = [print_callback, checkpoint, reduce_lr]

model.fit(x, y, batch_size=128, epochs=64, callbacks=callbacks)