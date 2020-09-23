from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
from DataHandler import DataHandler


datahandler = DataHandler()

chars = datahandler.get_chars()
max_len = datahandler.get_max_len()


class LyricsModel():

    def __init__(self):
        self.max_len = max_len
        self.chars = chars
        self.model = Sequential() 
        self.model.add(LSTM(128, input_shape=(max_len, len(chars))))
        self.model.add(Dense(len(chars)))
        self.model.add(Activation('softmax'))
        return self.model


class TrainedLyricsModel():

    def __init__(self, filepath):
        self.max_len = max_len
        self.chars = chars
        self.model = Sequential() 
        self.model.add(LSTM(128, input_shape=(max_len, len(chars))))
        self.model.add(Dense(len(chars)))
        self.model.add(Activation('softmax'))
        self.model.load_weights(filepath)
        return self.model

