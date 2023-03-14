from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, TimeDistributed, Activation
from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import load_model
import os as os
from celonis_ml.data_preprocessing import DataLoader
import configparser


class ActivityPredictor:
    def __init__(self,
                 mode='development',
                 sample=False,
                 sample_size=None,
                 data_dir='../data/',
                 test_size=None):
        """
        Parameters
        ----------
        mode : {'development', 'production'}
        run this module in 'development' or 'production'
        mode. This will affect the verbosity and size of
        the data. Not fully implemented yet.
        sample : bool
        restrict the dataset to a sample size. If true,
        `sample_size` attribute is mandatory.
        sample_size : int
        size of the final dataset.
        data_dir : path_like
        base folder where to store the data.
        test_size : float

        """
        self._sample = sample
        self._sample_size = sample_size
        self.mode = mode
        self.data_init = None
        self.test_size = test_size

        # Setup working directory
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self._data_dir = data_dir
        self._model_file = self._data_dir + "model.h5"
        self.model = None
        self.tokenizer = None
        self.data_init = DataLoader(mode=self.mode)
        self.activities = self.data_init.get_list_of_activities()
        self.prepare_data(purpose="training")

    def prepare_data(self, purpose="training"):

        df = self.data_init.get_process_variants(purpose=purpose)
        data = df["Variant"].to_string(index=False).split('\n')
        if purpose == "training":
            self.tokenizer = Tokenizer(split=", ")
            self.tokenizer.fit_on_texts([data])
        encoded = self.tokenizer.texts_to_sequences([data])[0]
        self.vocab_size = len(self.tokenizer.word_index) + 1
        print('Vocabulary Size: %d' % self.vocab_size)

        # encode 2 words -> 1 word
        sequences = list()
        for i in range(2, len(encoded)):
            sequence = encoded[i-2:i+1]
            sequences.append(sequence)
        print('Total Sequences: %d' % len(sequences))

        self.max_length = max([len(seq) for seq in sequences])
        sequences = pad_sequences(sequences,
                                  maxlen=self.max_length,
                                  padding='pre')
        print('Max Sequence Length: %d' % self.max_length)

        sequences = array(sequences)
        
        if purpose == "training":
            X, y = sequences[:, :-1], sequences[:, -1]
            y = to_categorical(y, num_classes=self.vocab_size)
            self.X_train = X
            self.y_train = y
        # TODO: how do you predict
        if purpose == "prediction":
            self.X_pred = sequences

    def train(self, epochs=20, verbose=2, batch_size=20):
        
        model = Sequential()
        model.add(Embedding(self.vocab_size, 50, input_length=self.max_length-1))
        model.add(LSTM(50))
        model.add(Dropout(0.1))
        model.add(Dense(self.vocab_size, activation='softmax'))
                                                                                                
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(self.X_train, self.y_train, epochs=epochs, verbose=verbose, batch_size=batch_size)

        print(model.summary())
        model.save(self._model_file)  # creates a HDF5 file
        self.model = model  # deletes the existing model

    def generate_seq(self, seed_text, n_words):
        if self.model is None:
            if os.path.isfile(self._model_file):
                self.model = load_model(self._model_file)
            else:
                print("kaputt")
        
        # Predicting next Activity		
        in_text = seed_text
        for _ in range(n_words):
            encoded = self.tokenizer.texts_to_sequences([in_text])[0]
            encoded = pad_sequences([encoded], maxlen=self.max_length, padding='pre')
            yhat = self.model.predict_classes(encoded, verbose=0)
            out_word = ''
            for word, index in self.tokenizer.word_index.items():
                if index == yhat:
                    out_word = word
                    break
            in_text += ', ' + out_word
        return in_text


if __name__ == '__main__':

    predictor = ActivityPredictor()
    # list of distinct activities
    activities = predictor.activities
    #predictor.train()
    print('Results: ')
    print('\n')
    for act in activities:
        print(predictor.generate_seq(act, 3))
    print("Done.")