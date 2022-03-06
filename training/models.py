import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from data_processor import CDataProcessor, MinMaxScaler
from matplotlib import pyplot
import pandas as pd
import numpy as np
import json
import os


class CNNModel:
    def __init__(self, model_name=None, model_path=None):
        self.model_path = os.path.abspath(model_path) if model_path else None
        self.model = Sequential()
        self.name = model_name
        self.info = dict({'y_scale_degree': 15})
        self.history = None

    def construct(self, input_size, out_size):
        self.model.add(Dense(128, input_shape=(input_size, )))
        self.model.add(LeakyReLU(alpha=0.2))
        self.model.add(Dropout(0.15))
        self.model.add(Dense(64, activation='relu'))
        # self.model.add(Dropout(0.12))
        # self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(out_size, activation='sigmoid'))
        # self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])

    def fit(self, x_train, y_train, x_cv, y_cv, epoch=200, batch_size=32, optimizer='adam',
            loss='mean_squared_error', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.history = self.model.fit(x_train, self.scale(y_train), validation_data=(x_cv, self.scale(y_cv)), epochs=epoch, batch_size=batch_size)

    def plot_learning_curve(self):
        pyplot.title('Learning Curves')
        pyplot.xlabel('Epoch')
        pyplot.ylabel('Mean Squared Error')
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='val')
        pyplot.legend()
        pyplot.show()

    def save(self):
        self.model.save(os.path.join(os.getcwd(), '{}.h5'.format(self.name)), save_format='h5') # Save models in CWD
        with open('model_columns.json', 'w+') as model_info:
            json.dump(self.info, model_info, indent=4)

    def load(self):
        self.model = tf.keras.models.load_model(self.model_path)

    @staticmethod
    def load_model(self, model_path):
        return CNNModel(model_path=model_path)

    def predict(self, input):
        df = input.copy(deep=True)
        data = CDataProcessor.prepare_address_columns(df)
        data = CDataProcessor.prepare_object_str_columns(data)
        data = CDataProcessor.prepare_datetime_columns(data)
        scaler = MinMaxScaler()
        scaler.fit(data)
        return self.unscale(
            self.model.predict(pd.DataFrame(scaler.fit_transform(data), columns=data.columns).values, batch_size=32)
        )

    def scale(self, d):
        y_scale_degree = self.info['y_scale_degree']
        if y_scale_degree == 1:
            return d
        scale = lambda t: np.sign(t) * np.absolute(t / 1000000.0) ** (1.0 / float(y_scale_degree))
        vscale = np.vectorize(scale)
        return vscale(d)

    def unscale(self, d):
        y_scale_degree = self.info['y_scale_degree']
        if y_scale_degree == 1:
            return d
        unscale = lambda t: (t ** y_scale_degree) * 1000000
        vunscale = np.vectorize(unscale)
        return vunscale(d)
