import numpy as np
import pandas as pd
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable("AQCalib")
class LSTNet(keras.Model):

    def __init__(self, input_dim:int, time_window:int, hidden_dim:int, **kwargs):
        super().__init__(**kwargs)
        output_dim = input_dim

        self.feat_conv = keras.layers.Conv2D(
            filters=hidden_dim,
            kernel_size=(1,input_dim),
            padding='valid',
            activation='relu'
        )
        self.time_conv = keras.layers.Conv1D(
            filters=hidden_dim,
            kernel_size=time_window,
            padding='causal',
            activation='relu'
        )
        self.lstm = keras.layers.LSTM(
            units=hidden_dim,
            return_sequences=True
        )
        self.latent_projection = keras.layers.Dense(
            units=output_dim,
            activation='relu'
        )
        self.auto_regressor = keras.layers.Dense(
            units=output_dim,
            activation=None
        )
    
    def call(self, inputs:tf.Tensor, training=None, mask=None) -> tf.Tensor:
        y = self.time_conv(inputs)

        y = tf.expand_dims(y, axis=-1)
        y = self.feat_conv(y)
        y = tf.squeeze(y, axis=-2)

        y = self.lstm(y)
        y = self.auto_regressor(inputs) \
            + self.latent_projection(y)
        return y