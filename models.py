import numpy as np
import pandas as pd
import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class LSTNet(keras.layers.Layer):

    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, p:float, **kwargs):
        super().__init__(**kwargs)

        self.conv1d = keras.layers.Conv1D(
            filters=hidden_dim,
            kernel_size=input_dim,
            padding='causal',
            activation='relu',
            training=kwargs['training']
        )

        self.rnn = keras.layers.LSTM(
            units=hidden_dim,
            return_sequences=True,
            training=kwargs['training']
        )
        self.projection = keras.layers.Dense(
            units=output_dim,
            activation='relu'
        )

        self.regularizer = keras.layers.Dense(
            units=output_dim,
            activation=None
        )
    
    def __call__(self, x:tf.Tensor) -> tf.Tensor:
        l1 = self.conv1d(x),
        l2 = self.rnn(l1),
        l3 = self.regularizer(x) + self.projection(l2)
        return l3
    
    # def train(self, batch:)


