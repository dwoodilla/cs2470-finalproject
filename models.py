import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from tqdm import tqdm

# class CustomModel(keras.Model):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#     def call(self):
#         raise NotImplementedError
    
#     def train(self, )


@keras.saving.register_keras_serializable("AQCalib")
class LSTNet(keras.Model):

    def __init__(self,  
        time_window:int, 
        hidden_dim:int,
        input_dim:int=12,
        output_dim:int=5, 
        **kwargs
    ):
        super().__init__(**kwargs)

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
        tf.print(
            "time_conv",
            y.shape
        )

        # y = tf.expand_dims(y, axis=-1)
        # y = self.feat_conv(y)
        # y = tf.squeeze(y, axis=-2)
        # tf.print(
        #     "feat_conv",
        #     y.shape
        # )

        y = self.lstm(y)
        tf.print(
            "lstm",
            y.shape
        )
        y = self.auto_regressor(inputs) \
            + self.latent_projection(y)
        tf.print(
            "proj",
            y.shape
        )
        return y