import keras
import tensorflow as tf
import numpy as np

@keras.saving.register_keras_serializable()
class LSTNet(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lstm = keras.layers.LSTM()
        self.attn = keras.layers.Attention()

    def call(self, x:tf.Tensor):
        pass

