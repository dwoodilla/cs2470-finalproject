import numpy as np
import pandas as pd
import tensorflow as tf
import keras

# class Conv(keras.layers.Layer):
#     def __init__(self, **kwargs):
#         self.conv = keras.layers.Conv1D(**kwargs)

class LSTNet(keras.layers.Layer):
    def __init__(
        self,
        filters:int,
        kernel_size:int,
        

    ):
        super().__init__()
        self.conv = keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="causal",
            activation='relu'
        )