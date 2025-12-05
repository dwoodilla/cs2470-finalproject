import tensorflow as tf
import keras
import pandas as pd

@keras.saving.register_keras_serializable(package='cs2470fp', name='lstnet')
class LSTNet(keras.Model):
    def __init__(self,  
        sequence_dim:int,
        hidden_dim:int,
        seq2seq:bool,
        omega:int,
        context:bool,
        output_dim:int=5, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.omega = omega
        self.hidden_dim = hidden_dim
        self.sequence_dim = sequence_dim
        self.output_dim = output_dim
        self.seq2seq = seq2seq
        self.context = context

        assert hidden_dim >= sequence_dim # Otherwise convolutions apparently break

        self.pad = keras.layers.ZeroPadding1D(
            padding=(self.omega-1, int(0))
        )
        self.conv = keras.layers.Conv2D(
            filters = hidden_dim,
            kernel_size = ((self.omega, self.sequence_dim)),
            data_format = "channels_last" # required for CPU compatability
        )
        self.gru = keras.layers.GRU(
            units=hidden_dim,
            # recurrent_activation='relu',
            return_sequences=self.seq2seq
        )
        self.latent_projection = keras.Sequential([
            keras.layers.Dense(128, 'relu'),
            keras.layers.Dense(64, 'relu'),
            keras.layers.Dense(self.output_dim, 'relu')
        ])
        self.highway_layer = keras.layers.Dense(
            units=output_dim,
            activation=None
        )
        self.flatten = keras.layers.Flatten()
    
    def call(self, inputs:tf.Tensor, training=None, mask=None) -> tf.Tensor:

        Xs, Xc = inputs 

        if self.context:
            # concatenate context and context mask along the feature dimension
            X = tf.concat([Xs, Xc], axis=-1) 
        else: X = Xs

        y = self.pad(X) 

        y = tf.expand_dims(y, 1) 
        y = tf.transpose(y, perm=[0,2,3,1]) 
        y = self.conv(y) 
        y = tf.squeeze(y, -2) 
        y = self.gru(y) 

        y = self.latent_projection(y) 

        highway_in = X
        if not self.seq2seq: highway_in = self.flatten(highway_in)
        highway_out = self.highway_layer(highway_in)

        y = highway_out + y 
        return y

    def get_config(self):
        base_config = super().get_config()
        config = {
            "sequence_dim":     self.sequence_dim,
            "context":          self.context,
            "output_dim":       self.output_dim,
            "omega":            self.omega,
            "hidden_dim":       self.hidden_dim,
            "seq2seq":          self.seq2seq,
            "conv_layer":         self.conv,
            "gru_layer":          self.gru,
            "latent_projection":  self.latent_projection,
            "highway_layer":      self.highway_layer
        }
        return {**base_config, **config}