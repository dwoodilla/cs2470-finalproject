import tensorflow as tf
import keras

LOG_PATH = 'file://C:/Users/dwoodill/csci2470/cs2470-finalproject/models.log'

# class SkipRNN(keras.layers.Layer):
#     def __init__(self,

#     )

@keras.saving.register_keras_serializable()
class LSTNet(keras.Model):
    def __init__(self,  
        time_convolutional_window:int,
        hidden_dim:int,
        input_dim:int=10,
        output_dim:int=5, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.omega = time_convolutional_window
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        assert hidden_dim >= input_dim # Otherwise convolutions break

        self.Xs_pad = keras.layers.ZeroPadding1D(
            padding=(self.omega-1, int(0)) 
        )
        self.conv = keras.layers.Conv2D(
            filters = hidden_dim,
            kernel_size = ((self.omega, self.input_dim)),
            data_format = "channels_first"
        )
        self.gru = keras.layers.GRU(
            units=256,
            recurrent_activation='relu',
            return_sequences=True
        )
        self.latent_projection = keras.Sequential([
            keras.layers.Dense(128, 'relu'),
            keras.layers.Dense(64, 'relu'),
            keras.layers.Dense(self.output_dim, 'relu')
        ])
        self.auto_regressor = keras.layers.Dense(
            units=output_dim,
            activation=None
        )
    
    def call(self, inputs:tf.Tensor, training=None, mask=None) -> tf.Tensor:

        Xs, Xc = inputs # Xs.shape = [bn,T=20,d=10]
        y = self.Xs_pad(Xs) # y.shape = [bn, T+omega-1=25, d=10 ]
        y = tf.expand_dims(y, 1) # y.shape = [bn, 1, T+omega-1 =25, d =10]

        y = self.conv(y) # y.shape = [bn, filters, T=20, 1]

        y = tf.squeeze(y, -1) 
        y = tf.transpose(y, perm=[0,2,1]) # y.shape = [bn, T=20, filters]

        y = self.gru(y) # y.shape = [bn, T=20, h2=256]

        y = self.latent_projection(y) # y.shape = [bn, T=20, d=5]

        y = self.auto_regressor(Xs) + y # y.shape = bn, T=20, d=5]
        return y