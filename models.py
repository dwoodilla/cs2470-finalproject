import tensorflow as tf
import keras

@keras.saving.register_keras_serializable()
class LSTNet(keras.Model):
    def __init__(self,  
        time_window:int, 
        hidden_dim:int,
        input_dim:int=24,
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
        tf.print(
            'Inputs: ',
            inputs,
            sep='\n',
            output_stream='file:///Users/mikewoodilla/csci2470/fp/models.log',
            summarize=-1
        )
        tf.debugging.assert_all_finite(inputs, 'inputs non-finite')

        y = self.time_conv(inputs)

        tf.print(
            'y: ',
            y,
            sep='\n',
            output_stream='file:///Users/mikewoodilla/csci2470/fp/models.log',
            summarize=-1
        )
        tf.debugging.assert_all_finite(
            y, 
            f'time_conv yields infinite values: y.shape = {y.shape}, inputs.shape = {inputs.shape}'
        )
        '''
        time_conv yields NaN values
        y.shape = (None, 20, 24), inputs.shape = (None, 20, 24)
        '''

        y = tf.expand_dims(y, axis=-1)
        y = self.feat_conv(y)
        y = tf.squeeze(y, axis=-2)

        y = self.lstm(y)
        y = self.auto_regressor(inputs) \
            + self.latent_projection(y)
        return y