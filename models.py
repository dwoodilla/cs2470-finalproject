import tensorflow as tf
import keras

LOG_PATH = 'file://C:/Users/dwoodill/csci2470/cs2470-finalproject/models.log'

@keras.saving.register_keras_serializable()
class LSTNet(keras.Model):
    def __init__(self,  
        time_window:int, 
        hidden_dim:int,
        input_dim:int=10,
        output_dim:int=5, 
        **kwargs
    ):
        super().__init__(**kwargs)

        assert hidden_dim >= input_dim # Otherwise convolutions break

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

        # If inputs is a tuple, take the sequence (i.e. not context)
        if type(inputs)==tuple:
            x = inputs[0]
        else: x = inputs

        tf.print(
            'x: ',
            x,
            sep='\n',
            # output_stream='file:///Users/mikewoodilla/csci2470/fp/models.log',
            # output_stream=LOG_PATH,
            summarize=-1
        )
        tf.debugging.assert_all_finite(x, 'inputs non-finite')

        y = self.time_conv(x)

        tf.print(
            'Y after time_conv: ',
            y,
            sep='\n',
            # output_stream=LOG_PATH,
            # output_stream='file:///Users/mikewoodilla/csci2470/fp/models.log',
            summarize=-1
        )

        y = tf.expand_dims(y, axis=-1)

        tf.print(
            'Y before feat_conv: ',
            y,
            sep='\n',
            # output_stream=LOG_PATH,
            # output_stream='file:///Users/mikewoodilla/csci2470/fp/models.log',
            summarize=-1
        )
        y = self.feat_conv(y)
        tf.print(
            'Y after feat_conv: ',
            y,
            sep='\n',
            # output_stream=LOG_PATH,
            # output_stream='file:///Users/mikewoodilla/csci2470/fp/models.log',
            summarize=-1
        )
        y = tf.squeeze(y, axis=-2)
        '''
        DEBUG: 
            y = tf.squeeze(y, axis=-2)
    
            ValueError: 
            Can not squeeze dim[2], 
            expected a dimension of 1, 
            got 15 for '{{node lst_net/Squeeze}} = Squeeze[T=DT_FLOAT, squeeze_dims=[-2]](lst_net/conv2d/Relu)' 
            with input shapes: [?,20,15,24].

        '''

        y = self.lstm(y)
        y = self.auto_regressor(x) \
            + self.latent_projection(y)
        return y