import tensorflow as tf
import keras
import pandas as pd
import numpy as np

@keras.saving.register_keras_serializable(package='cs2470fp', name='lstnet')
class LSTNet(keras.Model):
    def __init__(self,  
        sequence_dim:int,
        hidden_dim:int,
        seq2seq:bool,
        omega:int,
        context:bool,
        output_dim:int=5, 
        context_dim:int=24,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.omega = omega
        self.hidden_dim = hidden_dim
        self.sequence_dim = sequence_dim
        self.output_dim = output_dim
        self.seq2seq = seq2seq
        # self.context_dim = context_dim
        self.context = context
        self.context_dim = context_dim if context else 0

        assert hidden_dim >= sequence_dim # Otherwise convolutions apparently break

        self.pad = keras.layers.ZeroPadding1D(
            padding=(self.omega-1, int(0))
        )
        self.conv = keras.layers.Conv2D(
            filters = hidden_dim,
            kernel_size = ((self.omega, self.sequence_dim+self.context_dim)),
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

        # self.build(((None, 24, self.sequence_dim),(None,24,self.context_dim)))
    
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
    
    @tf.function
    def train_step(self, inputs):
        (Xs, Xc), Y = inputs
        BN = tf.shape(Xs)[0]
        Xs_obs = tf.expand_dims(Xs[0], 0)
        N = tf.constant(0)

        grads_running = [tf.zeros_like(v) for v in self.trainable_variables]
        loss_running = tf.constant(0.0, dtype=tf.float32)
        
        # forced_prediction_arr = tf.TensorArray(tf.float32, size=BN)
        pred_arr = tf.TensorArray(tf.float32, size=BN)
        # loss_arr = tf.TensorArray(tf.float32, size=BN)

        def cond(N, Xs_obs_3, loss_running, grads_running, pred_arr):
            return tf.less(N, BN)
        def body(N, Xs_obs_3, loss_running, grads_running, pred_arr):
            # tf.print("Batch progress: ", N, "/", BN, end="\r")

            Xc_obs_3 = tf.expand_dims(Xc[N], 0)
            Y_obs_3  = tf.stop_gradient(tf.expand_dims(Y[N], 0))

            with tf.GradientTape() as t:
                Y_pred_3 = self((Xs_obs_3, Xc_obs_3), training=True)
                loss = self.loss(Y_obs_3, Y_pred_3)
            grads = t.gradient(loss, self.trainable_variables)
            
            def running_mean(running, grad):
                return running + ((grad - running)/(tf.cast(N+1, tf.float32)))
            grads_running = tf.nest.map_structure(running_mean, grads_running, grads)
            loss_running  = loss_running + ((loss - loss_running)/tf.cast(N+1, tf.float32))

            # loss_arr = loss_arr.write(N, loss)
            pred_arr = pred_arr.write(N, Y_pred_3)

            finite_mask_3 = tf.stop_gradient(tf.math.is_finite(Y_obs_3))
            Y_obs_3_nansafe = tf.stop_gradient(tf.where(finite_mask_3, Y_obs_3, tf.zeros_like(Y_obs_3)))
            Y_pred_forced_3 = tf.where(finite_mask_3, Y_obs_3_nansafe, Y_pred_3)

            Xs_next = tf.concat([Y_pred_forced_3, tf.cast(finite_mask_3, tf.float32)], axis=-1)
            
            return N+1, Xs_next, loss_running, grads_running, pred_arr
        
        _, _, loss_running, grads_running, pred_arr = tf.while_loop(
            cond,
            body,
            loop_vars = (N, Xs_obs, loss_running, grads_running, pred_arr),
            parallel_iterations=1,
            maximum_iterations=BN
        ) # type: ignore

        batch_preds = pred_arr.concat()
        # batch_loss = loss_arr.concat()
        self.optimizer.apply_gradients(zip(grads_running, self.trainable_variables))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss_running)
            else:
                metric.update_state(Y, batch_preds)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    
    @tf.function
    def interpolate(self, Xs, Xc, Y):
        BN = tf.shape(Xs)[0] # type: ignore

        Xs_obs = Xs[0]

        ta_pred_forced = tf.TensorArray(tf.float32, size=BN)
        ta_pred        = tf.TensorArray(tf.float32, size=BN)
        
        N = tf.constant(0)

        def cond(N, Xs_obs, ta_pred_forced, ta_pred):
            return tf.less(N,BN)
        
        def body(N, Xs_obs, ta_pred_forced, ta_pred):
            tf.print("Progress: ", N, "/", BN, end="\r")
            Xs_obs_3 = tf.expand_dims(Xs_obs, 0)
            Xc_obs_3 = tf.expand_dims(Xc[N], 0)
            Y_obs_3  = tf.expand_dims(Y[N], 0)

            Y_pred_3 = self((Xs_obs_3, Xc_obs_3), training=False)

            finite_mask_3 = tf.stop_gradient(tf.math.is_finite(Y_obs_3))
            Y_obs_3_nansafe = tf.stop_gradient(tf.where(finite_mask_3, Y_obs_3, tf.zeros_like(Y_obs_3)))
            Y_pred_forced_3 = tf.where(finite_mask_3, Y_obs_3_nansafe, Y_pred_3)

            Xs_next = tf.squeeze(tf.concat([Y_pred_forced_3, tf.cast(finite_mask_3, tf.float32)], axis=-1))

            ta_pred_forced = ta_pred_forced.write(N, Y_pred_forced_3)
            ta_pred        = ta_pred.write(N, Y_pred_3)

            return N+1, Xs_next, ta_pred_forced, ta_pred
        
        _, _, ta_pred_forced, ta_pred = tf.while_loop(
            cond,
            body,
            loop_vars=(N, Xs_obs, ta_pred_forced, ta_pred),
            parallel_iterations=1,
            maximum_iterations=BN
        ) # type: ignore

        Y_interpolated = ta_pred_forced.concat()
        Y_predicted    = ta_pred.concat()
        return Y_interpolated, Y_predicted

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

    # def build(self, input_shape):
    #     Xs_shape, Xc_shape = input_shape
    #     if self.context_dim:
    #         in_d = Xs_shape[-1] + Xc_shape[-1]
    #     else: in_d = Xs_shape[-1]

    #     self.pad.build((None, 24, in_d))
    #     self.conv.build((None, 24+(self.omega-1), in_d, 1))
    #     self.gru.build((None, 24, self.hidden_dim))
    #     if self.seq2seq:
    #         self.latent_projection.build((None, self.hidden_dim))
    #     else: 
    #         self.latent_projection.build((None, 24, self.hidden_dim))

    #     self.flatten.build((None, 24, in_d))
    #     if not self.seq2seq:
    #         self.highway_layer.build((None, 24 * in_d))
        
    #     self.add_weight
        