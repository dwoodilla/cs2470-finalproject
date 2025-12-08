import tensorflow as tf
import keras
import models

@keras.saving.register_keras_serializable(package='cs2470fp', name='masked_mse')
class MaskedMSE(keras.losses.Loss):
    def __init__(self, seq2seq, name='masked_mse'):
        super().__init__(name=name)
        self.seq2seq = seq2seq
    
    def call(self, y_true, y_pred):
        mask = tf.stop_gradient(tf.math.is_finite(y_true))
        y_true_masked = tf.stop_gradient(tf.where(mask, y_true, 0.0))
        if not self.seq2seq: 
            mask = mask[:,-1,:]
            y_true_masked = y_true_masked[:,-1,:]

        se = tf.multiply(tf.square(tf.subtract(y_pred, y_true_masked)), tf.cast(mask, tf.float32)) # [bn, T]

        se_sum_d = tf.reduce_sum(se, axis=-1)
        valid_dims_per_obs = tf.reduce_sum(tf.cast(mask, tf.float32), -1) 
        
        se_mean_d = tf.math.divide_no_nan(se_sum_d, valid_dims_per_obs)

        se_mean_b_t = tf.reduce_mean(se_mean_d) # mean across bn and T dims.
        # tf.debugging.assert_all_finite(se_mean_b_t, 'loss not all finite')
        return se_mean_b_t

@keras.saving.register_keras_serializable(package='cs2470fp', name='masked_mae')
class MaskedMAE(keras.losses.Loss):
    def __init__(self, seq2seq, name='masked_mae'):
        super().__init__(name=name)
        self.seq2seq = seq2seq
    
    def call(self, y_true, y_pred):
        mask = tf.stop_gradient(tf.math.is_finite(y_true))
        y_true_masked = tf.stop_gradient(tf.where(mask, y_true, 0.0))
        if not self.seq2seq: 
            mask = mask[:,-1,:]
            y_true_masked = y_true_masked[:,-1,:]

        ae = tf.multiply(tf.abs(tf.subtract(y_pred, y_true_masked)), tf.cast(mask, tf.float32)) # [bn, T]

        ae_sum_d = tf.reduce_sum(ae, axis=-1)
        valid_dims_per_obs = tf.reduce_sum(tf.cast(mask, tf.float32), -1) 
        
        ae_mean_d = tf.math.divide_no_nan(ae_sum_d, valid_dims_per_obs)

        ae_mean_b_t = tf.reduce_mean(ae_mean_d) # mean across bn and T dims.
        return ae_mean_b_t

@keras.saving.register_keras_serializable(package='cs2470fp', name='R2CoD')
class R2CoD(keras.losses.Loss):
    def __init__(self, seq2seq, name='R2CoD'):
        super().__init__(name=name)
        self.seq2seq = seq2seq
    
    def call(self, y_true, y_pred):

        y_true_no_nan = tf.stop_gradient(tf.where(tf.math.is_finite(y_true), y_true, 0.0))
        num_not_nan_per_window = tf.reduce_sum(tf.cast(tf.math.is_finite(y_true), tf.float32), axis=1)
        if self.seq2seq: # [BN, dim], eliminate window axis
            y_pred_dewindowed = tf.reduce_mean(y_pred, axis=1) 
            y_true_dewindowed = tf.math.divide_no_nan(tf.reduce_sum(y_true_no_nan, 1), num_not_nan_per_window)
        else: #seq2tok
            y_pred_dewindowed = y_pred
            y_true_dewindowed = y_true[:,-1,:]
        y_true_mean = tf.reduce_mean(y_true_dewindowed, axis=0)
        ss_res = tf.reduce_sum(tf.square(tf.subtract(y_true_dewindowed, y_pred_dewindowed)), axis=0)
        ss_tot = tf.reduce_sum(tf.square(tf.subtract(y_true_dewindowed, y_true_mean)), axis=0)
        cod = 1 - tf.math.divide_no_nan(ss_res, ss_tot)
        return cod


@keras.saving.register_keras_serializable(package='cs2470fp', name='seq_completeness')
class SequenceCompleteness(keras.losses.Loss):
    def __init__(self, seq2seq, name='seq_completeness'):
        super().__init__(name=name)
        self.seq2seq = seq2seq
    
    def call(self, y_true, y_pred):

        is_finite_float = tf.stop_gradient(tf.cast(tf.math.is_finite(y_true), dtype=tf.float32))
        return tf.reduce_mean(is_finite_float)