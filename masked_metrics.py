import tensorflow as tf
import keras
import models

@keras.saving.register_keras_serializable(package='cs2470fp', name='masked_mse')
class MaskedMSE(keras.losses.Loss):
    def __init__(self, seq2seq, reduction=None, name='masked_mse'):
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
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'seq2seq':self.seq2seq
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['seq2seq'] = keras.losses.deserialize(config['seq2seq'])
        return cls(**config)

@keras.saving.register_keras_serializable(package='cs2470fp', name='masked_mae')
class MaskedMAE(keras.losses.Loss):
    def __init__(self, seq2seq, reduction=None, name='masked_mae'):
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
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'seq2seq':self.seq2seq
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['seq2seq'] = keras.losses.deserialize(config['seq2seq'])
        return cls(**config)

@keras.saving.register_keras_serializable(package='cs2470fp', name='R2CoD')
class R2CoD(keras.metrics.Metric):
    def __init__(self, seq2seq, reduction=None, output_dim=5, name='R2CoD'):
        super().__init__(name=name)
        self.seq2seq = seq2seq
        self.value = self.add_variable(
            shape=(output_dim,),
            initializer='zeros',
            name='value',
            aggregation='none'
        )
    
    def call(self, y_true, y_pred):

        y_true_no_nan = tf.stop_gradient(tf.where(tf.math.is_finite(y_true), y_true, 0.0))
        num_not_nan_per_window = tf.reduce_sum(tf.cast(tf.math.is_finite(y_true), tf.float32), axis=1)
        if self.seq2seq: # [BN, dim], eliminate window axis
            y_pred_dewindowed = tf.reduce_mean(y_pred, axis=1) 
            y_true_dewindowed = tf.math.divide_no_nan(tf.reduce_sum(y_true_no_nan, 1), num_not_nan_per_window)
        else: #seq2tok
            y_pred_dewindowed = y_pred
            y_true_dewindowed = y_true_no_nan[:,-1,:]
        y_true_mean = tf.reduce_mean(y_true_dewindowed, axis=0)
        ss_res = tf.reduce_sum(tf.square(tf.subtract(y_true_dewindowed, y_pred_dewindowed)), axis=0)
        ss_tot = tf.reduce_sum(tf.square(tf.subtract(y_true_dewindowed, y_true_mean)), axis=0)
        cod = 1 - tf.math.divide(ss_res, ss_tot) # allow nans/infs
        return cod
    
    def update_state(self, y_true, y_pred, **kwargs):
        self.value = self.call(y_true, y_pred)
    def result(self): 
        return {
            'R2_co' : self.value[0],
            'R2_no' : self.value[1], 
            'R2_no2': self.value[2], 
            'R2_o3' : self.value[3], 
            'R2_pm25':self.value[4]
        }
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'seq2seq':self.seq2seq
        })
        return config

    @classmethod
    def from_config(cls, config):
        if "dtype" in config and isinstance(config["dtype"], dict):
            policy = keras.dtype_policies.deserialize(config["dtype"])

        config['seq2seq'] = keras.metrics.deserialize(config['seq2seq'])
        return cls(name=config['name'],seq2seq=config['seq2seq'])



@keras.saving.register_keras_serializable(package='cs2470fp', name='seq_completeness')
class SequenceCompleteness(keras.losses.Loss):
    def __init__(self, seq2seq, reduction=None, name='seq_completeness'):
        super().__init__(name=name)
        self.seq2seq = seq2seq
    
    def call(self, y_true, y_pred):
        is_finite_float = tf.stop_gradient(tf.cast(tf.math.is_finite(y_true), dtype=tf.float32))
        return tf.reduce_mean(is_finite_float)

    def get_config(self):
        config = super().get_config()
        config.update({
            'seq2seq':self.seq2seq
        })
        return config

    @classmethod
    def from_config(cls, config):
        if "dtype" in config and isinstance(config["dtype"], dict):
            policy = keras.dtype_policies.deserialize(config["dtype"])
        config['seq2seq'] = keras.losses.deserialize(config['seq2seq'])
        return cls(**config)