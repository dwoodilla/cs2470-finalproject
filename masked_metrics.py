import tensorflow as tf
import keras

class MaskedMSE_seq2seq(keras.losses.Loss):
    def __init__(self, reduction=keras.losses.Reduction.AUTO, name='masked_mse'):
        super().__init__(reduction=reduction, name=name)
    
    def call(self, y_true, y_pred):
        mask = tf.stop_gradient(tf.math.is_finite(y_true))
        y_true_masked = tf.stop_gradient(tf.where(mask, y_true, 0.0))

        se = tf.multiply(tf.square(tf.subtract(y_pred, y_true_masked)), tf.cast(mask, tf.float32)) # [bn, T]

        se_sum_d = tf.reduce_sum(se, axis=-1)
        valid_dims_per_obs = tf.reduce_sum(tf.cast(mask, tf.float32), -1) 
        
        se_mean_d = tf.math.divide_no_nan(se_sum_d, valid_dims_per_obs)

        se_mean_b_t = tf.reduce_mean(se_mean_d) # mean across bn and T dims.
        return se_mean_b_t

class MaskedMAE(keras.losses.Loss):
    def __init__(self, reduction=keras.losses.Reduction.AUTO, name='masked_mae'):
        super().__init__(reduction=reduction, name=name)
    
    def call(self, y_true, y_pred):
        mask = tf.stop_gradient(tf.math.is_finite(y_true))
        y_true_masked = tf.stop_gradient(tf.where(mask, y_true, 0.0))

        ae = tf.multiply(tf.abs(tf.subtract(y_pred, y_true_masked)), tf.cast(mask, tf.float32)) # [bn, T]

        ae_sum_d = tf.reduce_sum(ae, axis=-1)
        valid_dims_per_obs = tf.reduce_sum(tf.cast(mask, tf.float32), -1) 
        
        ae_mean_d = tf.math.divide_no_nan(ae_sum_d, valid_dims_per_obs)

        ae_mean_b_t = tf.reduce_mean(ae_mean_d) # mean across bn and T dims.
        return ae_mean_b_t

# class MaskedR2CoD(keras.losses.Loss):
#     def __init__(self, reduction=keras.losses.Reduction.AUTO, name='masked_R2CoD'):
#         super().__init__(reduction=reduction, name=name)

#     def call(self, y_true, y_pred):

