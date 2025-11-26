import tensorflow as tf
import keras
import logging

# logger = logging.getLogger(__name__)

class MaskedMSE(keras.losses.Loss):
    def __init__(self, reduction=keras.losses.Reduction.AUTO, name='masked_mse'):
        super().__init__(reduction=reduction, name=name)
    
    def call(self, y_true, y_pred):

        mask = tf.cast(tf.math.is_finite(y_true), dtype=tf.float32)
        y_true_masked = tf.where(
            tf.math.is_finite(y_true),
            y_true,
            tf.zeros_like(y_true),
            name='masked_mse_apply_mask'
        )

        sq_err = tf.square(y_pred - y_true_masked) * mask
        sq_err_sum = tf.reduce_sum(sq_err, axis=[1,2])
        n_valid = tf.reduce_sum(mask, axis=[1,2])

        per_batch_loss = tf.where(n_valid > 0.0, sq_err_sum / n_valid, 0.0)
        return tf.reduce_mean(per_batch_loss)

class MaskedMAE(keras.losses.Loss):
    def __init__(self, reduction=keras.losses.Reduction.AUTO, name='masked_mae'):
        super().__init__(reduction, name)

    def call(self, y_true, y_pred):
        mask = tf.cast(tf.math.is_finite(y_true), dtype=tf.float32)
        y_true_masked = tf.where(
            tf.math.is_finite(y_true),
            y_true,
            tf.zeros_like(y_true),
            name='masked_mae_apply_mask'
        )
        abs_err = tf.abs(y_pred - y_true_masked) * mask
        abs_err_sum = tf.reduce_sum(abs_err, axis=[1,2])
        n_valid = tf.reduce_sum(mask, axis=[1,2])

        per_batch_loss = tf.where(n_valid > 0.0, abs_err_sum / n_valid, 0.0)
        return tf.reduce_mean(per_batch_loss)

class MaskedRMSE(keras.losses.Loss):
    def __init__(self, reduction=keras.losses.Reduction.AUTO, name='masked_rmse'):
        super().__init__(reduction, name)
    
    def call(self, y_true, y_pred):
        mask = tf.cast(tf.math.is_finite(y_true), dtype=tf.float32)
        y_true_masked = tf.where(
            tf.math.is_finite(y_true),
            y_true,
            tf.zeros_like(y_true),
            name='masked_mse_apply_mask'
        )
        sq_err = tf.square(y_pred - y_true_masked) * mask
        sq_err_sum = tf.reduce_sum(sq_err, axis=[1,2])
        n_valid = tf.reduce_sum(mask, axis=[1,2])

        per_batch_loss = tf.where(n_valid > 0.0, sq_err_sum / n_valid, 0.0)
        return tf.reduce_mean(tf.sqrt(per_batch_loss))
    