import argparse
import tensorflow as tf
import keras
import numpy as np
import models
import tensorflow_datasets as tfds

def parse_args(args=None):
    """ 
    This argument parser is adapted from HW4: Imcap. 
    Credit goes to HW4 authors.
    """
    parser = argparse.ArgumentParser(description="Let's train some neural nets!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--arch',           required=True,              choices=['lstnet', 'custom_forecaster', 'calibrator'],     
        help='Type of model to train. LSTNet and custom_forecaster perform AQS time series forecasting for use in linear regression calibration, \
            while calibrator performs a deep calibration of low-cost sensors using predicted AQS time series.')
    parser.add_argument('--task',           required=True,              choices=['train', 'test', 'both'],  help='Task to run')
    parser.add_argument('--data',           required=True,              help='File path to the assignment data file.')
    parser.add_argument('--epochs',         type=int,   default=3,      help='Number of epochs used in training.')
    parser.add_argument('--lr',             type=float, default=1e-3,   help='Model\'s learning rate')
    parser.add_argument('--optimizer',      type=str,   default='adam', choices=['adam', 'rmsprop', 'sgd'], help='Model\'s optimizer')
    parser.add_argument('--batch_size',     type=int,   default=100,    help='Model\'s batch size.')
    parser.add_argument('--hidden_size',    type=int,   default=256,    help='Hidden size used to instantiate the model.')
    parser.add_argument('--window_size',    type=int,   default=20,     help='Length of time sequence window.')
    parser.add_argument('--chkpt_path',     default='',                 help='Where the model checkpoint is.')
    parser.add_argument('--check_valid',    default=True,               action="store_true",  help='if training, also print validation after each epoch')

    return parser.parse_args()

def apply_na_mask(x:tf.Tensor) -> tf.Tensor:
    '''
    returns a new tensor with NaNs replaced with 0's
    '''
    mask = tf.math.is_nan(x)
    return tf.where(mask, 0.0, x)

def get_na_mask(x:tf.Tensor) -> tf.Tensor:
    '''
    returns mask for element-wise multiplication (i.e. for loss)
    '''
    mask = tf.math.is_nan(x)
    return tf.where(mask, 0.0, 1.0)

def main(args) :
    # === Load dataset ===
    ds = tf.data.Dataset.load(args.data)
    ds_train, ds_test = keras.utils.split_dataset(
        dataset   = ds,
        left_size = 0.8,
        shuffle   = True,
        seed      = 0
    )
    ds_train = ds_train.batch(args.batch_size)
    ds_test  = ds_test .batch(args.batch_size)

    # === Instantiate model ===
    model_class = {
        "lstnet" : models.LSTNet
    }[args.arch]

    model = model_class(
        input_dim=12,
        time_window=args.window_size,
        hidden_dim=args.hidden_size
    )

    # === Instantiate optimizer and loss ===
    optimizer_class = {
        'adam'      : keras.optimizers.Adam,
        'rmsprop'   : keras.optimizers.RMSprop,
        'sgd'       : keras.optimizers.SGD
    }[args.optimizer]

    optimizer = optimizer_class(
        learning_rate = args.lr
    )

    # === Compile model ===
    model.compile(
        optimizer = optimizer,
        loss = keras.metrics.MeanSquaredError(),
        metrics = [
            keras.metrics.MeanAbsoluteError(), 
            keras.metrics.RootMeanSquaredError()
        ]
    )

    # === Train model ===
    model.fit(
        x = ds_train,
        epochs = args.epochs
    )
    eval = model.evaluate(
        x = ds_test
    )
    print(eval)

if __name__=="__main__":
    main(parse_args())