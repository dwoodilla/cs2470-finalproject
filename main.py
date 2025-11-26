import argparse
import tensorflow as tf
import keras
import logging
# import numpy as np
# import tensorflow_datasets as tfds

# logging.basicConfig(
#     level=logging.INFO,
#     filename="training.log",
#     filemode="w",
#     format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
# )

import models
import metrics


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

def main(args) :
    # === Load dataset ===
    ds = tf.data.Dataset.load(args.data)

    card = ds.cardinality().numpy()
    train_sz = int(0.8*card)
    ds = ds.shuffle(buffer_size=card, seed=0)
    ds_train = ds.take(train_sz).batch(args.batch_size)
    ds_test  = ds.skip(train_sz).batch(args.batch_size)

    # === Instantiate model ===
    model_class = {
        "lstnet" : models.LSTNet
    }[args.arch]

    model = model_class(
        time_window = args.window_size,
        hidden_dim  = args.hidden_size
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
        loss = metrics.MaskedMSE(),
        metrics = [
            metrics.MaskedMAE(),
            metrics.MaskedRMSE()
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