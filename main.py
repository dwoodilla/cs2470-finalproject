import argparse
import tensorflow as tf
import keras
import numpy as np
import models
import masked_metrics
# import pandas as pd
from pandas import read_pickle

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
    parser.add_argument('--dataframe',      default='aqmet_pd.pkl',     help = 'File path to pandas dataframe pickle containing combined aq and met data.')
    # parser.add_argument('--data',           required=True,              help='File path to the assignment data file.')
    parser.add_argument('--epochs',         type=int,   default=3,      help='Number of epochs used in training.')
    parser.add_argument('--lr',             type=float, default=1e-3,   help='Model\'s learning rate')
    parser.add_argument('--optimizer',      type=str,   default='adam', choices=['adam', 'rmsprop', 'sgd'], help='Model\'s optimizer')
    parser.add_argument('--batch_size',     type=int,   default=100,    help='Model\'s batch size.')
    parser.add_argument('--hidden_size',    type=int,   default=256,    help='Hidden size used to instantiate the model.')
    parser.add_argument('--window_size',    type=int,   default=6,      help='Length of time sequence window.')
    parser.add_argument('--chkpt_path',     default='',                 help='Where the model checkpoint is.')
    parser.add_argument('--check_valid',    default=True,               action="store_true",  help='if training, also print validation after each epoch')

    return parser.parse_args()

def construct_dataset(args, T):

    dataframe = read_pickle(args.dataframe).to_numpy(dtype=np.float32)[:,1:] # omit the timestamp column; keep time encodings.

    dataframe = np.hstack([dataframe[:-1,:],dataframe[1:,-5:]])

    is_finite_bool = np.isfinite(dataframe)
    is_finite_float = is_finite_bool.astype(np.float32)
    finite_dataframe = np.where(is_finite_bool, dataframe, 0.0)

    Xs = finite_dataframe[:,-10:-5]
    Xc = finite_dataframe[:, :-10]
    Y  = dataframe[:, -5:]
    
    Xs_mask = is_finite_float[:,-10:-5]
    Xc_mask = is_finite_float[:,:-10]

    Xs_and_mask = np.hstack([Xs,Xs_mask])
    Xc_and_mask = np.hstack([Xc,Xc_mask])
    dataframe = np.hstack([Xc_and_mask, Xs_and_mask, Y])

    dataframe = np.squeeze(np.lib.stride_tricks.sliding_window_view(dataframe, (T, dataframe.shape[-1])))
    Xs = dataframe[:,:,-15:-5]
    Xc = dataframe[:,:,:-15]
    Y  = dataframe[:,:,-5:]

    dataset = tf.data.Dataset.from_tensor_slices(((Xs,Xc),Y))
    return dataset


def main(args) :

    tf.debugging.experimental.enable_dump_debug_info(
        "C:/Users/dwoodill/csci2470/fp/tmp/tfdbg2_logdir",
        tensor_debug_mode="FULL_HEALTH",
        circular_buffer_size=-1
    )

    tf.debugging.enable_check_numerics()

    # === Load dataset ===

    ds = construct_dataset(args, 24)

    card = ds.cardinality().numpy()
    # train_sz = int(0.8*card)
    ds = ds.shuffle(buffer_size=card, seed=0)
    ds_train = ds.take(3).batch(1)
    ds_test = ds.take(1).batch(1)
    # ds_train = ds.take(train_sz).batch(args.batch_size)
    # ds_test  = ds.skip(train_sz).batch(args.batch_size)

    # === Instantiate model ===
    model_class = {
        "lstnet" : models.LSTNet
    }[args.arch]

    model = model_class(
        time_convolutional_window = args.window_size,
        hidden_dim  = args.hidden_size
    )

    # === Instantiate optimizer and loss ===
    # optimizer_class = {
    #     'adam'      : keras.optimizers.Adam,
    #     'rmsprop'   : keras.optimizers.RMSprop,
    #     'sgd'       : keras.optimizers.SGD
    # }[args.optimizer]

    # optimizer = optimizer_class(
    #     learning_rate = args.lr
    # )
    optimizer = keras.optimizers.Adam(
        clipnorm=10.0
    )

    # === Compile model ===
    model.compile(
        optimizer = optimizer,
        loss = masked_metrics.MaskedMSE(),
        # metrics = [
        #     masked_metrics.MaskedMAE(),
        #     masked_metrics.MaskedRMSE()
        # ],
        run_eagerly=True
    )

    # batch = next(iter(ds_train.take(1)))
    # (Xs, Xc), Y = batch # type: ignore

    # with tf.GradientTape() as tape:
    #     y_pred = model((Xs, Xc), training=True)
    #     loss = model.compute_loss(y=Y, y_pred=y_pred)

    # variables = model.trainable_variables
    # grads = tape.gradient(loss, variables)

    # for var, grad in zip(variables, grads):
    #     if grad is None: #or tf.reduce_any(tf.math.is_nan(grad)):
    #         print("[DISCONNECTED]", var.name)
    #     elif tf.reduce_any(tf.math.is_nan(grad)):
    #         print("[HAS NANS]", var.name)
    #     else:
    #         print("[OK] ", var.name, grad.shape)
    # for var, grad in zip(variables, grads):
    #     tf.print(var, grad, sep='\n')

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
    # main()