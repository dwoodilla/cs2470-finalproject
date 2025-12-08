import argparse
from datetime import datetime
import tensorflow as tf
import keras
import numpy as np
import models
import masked_metrics
import pandas as pd
from pandas import read_pickle
import matplotlib.pyplot as plt
import os
from plot_helpers import plot_interpolated
from sklearn.metrics import r2_score

TIME = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def parse_args(args=None):
    """
    This argument parser is adapted from HW4: Imcap.
    Credit goes to HW4 authors.
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "1"):
            return True
        if v.lower() in ("no", "false", "f", "0"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")
    
    parser = argparse.ArgumentParser(description="Let's train some neural nets!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hidden_size',    type=int,   default=256,    help='Hidden size used to instantiate the model.')
    parser.add_argument('--conv_height',    type=int,   default=6,      help='Length of time sequence window.')
    parser.add_argument('--seq2seq',        type=str2bool,  required=True,   help='If false, forecaster treats first t-1 tokens as warmup.')
    parser.add_argument('--context',        type=str2bool, required=True, help='Include context sequence if true.')
    parser.add_argument('--eager',          type=str2bool, default=False)
    parser.add_argument('--batch_size',     type=int,   default=100)
    parser.add_argument('--epochs',         type=int,   default=3,      help='Number of epochs used in training.')

    return parser.parse_args()

def construct_dataset(args, T=24):

    horizon=1
    dataframe_base = read_pickle('aqmet_pd.pkl')

    dataframe = dataframe_base.to_numpy(dtype=np.float32)[:,1:]
    dataframe = np.hstack([dataframe[:-horizon,:],dataframe[horizon:,-5:]])

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

    ds = tf.data.Dataset.from_tensor_slices(((Xs,Xc),Y))
    card = ds.cardinality().numpy()
    train_sz = int(0.7*card)
    val_sz   = int(0.15*card)
    test_sz  = card - (train_sz + val_sz)
    
    train_ds = ds.take(train_sz).batch(args.batch_size)
    test_ds  = ds.take(test_sz).batch(args.batch_size)
    val_ds   = ds.take(val_sz).batch(args.batch_size)

    Xs, Xc, Y = map(tf.convert_to_tensor, [Xs,Xc,Y])

    return train_ds, test_ds, val_ds, Xs, Xc, Y, dataframe_base

def train_and_save(args, train_ds, test_ds, val_ds, Xs, Xc, Y):

    tbd_callback = keras.callbacks.TensorBoard(
        log_dir=f'./runs/{TIME}/callbacks', 
        histogram_freq=1,
        write_images=True
    )
    tf.debugging.experimental.enable_dump_debug_info(
        f"./runs/{TIME}/debug_logs",
        tensor_debug_mode='NO_TENSOR',
        circular_buffer_size=-1
    )

    # === Instantiate model ===
    tsf_model = models.LSTNet(
        sequence_dim = 10,
        hidden_dim = args.hidden_size,
        seq2seq = args.seq2seq,
        omega = args.conv_height,
        context = args.context
    )
    tsf_model((tf.zeros((1,24,10)), tf.zeros((1,24,24))))

    # === Instantiate optimizer and loss ===
    optimizer = keras.optimizers.Adam()

    # === Compile model ===
    tsf_model.compile(
        optimizer = optimizer,
        loss = masked_metrics.MaskedMSE(seq2seq=args.seq2seq),
        metrics = [
            masked_metrics.MaskedMAE(seq2seq=args.seq2seq),
            masked_metrics.R2CoD(seq2seq=args.seq2seq),
            masked_metrics.SequenceCompleteness(args.seq2seq)
        ],
        run_eagerly=args.eager
    )

    # === Train model ===
    tsf_model.fit(
        x = train_ds,
        validation_data=val_ds,
        epochs = args.epochs,
        callbacks=[tbd_callback]
    )
    tsf_model.evaluate(
        x = test_ds
    )

    save_model(tsf_model, args)

    # tsf_model.save(f'./runs/{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}/model_s2s={args.seq2seq}.keras')
    return tsf_model

def save_model(model:keras.models.Model, args):
    dirname = f'./runs/{TIME}/'
    filename = f'model_s2s={args.seq2seq}.keras'
    os.makedirs(dirname, exist_ok=True)
    model.save(os.path.join(dirname, filename))
# def load_model(time:str, seq2seq:bool):
#     dirname = f'runs/{time}'
#     path = os.path.join(dirname, name)
#     return keras.models.load_model(path)

def main(args):

    test_ds, train_ds, val_ds, Xs, Xc, Y, ds = construct_dataset(args)
    model = train_and_save(args, test_ds, train_ds, val_ds, Xs, Xc, Y)

    # model = load_model(TIME, args.seq2seq)
    # model = keras.models.load_model('runs/2025-12-07_21:53:58/model_s2s=True.keras')

    # model = keras.models.load_model(
    #     'runs/2025-12-07_21:53:58/model_s2s=True.keras',
    #     custom_objects={
    #         "MaskedMSE": masked_metrics.MaskedMSE,
    #         "MaskedMAE": masked_metrics.MaskedMAE,
    #         "SequenceCompleteness": masked_metrics.SequenceCompleteness
    #     }
    # )

    # Y_inter, Y_pred = interpolate(model, Xs, Xc, Y, time=TIME)
    Y_inter = np.load('runs/2025-12-07_22:38:10/y_inter.npz')['arr_0']
    Y_pred  = np.load('runs/2025-12-07_22:38:10/y_pred.npz')['arr_0']

    report_r2()

    # plot_interpolated(Y_inter, Y_pred, ds, args, time=TIME)

def report_r2(y_inter, y_pred, y_true):
    r2_score(y_inter, y_pred)

def interpolate(model, Xs, Xc, Y, time=TIME):
    Y_inter, Y_pred = model.interpolate(Xs, Xc, Y)
    os.makedirs(f'./runs/{time}', exist_ok=True)
    np.savez_compressed(f'./runs/{TIME}/y_inter.npz', Y_inter.numpy())
    np.savez_compressed(f'./runs/{TIME}/y_pred.npz', Y_pred.numpy())
    return Y_inter, Y_pred

if __name__=="__main__":
    main(parse_args())