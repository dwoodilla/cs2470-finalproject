import argparse
from datetime import datetime
import tensorflow as tf
import keras
import numpy as np
import models
import masked_metrics
import pandas as pd
import os
from plotly_plot_helpers import plot_interpolated
import pathlib
from warnings import warn
import re

DATETIME = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# DIR  = os.path.normpath(f'./runs/{DATETIME}')
# os.makedirs(DIR)

def parse_args(args=None):
    def str2path(arg:str):
        ret = pathlib.Path(arg)
        if not ret.exists(): raise argparse.ArgumentTypeError(f"Path {arg} does not exist")
        elif not ret.is_dir(): raise argparse.ArgumentTypeError(f"Path {arg} is not a directory")
        else: return ret

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Program arguments
    parser.add_argument('--task', choices=['all','train','interpolate'], default='all', type=str, help="Task the program should perform. Interpolation requires a read path to be specified.")
    parser.add_argument('--read_path', type=str2path, default=None)
    parser.add_argument('--write_path', type=str2path, default=None)
    parser.add_argument('--callbacks', action='store_true', help='Flag indicating whether to write callbacks to write_dir.')
    parser.add_argument('--eager', action='store_true', help='Flag indicating whether to train model eagerly. Ignored if not training.')
    
    # Model hyperparameters
    parser.add_argument('--seq2seq', action='store_true', help='Flag indicating whether to predict next sequence or next token.')
    parser.add_argument('--context', action='store_true', help='Flag indicating whether to include context sequence as model input.')
    parser.add_argument('--T', type=int, default=24, help="The length of the sliding time window; i.e. X.shape=(None, T, #pollutants)")
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs used in training.')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--hidden_layer_size', type=int, default=256, help='Hidden size used to instantiate the model.')
    parser.add_argument('--conv_height', type=int, default=6, help='Height of the convolution kernel (in hours).')

    args = parser.parse_args()

    require_write = args.task in ['all', 'train'] or args.callbacks
    require_read  = args.task in ['all', 'interpolate']
    require_same  = args.task == 'all'

    if require_write and args.write_path is None:
        print(f"args.write_path is required but not specified; defaulting to ./runs")
        args.write_path = pathlib.Path('./runs')
        args.write_path.mkdir(parents=False, exist_ok=True)
    if require_read and args.read_path is None:
        print(f"args.read_path is required but not specified; defaulting to ./runs/{DATETIME}") 
        args.write_path = pathlib.Path(f'./runs/{DATETIME}')
        args.write_path.mkdir(parents=False, exist_ok=True)
    if require_same and not args.write_path.joinpath(DATETIME) == args.read_path:
        raise ValueError("Argument 'task' is 'all' but write_path/DATETIME is not read_path.")

    return args

def construct_dataset(batch_size, T=24):

    horizon=1
    dataframe_base = pd.read_pickle('aqmet_pd.pkl')

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

    train_ds    = ds.take(train_sz).batch(batch_size)
    rest        = ds.skip(train_sz)
    val_ds      = rest.take(val_sz).batch(batch_size)
    test_ds     = rest.skip(val_sz).batch(batch_size)

    Xs, Xc, Y = map(tf.convert_to_tensor, [Xs,Xc,Y])
    return train_ds, test_ds, val_ds, Xs, Xc, Y, dataframe_base

def train_and_save(args, train_ds, test_ds, val_ds, Xs, Xc, Y):

    tbd_callback = keras.callbacks.TensorBoard(
        log_dir=str(args.write_path.joinpath('callbacks').absolute()),
        histogram_freq=1,
        write_images=True
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
        optimizer = optimizer, # type: ignore
        loss = masked_metrics.MaskedMSE(seq2seq=args.seq2seq),
        metrics = [
            masked_metrics.MaskedMAE(seq2seq=args.seq2seq),
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

    return tsf_model

def save_model(model:keras.models.Model, args):
    filename = f'model_s2s={args.seq2seq}_ctx={args.context}.keras'
    model.save(args.write_path.joinpath(filename))

def load_model(path:pathlib.Path):
    keras.models.load_model(path)

def main(args):
    train_ds, test_ds, val_ds, Xs, Xc, Y, ds = construct_dataset(args.batch_size, T=args.T)
    if args.task in ['all','train']:
        model = train_and_save(args, test_ds, train_ds, val_ds, Xs, Xc, Y)
    else: 
        ls = os.listdir(args.read_path)
        modelpath_list = [e for e in os.listdir(args.read_path) if re.match(r'model_*.keras', e)]
        if not len(modelpath_list)==1: raise ValueError("model_*.keras folder in args.read_dir is missing or not unique.")
        model = load_model(modelpath_list[0])
    if args.task in ['all','interpolate']:
        Y_inter, Y_pred = interpolate(model, Xs, Xc, Y)
        plot_interpolated(Y_inter, Y_pred, ds, seq2seq=args.seq2seq, _dir=args.write_path)


def interpolate(model, Xs, Xc, Y ):
    Y_inter, Y_pred = model.interpolate(Xs, Xc, Y)

    np.savez_compressed(os.path.join(DIR, 'y_inter.npz'), Y_inter.numpy())
    np.savez_compressed(os.path.join(DIR, 'y_pred.npz'), Y_pred.numpy())
    return Y_inter, Y_pred

if __name__=="__main__":
    main(parse_args())