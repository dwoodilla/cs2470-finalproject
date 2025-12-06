import argparse
from datetime import datetime
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
    parser.add_argument('--hidden_size',    type=int,   default=256,    help='Hidden size used to instantiate the model.')
    parser.add_argument('--window_size',    type=int,   default=6,      help='Length of time sequence window.')
    parser.add_argument('--seq2seq',        type=bool,  default=True,   help='If false, forecaster treats first t-1 tokens as warmup.')
    parser.add_argument('--batch_size',     type=int,   default=100)
    parser.add_argument('--epochs',         type=int,   default=3,      help='Number of epochs used in training.')

    return parser.parse_args()

def construct_dataset(args, T=24):

    horizon=1
    dataframe = read_pickle('aqmet_pd.pkl').to_numpy(dtype=np.float32)[:,1:] # omit the timestamp column; keep time encodings.

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

    return train_ds, test_ds, val_ds, Xs, Xc, Y

def train_and_save(args, dataset_tuple):

    logdir = f'./logs/fit/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tbd_callback = keras.callbacks.TensorBoard(
        log_dir=logdir, 
        histogram_freq=1,
        # update_freq='100',
        write_images=True
    )

    # tf.debugging.experimental.enable_dump_debug_info(
    #     "/Users/mikewoodilla/csci2470/fp/tmp/tfdbg2_logdir",
    #     tensor_debug_mode='FULL_HEALTH',
    #     circular_buffer_size=-1
    # )

    # === Load dataset ===
    train_ds, test_ds, val_ds = dataset_tuple


    # === Instantiate model ===
    tsf_model = models.LSTNet(
        omega = args.conv_height,
        hidden_dim = args.hidden_size,
        seq2seq = args.seq2seq
    )

    # === Instantiate optimizer and loss ===
    optimizer = keras.optimizers.Adam()

    # === Compile model ===
    tsf_model.compile(
        optimizer = optimizer,
        loss = masked_metrics.MaskedMSE(seq2seq=args.seq2seq),
        metrics = [
            masked_metrics.MaskedMAE(seq2seq=args.seq2seq),
            masked_metrics.SequenceCompleteness(args.seq2seq)
        ],
        run_eagerly=True
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
    tsf_model.save(f'./models/tsf_model_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_s2s={args.seq2seq}.keras')

def main(args):
    _,_,_, Xs, Xc, Y = construct_dataset(args)
    # train_and_save(args, dataset_tuple)

    model = models.LSTNet(
        sequence_dim = 2*5,
        hidden_dim=256,
        seq2seq=True,
        omega=6,
        context=False,
        output_dim=5
    )
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = masked_metrics.MaskedMSE(seq2seq=args.seq2seq),
        metrics = [
            masked_metrics.MaskedMAE(seq2seq=args.seq2seq),
            masked_metrics.SequenceCompleteness(args.seq2seq)
        ],
        # run_eagerly=True
    )

    x,y = model.interpolate_fast(
        Xs, Xc, Y
    )
    print(x)
    print(y)
    

if __name__=="__main__":
    main(parse_args())