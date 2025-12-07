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

def parse_args(args=None):
    """
    This argument parser is adapted from HW4: Imcap.
    Credit goes to HW4 authors.
    """
    parser = argparse.ArgumentParser(description="Let's train some neural nets!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hidden_size',    type=int,   default=256,    help='Hidden size used to instantiate the model.')
    parser.add_argument('--conv_height',    type=int,   default=6,      help='Length of time sequence window.')
    parser.add_argument('--seq2seq',        type=bool,  default=True,   help='If false, forecaster treats first t-1 tokens as warmup.')
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

    logdir = f'./logs/fit/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    # tbd_callback = keras.callbacks.TensorBoard(
    #     log_dir=logdir, 
    #     histogram_freq=1,
    #     write_images=True
    # )

    # tf.debugging.experimental.enable_dump_debug_info(
    #     "/Users/mikewoodilla/csci2470/fp/tmp/tfdbg2_logdir",
    #     tensor_debug_mode='FULL_HEALTH',
    #     circular_buffer_size=-1
    # )

    # === Instantiate model ===
    tsf_model = models.LSTNet(
        sequence_dim = 10,
        hidden_dim = args.hidden_size,
        seq2seq = args.seq2seq,
        omega = args.conv_height,
        context = False
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
        # run_eagerly=True
    )

    # === Train model ===
    tsf_model.fit(
        x = train_ds,
        validation_data=val_ds,
        epochs = args.epochs
        # callbacks=[tbd_callback]
    )
    tsf_model.evaluate(
        x = test_ds
    )
    tsf_model.save(f'./models/tsf_model_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}_s2s={args.seq2seq}.keras')
    return tsf_model

def main(args):
    test_ds, train_ds, val_ds, Xs, Xc, Y, ds = construct_dataset(args)
    model = train_and_save(args, test_ds, train_ds, val_ds, Xs, Xc, Y)

    Y_inter, Y_pred = model.interpolate(Xs, Xc, Y)

    plot_interpolated(Y_inter, Y_pred, ds)

# def plot_interpolated(Y_interpolated:tf.Tensor, Y_pred:tf.Tensor, pd_dataset:pd.DataFrame):

#     Y_inter_np = tf.make_ndarray(Y_interpolated)
#     Y_pred_np  = tf.make_ndarray(Y_pred)

#     Y_inter_sliced = Y_inter_np[:,-1,:]
#     Y_pred_sliced  = Y_pred_np[:,-1,:]

#     # timestamps = timestamps[23:]
#     timestamps = pd_dataset.index[23:]
#     Y_inter_pd = pd.DataFrame(Y_inter_sliced, columns=['co', 'no', 'no2', 'o3', 'pm25'])
#     Y_pred_pd  = pd.DataFrame(Y_pred_sliced,  columns=['co', 'no', 'no2', 'o3', 'pm25'])
#     pd.concat([pd.Series(timestamps), Y_inter_pd, Y_pred_pd])
    
def plot_interpolated(Y_interpolated, Y_pred, pd_dataset, start_offset=23):
    """
    Plot interpolated vs predicted time series.
    - Y_interpolated, Y_pred: tf.Tensor or numpy arrays. Expected shape often (N, T, F) or (N, F).
    - pd_dataset: DataFrame containing the timestamps in its index (used with start_offset).
    - feature_names: list of strings length F. Defaults to ['co','no','no2','o3','pm25'].
    - start_offset: integer (23 in your code) to slice timestamps: timestamps = pd_dataset.index[start_offset:].
    """
    feature_names = ['co', 'no', 'no2', 'o3', 'pm25']

    # Y_inter_np = tf.make_ndarray(Y_interpolated)
    # Y_pred_np  = tf.make_ndarray(Y_pred)
    Y_inter_np = Y_interpolated.numpy()
    Y_pred_np = Y_pred.numpy()

    # If tensors are 3D (batch, T, features) and you want the last time-step like your snippet:
    Y_inter_sliced = Y_inter_np[:, -1, :]
    Y_pred_sliced = Y_pred_np[:, -1, :]

    # timestamps (use same slicing as your snippet)
    timestamps = pd_dataset.index[start_offset:]

    # align lengths (defensive)
    n = min(len(timestamps), Y_inter_sliced.shape[0], Y_pred_sliced.shape[0])
    if n < max(len(timestamps), Y_inter_sliced.shape[0], Y_pred_sliced.shape[0]):
        print(f"Time sequences are unaligned:\nlen(timestamps)={len(timestamps)}\nY_inter_sliced.shape[0]={Y_inter_sliced.shape[0]}\nY_pred_sliced.shape[0]={Y_pred_sliced.shape[0]}")

    timestamps = timestamps[:n]
    Y_inter_sliced = Y_inter_sliced[:n]
    Y_pred_sliced = Y_pred_sliced[:n]

    # Build DataFrames (use timestamps as index)
    df_inter = pd.DataFrame(Y_inter_sliced, index=timestamps, columns=feature_names)
    df_pred  = pd.DataFrame(Y_pred_sliced,  index=timestamps, columns=feature_names)
    df_true  = pd_dataset.iloc[:n, -5:]
    

    # Plot: one subplot per feature
    n_features = 5
    figsize = (12, 2.5 * n_features)
    fig, axes = plt.subplots(nrows=n_features, ncols=1, figsize=figsize, sharex=True)

    for i, feat in enumerate(df_inter.columns):
        ax = axes[i]

        ax.plot(df_inter.index, df_inter[feat], label='interpolated')
        ax.plot(df_pred.index,  df_pred[feat],  label='predicted', linestyle='--')
        ax.plot(df_true.index, df_true[feat], label='true')

        # true_unknown = df_true[feat].isna()
                # ======== ADD SHADING FOR NAN REGIONS ========
        true_series = df_true[feat]
        is_nan = true_series.isna()

        # Identify contiguous NaN intervals
        nan_groups = []
        in_nan = False
        start = None

        prev_t = None
        for t, nan_flag in zip(true_series.index, is_nan):
            if prev_t is None: prev_t = t
            if nan_flag and not in_nan:
                in_nan = True
                start = t
            if not nan_flag and in_nan:
                in_nan = False
                nan_groups.append((start, prev_t))
            prev_t = t

        # If we ended inside a NaN block
        if in_nan:
            nan_groups.append((start, true_series.index[-1]))

        # Shade each NaN block
        for (t_start, t_end) in nan_groups:
            ax.axvspan(t_start, t_end, alpha=0.15, color='salmon')

        ax.set_ylabel(feat)
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right')

    # Improve x-axis labeling
    plt.xlabel('time')
    plt.tight_layout()
    fig.autofmt_xdate()
    plt.show()
    
    

if __name__=="__main__":
    main(parse_args())