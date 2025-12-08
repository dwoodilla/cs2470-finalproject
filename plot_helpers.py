import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def plot_interpolated(Y_interpolated, Y_pred, pd_dataset, _dir, seq2seq):
    feature_names = ['co', 'no', 'no2', 'o3', 'pm25']

    Y_inter_np = Y_interpolated#.numpy()
    Y_pred_np = Y_pred#.numpy()

    Y_inter_sliced = Y_inter_np[:, -1, :]
    if seq2seq: Y_pred_sliced = Y_pred_np[:, -1, :]
    else: Y_pred_sliced = Y_pred_np

    timestamps = pd_dataset.index[23:]

    n = min(len(timestamps), Y_inter_sliced.shape[0], Y_pred_sliced.shape[0])
    if n < max(len(timestamps), Y_inter_sliced.shape[0], Y_pred_sliced.shape[0]):
        print(f"Time sequences are unaligned:\nlen(timestamps)={len(timestamps)}\nY_inter_sliced.shape[0]={Y_inter_sliced.shape[0]}\nY_pred_sliced.shape[0]={Y_pred_sliced.shape[0]}")

    timestamps = timestamps[:n]
    Y_inter_sliced = Y_inter_sliced[:n]
    Y_pred_sliced = Y_pred_sliced[:n]

    df_inter = pd.DataFrame(Y_inter_sliced, index=timestamps, columns=feature_names)
    df_pred  = pd.DataFrame(Y_pred_sliced,  index=timestamps, columns=feature_names)
    df_true  = pd_dataset.iloc[:n, -5:]
    

    n_features = 5
    figsize = (12, 2.5 * n_features)
    fig, axes = plt.subplots(nrows=n_features, ncols=1, figsize=figsize, sharex=True)

    for i, feat in enumerate(df_inter.columns):
        ax = axes[i]

        ax.plot(df_inter.index, df_inter[feat], label='interpolated')
        ax.plot(df_pred.index,  df_pred[feat],  label='predicted', linestyle='--')
        ax.plot(df_true.index, df_true[feat], label='true')

        # # add red shading
        # true_series = df_true[feat]
        # is_nan = true_series.isna()

        # # Identify contiguous NaN intervals
        # nan_groups = []
        # in_nan = False
        # start = None

        # prev_t = None
        # for t, nan_flag in zip(true_series.index, is_nan):
        #     if prev_t is None: prev_t = t
        #     if nan_flag and not in_nan:
        #         in_nan = True
        #         start = t
        #     if not nan_flag and in_nan:
        #         in_nan = False
        #         nan_groups.append((start, prev_t))
        #     prev_t = t

        # # If we ended inside a NaN block
        # if in_nan:
        #     nan_groups.append((start, true_series.index[-1]))

        # # Shade each NaN block
        # for (t_start, t_end) in nan_groups:
        #     ax.axvspan(t_start, t_end, alpha=0.15, color='salmon')

        ax.set_ylabel(feat)
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right')

    plt.xlabel('time')
    plt.tight_layout()
    fig.autofmt_xdate()

    plt.savefig(os.path.join(_dir, 'timeseries.png'))