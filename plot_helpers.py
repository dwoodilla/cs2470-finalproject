import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def plot_interpolated(Y_interpolated, Y_pred, pd_dataset, seq2seq, _dir):
    feature_names = ['co', 'no', 'no2', 'o3', 'pm25']

    Y_inter_np = Y_interpolated
    Y_pred_np = Y_pred

    Y_inter_sliced = Y_inter_np[:, -1, :]
    Y_pred_sliced = Y_pred_np[:, -1, :] if seq2seq else Y_pred_np

    timestamps = pd_dataset.index[23:]

    n = min(len(timestamps), Y_inter_sliced.shape[0], Y_pred_sliced.shape[0])
    if n < max(len(timestamps), Y_inter_sliced.shape[0], Y_pred_sliced.shape[0]):
        print(f"Time sequences are unaligned:\n"
              f"len(timestamps)={len(timestamps)}\n"
              f"Y_inter_sliced.shape[0]={Y_inter_sliced.shape[0]}\n"
              f"Y_pred_sliced.shape[0]={Y_pred_sliced.shape[0]}")

    timestamps = timestamps[:n]
    Y_inter_sliced = Y_inter_sliced[:n]
    Y_pred_sliced = Y_pred_sliced[:n]

    df_inter = pd.DataFrame(Y_inter_sliced, index=timestamps, columns=feature_names)
    df_pred  = pd.DataFrame(Y_pred_sliced,  index=timestamps, columns=feature_names)
    df_true  = pd_dataset.iloc[:n, -5:]
    df_true.index = timestamps 

    n_features = len(feature_names)

    def make_plot(df, title, filename, color_cycle=None):
        figsize = (12, 2.5 * n_features)
        fig, axes = plt.subplots(nrows=n_features, ncols=1, figsize=figsize, sharex=True)

        for i, feat in enumerate(feature_names):
            ax = axes[i]
            ax.plot(df.index, df[feat], label=title)

            ax.set_ylabel(feat)
            ax.grid(alpha=0.25)
            ax.legend(loc='upper right')

        plt.xlabel("time")
        plt.tight_layout()
        fig.autofmt_xdate()

        plt.savefig(os.path.join(_dir, filename))
        plt.close(fig)

    make_plot(df_pred, "predicted", "timeseries_predicted.png")

    make_plot(df_inter, "interpolated", "timeseries_interpolated.png")

    make_plot(df_true, "true", "timeseries_true.png")

    figsize = (12, 2.5 * n_features)
    fig, axes = plt.subplots(nrows=n_features, ncols=1, figsize=figsize, sharex=True)

    for i, feat in enumerate(feature_names):
        ax = axes[i]

        ax.plot(df_true.index, df_true[feat], label='true', linewidth=1.5, alpha=0.8, color='black')

        ax.plot(df_inter.index, df_inter[feat], label='interpolated', color='blue')

        ax.plot(df_pred.index, df_pred[feat], label='predicted', linestyle='--', color='red')

        ax.set_ylabel(feat)
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right')

    plt.xlabel("time")
    plt.tight_layout()
    fig.autofmt_xdate()

    plt.savefig(os.path.join(_dir, "timeseries_overlay.png"))
    plt.close(fig)