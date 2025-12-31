import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def plot_interpolated(
    Y_interpolated,
    Y_pred,
    pd_dataset,
    seq2seq,
    _dir
):
    os.makedirs(_dir, exist_ok=True)

    feature_names = ['co', 'no', 'no2', 'o3', 'pm25']
    n_features = len(feature_names)

    # -----------------------------
    # Prepare arrays
    # -----------------------------
    Y_inter_np = np.asarray(Y_interpolated)
    Y_pred_np = np.asarray(Y_pred)

    Y_inter_sliced = Y_inter_np[:, -1, :]
    Y_pred_sliced = Y_pred_np[:, -1, :] if seq2seq else Y_pred_np

    timestamps = pd_dataset.index[23:]

    n = min(
        len(timestamps),
        Y_inter_sliced.shape[0],
        Y_pred_sliced.shape[0]
    )

    if n < max(len(timestamps), Y_inter_sliced.shape[0], Y_pred_sliced.shape[0]):
        print(
            "Time sequences are unaligned:\n"
            f"len(timestamps)={len(timestamps)}\n"
            f"Y_inter_sliced.shape[0]={Y_inter_sliced.shape[0]}\n"
            f"Y_pred_sliced.shape[0]={Y_pred_sliced.shape[0]}"
        )

    timestamps = timestamps[:n]
    Y_inter_sliced = Y_inter_sliced[:n]
    Y_pred_sliced = Y_pred_sliced[:n]

    df_inter = pd.DataFrame(Y_inter_sliced, index=timestamps, columns=feature_names)
    df_pred  = pd.DataFrame(Y_pred_sliced,  index=timestamps, columns=feature_names)
    df_true  = pd_dataset.iloc[:n, -5:].copy()
    df_true.index = timestamps

    # -----------------------------
    # Helper: make interactive plot
    # -----------------------------
    def make_plot(dfs, labels, styles, title, filename):
        fig = make_subplots(
            rows=n_features,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=feature_names
        )

        for i, feat in enumerate(feature_names, start=1):
            for df, label, style in zip(dfs, labels, styles):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[feat],
                        name=label,
                        mode='lines',
                        **style
                    ),
                    row=i,
                    col=1
                )

            fig.update_yaxes(title_text=feat, row=i, col=1)

        fig.update_layout(
            title=title,
            height=220 * n_features,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )

        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ]
            )
        )

        # Open in browser when run as a script
        pio.renderers.default = "browser"
        fig.show()

        # Save interactive HTML
        fig.write_html(
            os.path.join(_dir, filename),
            include_plotlyjs="cdn"
        )

    # -----------------------------
    # Individual plots
    # -----------------------------
    make_plot(
        dfs=[df_pred],
        labels=["predicted"],
        styles=[dict(line=dict(dash="dash"))],
        title="Predicted Time Series",
        filename="timeseries_predicted.html"
    )

    make_plot(
        dfs=[df_inter],
        labels=["interpolated"],
        styles=[dict()],
        title="Interpolated Time Series",
        filename="timeseries_interpolated.html"
    )

    make_plot(
        dfs=[df_true],
        labels=["true"],
        styles=[dict(line=dict(width=2))],
        title="True Time Series",
        filename="timeseries_true.html"
    )

    # -----------------------------
    # Overlay plot
    # -----------------------------
    make_plot(
        dfs=[df_true, df_inter, df_pred],
        labels=["true", "interpolated", "predicted"],
        styles=[
            dict(line=dict(width=2, color="black")),
            dict(line=dict(color="blue")),
            dict(line=dict(color="red", dash="dash"))
        ],
        title="Time Series Overlay",
        filename="timeseries_overlay.html"
    )
