# %%
# !module load cuda cudnn
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from plot_helpers import plot_interpolated
import os
import main
import keras
import masked_metrics
import models

#%%
# dataset = pd.read_pickle('aqmet_pd.pkl')
# Y = dataset.iloc[1:,-5:]
Y_pred = np.load('runs_from_ccv/2025-12-08_14-40-22/y_pred.npz')['arr_0']
Y_inter = np.load('runs_from_ccv/2025-12-08_14-40-22/y_pred.npz')['arr_0']
_,_,_,Xs,Xc,Y,ds = main.construct_dataset(100)
model:models.LSTNet = keras.models.load_model(
    'runs_from_ccv/2025-12-08_14-40-22/model_s2s=False.keras',
    custom_objects={
        'masked_mse':masked_metrics.MaskedMSE,
        'masked_mae':masked_metrics.MaskedMAE,
        'seq_completeness':masked_metrics.SequenceCompleteness
    }) # type: ignore

# %%
model.train_step(((Xs[:100],Xc[:100]),Y[:100]))
model.interpolate(Xs,Xc,Y)

print(Y.shape, Y_pred.shape, Y_inter.shape)
plot_interpolated(Y_inter, Y_pred, ds, os.path.normpath('runs/2025-12-08_14-40-22'), False)

# %%
model = models.LSTNet(10,256,False,6,False,5,24)

optimizer = keras.optimizers.Adam()

# === Compile model ===
model.compile(
    optimizer = optimizer,
    loss = masked_metrics.MaskedMSE(seq2seq=False),
    metrics = [
        masked_metrics.MaskedMAE(seq2seq=False),
        masked_metrics.R2CoD(seq2seq=False),
        masked_metrics.SequenceCompleteness(False)
    ],
    run_eagerly=False
)

model.save('save_test.keras')
keras.models.load_model('save_test.keras')




# %%
