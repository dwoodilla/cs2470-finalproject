# Time Series Forecasting with LSTNet

## Version information

I implemented this program in a python virtual environment with python 3.9.6 and the most current versions of Tensorflow and Keras.

## Reproducing results

Saved models are stored in the runs.zip file; data is stored in the data.zip file. The aqmet_pd.pkl is the pickled Pandas dataframe which I used for all of my model runs.

R^2 values are computed in the file `timeseries_analysis.ipynb`.

Every model training run produces a timestamped folder which should contain 
the model's `.keras` file. 

## Usage

```sh
usage: main.py [-h] [--hidden_size HIDDEN_SIZE] [--conv_height CONV_HEIGHT] --seq2seq SEQ2SEQ --context
               CONTEXT [--eager EAGER] [--batch_size BATCH_SIZE] [--epochs EPOCHS]

options:
  -h, --help            show this help message and exit
  --hidden_size HIDDEN_SIZE
                        Hidden size used to instantiate the model. (default: 256)
  --conv_height CONV_HEIGHT
                        Length of time sequence window. (default: 6)
  --seq2seq SEQ2SEQ     If false, forecaster treats first t-1 tokens as warmup. (default: None)
  --context CONTEXT     Include context sequence if true. (default: None)
  --eager EAGER
  --batch_size BATCH_SIZE
  --epochs EPOCHS       Number of epochs used in training. (default: 3)
```