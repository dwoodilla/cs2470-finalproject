import argparse
import tensorflow as tf
import keras
import numpy as np
import models

def parse_args(args=None):
    """ 
    This argument parser is adapted from HW4: Imcap. 
    Credit goes to HW4 authors.
    """
    parser = argparse.ArgumentParser(description="Let's train some neural nets!", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--arch',           required=True,              choices=['lstnet', 'custom_forecaster','calibrator'],     
        help='Type of model to train. LSTNet and custom_forecaster perform AQS time series forecasting for use in linear regression calibration, \
            while calibrator calibrates low-cost sensors directly.')
    parser.add_argument('--task',           required=True,              choices=['train', 'test', 'both'],  help='Task to run')
    parser.add_argument('--data',           required=True,              help='File path to the assignment data file.')
    parser.add_argument('--epochs',         type=int,   default=3,      help='Number of epochs used in training.')
    parser.add_argument('--lr',             type=float, default=1e-3,   help='Model\'s learning rate')
    parser.add_argument('--optimizer',      type=str,   default='adam', choices=['adam', 'rmsprop', 'sgd'], help='Model\'s optimizer')
    parser.add_argument('--batch_size',     type=int,   default=100,    help='Model\'s batch size.')
    parser.add_argument('--hidden_size',    type=int,   default=256,    help='Hidden size used to instantiate the model.')
    parser.add_argument('--window_size',    type=int,   default=20,     help='Length of time sequence window.')
    parser.add_argument('--chkpt_path',     default='',                 help='Where the model checkpoint is.')
    parser.add_argument('--check_valid',    default=True,               action="store_true",  help='if training, also print validation after each epoch')

#     return parser.parse_args()

def apply_na_mask(x:tf.Tensor) -> tf.Tensor:
    mask = tf.math.is_nan(x)
    return tf.where(mask, 0.0, x)

def main(args) :
    ds = tf.data.Dataset.load(args.data)
    batched = ds.batch(args.batch_size)

    batched = ds.batch(5) # Don't shuffle data, as it is chronological
    batch1 = apply_na_mask(next(iter(batched))[0])
    print(batch1.shape)
    pred = models.LSTNet(time_window=20, input_dim=12, hidden_dim=12)(batch1)
    print(pred.shape)
    print(pred)

def train_model(model, captions, img_feats, pad_idx, args, valid):
    '''
    Trains model and returns model statistics.
    This function is adapted from HW4: Imcap. Credit goes to HW4 authors.
    '''
    stats = []
    try:
        for epoch in range(args.epochs):
            stats += [model.train(captions, img_feats, pad_idx, batch_size=args.batch_size)]
            if args.check_valid:
                model.test(valid[0], valid[1], pad_idx, batch_size=args.batch_size)
    except KeyboardInterrupt as e:
        if epoch > 0:
            print("Key-value interruption. Trying to early-terminate. Interrupt again to not do that!")
        else: 
            raise e
        
    return stats

if __name__=="__main__":
    # main(parse_args())
    main()