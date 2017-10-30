"""Script to run an experiment.

Author: Stephan Rasp

TODO:
- reprodicibility
- learning rate schedule
- Save model
- pick outputs dataset
"""
import keras
import tensorflow as tf
from argparse import ArgumentParser
from models import *
from losses import *
from utils import *
from data_generator import *
from collections import OrderedDict

# This needs to be NOT hardcoded
feature_vars = OrderedDict({
    'TAP': 2,             # Temperature [z, sample]
    'QAP': 2,             # Specific humidity [z, sample]
    'OMEGA': 2,           # [z, sample]
    'dTdt_adiabatic': 2,  # [z, sample]
    'dQdt_adiabatic': 2,  # [z, sample]
    'QRL': 2,             # Long wave heating rate [z, sample]
    'QRS': 2,             # Short wave heating rate [z, sample]
    'SHFLX': 1,           # [sample]
    'LHFLX': 1,           # [sample]
    'LAT': 1,             # Latitude [sample]
})
target_vars = OrderedDict({
    'SPDT': 2,            # SP temperature tendency [z, sample]
    'SPDQ': 2,            # SP humidity tendency [z, sample]
})


def main(inargs):
    """Main function.
    """
    # set GPU usage
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = inargs.gpu_frac  # GPU RAM usage fraction of 4GB
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

    # Load train and valid set
    train_set = DataSet(inargs.data_dir, 'SPCAM_outputs_train.nc',
                        'SPCAM_mean.nc', 'SPCAM_std.nc', feature_vars.keys(),
                        convolution=inargs.convolution)
    valid_set = DataSet(inargs.data_dir, 'SPCAM_outputs_valid.nc',
                        'SPCAM_mean.nc', 'SPCAM_std.nc', feature_vars.keys(),
                        convolution=inargs.convolution)

    # Build and compile model
    if inargs.convolution:
        model = conv_model(train_set.features[0].shape[1:],
                           train_set.features[1].shape[1],
                           train_set.targets.shape[1],
                           inargs.conv_layers,
                           inargs.hidden_layers,
                           inargs.lr,
                           inargs.loss,
                           batch_norm=inargs.batch_norm,
                           kernel_size=inargs.kernel_size)
    else:   # Fully connected model
        model = fc_model(train_set.features.shape[1],
                         train_set.targets.shape[1],
                         inargs.hidden_layers,
                         inargs.lr,
                         inargs.loss)
    if inargs.verbose: print(model.summary())

    # Fit model
    model.fit(train_set.features, train_set.targets,
              batch_size=inargs.batch_size,
              epochs=inargs.epochs,
              validation_data=(valid_set.features, valid_set.targets),
              callbacks=[TensorBoard(log_dir=inargs.log_dir)])

    model.save()

if __name__ == '__main__':

    p = ArgumentParser()

    p.add_argument('--data_dir',
                   type=str,
                   help='Full outputs file.')
    p.add_argument('--log_dir',
                   type=str,
                   help='TensorBoard log dir')
    p.add_argument('--loss',
                   type=str,
                   help='Loss function.')
    p.add_argument('--gpu_frac',
                   type=float,
                   help='GPU usage fraction.')
    p.add_argument('--lr',
                   type=float,
                   help='Learning rate.')
    p.add_argument('--epochs',
                   type=int,
                   help='Number of epochs')
    p.add_argument('--batch_size',
                   type=int,
                   help='Batch size')
    p.add_argument('--kernel_size',
                   type=int,
                   help='Size of convolution kernel.')
    p.add_argument('--hidden_layers',
                   nargs='+',
                   type=int,
                   help='List with hidden nodes.')
    p.add_argument('--conv_layers',
                   nargs='+',
                   type=int,
                   help='List with feature maps')
    p.add_argument('--convolution',
                   dest='convolution',
                   action='store_true',
                   help='Use convolutional net.')
    p.set_defaults(convolution='False')
    p.add_argument('--batch_norm',
                   dest='batch_norm',
                   action='store_true',
                   help='Use batch_norm.')
    p.set_defaults(batch_norm='False')
    p.add_argument('--verbose',
                   dest='verbose',
                   action='store_true',
                   help='Print additional information.')
    p.set_defaults(verbose='False')

    args = p.parse_args()

    main(args)