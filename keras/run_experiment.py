"""Script to run an experiment.

Author: Stephan Rasp

TODO:
- reprodicibility
- learning rate schedule
- Save model
- pick outputs dataset
"""
import keras
from keras.callbacks import TensorBoard, LearningRateScheduler
import tensorflow as tf
from configargparse import ArgParser
from models import conv_model, fc_model
from losses import *
from utils import *
from data_generator import DataSet
from collections import OrderedDict


def main(inargs):
    """Main function.
    """
    # set GPU usage
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = inargs.gpu_frac  # GPU RAM usage fraction of 4GB
    config.gpu_options.allow_growth = True
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

    # Load train and valid set
    train_set = DataSet(inargs.data_dir, inargs.train_fn,
                        inargs.mean_fn, inargs.std_fn, inargs.feature_vars,
                        convolution=inargs.convolution,
                        flat_input=inargs.flat_input,
                        target_names=inargs.target_vars,
                        target_norm=inargs.target_norm,
                        target_norm_lev_weight=inargs.target_norm_lev_weight)
    valid_set = DataSet(inargs.data_dir, inargs.valid_fn,
                        inargs.mean_fn, inargs.std_fn, inargs.feature_vars,
                        convolution=inargs.convolution,
                        flat_input=inargs.flat_input,
                        target_names=inargs.target_vars,
                        target_norm=inargs.target_norm,
                        target_norm_lev_weight=inargs.target_norm_lev_weight)

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
                         inargs.loss,
                         batch_norm=inargs.batch_norm,
                         activation=inargs.activation)
    if inargs.verbose: print(model.summary())

    callbacks_list = []
    if inargs.log_dir is not None:
        callbacks_list.append(TensorBoard(log_dir=inargs.log_dir +
                                                  inargs.exp_name))
    if not inargs.lr_step == 0:
        def lr_update(epoch):
            # From goo.gl/GXQaK6
            init_lr = inargs.lr
            drop = 1./inargs.lr_divide
            epochs_drop = inargs.lr_step
            lr = init_lr * np.power(drop, np.floor((1+epoch)/epochs_drop))
            print('lr:', lr)
            return lr
        callbacks_list.append(LearningRateScheduler(lr_update))



    # Fit model
    model.fit(train_set.features, train_set.targets,
              batch_size=inargs.batch_size,
              epochs=inargs.epochs,
              validation_data=(valid_set.features, valid_set.targets),
              callbacks=callbacks_list)

    if inargs.exp_name is not None:
        model.save(inargs.model_dir + '/' + inargs.exp_name + '.h5')

if __name__ == '__main__':

    p = ArgParser()
    p.add('--config_file',
          is_config_file=True,
          help='Name of config file in this directory. '
               'Must contain in and out variable lists.')
    p.add_argument('--feature_vars',
                   type=str,
                   nargs='+',
                   help='Feature variables.')
    p.add_argument('--target_vars',
                   type=str,
                   nargs='+',
                   help='Target variables.')
    p.add_argument('--exp_name',
                   default=None,
                   type=str,
                   help='Experiment name.')
    p.add_argument('--model_dir',
                   type=str,
                   help='Directory to save model to.')
    p.add_argument('--data_dir',
                   type=str,
                   help='Full outputs file.')
    p.add_argument('--train_fn',
                   type=str,
                   help='Training set file.')
    p.add_argument('--valid_fn',
                   type=str,
                   help='Validation set file.')
    p.add_argument('--mean_fn',
                   type=str,
                   help='Mean file.')
    p.add_argument('--std_fn',
                   type=str,
                   help='Std file.')
    p.add_argument('--log_dir',
                   default=None,
                   type=str,
                   help='TensorBoard log dir')
    p.add_argument('--loss',
                   default='mae',
                   type=str,
                   help='Loss function.')
    p.add_argument('--activation',
                   default='relu',
                   type=str,
                   help='Activation function.')
    p.add_argument('--gpu_frac',
                   default=0.2,
                   type=float,
                   help='GPU usage fraction.')
    p.add_argument('--lr',
                   default=1e-4,
                   type=float,
                   help='Learning rate.')
    p.add_argument('--lr_step',
                   default=10,
                   type=int,
                   help='Step at which to divide learning rate by factor.')
    p.add_argument('--lr_divide',
                   default=5.,
                   type=float,
                   help='Factor to divide learning rate.')
    p.add_argument('--epochs',
                   default=50,
                   type=int,
                   help='Number of epochs')
    p.add_argument('--batch_size',
                   default=512,
                   type=int,
                   help='Batch size')
    p.add_argument('--kernel_size',
                   default=3,
                   type=int,
                   help='Size of convolution kernel.')
    p.add_argument('--hidden_layers',
                   default=[],
                   nargs='+',
                   type=int,
                   help='List with hidden nodes.')
    p.add_argument('--conv_layers',
                   default=[],
                   nargs='+',
                   type=int,
                   help='List with feature maps')
    p.add_argument('--convolution',
                   dest='convolution',
                   action='store_true',
                   help='Use convolutional net.')
    p.set_defaults(convolution=False)
    p.add_argument('--flat_input',
                   dest='flat_input',
                   action='store_true',
                   help='If true, assume flat input.')
    p.set_defaults(flat_input=False)
    p.add_argument('--batch_norm',
                   dest='batch_norm',
                   action='store_true',
                   help='Use batch_norm.')
    p.set_defaults(batch_norm=False)
    p.add_argument('--target_norm',
                   dest='target_norm',
                   action='store_true',
                   help='Normalize targets.')
    p.set_defaults(target_norm=False)
    p.add_argument('--target_norm_lev_weight',
                   dest='target_norm_lev_weight',
                   action='store_true',
                   help='Weigh target normalization by level.')
    p.set_defaults(target_norm_lev_weight=False)
    p.add_argument('--verbose',
                   dest='verbose',
                   action='store_true',
                   help='Print additional information.')
    p.set_defaults(verbose=False)

    args = p.parse_args()

    main(args)