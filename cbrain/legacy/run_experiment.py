"""Script to run an experiment.

Author: Stephan Rasp
"""
from cbrain.imports import *
from cbrain.data_generator import DataGenerator
from cbrain.models import *

# Loss dictionary. TODO: Solve this more cleverly (not Tom...)
loss_dict = {
    'mae': 'mae',
    'mse': 'mse',
    'log_loss': log_loss,
}


def main(inargs):
    """Main function.
    """
    # set GPU usage
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

    if inargs.norm_fn is None:
        norm_fn = inargs.train_fn.split('_shuffle')[0] + '_norm.nc'
    else:
        norm_fn = inargs.norm_fn

    # Load train and valid set
    train_gen = DataGenerator(
        inargs.data_dir,
        inargs.train_fn + '_features.nc',
        inargs.train_fn + '_targets.nc',
        inargs.batch_size,
        norm_fn,
        inargs.fsub, inargs.fdiv, inargs.tsub, inargs.tmult,
        shuffle=True, noise=inargs.noise,
    )
    valid_gen = DataGenerator(
        inargs.data_dir,
        inargs.valid_fn + '_features.nc',
        inargs.valid_fn + '_targets.nc',
        16384,  # Large batch size for speed!
        norm_fn,
        inargs.fsub, inargs.fdiv, inargs.tsub, inargs.tmult,
        shuffle=False,
    )
    feature_shape = train_gen.feature_shape
    target_shape = train_gen.target_shape

    # Build and compile model
    if inargs.convolution:
        model = conv_model(
            (30, 7) if inargs.tile else (30, 3),
            4,
            target_shape,
            inargs.conv_layers,
            inargs.hidden_layers,
            inargs.lr,
            loss_dict[inargs.loss],
            kernel_size=inargs.kernel_size, stride=inargs.stride,
            batch_norm=inargs.batch_norm,
            activation=inargs.activation,
            tile=inargs.tile,
            locally_connected=inargs.locally_connected,
            padding=inargs.padding,
            dr=inargs.dr,
        )
    else:   # Fully connected model
        model = fc_model(
            feature_shape,
            target_shape,
            inargs.hidden_layers,
            inargs.lr,
            loss_dict[inargs.loss],
            batch_norm=inargs.batch_norm,
            activation=inargs.activation,
            dr=inargs.dr,
            l2=inargs.l2,
            partial_relu=inargs.partial_relu,
	        eq=inargs.eq,
            fsub=train_gen.feature_norms[0],
            fdiv=train_gen.feature_norms[1],
        )
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
    model.fit_generator(
        train_gen.return_generator(inargs.convolution, inargs.tile),
        train_gen.n_batches,
        epochs=inargs.epochs - int(inargs.valid_after),
        validation_data=None if inargs.valid_after
            else valid_gen.return_generator(inargs.convolution, inargs.tile),
        validation_steps=valid_gen.n_batches,
        workers=inargs.n_workers,
        max_queue_size=50,
        callbacks=callbacks_list,
    )
    if inargs.valid_after:
        # Run last epoch with validation
        model.optimizer.lr = tf.Variable(lr_update(inargs.epochs))
        if len(callbacks_list) == 1:
            callbacks_list = []
        else:
            callbacks_list = [callbacks_list[0]]   # No LR scheduler
        model.fit_generator(
            train_gen.return_generator(inargs.convolution, inargs.tile),
            train_gen.n_batches,
            epochs=1,
            validation_data=valid_gen.return_generator(inargs.convolution, inargs.tile),
            validation_steps=valid_gen.n_batches,
            workers=inargs.n_workers,
            max_queue_size=inargs.max_queue_size,
            callbacks=callbacks_list,
        )
    if inargs.exp_name is not None:
        if not os.path.exists(inargs.model_dir): os.mkdir(inargs.model_dir)
        model.save(inargs.model_dir + '/' + inargs.exp_name + '.h5')

if __name__ == '__main__':

    p = ArgParser()
    p.add('-c', '--config_file',
          is_config_file=True,
          help='Name of config file.')
    p.add_argument('--exp_name',
                   default=None,
                   type=str,
                   help='Experiment name.')
    p.add_argument('--model_dir',
                   type=str,
                   default='./saved_models/',
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
    p.add_argument('--norm_fn',
                   type=str,
                   default=None,
                   help='Normalization file. Default: None -> Infer')
    p.add_argument('--log_dir',
                   default=None,
                   type=str,
                   help='TensorBoard log dir')
    p.add_argument('--fsub',
                   default=None,
                   type=str,
                   help='Subtract feature array by. Default: None')
    p.add_argument('--fdiv',
                   default=None,
                   type=str,
                   help='Divide feature array by. Special: range. Default: None')
    p.add_argument('--tsub',
                   default=None,
                   type=str,
                   help='Subtract target array by. Default: None')
    p.add_argument('--tmult',
                   default=None,
                   type=str,
                   help='Divide target array by, e.g. target_conv. Default: None')
    p.add_argument('--loss',
                   default='mae',
                   type=str,
                   help='Loss function.')
    p.add_argument('--mse_var_ratio',
                   default=10,
                   type=float,
                   help='If mse_var loss function, multiplier for var.')
    p.add_argument('--activation',
                   default='relu',
                   type=str,
                   help='Activation function.')
    p.add_argument('--lr',
                   default=1e-3,
                   type=float,
                   help='Learning rate.')
    p.add_argument('--lr_step',
                   default=5,
                   type=int,
                   help='Step at which to divide learning rate by factor.')
    p.add_argument('--lr_divide',
                   default=5.,
                   type=float,
                   help='Factor to divide learning rate.')
    p.add_argument('--epochs',
                   default=30,
                   type=int,
                   help='Number of epochs')
    p.add_argument('--batch_size',
                   default=1024,
                   type=int,
                   help='Batch size')
    p.add_argument('--kernel_size',
                   default=3,
                   type=int,
                   help='Size of convolution kernel.')
    p.add_argument('--stride',
                   default=1,
                   type=int,
                   help='stride')
    p.add_argument('--hidden_layers',
                   default=[],
                   nargs='+',
                   type=int,
                   help='List with hidden nodes.')
    p.add_argument('--dr',
                   default=None,
                   type=float,
                   help='Dropout rate.')
    p.add_argument('--noise',
                   default=None,
                   type=float,
                   help='Training noise std')
    p.add_argument('--l2',
                   default=None,
                   type=float,
                   help='L2 regularization for dense layers')
    p.add_argument('--conv_layers',
                   default=[],
                   nargs='+',
                   type=int,
                   help='List with feature maps')
    p.add_argument('--n_workers',
                   default=16,
                   type=int,
                   help='Workers for generator queue')
    p.add_argument('--max_queue_size',
                   default=50,
                   type=int,
                   help='Generator queue size')
    p.add_argument('--convolution',
                   dest='convolution',
                   action='store_true',
                   help='Use convolutional net.')
    p.set_defaults(convolution=False)
    p.add_argument('--tile',
                   dest='tile',
                   action='store_true',
                   help='tile')
    p.set_defaults(tile=False)
    p.add_argument('--padding',
                   default='same',
                   type=str,
                   help='padding')
    p.add_argument('--batch_norm',
                   dest='batch_norm',
                   action='store_true',
                   help='Use batch_norm.')
    p.set_defaults(batch_norm=False)
    p.add_argument('--partial_relu',
                   dest='partial_relu',
                   action='store_true',
                   help='...')
    p.set_defaults(partial_relu=False)
    p.add_argument('--eq',
                   dest='eq',
                   action='store_true',
                   help='...')
    p.set_defaults(eq=False)
    p.add_argument('--valid_after',
                   dest='valid_after',
                   action='store_true',
                   help='Only validate after training.')
    p.set_defaults(valid_after=False)
    p.add_argument('--locally_connected',
                   dest='locally_connected',
                   action='store_true',
                   help='Use locally connected convolutions.')
    p.set_defaults(locally_connected=False)
    p.add_argument('--verbose',
                   dest='verbose',
                   action='store_true',
                   help='Print additional information.')
    p.set_defaults(verbose=False)

    args = p.parse_args()

    loss_dict = {
        'mae': 'mae',
        'mse': 'mse',
        'log_loss': log_loss,
        'mse_var': mse_var(args.mse_var_ratio)
    }

    main(args)
