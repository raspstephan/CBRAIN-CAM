"""
Training script.

Created on 2019-01-28-12-21
Author: Stephan Rasp, raspstephan@gmail.com
"""

from cbrain.imports import *
from cbrain.utils import *
from cbrain.losses import *
from cbrain.data_generator import DataGenerator
from cbrain.models import *
from cbrain.learning_rate_schedule import LRUpdate
from cbrain.save_weights import save2txt, save_norm
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.losses import mse
import json

logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)


def main(args):
    """Main training script."""

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    limit_mem()

    # Load output scaling dictionary
    out_scale_dict = load_pickle(args.output_dict)

    logging.info('Create training and validation data generators')
    train_gen = DataGenerator(
        data_fn=args.data_dir + args.train_fn,
        input_vars=args.inputs,
        output_vars=args.outputs,
        norm_fn=args.data_dir + args.norm_fn,
        input_transform=(args.input_sub, args.input_div),
        output_transform=out_scale_dict,
        batch_size=args.batch_size,
        shuffle=True,
        var_cut_off=args.var_cut_off
    )

    if args.valid_fn is not None:
        valid_gen = DataGenerator(
            data_fn=args.data_dir + args.valid_fn,
            input_vars=args.inputs,
            output_vars=args.outputs,
            norm_fn=args.data_dir + args.norm_fn,
            input_transform=(args.input_sub, args.input_div),
            output_transform=out_scale_dict,
            batch_size=args.batch_size * 10,
            shuffle=False,
            var_cut_off=args.var_cut_off
        )
    else:
        valid_gen = None

    logging.info('Build model')
    model = fc_model(
        input_shape=train_gen.n_inputs,
        output_shape=train_gen.n_outputs,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        conservation_layer=args.conservation_layer,
        inp_sub=train_gen.input_transform.sub,
        inp_div=train_gen.input_transform.div,
        norm_q=out_scale_dict['PHQ']
    )
    print(model.summary())

    logging.info('Compile model')
    if args.loss == 'weak_loss':
        loss = WeakLoss(model.input, inp_div=train_gen.input_transform.div,
                        inp_sub=train_gen.input_transform.sub, norm_q=out_scale_dict['PHQ'],
                        alpha_mass=args.alpha_mass, alpha_ent=args.alpha_ent, noadiab=args.noadiab)
    else:
        loss = args.loss

    metrics = [mse]
    if args.conservation_metrics:
        mass_loss = WeakLoss(model.input, inp_div=train_gen.input_transform.div,
                        inp_sub=train_gen.input_transform.sub, norm_q=out_scale_dict['PHQ'],
                        alpha_mass=1, alpha_ent=0, name='mass_loss', noadiab=args.noadiab)
        ent_loss = WeakLoss(model.input, inp_div=train_gen.input_transform.div,
                             inp_sub=train_gen.input_transform.sub, norm_q=out_scale_dict['PHQ'],
                             alpha_mass=0, alpha_ent=1, name='ent_loss', noadiab=args.noadiab)
        metrics += [mass_loss, ent_loss]

    model.compile(args.optimizer, loss=loss, metrics=metrics)
    lrs = LearningRateScheduler(LRUpdate(args.lr, args.lr_step, args.lr_divide))

    logging.info('Train model')
    model.fit_generator(
        train_gen, epochs=args.epochs, validation_data=valid_gen, callbacks=[lrs])

    if args.exp_name is not None:
        exp_dir = args.model_dir + args.exp_name + '/'
        os.makedirs(exp_dir, exist_ok=True)
        model_fn = exp_dir + 'model.h5'
        logging.info(f'Saving model as {model_fn}')
        model.save(model_fn)

        if args.save_txt:
            weights_fn = exp_dir + 'weights.h5'
            logging.info(f'Saving weights as {weights_fn}')
            model.save_weights(weights_fn)
            save2txt(weights_fn, exp_dir)
            save_norm(train_gen.input_transform, train_gen.output_transform, exp_dir)

    logging.info('Done!')


# Create command line interface
if __name__ == '__main__':
    p = ArgParser()
    p.add('-c', '--config_file', default='config.yml', is_config_file=True, help='Path to config file.')

    # Data arguments
    p.add('--data_dir', type=str, help='Path to preprocessed data files.')
    p.add('--inputs', type=str, nargs='+', help='List of input variables.')
    p.add('--outputs', type=str, nargs='+', help='List of output variables.')
    p.add('--train_fn', type=str, help='File name of training file.')
    p.add('--norm_fn', type=str, help='File name of normalization file.')
    p.add('--input_sub', type=str, help='What to subtract from input array. E.g. "mean"')
    p.add('--input_div', type=str, help='What to divide input array by. E.g. "maxrs"')
    p.add('--output_dict', type=str, help='Output scaling dictionary.')
    p.add('--var_cut_off', type=json.loads, help='Input variable cut off for upper levels.')

    p.add('--valid_fn', type=str, default=None, help='File name of training file.')

    # Neural network hyperparameteris
    p.add('--batch_size', type=int, default=1024, help='Batch size of training generator.')
    p.add('--hidden_layers', type=int, nargs='+', help='Hidden layer sizes.')
    p.add('--activation', type=str, default='LeakyReLU', help='Activation function.')
    p.add('--optimizer', type=str, default='adam', help='Optimizer.')
    p.add('--conservation_layer', dest='conservation_layer', action='store_true', help='Add conservation layer.')
    p.set_defaults(conservation_layer=False)

    # Loss parameters
    p.add('--loss', type=str, default='mse', help='Loss function.')
    p.add('--conservation_metrics', dest='conservation_metrics', action='store_true', help='Add conservation metrics.')
    p.set_defaults(conservation_metrics=False)
    p.add('--alpha_mass', type=float, default=0.25, help='If weak_loss, weight of mass loss.')
    p.add('--alpha_ent', type=float, default=0.25, help='If weak_loss, weight of ent loss.')
    p.add('--noadiab', dest='noadiab', action='store_true',
          help='noadiab')
    p.set_defaults(noadiab=False)

    # Learning rate schedule
    p.add('--lr', type=float, default=0.001, help='Initial learning rate.')
    p.add('--lr_step', type=int, default=2, help='Divide every step epochs.')
    p.add('--lr_divide', type=float, default=5, help='Divide by this number.')
    p.add('--epochs', type=int, default=10, help='Number of epochs.')

    # Save parameters
    p.add('--exp_name', type=str, default=None, help='Experiment identifier.')
    p.add('--model_dir', type=str, default='./saved_models/', help='Model save path.')
    p.add('--save_txt', dest='save_txt', action='store_true', help='Save F90 txt files.')
    p.set_defaults(save_txt=True)

    p.add('--gpu', type=str, default=None, help='Which GPU to use.')

    args = p.parse_args()
    main(args)



