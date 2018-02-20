"""Little script to save weights from saved model.

Author: Stephan Rasp
"""

from keras.models import load_model
from configargparse import ArgParser
from losses import *
from keras.utils.generic_utils import get_custom_objects
metrics_dict = dict([(f.__name__, f) for f in all_metrics])
get_custom_objects().update(metrics_dict)

def main(inargs):
    """Load saved model and save weights

    Args:
        inargs: Namespace

    """

    model = load_model(inargs.model_path)

    model.save_weights(inargs.save_path)


if __name__ == '__main__':

    p = ArgParser()
    p.add_argument('--model_path',
                   type=str,
                   help='Path to model')
    p.add_argument('--save_path',
                   type=str,
                   help='Path for saved weights.')

    args = p.parse_args()
    main(args)
