"""
Script to preprocess SPCAM data.

Created on 2019-01-23-14-49
Author: Stephan Rasp, raspstephan@gmail.com
"""

from cbrain.imports import *
from cbrain.preprocessing.convert_dataset import preprocess
from cbrain.preprocessing.shuffle_dataset import shuffle

# Set up logging, mainly to get timings easily.
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)


def main(args):
    """

    Returns
    -------

    """
    # Create training dataset
    logging.info('Preprocess training dataset')
    preprocess(args.in_dir, args.in_fns, args.out_dir, args.out_fn, args.vars)

    # Shuffle training dataset
    if args.shuffle:
        logging.info('Shuffle training dataset')
        shuffle(args.out_dir, args.out_fn, args.chunk_size)

    # Potentially
    if args.val_in_fns is not None:
        logging.info('Preprocess validation dataset')
        preprocess(args.in_dir, args.val_in_fns, args.out_dir, args.val_out_fn, args.vars)


# Create command line interface
if __name__ == '__main__':

    p = ArgParser()
    p.add('-c', '--config_file', default='config.yml', is_config_file=True, help='Path to config file.')
    p.add('--vars', type=str, nargs='+', help='List of variables.')

    # For first file
    p.add('--in_dir', type=str, help='Directory containing SPCAM files.')
    p.add('--in_fns', type=str, help='SPCAM file names, * is allowed.')
    p.add('--out_dir', type=str, help='Directory where processed files will be stored.')
    p.add('--out_fn', type=str, help='Name of processed file.')

    # For shuffling
    p.add('--shuffle', dest='shuffle', action='store_true', help='Shuffle data along sample dimension.')
    p.add('--chunk_size', type=int, default=10_000_000, help='Chunk size for shuffling/')

    # For potential validation file
    p.add('--val_in_fns', type=str, default=None, help='Validation: SPCAM file names, * is allowed.')
    p.add('--val_out_fn', type=str, default=None, help='Validation: Name of processed file.')

    args = p.parse_args()
    main(args)


