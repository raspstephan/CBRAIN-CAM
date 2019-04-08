"""
Shuffle a saved dataset.

Created on 2019-01-23-16-03
Author: Stephan Rasp, raspstephan@gmail.com
"""

from ..imports import *

logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG
)


def fast_shuffle(orig_ds, shuffle_ds, chunk_size):
    # Try shuffling in batches that fit into RAM
    n_samples = orig_ds.dimensions['sample'].size
    n_chunks = int(np.ceil(n_samples / float(chunk_size)))

    for i in tqdm(range(n_chunks)):
        start_idx = i * chunk_size
        stop_idx = np.min([(i+1) * chunk_size, n_samples])
        rand_idxs = np.arange(stop_idx - start_idx)
        np.random.shuffle(rand_idxs)

        chunk = orig_ds.variables['vars'][start_idx:stop_idx]
        chunk = chunk[rand_idxs]
        shuffle_ds.variables['vars'][start_idx:stop_idx] = chunk


def shuffle(dir, fn, random_seed=42, chunk_size=10_000_000):
    """

    Parameters
    ----------
    dir: Directory where preprocessed file is stored
    fn: Name of that file in the directory
    random_seed: Random seed
    chunk_size: Chunk is sample dimension that is processed at one time. Has to fit into RAM!

    Returns
    -------

    """

    np.random.seed(random_seed)
    shuffle_fn = fn[:-3] + '_shuffle.nc'
    fn, shuffle_fn = [path.join(dir, f) for f in [fn, shuffle_fn]]

    logging.info(f'Start shuffling {fn} into {shuffle_fn}. Open and create datasets.')
    orig_ds = nc.Dataset(fn, 'r')
    shuffle_ds = nc.Dataset(shuffle_fn, 'w')
    for dim_name, dim in orig_ds.dimensions.items():
        shuffle_ds.createDimension(dim_name, dim.size)
    for var_name, var in orig_ds.variables.items():
        shuffle_ds.createVariable(var_name, var.dtype, var.dimensions)
        if var_name != 'vars':
            shuffle_ds.variables[var_name][:] = var[:]

    logging.info('Shuffle!')
    fast_shuffle(orig_ds, shuffle_ds, chunk_size)

    logging.info('Closing datasets')
    orig_ds.close()
    shuffle_ds.close()

    logging.info('Done!')


if __name__ == '__main__':
    fire.Fire(shuffle)

