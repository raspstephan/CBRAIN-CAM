"""Define DataGenerator class

Author: Stephan Rasp

"""

import numpy as np
import h5py
import pdb
import threading


# To make generators thread safe for multithreading
def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


class threadsafe_iter(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    https://github.com/fchollet/keras/issues/1638
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):   # Py3
        with self.lock:
            return next(self.it)


@threadsafe_generator
def data_generator(data_dir, feature_fn, target_fn, shuffle=True,
                   batch_size=512, feature_norms=None, target_norms=None, noise=None):
    """Works on pre-stacked targets with truely random batches
    """
    # Open files
    feature_file = h5py.File(data_dir + feature_fn, 'r')
    target_file = h5py.File(data_dir + target_fn, 'r')

    # Determine sizes
    n_samples = feature_file['features'].shape[0]
    n_batches = int(np.floor(n_samples / batch_size))
    # Create ID list
    idxs = np.arange(0, n_samples, batch_size)
    if shuffle:
        np.random.shuffle(idxs)

    # generate
    while True:
        for i in range(n_batches):
            batch_idx = idxs[i]
            x = feature_file['features'][batch_idx:batch_idx + batch_size, :]
            y = target_file['targets'][batch_idx:batch_idx + batch_size, :]
            if feature_norms is not None: x = (x - feature_norms[0]) / feature_norms[1]
            if target_norms is not None: y = (y - target_norms[0]) * target_norms[1]
            if noise is not None:
                x += np.random.normal(0, noise, x.shape)
            yield x, y

@threadsafe_generator
def data_generator_convo(data_dir, feature_fn, target_fn, shuffle=True,
                         batch_size=512, feature_norms=None, target_norms=None, noise=None,
                         tile=False):
    """Works on pre-stacked targets with truely random batches
    Hard coded right now for
    features = [TBP, QBP, VBP, PS, SOLIN, SHFLX, LHFLX]
    and lev = 30
    """
    # Open files
    feature_file = h5py.File(data_dir + feature_fn, 'r')
    target_file = h5py.File(data_dir + target_fn, 'r')

    # Determine sizes
    n_samples = feature_file['features'].shape[0]
    n_batches = int(np.floor(n_samples / batch_size))
    # Create ID list
    idxs = np.arange(0, n_samples, batch_size)
    if shuffle:
        np.random.shuffle(idxs)

    # generate
    while True:
        for i in range(n_batches):
            batch_idx = idxs[i]
            x = feature_file['features'][batch_idx:batch_idx + batch_size, :]
            if feature_norms is not None: x = (x - feature_norms[0]) / feature_norms[1]
            if tile:
                x = np.concatenate(
                    [
                        x[:, :90].reshape((x.shape[0], 30, -1)),
                        np.rollaxis(np.tile(x[:, 90:], (30, 1, 1)), 0, 2)
                    ],
                    axis=-1,
                )
            else:
                x1 = x[:, :90].reshape((x.shape[0], 30, -1))
                x2 = x[:, 90:]
                x = [x1, x2]
            y = target_file['targets'][batch_idx:batch_idx + batch_size, :]
            if target_norms is not None: y = (y - target_norms[0]) * target_norms[1]
            yield x, y


class DataGenerator(object):
    """Class wrapper around data_generator function
    """

    def __init__(self, data_dir, feature_fn, target_fn, batch_size, norm_fn=None,
                 fsub=None, fdiv=None, tsub=None, tmult=None, shuffle=True, verbose=True,
                 noise=None):
        """Initialize DataGenerator object

        Args:
            data_dir: Path to directory with feature and target files
            feature_fn: Feature file name
            target_fn: Target file name
            batch_size: batch size
            shuffle: Shuffle batches
        """
        self.data_dir = data_dir
        self.feature_fn = feature_fn
        self.target_fn = target_fn
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature_norms = None
        self.target_norms = None
        self.noise = noise

        # Determine n_batches and shapes
        with h5py.File(data_dir + feature_fn, 'r') as feature_file:
            n_samples = feature_file['features'].shape[0]
            self.feature_shape = feature_file['features'].shape[1]
        self.n_batches = int(np.floor(n_samples / batch_size))
        with h5py.File(data_dir + target_fn, 'r') as target_file:
            self.target_shape = target_file['targets'].shape[1]
        if fsub is not None or fdiv is not None:
            self.feature_norms = [0., 1.]   # This does nothing...
            with h5py.File(data_dir + norm_fn, 'r') as norm_file:
                if fsub is not None: self.feature_norms[0] = norm_file[fsub][:]
                if fdiv is not None:
                    if fdiv == 'range':
                        self.feature_norms[1] = (norm_file['feature_maxs'][:] -
                                                 norm_file['feature_mins'][:])
                    elif fdiv == 'max_rs':  # Max range, std_by_var
                        self.feature_norms[1] = np.maximum(
                            norm_file['feature_maxs'][:] - norm_file['feature_mins'][:],
                            norm_file['feature_stds_by_var']
                        )
                    elif fdiv == 'feature_stds_eps':
                        eps = 1e-10
                        self.feature_norms[1] = np.maximum(norm_file['feature_stds'][:], eps)
                    else:
                        self.feature_norms[1] = norm_file[fdiv][:]
        if tsub is not None or tmult is not None:
            self.target_norms = [0., 1.]   # This does nothing...
            with h5py.File(data_dir + norm_fn, 'r') as norm_file:
                if tsub is not None: self.target_norms[0] = norm_file[tsub][:]
                if tmult is not None: self.target_norms[1] = norm_file[tmult][:]
        if verbose:
            print('Generator will have %i samples in %i batches' %
                  (n_samples, self.n_batches))
            print('Features have shape %i; targets have shape %i' %
                  (self.feature_shape, self.target_shape))

    def return_generator(self, convo=False, tile=False):
        """Return data_generator

        Returns: data_generator
        """
        if convo:
            return data_generator_convo(
                self.data_dir,
                self.feature_fn,
                self.target_fn,
                self.shuffle,
                self.batch_size,
                self.feature_norms,
                self.target_norms,
                self.noise,
                tile
            )
        else:
            return data_generator(
                self.data_dir,
                self.feature_fn,
                self.target_fn,
                self.shuffle,
                self.batch_size,
                self.feature_norms,
                self.target_norms,
                self.noise
            )