"""Define DataGenerator class

Author: Stephan Rasp

TODO:
- Add option to read detailed data and then flatten
    - all
- Add option to create spatial input
- Add option to add timeseries input
"""

import numpy as np
import h5py
import pdb
import threading


# Define conversion dict
L_V = 2.5e6   # Latent heat of vaporization is actually 2.26e6
C_P = 1e3 # Specific heat capacity of air at constant pressure
conversion_dict = {
    'SPDT': C_P,
    'SPDQ': L_V,
    'QRL': C_P,
    'QRS': C_P,
    'PRECT': 1e3*24*3600 * 1e-3,
    'FLUT': 1. * 1e-5,
}


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

    def __next__(self): # Py3
        with self.lock:
            return next(self.it)




class DataSet(object):
    """Gets a dataset of train data.
    """

    def __init__(self, data_dir, out_fn, mean_fn, std_fn, feature_names,
                 target_names=['SPDT', 'SPDQ'], convolution=False,
                 dtype='float32', flat_input=False, mean_std_dir='same',
                 target_norm=False, target_norm_lev_weight=False):
        """
        Initialize dataset

        Args:
            data_dir: directory where data is stored
            out_fn: filename of outputs file
            mean_fn: filename of mean file
            std_fn: filename of std file
            target_names: target variable names
            convolution: get data with channels
            dtype: numpy precision
            flat_input: If true, assumes already flattened input array
            mean_std_dir: Directory for mean and std data. Default = 'same'
            target_norm: If given, normalize outputs
            target_norm_lev_weight: If given, weigh target norm by levels
        """
        # File names
        self.data_dir = data_dir
        self.out_fn = data_dir + out_fn
        if mean_std_dir == 'same':
            mean_std_dir = data_dir
        self.mean_fn = mean_std_dir + mean_fn
        self.std_fn = mean_std_dir + std_fn

        # Other settings
        self.convolution = convolution
        self.feature_names = feature_names
        self.target_names = target_names
        self.dtype = dtype
        self.target_norm = target_norm
        self.target_norm_lev_weight = target_norm_lev_weight

        # Load data
        self.features = self.__get_features(flat_input)
        self.targets = self.__get_targets(flat_input)

    # These functions are copied from the data generator function
    def __get_features(self, flat_input=False):
        """Load and scale the features
        """
        # Load features
        with h5py.File(self.out_fn, 'r') as out_file, \
                h5py.File(self.mean_fn, 'r') as mean_file, \
                h5py.File(self.std_fn, 'r') as std_file:

            # Get total number of samples
            if flat_input:
                self.n_samples = out_file['LAT'].shape[0]
            else:
                self.n_samples = np.prod(out_file['LAT'].shape[:])

            f_list = []
            for v in self.feature_names:
                if flat_input:
                    f = np.atleast_2d(out_file[v][:]).T
                else:
                    f = out_file[v][:]
                    # Either [date,time,lat,lon] or [date,time,lev,lat,lon]
                    # convert to [sample, 1] or [sample, lev]
                    if f.ndim == 4:
                        f = f.reshape((self.n_samples, 1))
                    elif f.ndim == 5:
                        f = np.rollaxis(f, 2, 5)
                        f = f.reshape((self.n_samples, -1))
                # normalize
                f = (f - mean_file[v].value) / std_file[v].value
                # Adjust data type
                f = np.asarray(f, dtype=self.dtype)
                f_list.append(f)
            if self.convolution:
                f_2d = [f for f in f_list if f.shape[1] > 1]
                f_1d = [f for f in f_list if f.shape[1] == 1]
                f_2d = np.stack(f_2d, axis=-1)
                # I do not think this is necessary!
                # f_2d = np.reshape(
                #     f_2d, (f_2d.shape[0], f_2d.shape[1], 1, f_2d.shape[2])
                # )
                f_1d = np.concatenate(f_1d, axis=1)
                return [f_2d, f_1d]
                # [sample, z, feature]
            else:
                return np.concatenate(f_list, axis=1)
                # [sample, flattened features]

    def __get_targets(self, flat_input):
        """Load and convert the targets [sample, target dim]
        """
        with h5py.File(self.out_fn, 'r') as out_file, \
                h5py.File(self.mean_fn, 'r') as mean_file, \
                h5py.File(self.std_fn, 'r') as std_file:
            # [date,time,lev,lat,lon]
            if flat_input:
                t_list = []
                if self.target_norm:
                    # Store converted means and std's to reconvert them later
                    cm_list = []
                    cs_list = []
                for v in self.target_names:
                    t = np.atleast_2d(out_file[v][:] * conversion_dict[v]).T
                    if self.target_norm:
                        conv_mean = mean_file[v].value * conversion_dict[v]
                        conv_std = std_file[v].value * conversion_dict[v]
                        if self.target_norm_lev_weight:
                            conv_std = (conv_std *
                                        np.atleast_1d(conv_std).shape[0])
                        cm_list.append(np.atleast_1d(conv_mean))
                        cs_list.append(np.atleast_1d(conv_std))
                        t = (t - conv_mean) / conv_std
                        # t = t / conv_std
                    t_list.append(t)
                targets = np.concatenate(t_list, axis=1)
                if self.target_norm:
                    self.target_mean = np.concatenate(cm_list, axis=0)
                    self.target_std = np.concatenate(cs_list, axis=0)
            else:
                assert self.target_names == ['SPDT', 'SPDQ'], 'Not implemented.'
                # [date,time,lev,lat,lon]
                t_list = []
                for var, fac in zip(['SPDT', 'SPDQ'], [1000., 2.5e6]):
                    t = out_file[var][:] * fac
                    t = np.rollaxis(t, 2, 5)
                    t_list.append(t.reshape((self.n_samples, -1)))
                targets = np.concatenate(t_list, axis=1)
            return np.asarray(targets, dtype=self.dtype)

    def renorm_outputs(self, x):
        """If targets and preds are normalized, reconvert them
        """
        assert self.target_norm, 'Only for normalized targets'

        return x * self.target_std + self.target_mean
        # return x * self.target_std


def get_n_batches(data_dir, out_fn, batch_size=512):
    with h5py.File(data_dir + out_fn) as out_file:
        # Determine sizes
        n_samples = out_file['TAP'].shape[1]
        n_batches = int(n_samples / batch_size)
        print(n_samples, n_batches)
    return n_batches

@threadsafe_generator
def data_generator1(data_dir, out_fn, mean_fn, std_fn, feature_names,
                    target_names=['SPDT', 'SPDQ'], shuffle=True,
                    batch_size=512):
    # Open files
    out_file = h5py.File(data_dir + out_fn)
    mean_file = h5py.File(data_dir + mean_fn)
    std_file = h5py.File(data_dir + std_fn)

    # Determine sizes
    n_samples = out_file['TAP'].shape[1]
    n_batches = int(n_samples / batch_size)

    # Create ID list
    idxs = np.arange(0, n_samples, batch_size)
    if shuffle:
        np.random.shuffle(idxs)

    # generate
    while True:
        for i in range(n_batches):
            batch_idx = idxs[i]
            # x
            f_list = []
            for v in feature_names:
                if out_file[v].ndim == 2:
                    f = out_file[v][:, batch_idx:batch_idx + batch_size].T
                elif out_file[v].ndim == 1:
                    f = np.atleast_2d(
                        out_file[v][batch_idx:batch_idx + batch_size]).T
                else:
                    raise ValueError('Wrong feature dimensions.')
                # normalize
                f = (f - mean_file[v].value) / std_file[v].value
                f_list.append(f)
            x = np.concatenate(f_list, axis=1)

            # y
            t_list = []
            for v in target_names:
                if out_file[v].ndim == 2:
                    t = out_file[v][:, batch_idx:batch_idx + batch_size].T
                elif out_file[v].ndim == 1:
                    t = np.atleast_2d(
                        out_file[v][batch_idx:batch_idx + batch_size]).T
                t = t * conversion_dict[v]
                t_list.append(t)
            y = np.concatenate(t_list, axis=1)
            yield x, y



import multiprocessing
def DataGenerator():
    """New multiprocessing version
    https://gist.github.com/tdeboissiere/195dde7fddfcf622a82a895b90d2c800
    """
    try:
        queue = multiprocessing.Queue(maxsize=max_q_size)

        # define producer (putting items into queue)
        def producer():

            try:
                # Load the data (not in memory)
                arr = np.load("data_dummy.npz")

                batch_index = np.random.randint(0, n_samples - batch_size)
                start = batch_index
                end = start + batch_size
                X = arr["data"][start: end]
                y = arr["labels"][start: end]
                y = np_utils.to_categorical(y, nb_classes=2)
                # Put the data in a queue
                queue.put((X, y))

            except:
                print("Nothing here")

        processes = []

        def start_process():
            for i in range(len(processes), maxproc):
                thread = multiprocessing.Process(target=producer)
                time.sleep(0.01)
                thread.start()
                processes.append(thread)

        # run as consumer (read items from queue, in current thread)
        while True:
            processes = [p for p in processes if p.is_alive()]

            if len(processes) < maxproc:
                start_process()

            yield queue.get()

    except:
        print("Finishing")
        for th in processes:
            th.terminate()
        queue.close()
        raise



class oldDataGenerator(object):
    """Generate batches
    """

    def __init__(self, data_dir, out_name, batch_size, feature_names,
                 target_names=['SPDT', 'SPDQ'], shuffle=True):
        """Initialize generator
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature_names = feature_names
        assert target_names == ['SPDT', 'SPDQ'], 'No other targets implemented.'
        self.target_names = target_names

        # Open files
        self.out_file = h5py.File(data_dir + out_name)
        self.mean_file = h5py.File(data_dir + 'SPCAM_mean.nc')
        self.std_file = h5py.File(data_dir + 'SPCAM_std.nc')

        # Determine sizes
        self.n_samples = self.out_file['TAP'].shape[1]
        self.n_batches = int(self.n_samples / batch_size)

        # Create ID list
        idxs = np.arange(0, self.n_samples, self.batch_size)
        if self.shuffle:
            np.random.shuffle(idxs)
        self.idxs = idxs

    def generate(self):
        """Generate data batches
        """
        while True:
            for i in range(self.n_batches):
                batch_idx = self.idxs[i]
                x = self.__get_features(batch_idx)
                y = self.__get_targets(batch_idx)
                yield x, y

    def __get_features(self, batch_idx):
        """Load and scale the features
        """
        # Load features
        f_list = []
        for v in self.feature_names:
            # NOTE to self: This below is much (factor 5) faster than
            # self.out_file[v].value[:,...]
            if self.out_file[v].ndim == 2:
                f = self.out_file[v][:, batch_idx:batch_idx + self.batch_size].T
            elif self.out_file[v].ndim == 1:
                f = np.atleast_2d(
                    self.out_file[v][batch_idx:batch_idx + self.batch_size]).T
            else:
                raise ValueError('Wrong feature dimensions.')
            # normalize
            f = (f - self.mean_file[v].value) / self.std_file[v].value
            f_list.append(f)
        return np.concatenate(f_list, axis=1)

    def __get_targets(self, batch_idx):
        """Load and convert the targets
        """
        targets = np.concatenate([
            self.out_file['SPDT'][:,
            batch_idx:batch_idx + self.batch_size] * 1000.,
            self.out_file['SPDQ'][:,
            batch_idx:batch_idx + self.batch_size] * 2.5e6,
        ], axis=0)
        return targets.T

@threadsafe_generator
def test_data_generator(data_dir, out_name, batch_size, feature_names,
                        target_names=['SPDT', 'SPDQ'], shuffle=True):
    # Open files
    out_file = h5py.File(data_dir + out_name)
    mean_file = h5py.File(data_dir + 'SPCAM_mean.nc')
    std_file = h5py.File(data_dir + 'SPCAM_std.nc')

    # Determine sizes
    n_samples = out_file['TAP'].shape[1]
    n_batches = int(n_samples / batch_size)

    # Create ID list
    idxs = np.arange(0, n_samples, batch_size)
    if shuffle:
        np.random.shuffle(idxs)

    # generate
    while True:
        for i in range(n_batches):
            batch_idx = idxs[i]
            # x
            f_list = []
            for v in feature_names:
                # NOTE to self: This below is much (factor 5) faster than
                # self.out_file[v].value[:,...]
                if out_file[v].ndim == 2:
                    f = out_file[v][:, batch_idx:batch_idx + batch_size].T
                elif out_file[v].ndim == 1:
                    f = np.atleast_2d(
                        out_file[v][batch_idx:batch_idx + batch_size]).T
                else:
                    raise ValueError('Wrong feature dimensions.')
                # normalize
                f = (f - mean_file[v].value) / std_file[v].value
                f_list.append(f)
            x = np.concatenate(f_list, axis=1)

            # y
            targets = np.concatenate([
                out_file['SPDT'][:, batch_idx:batch_idx + batch_size] * 1000.,
                out_file['SPDQ'][:, batch_idx:batch_idx + batch_size] * 2.5e6,
            ], axis=0)
            y = targets.T
            yield x, y