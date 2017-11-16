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


class DataGenerator(object):
    """Generate batches
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    Note that this function loads the entire dataset into RAM.
    This can be quite memory intensive!
    """

    def __init__(self, data_dir, out_name, batch_size, feature_names,
                 target_names=['SPDT', 'SPDQ'],
                 shuffle_mode='batches',
                 convolution=False):
        """Initialize generator
        If shuffle_mode is "batches", only the order of the batches will be shuffled.
        If it is something else, everything will be shuffled, but this is much much much
        slower!
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle_mode = shuffle_mode
        self.feature_names = feature_names
        assert target_names == ['SPDT', 'SPDQ'], 'No other targets implemented.'
        self.target_names = target_names
        self.convolution = convolution

        # Open files
        self.out_fn = data_dir + out_name
        self.mean_fn = data_dir + 'SPCAM_mean.nc'
        self.std_fn = data_dir + 'SPCAM_std.nc'

        # Load features
        self.features = self.__get_features()
        self.targets = self.__get_targets()

        # Determine sizes

        self.n_batches = int(self.n_samples / batch_size)

        # Create ID list
        if self.shuffle_mode == 'batches':
            self.idxs = np.arange(0, self.n_samples, self.batch_size)
        else:
            self.idxs = np.arange(self.n_samples)

    def generate(self, shuffle=True):
        """Generate data batches
        """
        gen_idxs = np.copy(self.idxs)
        if shuffle:
            np.random.shuffle(gen_idxs)

        while True:
            for i in range(self.n_batches):
                if self.shuffle_mode == 'batches':
                    batch_idx = gen_idxs[i]
                    if self.convolution:
                        x = [
                            self.features[0][batch_idx:batch_idx + self.batch_size],
                            self.features[1][batch_idx:batch_idx + self.batch_size]
                        ]
                    else:
                        x = self.features[batch_idx:batch_idx + self.batch_size]
                    y = self.targets[batch_idx:batch_idx + self.batch_size]
                else:
                    batch_idx = [self.idxs[k] for k in
                        self.idxs[i * self.batch_size:(i + 1) * self.batch_size]]
                    if self.convolution:
                        x = [
                            self.features[0][batch_idx],
                            self.features[1][batch_idx]
                        ]
                    else:
                        x = self.features[batch_idx]
                    y = self.targets[batch_idx]

                yield x, y

    def __get_features(self):
        """Load and scale the features
        """
        # Load features
        with h5py.File(self.out_fn, 'r') as out_file, \
             h5py.File(self.mean_fn, 'r') as mean_file, \
             h5py.File(self.std_fn, 'r') as std_file:

            # Get total number of samples
            self.n_samples = out_file['TAP'].shape[1]

            f_list = []
            for v in self.feature_names:
                f = np.atleast_2d(out_file[v][:]).T
                # normalize
                f = (f - mean_file[v].value) / std_file[v].value
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

    def __get_targets(self):
        """Load and convert the targets
        """
        with h5py.File(self.out_fn, 'r') as out_file:
            targets = np.concatenate([
                out_file['SPDT'][:] * 1000.,
                out_file['SPDQ'][:] * 2.5e6,
            ], axis=0)
            return targets.T
