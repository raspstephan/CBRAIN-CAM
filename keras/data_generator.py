"""Define DataGenerator class

Author: Stephan Rasp
"""

import numpy as np
import h5py


class DataGenerator(object):
    """Generate batches
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
    Note that this function loads the entire dataset into RAM.
    This can be quite memory intensive!
    """

    def __init__(self, data_dir, out_name, batch_size, feature_names,
                 target_names=['SPDT', 'SPDQ'], shuffle=True,
                 shuffle_mode='batches',
                 convolution=False):
        """Initialize generator
        If shuffle_mode is "batches", only the order of the batches will be shuffled.
        If it is something else, everything will be shuffled, but this is much much much
        slower!
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_mode = shuffle_mode
        self.feature_names = feature_names
        assert target_names == ['SPDT', 'SPDQ'], 'No other targets implemented.'
        self.target_names = target_names
        self.convolution = convolution

        # Open files
        self.out_file = h5py.File(data_dir + out_name)
        self.mean_file = h5py.File(data_dir + 'SPCAM_mean.nc')
        self.std_file = h5py.File(data_dir + 'SPCAM_std.nc')

        # Load features
        self.features = self.__get_features()
        self.targets = self.__get_targets()

        # Determine sizes
        self.n_samples = self.out_file['TAP'].shape[1]
        self.n_batches = int(self.n_samples / batch_size)

        # Create ID list
        if self.shuffle_mode == 'batches':
            idxs = np.arange(0, self.n_samples, self.batch_size)
        else:
            idxs = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(idxs)
        self.idxs = idxs

    def generate(self):
        """Generate data batches
        """
        while True:
            for i in range(self.n_batches):
                if self.shuffle_mode == 'batches':
                    batch_idx = self.idxs[i]
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
        f_list = []
        for v in self.feature_names:
            f = np.atleast_2d(self.out_file[v][:]).T
            # normalize
            f = (f - self.mean_file[v].value) / self.std_file[v].value
            f_list.append(f)
        if self.convolution:
            f_2d = [f for f in f_list if f.shape[1] > 1]
            f_1d = [f for f in f_list if f.shape[1] == 1]
            f_2d = np.stack(f_2d, axis=-1)
            f_2d = np.reshape(f_2d,
                              (f_2d.shape[0], f_2d.shape[1], 1, f_2d.shape[2]))
            f_1d = np.concatenate(f_1d, axis=1)
            return [f_2d, f_1d]  # [sample, z, feature]
        else:
            return np.concatenate(f_list, axis=1) # [sample, flattened features]

    def __get_targets(self):
        """Load and convert the targets
        """
        targets = np.concatenate([
            self.out_file['SPDT'][:] * 1000.,
            self.out_file['SPDQ'][:] * 2.5e6,
        ], axis=0)
        return targets.T