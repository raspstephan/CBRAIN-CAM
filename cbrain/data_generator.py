"""
Data generator class.

Created on 2019-01-28-10-39
Author: Stephan Rasp, raspstephan@gmail.com
"""

from .imports import *
from .utils import *
from .normalization import *


class DataGenerator(tf.keras.utils.Sequence):
    """
    https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

    Data generator class.
    """

    def __init__(self, data_fn, input_vars, output_vars,
                 norm_fn=None, input_transform=None, output_transform=None,
                 batch_size=1024, shuffle=True, xarray=False, var_cut_off=None):
        # Just copy over the attributes
        self.data_fn, self.norm_fn = data_fn, norm_fn
        self.input_vars, self.output_vars = input_vars, output_vars
        self.batch_size, self.shuffle = batch_size, shuffle

        # Open datasets
        self.data_ds = xr.open_dataset(data_fn)
        if norm_fn is not None: self.norm_ds = xr.open_dataset(norm_fn)

        # Compute number of samples and batches
        self.n_samples = self.data_ds.vars.shape[0]
        self.n_batches = int(np.floor(self.n_samples) / self.batch_size)

        # Get input and output variable indices
        self.input_idxs = return_var_idxs(self.data_ds, input_vars, var_cut_off)
        self.output_idxs = return_var_idxs(self.data_ds, output_vars)
        self.n_inputs, self.n_outputs = len(self.input_idxs), len(self.output_idxs)

        # Initialize input and output normalizers/transformers
        if input_transform is None:
            self.input_transform = Normalizer()
        elif type(input_transform) is tuple:
            self.input_transform = InputNormalizer(
                self.norm_ds, input_vars, input_transform[0], input_transform[1], var_cut_off)
        else:
            self.input_transform = input_transform  # Assume an initialized normalizer is passed

        if output_transform is None:
            self.output_transform = Normalizer()
        elif type(output_transform) is dict:
            self.output_transform = DictNormalizer(self.norm_ds, output_vars, output_transform)
        else:
            self.output_transform = output_transform  # Assume an initialized normalizer is passed

        # Now close the xarray file and load it as an h5 file instead
        # This significantly speeds up the reading of the data...
        if not xarray:
            self.data_ds.close()
            self.data_ds = h5py.File(data_fn, 'r')

    def __len__(self):
        return self.n_batches

    def __getitem__(self, index):
        # Compute start and end indices for batch
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size

        # Grab batch from data
        batch = self.data_ds['vars'][start_idx:end_idx]

        # Split into inputs and outputs
        X = batch[:, self.input_idxs]
        Y = batch[:, self.output_idxs]

        # Normalize
        X = self.input_transform.transform(X)
        Y = self.output_transform.transform(Y)

        return X, Y

    def on_epoch_end(self):
        self.indices = np.arange(self.n_batches)
        if self.shuffle: np.random.shuffle(self.indices)

