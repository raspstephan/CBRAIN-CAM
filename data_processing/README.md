# Preprocessing of SPCAM files

The scripts in this directory preprocess the raw aquaplanet NetCDF files to a handy format for the neural networks. 

`preprocess_aqua.py` extracts the required variables from the raw Aquaplanet files and saves them in one NetCDF file along with a mean and std file for normalization. Note that at the moment the means and stds are computed from the entire dataset before the train valid split. This is technically cheating, but I think should be fine for our purposes. This script preserves the time and space dimensions which might be necessary for later tests with CNNs and RNNs. Dimensions: --> [date, time step, (lev), lat, lon]

`train_valid_split_and_flatten.py` serves two purposes: First, it splits the file created by `preprocess_aqua.py` to create separate training and validation set files. Second, if requested it flattens the time and space dimensions to a simple sample dimension --> [(lev), sample] which can be used for our current simple neural nets. Having the train/valid split permanently is useful because the split is reproducible, thereby making experiments more comparable, and we can chose how we want to split the data. At the moment there are two options: First, a random split (only works if data is also flattened) which simple assigns samples to the train and valid set randomly; and second, a split by longitude. Here, we pick continuous longitude ranges for our train and validation set. So for example, 0 to 284 degrees for the train and 287 to 357 degrees for the validation set. I believe that this could be a fairer validation because otherwise the train and validation set are not really independent. This, of course, only works for the aquaplanet data. Better even would be a split by time, but for this we would need a several year dataset.




## preprocess_aqua.py

### Command line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
|`--config_file`| | Name of config file in this directory. Must contain in and out variable lists. Described in detail below. |
|`--vars`| | All variables. |
| `--current_vars` | | Variables from current time step. |
| `--in_dir` | | Directory with input (aqua) files. |
| `--out_dir` | | Directory to write preprocessed file. |
| `--aqua_pref` | AndKua_aqua_ | Prefix of aqua files. |
| `--out_fn` | SPCAM_outputs_detailed.nc | Filename of preprocessed file. |
| `--mean_fn` | SPCAM_mean_detailed.nc | Filename of mean file. |
| `--std_fn` | SPCAM_std_detailed.nc | Filename of std file. |
| `--min_lev` | 9 | Minimum level index. Python index starting at 0 |
| `--lat_range` | [-90, 90] | Latitude range. |
| `--dtype` | float32 | Datatype of processed variables. |
| `--flatten` | False | If given: flatten date, time, lat and lon to sample in a separate file. NOTE: Twice the memory! |
| `--flat_fn` | SPCAM_outputs_flat.nc | Filename of flat file. |
| `--verbose` | False | If given, print debugging information. |

### Config file

The variables to be preprocessed are listed in the config file. `vars` contains all files. If the data for the current time step needs to be taken (target variables, dynamics), these need to be listed in `current_vars`:

```
vars: [TAP, QAP, OMEGA, SHFLX, LHFLX, LAT, dTdt_adiabatic, dQdt_adiabatic, QRL, QRS, SPDT, SPDQ]   # All variables, feature and target
current_vars: [SPDT, SPDQ, OMEGA]          # Variables to take from current time step
```

Note that all variables are case-sensitive. `LAT` is the latitude repeated along all other dimensions.

