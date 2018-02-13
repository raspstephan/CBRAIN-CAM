# Preprocessing of SPCAM files

This directory contains two Python scripts to preprocess the Aquaplanet raw output files for use in the Keras models.

`preprocess_aqua.py` extracts the variables requested in the config file from the specified Aqua files, computed additional variables and saves the prepared variables in separate files for the features and targets. The variables are already stacked to allow for faster access with dimensions [sample, lev]. The sample dimensions is flattened from time, lat, lon, so that these can be reconstructed. Additionally a normalization file containing mean and standard deviation is either created for the current data or read in externally.

`shuffle_ds.py` randomly shuffles the sample dimensions of the feature and target datasets. It in fact shuffles in chunks if method=fast, but the difference is negligible. Shuffling is crucial for the network optimization.

### Naming conventions for variables

- ?BP = ?AP - ?PHYSTND (or appropriate name) \* dt
- ?_C = ?AP[t-1] - DTV/VD01[t-1] \* dt (if required)
- d?dt_adiabatic = (?BP - ?_C)*dt

### Dependencies

The scripts are designed for Python 3.6 and require the following packages:
- configargparse
- netCDF4
- xarray

Optional (but nice for a reproducible log string):
- gitpython


### Example usage

#### Define input and output variables in config file

```yaml
inputs : [TBP, QBP, VBP, PS, SOLIN]
outputs : [TPHYSTND, PHQ]
```

#### Process the training file and produce a new normalization file

```commandline
python preprocess_aqua.py --config_file ../config/full_physics_essentials.yml --in_dir /project/meteo/w2w/A6/S.Rasp/SP-CAM/Aquaplanet_enhance05/ --aqua_names AndKua_aqua_SPCAM3.0.cam2.h1.0000-01-* --out_dir /local/S.Rasp/cbrain_data/ --out_pref full_physics_essentials_train_month01
```




First preprocess the aquaplanet files using the following config.yml
```yaml
vars: [TAP, QAP, OMEGA, SHFLX, LHFLX, LAT, dTdt_adiabatic, dQdt_adiabatic, QRL, QRS, SPDT, SPDQ, PS]   # All variables, feature and target
current_vars: [SPDT, SPDQ, OMEGA]          # Variables to take from current time step
```
and this python command
```commandline
python preprocess_aqua.py --in_dir /project/meteo/w2w/A6/S.Rasp/SP-CAM/Aquaplanet/ --out_dir /project/meteo/w2w/A6/S.Rasp/SP-CAM/preprocessed_data/detailed_files/ --aqua_pref AndKua_aqua_SPCAM3.0.cam2.h1.0000
```
which creates these files in `out_dir`: `SPCAM_outputs_detailed.nc`, `SPCAM_mean_detailed.nc` and `SPCAM_std_detailed.nc`.

Then to create a flattened train/valid split by longitude we execute:
```commandline
python train_valid_split_and_flatten.py --full_fn /project/meteo/w2w/A6/S.Rasp/SP-CAM/preprocessed_data/detailed_files/SPCAM_outputs_detailed.nc --out_dir /project/meteo/w2w/A6/S.Rasp/SP-CAM/preprocessed_data/detailed_files/ --out_pref SPCAM_outputs --train_fraction 0.8 --split_by_lon --flatten --delete_intermediate
```
which creates these files: `SPCAM_outputs_train_by_lon_flat.nc`  and  `SPCAM_outputs_valid_by_lon_flat.nc`. Or we could create a randomly split train and valid set like this:

```commandline
python train_valid_split_and_flatten.py --full_fn /project/meteo/w2w/A6/S.Rasp/SP-CAM/preprocessed_data/detailed_files/SPCAM_outputs_detailed.nc --out_dir /project/meteo/w2w/A6/S.Rasp/SP-CAM/preprocessed_data/detailed_files/ --out_pref SPCAM_outputs --train_fraction 0.8 --flatten
```
which creates these files: `SPCAM_outputs_flat_train_random.nc`, `SPCAM_outputs_flat_valid_random.nc` and also `SPCAM_outputs_flat.nc` because we left out `--delete_intermediate`. This last file is identical to the `SPCAM_outputs.nc` file produced by Pierre, except for the variables in them. 


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
| `--flat_fn` | SPCAM_outputs_flat.nc | Filename of flat file. |
| `--verbose` | | If given, print debugging information. |

### Config file

The variables to be preprocessed are listed in the config file. `vars` contains all files. If the data for the current time step needs to be taken (target variables, dynamics), these need to be listed in `current_vars`:

```
vars: [TAP, QAP, OMEGA, SHFLX, LHFLX, LAT, dTdt_adiabatic, dQdt_adiabatic, QRL, QRS, SPDT, SPDQ]   # All variables, feature and target
current_vars: [SPDT, SPDQ, OMEGA]          # Variables to take from current time step
```

Note that all variables are case-sensitive. `LAT` is the latitude repeated along all other dimensions.


## train_valid_split_and_flatten.py

### Command line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
|`--full_fn`| | Path of full dataset produced by `preprocess_aqua.py` |
|`--out_dir`| | Directory to store output file in. |
|`--out_pref` | SPCAM_outputs | Prefix for output files. |
|`--train_fraction` | | Fraction of data in training set. If train_fraction = 1, no split is performed. |
|`--split_by_lon` | | If given, Split the data by longitude. |
|`--flatten` | | If given: flatten time, lat and lon in a separate file. NOTE: Twice the memory! |
|`--flatten` | | If given: Delete intermediate files. |
| `--verbose` | | If given, print debugging information. |