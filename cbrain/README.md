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
python preprocess_aqua.py --config_file ../config/full_physics_essentials.yml --in_dir /beegfs/DATA/pritchard/srasp/Aquaplanet_enhance05/ --aqua_names AndKua_aqua_SPCAM3.0_enhance05.cam2.h1.0000-01-* --out_dir /beegfs/DATA/pritchard/srasp/preprocessed_data/ --out_pref full_physics_essentials_train_month01
```

Which will produce the files out_pref + features.nc, + targets.nc and + norm.nc in the specified out_dir.


#### Process the validation file with the training normalization file

```commandline
python preprocess_aqua.py --config_file ../config/full_physics_essentials.yml --in_dir /beegfs/DATA/pritchard/srasp/Aquaplanet_enhance05/ --aqua_names AndKua_aqua_SPCAM3.0_enhance05.cam2.h1.0000-02-* --out_dir /beegfs/DATA/pritchard/srasp/prepr0cessed_data/ --out_pref full_physics_essentials_valid_month02 --ext_norm /beegfs/DATA/pritchard/srasp/preprocessed_data/full_physics_essentials_train_month01_norm.nc
```


#### Shuffle the sample dimension in the training file

```commandline
python shuffle_ds.py --method fast --pref /beegfs/DATA/pritchard/srasp/preprocessed_data/full_physics_essentials_train_month01 --chunk_size 10000000
```
