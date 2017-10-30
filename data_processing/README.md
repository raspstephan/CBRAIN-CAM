# Preprocessing

The scripts in this directory preprocess the raw aquaplanet NetCDF files to a handy format for the neural networks.

## preprocess_aqua.py

### Command line arguments:

| Argument | Default | Description |
|----------|---------|-------------|
|`--config_file`| | Name of config file in this directory. Must contain in and out variable lists. Described in detail below. |
|`--in_vars`| | Variables for neural net input. |
| `--out_vars` | | Variables for neural net output. |
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

The variables to be preprocessed are listed in the config file. Input variables (features) and output variables (targets) of the neural net have to be listed separately. Here is an example:

```
in_vars: [TAP, QAP, OMEGA, SHFLX, LHFLX, LAT, dTdt_adiabatic, dQdt_adiabatic, QRL, QRS]   # Inputs/features
out_vars: [SPDT, SPDQ]          # Outputs/targets
```

Note that all variables are case-sensitive. `LAT` is the latitude repeated along all other dimensions.
