# CBRAIN-CAM - a neural network climate model parameterization

Author: Stephan Rasp - <raspstephan@gmail.com> - https://raspstephan.github.io

Hi, thanks for checking out this repository. It contains the code that was used for Rasp et al. 2018 and serves as the basis for ongoing work. In particular, check out the [`climate_invariant`](https://github.com/raspstephan/CBRAIN-CAM/tree/climate_invariant) branch for [Tom Beucler's](http://tbeucler.scripts.mit.edu/tbeucler/) work on physically consistent ML parameterizations.

If you are looking for the exact version of the code that corresponds to the PNAS paper, check out this release: https://github.com/raspstephan/CBRAIN-CAM/releases/tag/PNAS_final [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1402384.svg)](https://doi.org/10.5281/zenodo.1402384)

For a sample of the SPCAM data used, click here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2559313.svg)](https://doi.org/10.5281/zenodo.2559313)


The modified climate model code is available at https://gitlab.com/mspritch/spcam3.0-neural-net. 
To create the training data (regular SPCAM) the correct branch is `fluxbypass`. To implement the trained neural network, check out `revision`.

### Papers

> T. Beucler, M. Pritchard, P. Gentine and S. Rasp, 2020.
> Towards Physically-consistent, Data-driven Models of Convection.
> https://arxiv.org/abs/2002.08525

> T. Beucler, M. Pritchard, S. Rasp, P. Gentine, J. Ott and P. Baldi, 2019.
> Enforcing Analytic Constraints in Neural-Networks Emulating Physical Systems.
> https://arxiv.org/abs/1909.00912

> S. Rasp, M. Pritchard and P. Gentine, 2018.
> Deep learning to represent sub-grid processes in climate models.
> PNAS. https://doi.org/10.1073/pnas.1810286115
 
> P. Gentine, M. Pritchard, S. Rasp, G. Reinaudi and G. Yacalis, 2018. 
> Could machine learning break the convection parameterization deadlock? 
> Geophysical Research Letters. http://doi.wiley.com/10.1029/2018GL078202


## Repository description

The main components of the repository are:

- `cbrain`: Contains the cbrain module with all code to preprocess the raw data, run the neural network experiments and analyze the data.
- `pp_config`: Contains configuration files and shell scripts to preprocess the climate model data to be used as neural network inputs
- `nn_config`: Contains neural network configuration files to be used with `run_experiment.py`.
- `notebooks`: Contains Jupyter notebooks used to analyze data. All plotting and data analysis for the papers is done in the subfolder `presentation`. `dev` contains development notebooks.
- `wkspectra`: Contains code to compute Wheeler-Kiladis figures. These were created by [Mike S. Pritchard](http://sites.uci.edu/pritchard/)
- `save_weights.py`: Saves the weights, biases and normalization vectors in text files. These are then used as input for the climate model.

