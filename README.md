# CBRAIN-CAM - a neural network climate model parameterization

Author: Stephan Rasp - <raspstephan@gmail.com> - https://raspstephan.github.io

Hi, thanks for checking out this repository. This is a working repository, which means that the most corrent commit might not always be the most functional or documented. 

**A Guide for collaborators**
People hoping to collaborate with me on this project, please check out some guidelines here: https://github.com/raspstephan/CBRAIN-CAM/wiki/A-guide-for-collaborators

If you are looking for the version of the code that corresponds to the PNAS paper. Check out this release: https://github.com/raspstephan/CBRAIN-CAM/releases/tag/PNAS_final

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1402384.svg)](https://doi.org/10.5281/zenodo.1402384)

The modified climate model code is available at https://gitlab.com/mspritch/spcam3.0-neural-net. 
To create the training data (regular SPCAM) the correct branch is `fluxbypass`. To implement the trained neural network, check out `revision`.

### Papers

> S. Rasp, M. Pritchard and P. Gentine, 2018.
> Deep learning to represent sub-grid processes in climate models
> https://arxiv.org/abs/1806.04731
 
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

