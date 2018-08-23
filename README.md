# CBRAIN-CAM - a neural network climate model parameterization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1402384.svg)](https://doi.org/10.5281/zenodo.1402384)

Author: Stephan Rasp - <raspstephan@gmail.com> - https://raspstephan.github.io

This is my working directory for the CBRAIN-CAM project. It contains all code used to preprocess the raw climate model data, run the neural networks and analyze the results.

The modified climate model code is available at https://gitlab.com/mspritch/spcam3.0-neural-net (branch: `nn_fbp_engy_ess`)

## Papers and code

### Climate model parameterization paper

The second paper with the prognostic climate simulations is available as a preprint:
> S. Rasp, M. Pritchard and P. Gentine, 2018.
> Deep learning to represent sub-grid processes in climate models
> https://arxiv.org/abs/1806.04731

For a snapshot of the repository as it was for the GRL paper, see release [paper2_submission](https://github.com/raspstephan/CBRAIN-CAM/releases/tag/paper2_submission). All figures for the paper were produced in [this Jupyter notebook](https://github.com/raspstephan/CBRAIN-CAM/blob/master/notebooks/presentation/paper2.ipynb)


### GRL paper

The first paper showing offline performance has been published: 
> P. Gentine, M. Pritchard, S. Rasp, G. Reinaudi and G. Yacalis, 2018. 
> Could machine learning break the convection parameterization deadlock? 
> Geophysical Research Letters. http://doi.wiley.com/10.1029/2018GL078202

For a snapshot of the repository as it was for the GRL paper, see release [grl_submission](https://github.com/raspstephan/CBRAIN-CAM/releases/tag/grl_submission). All figures for the paper were produced in [this Jupyter notebook](https://github.com/raspstephan/CBRAIN-CAM/blob/master/notebooks/presentation/grl_paper.ipynb)

## Repository description

The main components of the repository are:

- `cbrain`: Contains the cbrain module with all code to preprocess the raw data, run the neural network experiments and analyze the data.
- `pp_config`: Contains configuration files and shell scripts to preprocess the climate model data to be used as neural network inputs
- `nn_config`: Contains neural network configuration files to be used with `run_experiment.py`.
- `notebooks`: Contains Jupyter notebooks used to analyze data. All plotting and data analysis for the papers is done in the subfolder `presentation`. `dev` contains development notebooks.
- `wkspectra`: Contains code to compute Wheeler-Kiladis figures. These were created by [Mike S. Pritchard](http://sites.uci.edu/pritchard/)
- `save_weights.py`: Saves the weights, biases and normalization vectors in text files. These are then used as input for the climate model.

