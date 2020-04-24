# CBRAIN-CAM - a neural network for climate invariant parameterization





## About

This branch consists code for climate invariant cloud convection parameterization.  
Set up the conda environment using the [env.yml](env.yml) file



## Repository description

The main components of the repository are:

- `cbrain`: Contains the cbrain module with all code to preprocess the raw data, run the neural network experiments and analyze the data.

The process of creating a model is as follows.

### Preprocessing

To preprocess the data you the major files are  

**preprocessing.py** and **convert_dataset.py**.
You can check out [this](notebooks/ankitesh-devlog/01_Preprocessing.ipynb) notebook for more information about it.

### Model Training

Once the data is processed we can train the model as a whole or in progression. Below is the architecture of the whole network

Inp -> RH Transformation -> LH Transformation -> T-TNS Transformation -> Split + Scaling -> Vertical Interpolation.

Check out [this](notebooks/ankitesh-devlog/02_Model.ipynb) notebook to know more about training the network.

### Model Diagnostics


Once the model is trained you can run model diagnostics to visualize the learnings.

Check out [this](notebooks/ankitesh-devlog/03_ModelDiagnostics.ipynb) notebook to know more about model diagnostics.
