# SMC-T

**README in construction**

### TO DO (by order of priority): 
* take care of the #TODOs in the code linked to the debugging of the training algos & the computation of the loss. 
* Test the formula of the SMC loss by replacing it by the classic one in the SMC Transformer training (for num_particles=1). 
* IN THE TRAINING SCRIPT: store all the information needed (with callbacks, checkpoints, & logging library)
* Automatization of the experiments (cf GuessWhat training script or uncertainties's one). 
#### if enough time
* implement the multivariate case. (input_data=multivariate time-series, output data=univariate time-series.)
* regression case: computation of the 'customized mse' to allow to have an omega (std of the sampling weights) different of one. 
* inference function (to see with Sylvain). 

### Download
```
git clone https://github.com/AMDonati/SMC-T.git
```

### Requirements

The code works on  python 3 and tensorflow 2.1.
To install tensorflow 2.1, please follow the official instructions [here](https://www.tensorflow.org/install/pip?lang=python3)

### File architecture
```
├── config         # store the configuration file to create/train models

├── ouput            # store the output experiments (checkpoint, logs etc.)

├── data          # contains the data for the experiments

├── notebooks  # notebooks fo plots, etc... 

└── src            # source files
```

## Launching the experiments

To launch the experiments in the local directory, you first have to set the python path:
```
export PYTHONPATH=src:${PYTHONPATH} 
```

### training on synthetic data
```
python src/train/test_training_dummy_dataset.py

```

### training on toy datasets 
```
python src/train/test_training_toy_datasets.py

```
