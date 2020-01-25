# SMC-T

**README in construction**

### TO DO (remaining implementation by order of priority): 
* finalize SMC_loss for the one-layer case > ok done & tested on the simple case of sigma=identity matrix & epsilon=zero tensor.
* implement the multivariate case. 
* computation of attention weights > ok done.
* regression case: computation of weights + mean square error > ok done for simple case of omega (stddev of the gaussian distrbution) is equal to one (no customized loss implemented yet).
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
python src/train/train_SMC_Transformer.py

```
