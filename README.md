# SMC-T

**README in construction**

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
