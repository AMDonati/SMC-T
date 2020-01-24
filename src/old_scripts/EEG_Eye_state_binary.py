# https://machinelearningmastery.com/how-to-predict-whether-eyes-are-open-or-closed-using-brain-waves/

import pandas as pd
import tensorflow as tf

file_abs_path='/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-RNN---Transformers/data/EEG_Eye_State_no_outliers.csv'

# load the dataset
data = pd.read_csv(file_abs_path, header=None)
values = data.values

# data normalization:
values=tf.keras.utils.normalize(
    values,
    axis=-1,
    order=2)

# split data into inputs and outputs
X, y = values[:, :-1], values[:, -1]

print('shape of input data', X.shape)
print('extract of input data', X[:5,:])
print('shape of target data', y.shape)


if __name__ == "__main__":
  # conversion to tensors:
  inputs=tf.convert_to_tensor(X, dtype=tf.float32, name='inputs') # shape (B,S)
  targets=tf.convert_to_tensor(y, dtype=tf.int32, name='targets') # shape (B,)

 # reshaping to get the right size:
 inputs=tf.expand_dims(inputs, axis=-1) # shape (B,S,D) here D=1
 targets=tf.expand_dims(targets, axis=-1) # shape (B,1)
# CAUTION: here this is not a sequence to sequence framework, and then a 'sequential classification" task.


