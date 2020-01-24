from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib as mpl
import numpy as np
import os
import pandas as pd

## The weather dataset

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)

TRAIN_SPLIT = 300000

tf.random.set_seed(13)

# creating an univariate timeseries

uni_data = df['T (degC)']
uni_data.index = df['Date Time']

univariate_past_history = 20
univariate_future_target = 0

# create training, validation sets.
x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

# convert to tensors
train_data=tf.convert_to_tensor(x_train_uni, dtype=tf.float32)
train_targets=tf.convert_to_tensor(y_train_uni, dtype=tf.float32)

val_data=tf.convert_to_tensor(x_val_uni, dtype=tf.float32)
val_targets=tf.convert_to_tensor(y_val_uni, dtype=tf.float32)




