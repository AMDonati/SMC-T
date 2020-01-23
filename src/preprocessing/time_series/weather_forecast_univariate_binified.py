import tensorflow as tf
import os
import pandas as pd
from collections import OrderedDict
from preprocessing.ts_classif_utils import create_bins
from preprocessing.ts_classif_utils import map_uni_data_classes
from sklearn.model_selection import train_test_split
import numpy as np

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)

TRAIN_SPLIT = 300000

tf.random.set_seed(13)

uni_data_df = df['T (degC)']
uni_data_df.index = df['Date Time']

#TODO: create even an artificial binary dataset.

#create_bins
bins_12=create_bins(-25,5,12)
bins_binary=create_bins(-25,35,2)

#labels_12=create_categorical_dataset_from_bins(df=uni_data_df,
                                               #min_val=-25,
                                               #bin_interval=5,
                                               #num_bins=12)

#labels_2=create_categorical_dataset_from_bins(df=uni_data_df,
                                              #min_value=-25,
                                              #bin_interval=35,
                                              #num_bins=2)

#bins2=pd.IntervalIndex.from_tuples(bins_temp, closed='left')

temp_bins_binary=pd.cut(uni_data_df, bins_binary)
#
bins_temp_binary=list(set(list(temp_bins_binary.values)))
dict_temp_binary=OrderedDict(zip(range(2), bins_temp_binary))
#
classes_data_binary=map_uni_data_classes(continuous_data=uni_data_df,
                                         list_interval=list(bins_binary),
                                         dict_temp=dict_temp_binary)

print('binary classes value counts', classes_data_binary.value_counts())

#transform the series in a numpy array
x_data_binary=np.array(classes_data_binary)


# split Test/ train set
x_train_binary, x_val_binary=train_test_split(x_data_binary, train_size=TRAIN_SPLIT, shuffle=False)

if __name__ == "__main__":
  print('head of original regression dataset', uni_data_df.head())  # to put in the main function.
  print('head of numpy array', x_data_binary[:20])

  BATCH_SIZE = 256
  BUFFER_SIZE = 10000
  EVALUATION_INTERVAL = 200
  EPOCHS = 10
  seq_len=9

  # Transform the numpy_arrays in tf.data.Dataset.

  train_univariate = tf.data.Dataset.from_tensor_slices(x_train_binary)
  #train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

  sequences_train = train_univariate.batch(seq_len + 1, drop_remainder=True)

  val_univariate = tf.data.Dataset.from_tensor_slices(x_val_binary)
  val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

  sequences_val=val_univariate.batch(seq_len + 1, drop_remainder=True)

  def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

  dataset=sequences_train.map(split_input_target)

  dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

  print('dataset', dataset)

  #print('input data - train', input_train.shape)
  #print('target data - train', target_train.shape)

  #input_val, target_val=split_input_target(val_univariate)



