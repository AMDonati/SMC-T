import tensorflow as tf
import os
import pandas as pd
from collections import OrderedDict
from preprocessing.utils import create_bins
from preprocessing.utils import map_uni_data_classes
from sklearn.model_selection import train_test_split
import numpy as np

def df_to_dataset(file_path, fname, col_name, index_name, min_value, size_bin, num_bins, train_split, batch_size, buffer_size, seq_len, reduce_for_test=None):

  zip_path = tf.keras.utils.get_file(
      origin=file_path,
      fname=fname,
      extract=True)

  csv_path, _ = os.path.splitext(zip_path)

  df = pd.read_csv(csv_path)

  tf.random.set_seed(13)

  uni_data_df = df[col_name]
  uni_data_df.index = df[index_name]

  #reduce size of dataset for testing more quickly:
  if reduce_for_test is not None:
    uni_data_df=uni_data_df[:reduce_for_test]

  #create_bins
  bins=create_bins(min_value, size_bin, num_bins)

  df_bins=pd.cut(uni_data_df, bins)
  bins_list=list(set(list(df_bins.values)))
  dict_bins=OrderedDict(zip(range(num_bins), bins_list))
  df_categorized=map_uni_data_classes(continuous_data=uni_data_df,
                                      list_interval=list(bins),
                                      dict_temp=dict_bins)

  #transform the series in a numpy array
  data_array=np.array(df_categorized)

  # split Test/ train set
  x_train, x_val=train_test_split(data_array, train_size=train_split, shuffle=False)

  # Transform the numpy_arrays in tf.data.Dataset.
  train_univariate = tf.data.Dataset.from_tensor_slices(x_train)
  sequences_train = train_univariate.batch(seq_len + 1, drop_remainder=True) # seq_len + 1 to have seq_len when splitting between inputs and targets.

  val_univariate = tf.data.Dataset.from_tensor_slices(x_val)
  # val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
  sequences_val = val_univariate.batch(seq_len + 1, drop_remainder=True)

  def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

  train_dataset = sequences_train.map(split_input_target)
  val_dataset = sequences_val.map(split_input_target)

  train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

  return train_dataset, val_dataset, uni_data_df, df_categorized

if __name__ == "__main__":
  file_path = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
  fname = 'jena_climate_2009_2016.csv.zip'
  col_name='T (degC)'
  index_name='Date Time'
  TRAIN_SPLIT = 300000
  min_value = -25
  size_bin = 5
  num_bins = 12
  BATCH_SIZE = 256
  buffer_size = 10000
  seq_len = 9

  train_dataset, val_dataset, uni_data_df, df_categorized=df_to_dataset(file_path=file_path,
                                                               fname=fname,
                                                               col_name=col_name,
                                                               index_name=index_name,
                                                               min_value=min_value,
                                                               size_bin=size_bin,
                                                               train_split=TRAIN_SPLIT,
                                                               num_bins=num_bins,
                                                               batch_size=BATCH_SIZE,
                                                               buffer_size=buffer_size,
                                                               seq_len=seq_len)


  print('multi classes value counts', df_categorized.value_counts())
  print('head of original regression dataset', uni_data_df.head())