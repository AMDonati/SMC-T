import tensorflow as tf
import os
import pandas as pd
from collections import OrderedDict
from preprocessing.ts_classif_utils import create_bins
from preprocessing.ts_classif_utils import map_uni_data_classes
from sklearn.model_selection import train_test_split
import numpy as np

#TODO: Makes sure that all the classes are represented in the training dataset for the function df_to_dataset.

def df_to_dataset(file_path, fname, col_name, index_name, min_value, size_bin, num_bins, train_split, batch_size, buffer_frac, seq_len, reduce_for_test=None):

  zip_path = tf.keras.utils.get_file(
      origin=file_path,
      fname=fname,
      extract=True)

  csv_path, _ = os.path.splitext(zip_path)

  df = pd.read_csv(csv_path)

  tf.random.set_seed(13)

  uni_data_df = df[col_name]
  uni_data_df.index = df[index_name]
  print('length of original continuous dataset: {}'.format(len(uni_data_df)))

  #temp_min: env -25.
  # temp max: env. 40.
  # temp_range: 65.

  #create_bins
  bins=create_bins(min_value, size_bin, num_bins)

  df_bins=pd.cut(uni_data_df, bins)
  bins_list=list(set(list(df_bins.values)))

  print('list of bins...', bins_list)

  dict_bins=OrderedDict(zip(range(num_bins), bins_list))
  df_categorized=map_uni_data_classes(continuous_data=uni_data_df,
                                      list_interval=list(bins),
                                      dict_temp=dict_bins)

  # reduce size of dataset for testing more quickly:
  if reduce_for_test is not None:
    df_categorized = df_categorized[:reduce_for_test]
    print('selecting for testing {} samples...'.format(len(df_categorized)))

  reduced_number_of_classes = len(list(set(df_categorized.values)))

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

  # compute buffer_size from buffer_frac:
  buffer_size=int(len(df_categorized)*buffer_frac)

  train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

  return train_dataset, val_dataset, df_categorized, x_train, reduced_number_of_classes

def df_continuous_to_dataset(file_path, fname, col_name, index_name, train_split, batch_size, buffer_frac, seq_len, reduce_for_test=None):

  zip_path = tf.keras.utils.get_file(
      origin=file_path,
      fname=fname,
      extract=True)

  csv_path, _ = os.path.splitext(zip_path)

  df = pd.read_csv(csv_path)

  tf.random.set_seed(13)

  uni_data_df = df[col_name]
  uni_data_df.index = df[index_name]
  print('length of original continuous dataset: {}'.format(len(uni_data_df)))

  # reduce size of dataset for testing more quickly:
  if reduce_for_test is not None:
    uni_data_df = uni_data_df[:reduce_for_test]
    print('selecting for testing {} samples...'.format(len(uni_data_df)))

  #transform the series in a numpy array
  data_array=np.array(uni_data_df)

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

  # compute buffer_size from buffer_frac:
  buffer_size=int(len(uni_data_df)*buffer_frac)

  train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

  return train_dataset, val_dataset, uni_data_df, x_train

if __name__ == "__main__":

  file_path = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
  fname = 'jena_climate_2009_2016.csv.zip'
  col_name='T (degC)'
  index_name='Date Time'
  TRAIN_SPLIT = 0.8
  min_value = -25
  size_bin = 5
  num_bins = 12
  BATCH_SIZE = 256
  buffer_frac = 0.05
  seq_len = 9
  reduce_for_test=10000

#-------------------------test of df_to_dataset_function--------------------------------------------------------------------------------------------------------

  train_dataset, val_dataset, df_categorized, x_train, reduced_number_of_classes=df_to_dataset(file_path=file_path,
                                                                        fname=fname,
                                                                        col_name=col_name,
                                                                        index_name=index_name,
                                                                        min_value=min_value,
                                                                        size_bin=size_bin,
                                                                        train_split=TRAIN_SPLIT,
                                                                        num_bins=num_bins,
                                                                        batch_size=BATCH_SIZE,
                                                                        buffer_frac=buffer_frac,
                                                                        seq_len=seq_len,
                                                                        reduce_for_test=reduce_for_test)

  print('multi classes value counts', df_categorized.value_counts())
  print('reduced_number of classes:', reduced_number_of_classes)

#------------------------ test of dataset_continuous_to_dataset function-----------------------------------------------------------------------------

  train_dataset, val_dataset, uni_data_df, x_train=df_continuous_to_dataset(file_path=file_path,
                                                                          fname=fname,
                                                                          col_name=col_name,
                                                                          index_name=index_name,
                                                                          train_split=TRAIN_SPLIT,
                                                                          batch_size=BATCH_SIZE,
                                                                          buffer_frac=buffer_frac,
                                                                          seq_len=seq_len,
                                                                          reduce_for_test=reduce_for_test)