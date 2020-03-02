import tensorflow as tf
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

def split_dataset_into_seq(dataset, start_index, end_index, history_size, step):
  data = []
  start_index = start_index + history_size

  if end_index is None:
    end_index=len(dataset)

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

  return np.array(data)

def df_to_data_regression(file_path, fname, col_name, index_name, history, step, TRAIN_SPLIT, VAL_SPLIT=0.5, VAL_SPLIT_cv=0.9, cv=False):
  #TODO: col_name is now a list of selected features.
  zip_path = tf.keras.utils.get_file(
      origin=file_path,
      fname=fname,
      extract=True)

  csv_path, _ = os.path.splitext(zip_path)
  df = pd.read_csv(csv_path)
  uni_data_df = df[col_name]
  uni_data_df.index = df[index_name]
  print('length of original continuous dataset: {}'.format(len(uni_data_df)))

  #TRAIN_SPLIT=int(TRAIN_SPLIT*len(uni_data_df))

  uni_data=uni_data_df.values

  # normalization
  #data_mean = uni_data[:TRAIN_SPLIT].mean(axis=0)
  #data_std = uni_data[:TRAIN_SPLIT].std(axis=0)
  data_mean = uni_data.mean(axis=0)
  data_std = uni_data.std(axis=0)
  uni_data = (uni_data - data_mean) / data_std

  stats = (data_mean[0], data_std[0])

  data_in_seq = split_dataset_into_seq(uni_data, 0, None, history, step)
  if not cv:

    # split between validation dataset and test set:
    train_data, val_data = train_test_split(data_in_seq, train_size=TRAIN_SPLIT, shuffle=True)
    val_data, test_data=train_test_split(val_data, train_size=VAL_SPLIT, shuffle=True)

    # reshaping arrays to have a (future shape) of (B,S,1):
    if len(col_name) == 1:
      train_data = np.reshape(train_data, newshape=(train_data.shape[0], train_data.shape[1], 1))
      val_data = np.reshape(val_data, newshape=(val_data.shape[0], val_data.shape[1], 1))
      test_data = np.reshape(test_data, newshape=(test_data.shape[0], test_data.shape[1], 1))

    return (train_data, val_data, test_data), uni_data_df, stats

  else:
    train_val_data, test_data = train_test_split(data_in_seq, train_size=VAL_SPLIT_cv)
    kf = KFold(n_splits=5)
    list_train_data, list_val_data = [], []
    for train_index, val_index in kf.split(train_val_data):
      train_data = train_val_data[train_index, :, :]
      val_data = train_val_data[val_index, :, :]
      list_train_data.append(train_data)
      list_val_data.append(val_data)

    if len(col_name)==1:
      list_train_data = [np.reshape(d, newshape=(d.shape[0], d.shape[1], 1)) for d in list_train_data]
      list_val_data = [np.reshape(d, newshape=(d.shape[0], d.shape[1], 1)) for d in list_val_data]
      test_data = np.reshape(test_data, newshape=(test_data.shape[0], test_data.shape[1], 1))

    return (list_train_data, list_val_data, test_data), uni_data_df, stats

def split_input_target_uni_step(chunk):
  if len(chunk.shape) == 3:
    input_text = chunk[:,:-1,:]
    target_text = chunk[:,1:,:]
  elif len(chunk.shape) == 2:
    input_text = chunk[:,:-1]
    target_text = chunk[:,1:]
  return input_text, target_text

def split_synthetic_dataset(x_data, TRAIN_SPLIT, VAL_SPLIT=0.5, VAL_SPLIT_cv=0.9, cv=False):
  if not cv:
    train_data, val_test_data = train_test_split(x_data, train_size=TRAIN_SPLIT)
    val_data, test_data = train_test_split(val_test_data, train_size=VAL_SPLIT)

    return train_data, val_data, test_data

  else:
    train_val_data, test_data = train_test_split(x_data, train_size=VAL_SPLIT_cv)
    kf = KFold(n_splits=5)
    list_train_data, list_val_data = [], []
    for train_index, val_index in kf.split(train_val_data):
      train_data = train_val_data[train_index, :, :]
      val_data = train_val_data[val_index, :, :]
      list_train_data.append(train_data)
      list_val_data.append(val_data)

    return list_train_data, list_val_data, test_data


def data_to_dataset_uni_step(train_data, val_data, test_data, split_fn, BUFFER_SIZE, BATCH_SIZE, cv, target_feature=None):
  '''
  :param train_data: input data for training > shape (N_train, S+1, F) ; N_train = number of samples in training dataset.
  :param val_data: input data used for validation set > shape (N_val, S+1, F)
  :param split_fn: used to split between input data and target.
  :param BUFFER_SIZE: to shuffle the dataset.
  :param BATCH_SIZE:
  :param: cv: boolean; True if multiple train datasets / val datasets for cross-validation; False otherwise.
  :param target_feature: used to select the target feature to be predicted. Case of multivariate ts as input data > prediction of a univariate ts.
  :return:
  2 tf.data.Dataset, one for the training set, and one for the validation set, with:
  input data:  batches of train data > shape (B, S+1, F) > S+1 because the data is split in the SMC_Transformer.Py script.
  target data: shape (B,S,1) > univariate ts to be predicted (shifted from one timestep compared to the input data).
  '''
  if not cv:
    list_train_data = [train_data]
    list_val_data = [val_data]
  else:
    list_train_data = train_data
    list_val_data = val_data

  list_train_dataset, list_val_dataset = [], []
  list_train_dataset_for_RNN, list_val_dataset_for_RNN = [], []

  for (train_data, val_data) in zip(list_train_data, list_val_data):
    x_train, y_train = split_fn(train_data)
    x_val, y_val = split_fn(val_data)
    if target_feature is not None:
      y_train = y_train[:, :, target_feature]
      y_train = np.reshape(y_train, newshape=(y_train.shape[0], y_train.shape[1], 1))
      y_val = y_val[:, :, target_feature]
      y_val = np.reshape(y_val, newshape=(y_val.shape[0], y_val.shape[1], 1))
    # turning it into a tf.data.Dataset.
    # use train_data and val_data instead to have as input data a tensor of shape (B,S+1,F)
    # > trick because then the dataset is split again in the SMC_Transformer cell.
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, y_train))
    train_dataset = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
    train_dataset_for_RNN = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset_for_RNN = train_dataset_for_RNN.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    val_dataset_for_RNN = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset_for_RNN = val_dataset_for_RNN.batch(BATCH_SIZE, drop_remainder=True)
    list_train_dataset.append(train_dataset)
    list_val_dataset.append(val_dataset)
    list_train_dataset_for_RNN.append(train_dataset_for_RNN)
    list_val_dataset_for_RNN.append(val_dataset_for_RNN)

  x_test, y_test = split_fn(test_data)
  if target_feature is not None:
    y_test = y_test[:, :, target_feature]
    y_test = np.reshape(y_test, newshape=(y_test.shape[0], y_test.shape[1], 1))
    print('univariate timeseries forecasting...')
  else:
    print('multivariate timeseries forecasting with {} features'.format(y_train.shape[-1]))

  BATCH_SIZE_test = test_data.shape[0]
  test_dataset = tf.data.Dataset.from_tensor_slices((test_data, y_test))
  test_dataset = test_dataset.batch(BATCH_SIZE_test)

  test_dataset_for_RNN = tf.data.Dataset.from_tensor_slices((x_test, y_test))
  test_dataset_for_RNN = test_dataset_for_RNN.batch(BATCH_SIZE_test)

  if not cv:
    train_dataset, val_dataset = list_train_dataset[0], list_val_dataset[0]
    train_dataset_for_RNN, val_dataset_for_RNN = list_train_dataset_for_RNN[0], list_val_dataset_for_RNN[0]
    return train_dataset, val_dataset, test_dataset, train_dataset_for_RNN, val_dataset_for_RNN, test_dataset_for_RNN
  else:
    return list_train_dataset, list_val_dataset, test_dataset, list_train_dataset_for_RNN, list_val_dataset_for_RNN, test_dataset_for_RNN

### ----- for classification case (categorized time_series... ----------------------------------------------------------------

def df_to_data_uni_step(file_path, fname, col_name, index_name, q_cut, history, step, TRAIN_SPLIT):

  zip_path = tf.keras.utils.get_file(
      origin=file_path,
      fname=fname,
      extract=True)

  csv_path, _ = os.path.splitext(zip_path)
  df = pd.read_csv(csv_path)
  uni_data_df = df[col_name]
  uni_data_df.index = df[index_name]
  print('length of original continuous dataset: {}'.format(len(uni_data_df)))

  # temp_min: env -25.
  # temp max: env. 40.
  # temp_range: 65.

  uni_data_categorized = pd.qcut(uni_data_df, q_cut, False)
  uni_data_intervals = pd.qcut(uni_data_df, q_cut)
  uni_data_merged = pd.merge(uni_data_categorized, uni_data_intervals, left_index=True, right_index=True)
  print(uni_data_intervals.value_counts())

  TRAIN_SPLIT=int(TRAIN_SPLIT*len(uni_data_merged))

  train_data = split_dataset_into_seq(uni_data_categorized, 0, TRAIN_SPLIT, history, step)
  val_data = split_dataset_into_seq(uni_data_categorized, TRAIN_SPLIT, None, history,step)

  # split between validation dataset and test set:
  val_data, test_data = train_test_split(val_data, train_size=0.5)
  #val_data, test_data=train_test_split(val_data, train_size=0.8)

  return (train_data, val_data, test_data), uni_data_merged, uni_data_df

if __name__ == "__main__":

  #------------ REGRESSION CASE -------------------------------------------------------------------------------------
  file_path = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
  fname = 'jena_climate_2009_2016.csv.zip'
  #col_name='T (degC)'
  col_name = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
  index_name='Date Time'
  #q_cut=10
  TRAIN_SPLIT = 0.8
  BATCH_SIZE = 256
  BUFFER_SIZE = 10000
  history = 144+6
  step= 6 # sample a temperature every 4 hours.

  cv=False

  (train_data, val_data, test_data), original_df, stats = df_to_data_regression(file_path=file_path,
                                                                         fname=fname,
                                                                         col_name=col_name,
                                                                         index_name=index_name,
                                                                         TRAIN_SPLIT=TRAIN_SPLIT,
                                                                         history=history,
                                                                         step=step,
                                                                         cv=cv)

  print(train_data[:5])

  BUFFER_SIZE = 10000
  BATCH_SIZE = 64

  train_dataset, val_dataset, test_dataset, _, _, _ = data_to_dataset_uni_step(train_data=train_data,
                                                      val_data=val_data,
                                                      test_data = test_data,
                                                      split_fn=split_input_target_uni_step,
                                                      BUFFER_SIZE=BUFFER_SIZE,
                                                      BATCH_SIZE=BATCH_SIZE,
                                                      target_feature=0,
                                                      cv=cv)
  if cv:
    train_dataset = train_dataset[0]
    val_dataset = val_dataset[0]

  for (inp, tar) in train_dataset.take(2):
    print('training inputs', inp[0])
    print('training targets', tar[0])

  for (inp, tar) in val_dataset.take(1):
    print('validation inputs', inp[0])
    print('validation targets', tar[0])

  for (test_data, y_test) in test_dataset:
    print('test data', test_data[0])
    print('targets test', y_test[0])

  # ----- load and prepare synthetic dataset -----------------------------------------------------------------------------------------------------------------------------------

  file_path="/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/data/synthetic_dataset.npy"
  X_data = np.load(file_path)
  train_data_synt, val_data_synt, test_data_synt = split_synthetic_dataset(x_data=X_data, TRAIN_SPLIT=0.7, cv=cv)

  if not cv:
    print('train data', train_data_synt.shape)
    print('val_data', val_data_synt.shape)
    print('test data', test_data_synt.shape)

  train_dataset_synt, val_dataset_synt, test_dataset_synt, _, _, _ = data_to_dataset_uni_step(train_data=train_data_synt,
                                                      val_data=val_data_synt,
                                                      test_data=test_data_synt,
                                                      split_fn=split_input_target_uni_step,
                                                      BUFFER_SIZE=2000,
                                                      BATCH_SIZE=64,
                                                      target_feature=0,
                                                      cv=cv)

  print('train synthetic dataset', train_dataset_synt)
  print('val dataset synthetic', val_dataset_synt)
  print('test dataset synthetic', test_dataset_synt)


  #TODO save datasets in .npy files.
  # #print(data_categorized_df.head())
  # # for i in range(800,850):
  # #   print ('Single window of past history')
  # #   print (train_data[i])
  # data_folder='data'
  # train_array_path=os.path.join(data_folder, 'ts_weather_train_data_c10')
  # val_array_path = os.path.join(data_folder, 'ts_weather_val_data_c10')
  # test_array_path = os.path.join(data_folder, 'ts_weather_test_data_c10')
  #
  # # saving arrays in .npy files
  # np.save(train_array_path, train_data)
  # np.save(val_array_path, val_data)
  # np.save(test_array_path, test_data)
  # print('arrays saved on npy files...')

  # # load numpy arrays with preprocess data:
  # data_folder='/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/data'
  # train_data=np.load(data_folder+'/ts_weather_train_data.npy')
  # val_data = np.load(data_folder + '/ts_weather_val_data.npy')
  # test_data = np.load(data_folder + '/ts_weather_test_data.npy')
  #
  # print('train_data', train_data.shape)
  # print('test_data', test_data.shape)
  #
  # train_dataset, val_dataset=data_to_dataset_uni_step(train_data=train_data,
  #                                                     val_data=val_data,
  #                                                     split_fn=split_input_target_uni_step,
  #                                                     BUFFER_SIZE=BUFFER_SIZE,
  #                                                     BATCH_SIZE=BATCH_SIZE)
  #
  # print(train_dataset)
  #
  # for (inp, tar) in train_dataset.take(5):
  #   print('input example', inp[0])
  #   print('target example', tar[0])
  #
  # for (inp, tar) in val_dataset.take(5):
  #   print('inp val ex', inp[0])
  #   print('inp tar ex', tar[0])


# #-------------------------test of df_to_dataset_function--------------------------------------------------------------------------------------------------------
#
#   train_dataset, val_dataset, df_categorized, x_train, reduced_number_of_classes=df_to_dataset(file_path=file_path,
#                                                                         fname=fname,
#                                                                         col_name=col_name,
#                                                                         index_name=index_name,
#                                                                         min_value=min_value,
#                                                                         size_bin=size_bin,
#                                                                         train_split=TRAIN_SPLIT,
#                                                                         num_bins=num_bins,
#                                                                         batch_size=BATCH_SIZE,
#                                                                         buffer_frac=buffer_frac,
#                                                                         seq_len=seq_len,
#                                                                         reduce_for_test=reduce_for_test)
#
#   print('multi classes value counts', df_categorized.value_counts())
#   print('reduced_number of classes:', reduced_number_of_classes)
#
# #------------------------ test of dataset_continuous_to_dataset function-----------------------------------------------------------------------------
#
#   train_dataset, val_dataset, uni_data_df, x_train=df_continuous_to_dataset(file_path=file_path,
#                                                                           fname=fname,
#                                                                           col_name=col_name,
#                                                                           index_name=index_name,
#                                                                           train_split=TRAIN_SPLIT,
#                                                                           batch_size=BATCH_SIZE,
#                                                                           buffer_frac=buffer_frac,
#                                                                           seq_len=seq_len,
#                                                                           reduce_for_test=reduce_for_test)

 #---------old functions-----------