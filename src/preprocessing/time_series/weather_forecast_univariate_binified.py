import tensorflow as tf
import os
import pandas as pd
from collections import OrderedDict
from preprocessing.utils import univariate_data
from preprocessing.utils import create_bins
from preprocessing.utils import map_uni_data_classes
from preprocessing.utils import get_key
from preprocessing.utils import baseline
from preprocessing.utils import create_categorical_dataset_from_bins

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

bins_temp=create_bins(-25,5,12)
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

# TODO save this on a csv file.
# temp_classes=pd.Series(list_temp_classes)
#
uni_data = uni_data_df.values
# labels=temp_classes.values
#assert len(uni_data)==len(labels_12)

if __name__ == "__main__":
  print('head of original regression dataset', uni_data_df.head())  # to put in the main function.

  # range of temperature values.
  # max_temp = max(uni_data)
  # print('temp max', max_temp)
  # min_temp = min(uni_data)
  # print('temp min', min_temp)

  #print('binified dataset statistics', labels_2.value_counts())

  univariate_past_history = 10
  univariate_future_target = 10

  x_train_uni, y_train_uni = univariate_data(dataset=classes_data_binary,
                                             start_index=0,
                                             end_index=TRAIN_SPLIT,
                                             history_size=univariate_past_history,
                                             target_size=univariate_future_target)
  # x_val_uni, y_val_uni = univariate_data(dataset=labels_12,
  #                                        start_index=TRAIN_SPLIT,
  #                                        end_index=None,
  #                                        history_size=univariate_past_history,
  #                                        target_size=univariate_future_target)

  print ('Single window of past history')
  print (x_train_uni[0])
  print ('\n Target temperature to predict')
  print (y_train_uni[0])

  BATCH_SIZE = 256
  BUFFER_SIZE = 10000
  EVALUATION_INTERVAL = 200
  EPOCHS = 10

  train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
  train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

  #val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
  #val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

  #----simple baselines----------------------

  # #predictions_baseline=baseline(history=x_train_uni)
  #
  # #simple_lstm_model = tf.keras.models.Sequential([
  #   tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
  #   tf.keras.layers.Dense(1)
  # ])
  #
  # simple_lstm_model.compile(optimizer='adam',
  #                           loss='categorical_crossentropy',
  #                           metrics=['accuracy'])
  #
  # for x, y in val_univariate.take(1):
  #   print(simple_lstm_model.predict(x).shape)
  #
  # simple_lstm_model.fit(train_univariate,
  #                       epochs=EPOCHS,
  #                       steps_per_epoch=EVALUATION_INTERVAL,
  #                       validation_data=val_univariate,
  #                       validation_steps=50)

