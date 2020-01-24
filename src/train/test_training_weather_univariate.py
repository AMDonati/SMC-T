#TODO: debug and add tran_accuracy.

import tensorflow as tf
from models.Baselines.Transformer_without_enc import Transformer
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from train.training_algos import train_step_classic_T
from train.training_algos import train_step_SMC_T

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer

import time
import sys

from preprocessing.time_series.df_to_dataset import df_to_dataset

#------------------UPLOAD the training dataset------------------------------------------------------------------------------------------------

file_path = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
fname = 'jena_climate_2009_2016.csv.zip'
col_name = 'T (degC)'
index_name = 'Date Time'
TRAIN_SPLIT = 0.8
min_value = -25
size_bin = 5
num_bins = 12
buffer_frac = 0.2 # fraction of total dataset to be taken in the buffer when shuffling.
seq_len_dataset =10 # one more than for the transformer.
BATCH_SIZE = 16 # small batch_size to avoid memory errors.
num_bins = 12 # correspond to the number of classes for a classification task
num_classes=num_bins
reduce_for_test = None # taking only 10,000 samples for testing.

train_dataset, val_dataset, df_categorized, x_train, num_classes=df_to_dataset(file_path=file_path,
                                                                      fname=fname,
                                                                      col_name=col_name,
                                                                      index_name=index_name,
                                                                      min_value=min_value,
                                                                      size_bin=size_bin,
                                                                      train_split=TRAIN_SPLIT,
                                                                      num_bins=num_bins,
                                                                      batch_size=BATCH_SIZE,
                                                                      buffer_frac=buffer_frac,
                                                                      seq_len=seq_len_dataset,
                                                                      reduce_for_test=reduce_for_test)

#TODO: save the data pre-processed in a .npy file so that the function is not called every-time.

print('multi-class classification problem with {} classes...'.format(num_classes))
num_samples_training=x_train.shape[0]
print('number of training samples...', num_samples_training)
df_train_samples=df_categorized[:num_samples_training]

#TODO: replace class numbers in original df_to_dataset function so that it works when reducing nulmber of smaples for testing.
print('classes distributions for training data', df_train_samples.value_counts())

# -------define hyperparameters----------------------------------------------------------------------------------------------------------------
## Optimizer
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# Model's hyperparameters.
num_particles = 1
num_heads = 2
d_model = 4
dff = 8
maximum_position_encoding = None  # no positional encoding for time_series dataset.
target_vocab_size = num_classes # correspond to the number of classes: multi-class classification problem.
num_layers = 1
data_type = 'time_series'
task_type = 'classification'
seq_len=9
sigma=1
noise=False

#----DEFINE THE MODEL---------------------------------------------------------------------------------------------------------------------------------------------------------------
# SMC_Transformer
smc_transformer = SMC_Transformer(num_layers=num_layers,
                        d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        target_vocab_size=target_vocab_size,
                        maximum_position_encoding=maximum_position_encoding,
                        num_particles=num_particles,
                        sigma=sigma,
                        noise=noise,
                        seq_len=seq_len,
                        data_type=data_type,
                        task_type=task_type)

# Transformer - baseline.
transformer=Transformer(num_layers=num_layers,
                        d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        target_vocab_size=target_vocab_size,
                        maximum_position_encoding=maximum_position_encoding,
                        data_type=data_type)

#-------------------- SIMPLE BASELINE FOR COMPARISON --------------------------------------------------------------------
#TODO: adapt this with the variables of this script.
# input_shape=0 #TODO: put the right input_shape here.
# simple_lstm_model = tf.keras.models.Sequential([
#     tf.keras.layers.LSTM(8, input_shape=input_shape),
#     tf.keras.layers.Dense(1)
# ])
#
# simple_lstm_model.compile(optimizer='adam', loss='categorical_crossentropy') #TODO change into the loss & optimizer of the classic Transformer.
#
# EVALUATION_INTERVAL = 200
# EPOCHS = 10
#
# simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
#                       steps_per_epoch=EVALUATION_INTERVAL,
#                       validation_data=val_univariate, validation_steps=50)

#-----------------TRAINING-----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
  # TF-2.0
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  sys.setrecursionlimit(100000)
  tf.config.experimental_run_functions_eagerly(True)  # to remove TensorInacessibleError

  dataset = train_dataset
  EPOCHS = 5

  # -------------------------------------------TRAIN ON THE DATASET - CLASSIC TRANSFORMER -------------------------------------------

  # Transformer - baseline.
  transformer = Transformer(num_layers=num_layers,
                            d_model=d_model,
                            num_heads=num_heads,
                            dff=dff,
                            target_vocab_size=target_vocab_size,
                            maximum_position_encoding=maximum_position_encoding,
                            data_type=data_type)

  for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()

    for (batch, (inp, tar)) in enumerate(dataset):
      loss_baseline = train_step_classic_T(inputs=inp,
                                           targets=tar,
                                           transformer=transformer,
                                           train_loss=train_loss,
                                           optimizer=optimizer)

      if batch % 500 == 0:
        print('epoch', epoch)
        print('batch', batch)
        print('loss -  Baseline Transformer', loss_baseline.numpy())
        print('average loss', train_loss(loss_baseline).numpy())

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

  print('training of SMC Transformer for weather dataset done...')


#-------------------TRAINING ON THE DATASET - SMC_TRANSFORMER-----------------------------------------------------------------------------------------------------------

  smc_transformer=SMC_Transformer(num_layers=num_layers,
                        d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        target_vocab_size=target_vocab_size,
                        maximum_position_encoding=maximum_position_encoding,
                        num_particles=num_particles,
                        sigma=sigma,
                        noise=noise,
                        seq_len=seq_len,
                        data_type=data_type,
                        task_type=task_type)

  #print('SMC transformer model summary...', smc_transformer.summary())

  # check the pass forward.
  for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions, _, _ = smc_transformer(inputs=input_example_batch,
                                            training=False,
                                            mask=create_look_ahead_mask(seq_len))
    print("predictions shape", example_batch_predictions.shape)

  for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    #train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(dataset):
      loss_smc=train_step_SMC_T(inputs=inp,
                                targets=tar,
                                smc_transformer=smc_transformer,
                                optimizer=optimizer,
                                train_loss=train_loss,
                                classic_loss=True,
                                SMC_loss=False)

      if batch % 500 == 0:
        print('epoch', epoch)
        print('batch', batch)
        print('loss - SMC Transformer', loss_smc.numpy())
        print('average SMC loss - SMC Transformer', train_loss(loss_smc.numpy()))

    # if (epoch + 1) % 5 == 0:
    #   ckpt_save_path = ckpt_manager.save()
    #   print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
    #                                                       ckpt_save_path))
    # print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
    #                                                     train_loss.result(),
    #                                                     train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

  print('training of SMC Transformer for weather dataset done...')

