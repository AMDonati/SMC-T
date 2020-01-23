#TODO: to adapt for the categorized weather dataset.

import tensorflow as tf
from models.Baselines.Transformer_without_enc import Transformer
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from neural_toolbox.training_algos import loss_function_classification
from neural_toolbox.training_algos import loss_function_regression
from neural_toolbox.training_algos import loss_function_classic_T_classif

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer

import time
import sys

from preprocessing.time_series.df_to_dataset import df_to_dataset

#------------------UPLOAD the training dataset------------------------------------------------------------------------------------------------

file_path = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
fname = 'jena_climate_2009_2016.csv.zip'
col_name = 'T (degC)'
index_name = 'Date Time'
#TODO: put this as a fraction to work with number of samples.
TRAIN_SPLIT = 300000
min_value = -25
size_bin = 5
num_bins = 12
buffer_frac = 10000
seq_len_dataset =10 # one more than for the transformer.
BATCH_SIZE = 64
num_bins = 12 # correspond to the number of classes for a classification task

#TODO: add a reduce_for_test parameter.
train_dataset, val_dataset, uni_data_df, df_categorized=df_to_dataset(file_path=file_path,
                                                                      fname=fname,
                                                                      col_name=col_name,
                                                                      index_name=index_name,
                                                                      min_value=min_value,
                                                                      size_bin=size_bin,
                                                                      train_split=TRAIN_SPLIT,
                                                                      num_bins=num_bins,
                                                                      batch_size=BATCH_SIZE,
                                                                      buffer_frac=buffer_frac,
                                                                      seq_len=seq_len_dataset)

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
target_vocab_size = 12 # correspond to the number of classes: multi-class classification problem.
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


# ------ DEF the TRAIN STEP FUNCTION -----------------------------------------------------------------------------------

# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
  tf.TensorSpec(shape=(None, None), dtype=tf.int32),
  tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

@tf.function(input_signature=train_step_signature)
def train_step_SMC_T(inputs, targets=None, SMC_loss=True, classic_loss=True):
  '''
  compute a gradient descent step using categorical crossentropy loss by updating the trainable parameters.
  :param input: input data > shape (B,S) for nlp and univariate time_series.
  multivariate case needs to be implemented still.
  :param model: to choose between the classic transformer model and the smc one.
  :param target: target data (sequential one)
  :param SMC_loss: boolean to compute SMC_loss or not. Default is False.
  :param classic_loss: boolean to compute classic cross-entropy loss, or not. Default is True.
  :return:
  The updated loss.
  '''
  if targets is None:
    tar_inp = inputs[:, :-1]
    tar_real = inputs[:, 1:]
  else:
    tar_inp=inputs
    tar_real=targets

  if len(tf.shape(tar_inp))==3:
    assert tf.shape(tar_inp)[-1]==1
    tar_inp=tf.squeeze(tar_inp, axis=-1)

  if len(tf.shape(tar_real))==3:
    assert tf.shape(tar_real)[-1]==1
    tar_real=tf.squeeze(tar_real, axis=-1)

  assert len(tf.shape(tar_inp))==2
  assert len(tf.shape(tar_real))==2

  seq_len=tf.shape(tar_inp)[1]
  mask_transformer = create_look_ahead_mask(seq_len)

  with tf.GradientTape() as tape:
    predictions, trajectories, weights = smc_transformer(inputs=tar_inp,
                                               training=True,
                                               mask=mask_transformer)

    # predictions: shape (B,P,S,C) > sequence of log_probas for the classification task.
    # trajectories: shape (B,P,S,D) = [z0,z1,z2,...,zT]
    # weights: shape (B,P,1) = w_T: used in the computation of the loss.

    if smc_transformer.task_type== 'classification':
      assert tf.shape(predictions)[-1]>2
      loss = loss_function_classification(real=tar_real,
                                          predictions=predictions,
                                          weights=weights,
                                          transformer=smc_transformer,
                                          SMC_loss=SMC_loss,
                                          classic_loss=classic_loss)
    elif smc_transformer.task_type== 'regression':
      loss=loss_function_regression(real=tar_real,
                                    predictions=predictions,
                                    weights=weights,
                                    tranformer=smc_transformer,
                                    SMC_loss=SMC_loss,
                                    classic_loss=classic_loss)
    else:
      raise ValueError('task_type argument in Transformer class is not supported.'
                       'Please choose between "classification" or "regression"')

    gradients = tape.gradient(loss, smc_transformer.trainable_variables)

  optimizer.apply_gradients(zip(gradients, smc_transformer.trainable_variables))

  #train_loss(loss)
  #train_accuracy(tar_real, predictions)

  return loss

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

@tf.function(input_signature=train_step_signature)
def train_step_classic_T(inputs, targets=None):
  '''training step for the classic Transformer model (dummy dataset)'''
  if targets is None:
    tar_inp = inputs[:, :-1]
    tar_real = inputs[:, 1:]
  else:
    tar_inp=inputs
    tar_real=targets

  if len(tf.shape(tar_inp))==2:
    tar_inp=tf.expand_dims(tar_inp, axis=-1)
  if len(tf.shape(tar_real))==2:
    tar_real=tf.expand_dims(tar_real, axis=-1)

  # CAUTION. Unlike the SMC_Transformer, the inputs and targets need to be of shape (B,S,1).
  assert len(tf.shape(tar_inp))==3
  assert len(tf.shape(tar_real))==3

  seq_len = tf.shape(tar_inp)[1]
  mask_transformer = create_look_ahead_mask(seq_len)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(inputs=tar_inp, training=True, mask=mask_transformer)

    loss = loss_function_classic_T_classif(real=tar_real, pred=predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  #train_loss(loss)
  #train_accuracy(tar_real, predictions)

  return loss

#-----------------TRAINING-----------------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
  # TF-2.0
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  sys.setrecursionlimit(100000)
  tf.config.experimental_run_functions_eagerly(True)  # to remove TensorInacessibleError

#-------------------TRAINING ON DUMMY DATASET - SMC_TRANSFORMER-----------------------------------------------------------------------------------------------------------

  dataset=train_dataset
  EPOCHS=10

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

  # # check the pass forward.
  # for input_example_batch, target_example_batch in dataset.take(1):
  #   example_batch_predictions, _, _ = smc_transformer(inputs=input_example_batch,
  #                                           training=False,
  #                                           mask=create_look_ahead_mask(seq_len))
  #   print("predictions shape", example_batch_predictions.shape)

  # for epoch in range(EPOCHS):
  #   start = time.time()
  #
  #   #train_loss.reset_states()
  #   #train_accuracy.reset_states()
  #
  #   for (batch, (inp, tar)) in enumerate(dataset):
  #     if len(tf.shape(inp))==2:
  #       inp=tf.expand_dims(inp, axis=-1)
  #     if len(tf.shape(tar))==2:
  #       tar=tf.expand_dims(tar, axis=-1)
  #     loss_smc=train_step_SMC_T(inputs=inp,
  #                               targets=tar,
  #                               classic_loss=True,
  #                               SMC_loss=False)
  #
  #     if batch % 10 == 0:
  #       print('epoch', epoch)
  #       print('batch', batch)
  #       print('loss - SMC Transformer', loss_smc)
  #
  #   # if (epoch + 1) % 5 == 0:
  #   #   ckpt_save_path = ckpt_manager.save()
  #   #   print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
  #   #                                                       ckpt_save_path))
  #
  #   # print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
  #   #                                                     train_loss.result(),
  #   #                                                     train_accuracy.result()))
  #
  #   print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
  #
  # print('training of SMC Transformer for weather dataset done...')

 #-------------------------------------------TRAIN ON DUMMY DATASET - CLASSIC TRANSFORMER -------------------------------------------

# Transformer - baseline.
transformer=Transformer(num_layers=num_layers,
                        d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        target_vocab_size=target_vocab_size,
                        maximum_position_encoding=maximum_position_encoding,
                        data_type=data_type)

for epoch in range(EPOCHS):
  start = time.time()

  for (batch, (inp, tar)) in enumerate(dataset):
    loss_baseline = train_step_classic_T(inputs=inp,
                                targets=tar)

    if batch % 10 == 0:
      print('epoch', epoch)
      print('batch', batch)
      print('loss -  Baseline Transformer', loss_baseline)

  print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

print('training of SMC Transformer for weather dataset done...')