#TODO: the output of the Model should return predictions of shape (B,P,S,2) (one_hot_encoding) so that it works well. 

import tensorflow as tf
import os
import pandas as pd
from collections import OrderedDict
from preprocessing.ts_classif_utils import create_bins
from preprocessing.ts_classif_utils import map_uni_data_classes
from sklearn.model_selection import train_test_split
import numpy as np

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from neural_toolbox.training_algos import categorical_ce_with_particules
from neural_toolbox.training_algos import binary_ce_with_particules
from neural_toolbox.training_algos import mse_with_particles

import time
import sys

#--------------CREATE THE DATASET--------------------------------------------

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

#create_bins
bins_12=create_bins(-25,5,12)

temp_bins_12=pd.cut(uni_data_df, bins_12)
bins_temp_12=list(set(list(temp_bins_12.values)))
dict_temp_12=OrderedDict(zip(range(12), bins_temp_12))
classes_data_12=map_uni_data_classes(continuous_data=uni_data_df,
                                     list_interval=list(bins_12),
                                     dict_temp=dict_temp_12)
print('binary classes value counts', classes_data_12.value_counts())

#transform the series in a numpy array
x_data_multiclasses=np.array(classes_data_12)

# split Test/ train set
x_train, x_val=train_test_split(x_data_multiclasses, train_size=TRAIN_SPLIT, shuffle=False)

if __name__ == "__main__":
  print('head of original regression dataset', uni_data_df.head())  # to put in the main function.
  print('head of numpy array', x_data_multiclasses[:20])

  BATCH_SIZE = 256
  BUFFER_SIZE = 10000
  EVALUATION_INTERVAL = 200
  EPOCHS = 10
  seq_len=9

  # Transform the numpy_arrays in tf.data.Dataset.

  train_univariate = tf.data.Dataset.from_tensor_slices(x_train)
  sequences_train = train_univariate.batch(seq_len + 1, drop_remainder=True)

  val_univariate = tf.data.Dataset.from_tensor_slices(x_val)
  #val_univariate = val_univariate.batch(BATCH_SIZE).repeat()
  sequences_val=val_univariate.batch(seq_len + 1, drop_remainder=True)

  def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

  train_dataset=sequences_train.map(split_input_target)
  val_dataset=sequences_val.map(split_input_target)

  train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

  print('train dataset', train_dataset)

  # look examples of the train dataset
  for input_example, target_example in train_dataset.take(1):
    print('Input data: ', input_example)
    print('Target data:', target_example)

  #-------define hyperparameters------------------------
  ## Optimizer
  learning_rate = 0.001
  optimizer = tf.keras.optimizers.Adam(learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.98,
                                       epsilon=1e-9)

  EPOCHS = 5

  # generate dummy dataset
  num_particles = 5
  num_heads = 2
  d_model = 12
  dff = 24
  target_vocab_size = 12 # binary classification problem.
  # mean square error...
  pe_target = None
  num_layers = 3
  task_type='classification'
  data_type='time_series'

  #----create the SMC Transformer:
  transformer = SMC_Transformer(num_layers=num_layers,
                                d_model=d_model,
                                num_heads=num_heads,
                                dff=dff,
                                target_vocab_size=target_vocab_size,
                                maximum_position_encoding=pe_target,
                                num_particles=num_particles,
                                sigma=1,
                                seq_len=seq_len,
                                data_type=data_type,
                                task_type=task_type)

  #------ LOSS FUNCTIONS--------------

  # The @tf.function trace-compiles train_step into a TF graph for faster
  # execution. The function specializes to the precise shape of the argument
  # tensors. To avoid re-tracing due to the variable sequence lengths or variable
  # batch sizes (the last batch is smaller), use input_signature to specify
  # more generic shapes.

  train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),
  ]


  def loss_function_classification(real, predictions, weights, transformer, classic_loss=True, SMC_loss=True):
    '''
    :param real: targets > shape (B,P,S)
    :param predictions (log_probas) > shape (B,P,S,V)
    :param weights: re-sampling_weights for the last element > shape (B,P)
    :param classic_loss: boolean to compute the classic loss or not (default=True)
    :param SMC_loss: boolean to compute the SMC loss (default=True)
    :return:
    a scalar computing the SMC loss as defined in the paper.
    '''
    if classic_loss:
      loss_ce = categorical_ce_with_particules(real=real, pred=predictions, sampling_weights=weights)
    else:
      loss_ce = 0
    if SMC_loss:
      loss_smc = -transformer.compute_SMC_log_likelihood(real=real,
                                                         sampling_weights=weights)  # we take a minus because we want to minimize -maximum_likelihood.
    else:
      loss_smc = 0
    loss = loss_ce + loss_smc
    return loss

  def loss_function_binary(real, predictions, weights, transformer, classic_loss=True, SMC_loss=True):
    '''
    :param real: targets > shape (B,P,S)
    :param predictions (log_probas) > shape (B,P,S,V)
    :param weights: re-sampling_weights for the last element > shape (B,P)
    :param classic_loss: boolean to compute the classic loss or not (default=True)
    :param SMC_loss: boolean to compute the SMC loss (default=True)
    :return:
    a scalar computing the SMC loss as defined in the paper.
    '''
    if classic_loss:
      loss_ce = binary_ce_with_particules(real=real, pred=predictions, sampling_weights=weights)
    else:
      loss_ce = 0
    if SMC_loss:
      loss_smc = -transformer.compute_SMC_log_likelihood(real=real,
                                                         sampling_weights=weights)  # we take a minus because we want to minimize -maximum_likelihood.
    else:
      loss_smc = 0
    loss = loss_ce + loss_smc
    return loss


  def loss_function_regression(real, predictions, weights, transformer, classic_loss=True, SMC_loss=True):
    '''
    :param real: targets > shape (B,P,S)
    :param predictions (log_probas) > shape (B,P,S,V)
    :param weights: re-sampling_weights for the last element > shape (B,P)
    :param classic_loss: boolean to compute the classic loss or not (default=True)
    :param SMC_loss: boolean to compute the SMC loss (default=True)
    :return:
    a scalar computing the SMC loss as defined in the paper.
    '''
    if classic_loss:
      # TODO: if sigma of weights_computation is not equal to 1. change the mse by a custom SMC_log_likelihood.
      loss_ce = mse_with_particles(real=real, pred=predictions, sampling_weights=weights)
    else:
      loss_ce = 0
    if SMC_loss:
      # take minus the log_likelihood.
      loss_smc = -transformer.compute_SMC_log_likelihood(real=real,
                                                         sampling_weights=weights)  # we take a minus because we want to minimize -maximum_likelihood.
    else:
      loss_smc = 0
    loss = loss_ce + loss_smc
    return loss

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy=tf.keras.metrics.BinaryCrossentropy(name='train_accuracy')

### ------- FONCTION TRAIN_STEP---------------------------------------------
@tf.function(input_signature=train_step_signature)
def train_step(inputs, targets=None, SMC_loss=True, classic_loss=True):
  '''
  compute a SGD step using categorical crossentropy loss by updating the trainable parameters.
  :param input: input data > shape (B,S,F) (for time-series). (B,S) for nlp.
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

  mask = create_look_ahead_mask(seq_len)

  with tf.GradientTape() as tape:
    predictions, trajectories, weights = transformer(inputs=tar_inp,
                                                     training=True,mask=mask)

    # predictions: shape (B,inp_seq,P,1,V) for nlp or (B,inP_seq,P,1,F) for time-series. > log probas.
    # trajectories: shape (B,inp_seq,P,1,D) compute Z0,Z1,Z2,...,ZT
    # weights: shape (B,P,1)

    #transformer.summary()

    if transformer.task_type=='classification':
      if tf.shape(predictions)[-1]==1:
        loss = loss_function_binary(real=tar_real,
                                            predictions=predictions,
                                            weights=weights,
                                            transformer=transformer,
                                            SMC_loss=SMC_loss,
                                            classic_loss=classic_loss)
      else:
        loss=loss_function_classification(real=tar_real,
                                      predictions=predictions,
                                      weights=weights,
                                      transformer=transformer,
                                      SMC_loss=SMC_loss,
                                      classic_loss=classic_loss)
    elif transformer.task_type=='regression':
      loss=loss_function_regression(tar_real,
                                      predictions,
                                      weights,
                                      transformer,
                                      SMC_loss=SMC_loss,
                                      classic_loss=classic_loss)
    else:
      raise ValueError('task_type argument in Transformer class is not supported. '
                       'Please choose between "classification" or "regression"')

    print ('loss computed...')

    gradients = tape.gradient(loss, transformer.trainable_variables)

    print('gradients computed...')
  # TO DO: print the list of trainable variables to check that all the PF related ones are not trainable.
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

  print('parameters updated...')

  train_loss(loss)
  train_accuracy(tar_real, predictions)

#-----------------TRAINING--------------------------

if __name__ == "__main__":

  # TF-2.0
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  sys.setrecursionlimit(100000)

  tf.config.experimental_run_functions_eagerly(True)  # to remove TensorInacessibleError
  for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
      if len(tf.shape(inp))==2:
        inp=tf.expand_dims(inp, axis=-1)
      if len(tf.shape(inp))==2:
        tar=tf.expand_dims(tar, axis=-1)
      loss=train_step(inp, tar)

      if batch % 10 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(
          epoch + 1, batch, loss))

    # if (epoch + 1) % 5 == 0:
    #   ckpt_save_path = ckpt_manager.save()
    #   print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
    #                                                       ckpt_save_path))

    # print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
    #                                                     train_loss.result(),
    #                                                     train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
