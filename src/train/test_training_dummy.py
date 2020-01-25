import tensorflow as tf

from models.Baselines.Transformer_without_enc import Transformer
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from train.training_algos import loss_function_classification
from train.training_algos import loss_function_regression
from train.training_algos import loss_function_classic_T_classif

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer

import time
import sys
import numpy as np

#------------------DUMMY DATASET TO TEST AS A START-------------------------------------------------------------------------------------------
seq_len_dataset =10 # one more than for the transformer.
BATCH_SIZE = 64
num_bins = 12 # correspond to the number of classes for a classification task.
dummy_sample = np.random.choice(np.arange(num_bins), size=seq_len_dataset)
dummy_list = [np.random.choice(np.arange(num_bins), size=seq_len_dataset) for _ in range(BATCH_SIZE)]
dummy_array = np.array(dummy_list)
dummy_dataset = tf.constant(dummy_array, dtype=tf.int32)

# -------define hyperparameters----------------------------------------------------------------------------------------------------------------

## Optimizer
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# generate dummy dataset
num_particles = 1
num_heads = 2
d_model = 4
dff = 8
maximum_position_encoding = None  # no positional encoding for time_series dataset.
target_vocab_size = 12 # correspond to the number of classes: multi-class classification problem.
num_layers = 2
data_type = 'time_series'
task_type = 'classification'
seq_len=9
sigma=1
noise_encoder=False
noise_SMC_layer=False
#----DEFINE THE MODELS---------------------------------------------------------------------------------------------------------------------------------------------------------------
# SMC_Transformer
smc_transformer = SMC_Transformer(num_layers=num_layers,
                        d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        target_vocab_size=target_vocab_size,
                        maximum_position_encoding=maximum_position_encoding,
                        num_particles=num_particles,
                        sigma=sigma,
                        noise_encoder=noise_encoder,
                        noise_SMC_layer=noise_SMC_layer,
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
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------
@tf.function(input_signature=train_step_signature)
def train_step_dummy_SMC_T(inputs, targets=None, SMC_loss=True, classic_loss=True):
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
    assert len(tf.shape(inputs))==2
    tar_inp = inputs[:, :-1]
    tar_real = inputs[:, 1:]
  else:
    assert len(tf.shape(inputs)) == 2
    tar_inp=inputs
    assert len(tf.shape(inputs)) == 2
    tar_real=targets

  seq_len=tf.shape(tar_inp)[1]
  mask = create_look_ahead_mask(seq_len)

  with tf.GradientTape() as tape:
    predictions, trajectories, weights = smc_transformer(inputs=tar_inp,
                                                     training=True,
                                                     mask=mask)
    # predictions: shape (B,inp_seq,P,1,V) for nlp or (B,inP_seq,P,1,F) for time-series. > log probas.
    # trajectories: shape (B,inp_seq,P,1,D) compute Z0,Z1,Z2,...,ZT
    # weights: shape (B,P)
    if smc_transformer.task_type=='classification':
      loss=loss_function_classification(real=tar_real,
                                      predictions=predictions,
                                      weights=weights,
                                      transformer=smc_transformer,
                                      SMC_loss=SMC_loss,
                                      classic_loss=classic_loss)
    elif smc_transformer.task_type=='regression':
      loss=loss_function_regression(real=tar_real,
                                      predictions=predictions,
                                      weights=weights,
                                      transformer=smc_transformer,
                                      SMC_loss=SMC_loss,
                                      classic_loss=classic_loss)
    else:
      raise ValueError('task_type argument in Transformer class is not supported. '
                       'Please choose between "classification" or "regression"')

    gradients = tape.gradient(loss, smc_transformer.trainable_variables)

  # TO DO: print the list of trainable variables to check that all the PF related ones are not trainable.
  optimizer.apply_gradients(zip(gradients, smc_transformer.trainable_variables))

  #train_loss(loss)

  return loss

#----------------------------------------------------------------------------------------------------------------------------------------------------------------

@tf.function(input_signature=train_step_signature)
def train_step_dummy_classic_T(inputs, targets=None):
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
  EPOCHS=5

  dataset=dummy_dataset
  smc_transformer=SMC_Transformer(num_layers=num_layers,
                        d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        target_vocab_size=target_vocab_size,
                        maximum_position_encoding=maximum_position_encoding,
                        num_particles=num_particles,
                        sigma=sigma,
                        noise_encoder=noise_encoder,
                        noise_SMC_layer=noise_SMC_layer,
                        seq_len=seq_len,
                        data_type=data_type,
                        task_type=task_type)



  for epoch in range(EPOCHS):
    start = time.time()
    loss_smc=train_step_dummy_SMC_T(inputs=dummy_dataset,
                                    targets=None,
                                    classic_loss=True,
                                    SMC_loss=False)
    loss_smc_w_smc_part=train_step_dummy_SMC_T(inputs=dummy_dataset,
                                    targets=None,
                                    classic_loss=True,
                                    SMC_loss=True)

    print('epoch', epoch)
    print('loss - SMC_Transformer - classic part', loss_smc)
    print('loss - SMC_transformer - total loss', loss_smc_w_smc_part)

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

  print('SMC Transformer model summary...', smc_transformer.summary())

  print('training of SMC Transformer for dummy dataset done...')

 #-------------------------------------------TRAIN ON DUMMY DATASET - CLASSIC TRANSFORMER -------------------------------------------
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

    loss_baseline=train_step_dummy_classic_T(inputs=dummy_dataset)

    print('epoch', epoch)
    print('loss - Baseline', loss_baseline)

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

  print('training of Classic Transformer for dummy dataset done...')
