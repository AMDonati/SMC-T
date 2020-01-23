#TODO: add the attention weights
#TODO: implement the time-window lag.

#TODO: error raise sur la dimension. > tar_inp should be dim (B,S)
#TODO: uniformiser le processus experimental.

import time
import numpy as np
import sys
import tensorflow as tf

# additional imports
from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from neural_toolbox.training_algos import categorical_ce_with_particules
from neural_toolbox.training_algos import mse_with_particles

## Optimizer
learning_rate=0.001
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

EPOCHS = 5

# generate dummy dataset
num_particles = 5
seq_len = 10
num_heads = 2
d_model = 12
dff = 24
target_vocab_size = 50 # here for time_series > regression problem.
# mean square error...
pe_target = 25
num_layers = 3
batch_size=8
data_type = 'time_series'
task_type = 'classification'
noise=False

dummy_sample = np.random.choice(np.arange(target_vocab_size), size=seq_len)
dummy_list = [np.random.choice(np.arange(target_vocab_size), size=seq_len) for _ in range(batch_size)]
dummy_array = np.array(dummy_list)
dummy_dataset = tf.constant(dummy_array, dtype=tf.int32)


#----create the SMC Transformer:
transformer = SMC_Transformer(num_layers=num_layers,
                              d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,
                              target_vocab_size=target_vocab_size,
                              maximum_position_encoding=pe_target,
                              num_particles=num_particles,
                              sigma=1,
                              seq_len=seq_len-1,
                              data_type=data_type,
                              task_type=task_type,
                              noise=noise)

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
    loss_ce=categorical_ce_with_particules(real=real, pred=predictions, sampling_weights=weights)
  else:
    loss_ce=0
  if SMC_loss:
    loss_smc=-transformer.compute_SMC_log_likelihood(real=real, sampling_weights=weights) # we take a minus because we want to minimize -maximum_likelihood.
  else:
    loss_smc=0
  loss=loss_ce+loss_smc
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
    #TODO: if sigma of weights_computation is not equal to 1. change the mse by a custom SMC_log_likelihood.
    loss_ce=mse_with_particles(real=real, pred=predictions, sampling_weights=weights)
  else:
    loss_ce=0
  if SMC_loss:
    # take minus the log_likelihood.
    loss_smc=-transformer.compute_SMC_log_likelihood(real=real, sampling_weights=weights) # we take a minus because we want to minimize -maximum_likelihood.
  else:
    loss_smc=0
  loss=loss_ce+loss_smc
  return loss

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

  mask = create_look_ahead_mask(seq_len-1)

  with tf.GradientTape() as tape:
    predictions, trajectories, weights = transformer(inputs=tar_inp,
                                                     training=True,mask=mask)

    # predictions: shape (B,inp_seq,P,1,V) for nlp or (B,inP_seq,P,1,F) for time-series. > log probas.
    # trajectories: shape (B,inp_seq,P,1,D) compute Z0,Z1,Z2,...,ZT
    # weights: shape (B,P)

    # reshaping Tranformer outputs to put them in the loss.
    #transformer.summary()

    #TODO: solve this issue of sequence length.
    #tar_real=tar_real[:,:seq_length-2] # trick to have the right shape.

    if transformer.task_type=='classification':
      loss=loss_function_classification(tar_real,
                                      predictions,
                                      weights,
                                      transformer,
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

  return loss

if __name__ == "__main__":

  noise_update_step=1

  # TF-2.0
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  # tf.debugging.set_log_device_placement(True)

  sys.setrecursionlimit(100000)

  tf.config.experimental_run_functions_eagerly(True)  # to remove TensorInacessibleError
  for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    loss = train_step(dummy_dataset) # issue here: categorical_crossentropy_loss is not a scalar...
    print('epoch', epoch)
    print('loss', loss)

    #if (epoch+1) % noise_update_step ==0:
      #transformer.compute_direct_update_cov_matrix() # put this inside a tf.gradient?
      #print('updating covariance matrix parameter for the reparametrized noise')

    #print('Epoch {} Loss {}'.format((epoch, loss)))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

