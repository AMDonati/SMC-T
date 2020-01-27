#TODO: test the classification loss for a number of classes equal to 2.
#TODO: add a mask option in the loss for nlp datasets.

#TODO: debug the mse_with_particles function for the regression case.
import tensorflow as tf
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


### ----------------------- LOSS FUNCTIONS------------------------------------------------------------------------------

def loss_function_classic_T_classif(real, pred):
  '''add a mask in the loss for padded sequences - only useful for nlp dataset.'''
  #mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss_ = loss_object(real, pred)
  #mask = tf.cast(mask, dtype=loss_.dtype)
  #loss_ *= mask
  return tf.reduce_mean(loss_)


def categorical_ce_with_particules(real, pred, sampling_weights):
  '''
  :param real: targets tensor > shape (B,S)
  :param pred: predictions (particules logits) > shape (B,P,S,V)
  :param sampling_weights: re-sampling weights for last timestep > shape (B,P)
  :return:
  '''
  # tiling the targets to have a shape (B,P,S)
  num_particles = tf.shape(pred)[1]

  if len(tf.shape(real)) < 3:
    real = tf.expand_dims(real, axis=1)
    real = tf.tile(real, multiples=[1, num_particles, 1])

  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  loss_ = loss_object(real, pred)  # shape (B,P,S)

  # mean over sequence elements
  #TODO: ask / check if the reduction over the seq dimension should be a sum or a mean... (a sum according to the blog post below)
  # https://towardsdatascience.com/recurrent-neural-networks-rnns-3f06d7653a85
  loss_ = tf.reduce_mean(loss_, axis=-1)  # shape (B,P)
  # weighted sum over number of particles
  loss_ = tf.reduce_sum(sampling_weights * loss_, axis=-1)
  # mean over batch elements
  loss = tf.reduce_mean(loss_, axis=0)
  return loss


def binary_ce_with_particules(real, pred, sampling_weights, from_logits=True):
  '''
  DOES NOT WORK. USE THE Categorical_one instead, event for 2 classes.
  :param real: targets tensor > shape (B,S)
  :param pred: predictions (particules logits) > shape (B,P,S,1)
  :param sampling_weights: re-sampling weights for last timestep > shape (B,P)
  :return:
  '''
  # tiling the targets to have a shape (B,P,S)
  num_particles = tf.shape(pred)[1]
  if len(tf.shape(real)) < 3:
    real = tf.expand_dims(real, axis=1)
    real = tf.tile(real, multiples=[1, num_particles, 1])

  # One-hot encoding of real to have a shape (B,P,S,2)
  real = tf.cast(real, dtype=tf.int32)
  real = tf.one_hot(real, depth=2)

  # TODO: ask Florian if necessary to have padded sequences.
  loss_ = tf.keras.losses.binary_crossentropy(
    y_true=real,
    y_pred=pred,
    from_logits=from_logits,
    label_smoothing=0)  # shape (B,P,S)

  # mean over sequence elements
  loss_ = tf.reduce_mean(loss_, axis=-1)  # shape (B,P)

  # weighted sum over number of particles

  loss_ = tf.reduce_sum(sampling_weights * loss_, axis=-1)  # squeezing weights to have shape (B,P)

  # mean over batch elements
  loss = tf.reduce_mean(loss_, axis=0)

  return loss


def mse_with_particles(real, pred, sampling_weights):
  '''
  :param real: shape (B,P,S,F)
  :param pred: shape (B,P,S,F)
  :param sampling_weights: shape (B,P)
  :return:
  the average mse scalar loss (with a weighted average over the dim number of particles)
  '''
  # reshaping real from (B,P,S) to (B,P,S,1)
  if len(tf.shape(real))==3:
    real=tf.expand_dims(real, axis=-1)
  mse = tf.keras.losses.mse
  loss = mse(y_true=real, y_pred=pred)  # shape (B,P,S)
  #TODO: do a sum over sequence elements instead of a mean?
  loss = tf.reduce_mean(loss, axis=-1)  # shape (B,P)
  # squeezing sampling_weights to have a shape (B,P)
  sampling_weights=tf.squeeze(sampling_weights, axis=-1)
  loss = tf.reduce_sum(sampling_weights * loss)  # shape (B,)
  loss = tf.reduce_mean(loss)
  return loss


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
    loss_smc = -transformer.compute_SMC_log_likelihood(sampling_weights=weights)  # we take a minus because we want to minimize -maximum_likelihood.
  else:
    loss_smc = 0
  loss = loss_ce + loss_smc
  return loss


def loss_function_binary(real, predictions, weights, transformer, classic_loss=True, SMC_loss=True):
  '''DOES NOT WORK...
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
    loss_smc = -transformer.compute_SMC_log_likelihood(real=real, sampling_weights=weights)  # we take a minus because we want to minimize -maximum_likelihood.
  else:
    loss_smc = 0
  loss = loss_ce + loss_smc
  return loss


def loss_function_regression(real, predictions, weights, transformer, classic_loss=True, SMC_loss=True):
  '''
  :param real: targets > shape (B,P,S)
  :param predictions > shape (B,P,S,1)
  :param weights: re-sampling_weights for the last element > shape (B,P)
  :param classic_loss: boolean to compute the classic loss or not (default=True)
  :param SMC_loss: boolean to compute the SMC loss (default=True)
  :return:
  a scalar computing the SMC loss as defined in the paper.
  '''
  num_particles=tf.shape(weights)[1]
  if len(tf.shape(real))==2:
    # tiling targets to add the particle dimensions
    real=tf.expand_dims(real, axis=1)
    real=tf.tile(real, multiples=[1,num_particles,1])
  if classic_loss:
    # TODO: if sigma of weights_computation is not equal to 1. change the mse by a custom SMC_log_likelihood.
    loss_ce = mse_with_particles(real=real, pred=predictions, sampling_weights=weights)
  else:
    loss_ce = 0
  if SMC_loss:
    # take minus the log_likelihood.
    loss_smc = -transformer.compute_SMC_log_likelihood(sampling_weights=weights)  # we take a minus because we want to minimize -maximum_likelihood.
  else:
    loss_smc = 0
  loss = loss_ce + loss_smc

  return loss


# -------------------------------- TRAIN STEP FUNCTIONS ---------------------------------------------------------------------
train_step_signature = [
  tf.TensorSpec(shape=(None, None), dtype=tf.int32),
  tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]
@tf.function(input_signature=train_step_signature)
def train_step_classic_T(inputs, transformer, optimizer, train_loss, targets=None):
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

  train_loss(loss)

  # here train_accuracy=tf.keras.metrics.SparseCategoricalCrossEntropy()

  # tar_real should be the last element of the sequence tar_real=tar_real[:,-1,:], axis=-1) > shape (B,1)
  # predictions should be the last element of the sequence of predictions (of shape (B,S,V)
  # prediction=prediction[:,-1,:] > shape (B,V)

  #train_accuracy(tar_real, predictions)

  return loss

#--------------SMC Transformer train_step------------------------------------

@tf.function(input_signature=train_step_signature)
def train_step_SMC_T(inputs, smc_transformer, optimizer, train_loss, targets=None, SMC_loss=True, classic_loss=True):
  '''
  compute a gradient descent step using categorical crossentropy loss by updating the trainable parameters.
  :param input: input data > shape (B,S) for nlp and univariate time_series.
  multivariate case needs to be implemented still.
  :param target: target data (sequential one) > shape (B,S).
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

  assert len(tf.shape(tar_inp))==2
  assert len(tf.shape(tar_real))==2

  seq_len=tf.shape(tar_inp)[1]
  mask_transformer = create_look_ahead_mask(seq_len)

  with tf.GradientTape() as tape:
    (predictions, trajectories, weights), attn_weights = smc_transformer(inputs=tar_inp,
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
                                    transformer=smc_transformer,
                                    SMC_loss=SMC_loss,
                                    classic_loss=classic_loss)
    else:
      raise ValueError('task_type argument in Transformer class is not supported.'
                       'Please choose between "classification" or "regression"')

    gradients = tape.gradient(loss, smc_transformer.trainable_variables)

  optimizer.apply_gradients(zip(gradients, smc_transformer.trainable_variables))

  train_loss(loss)
  #train_accuracy(tar_real, predictions)

  return loss



# #------ old function not working------
# def categorical_crossentropy(real, logits, sampling_weights):
#     '''formula: mean(over batch)[sum(w(m)*-sum(real*log pred))
#     -args:
#       -real: tensor of dim (B,P,S) or dim (B,S)
#       -pred (logits):tensor of dim (B,P,S,V) with V the vocabulary size.
#       -sampling_weights: tensor of dim (B,P)'''
#     num_particles = tf.shape(sampling_weights)[-1]
#     #if len(tf.shape(real)) < 3:
#       #real = tf.tile(real[:, tf.newaxis, :], multiples=[1, num_particles, 1])  # add particles dimension > dim (B,P,S)
#     pred = tf.reduce_max(logits, axis=-1)  # dim (B, P, S)
#     pred = tf.cast(pred, dtype=tf.float32)
#     real = tf.cast(real, dtype=tf.float32)
#     loss = -tf.reduce_sum(real * tf.math.log(pred), axis=-1)  # dim (B,P)
#     # weighted sum using sampling_weights.
#     loss = tf.reduce_sum(sampling_weights * loss, axis=-1)  # dim (B,)
#     # averaging over the batch
#     loss = tf.reduce_mean(loss, axis=0)
#     print(loss.shape)
#     return loss


if __name__ == "__main__":

  #------------------------ testing of categorical ce with particules function......-----------------------------------------------------
  B=8
  P=1
  S=10
  V=50

  real=tf.ones(shape=(B,P,S))
  logits=tf.random.uniform(shape=(B,P,S,V))
  sampling_weights=tf.ones(shape=(B,P))
  loss=categorical_ce_with_particules(real, logits, sampling_weights)

  print('categorical ce loss for {} classes'.format(V), loss.numpy())

  # test in the binary case:
  V=2
  logits = tf.random.uniform(shape=(B, P, S, V))
  sampling_weights = tf.ones(shape=(B, P))
  loss_binary = categorical_ce_with_particules(real, logits, sampling_weights)

  print('categorical ce loss - binary case', loss_binary.numpy())

  #-------------------- test of mse with particles loss -----------------------------------------------------------------------------------
  B = 8
  P = 1
  S = 10
  V = 1

  real = tf.random.uniform(shape=(B, P, S))
  logits = tf.random.uniform(shape=(B, P, S, V))
  sampling_weights = tf.ones(shape=(B, P, 1))

  loss_regression=mse_with_particles(real=real, pred=logits, sampling_weights=sampling_weights)

  print('regression loss', loss_regression.numpy())









