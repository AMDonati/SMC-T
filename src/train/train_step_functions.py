# TODO: test the classification loss for a number of classes equal to 2.
# TODO: debug the mse_with_particles function for the regression case.
import tensorflow as tf
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask

from train.loss_functions import loss_function_classic_T_classif
from train.loss_functions import loss_function_classification
from train.loss_functions import loss_function_regression

# -------------------------------- TRAIN STEP FUNCTIONS ---------------------------------------------------------------------
train_step_signature = [
  tf.TensorSpec(shape=(None, None), dtype=tf.int32),
  tf.TensorSpec(shape=(None, None), dtype=tf.int32),
]

@tf.function(input_signature=train_step_signature)
def train_step_classic_T(inputs,
                         transformer,
                         optimizer,
                         train_accuracy,
                         data_type,
                         task_type,
                         targets=None,
                         perplexity_metric=None):
  '''training step for the classic Transformer model'''
  if targets is None:
    tar_inp = inputs[:, :-1]
    tar_real = inputs[:, 1:]
  else:
    tar_inp = inputs
    tar_real = targets

  if len(tf.shape(tar_inp)) == 2:
    tar_inp = tf.expand_dims(tar_inp, axis=-1)
  if len(tf.shape(tar_real)) == 2:
    tar_real = tf.expand_dims(tar_real, axis=-1)

  # CAUTION. Unlike the SMC_Transformer, the inputs and targets need to be of shape (B,S,1).
  assert len(tf.shape(tar_inp)) == 3
  assert len(tf.shape(tar_real)) == 3

  seq_len = tf.shape(tar_inp)[1]
  mask_transformer = create_look_ahead_mask(seq_len)

  with tf.GradientTape() as tape:
    predictions, _ = transformer(inputs=tar_inp, training=True, mask=mask_transformer)

    if task_type == 'classification':
      loss = loss_function_classic_T_classif(real=tar_real, pred=predictions, data_type=data_type)
    elif task_type == 'regression':
      loss = tf.keras.losses.MSE(tar_real, predictions)
    else:
      raise ValueError("task_type should be either 'regression' or 'classification'")

    # averaging loss over the seq and batch dims
    loss = tf.reduce_mean(loss, axis=-1) # (B,)
    loss = tf.reduce_mean(loss, axis=-1)

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))


  if task_type == 'classification':
    train_accuracy_batch = train_accuracy(tar_real, predictions)

    if perplexity_metric is not None:
      # input predictions of the perplexity metric needs to be of shape (B,S) (and not (B,S,1)
      train_perplexity = perplexity_metric(tf.expand_dims(tar_real, axis=-1), predictions)
    else:
      train_perplexity = None

    return loss, train_accuracy_batch, train_perplexity

  elif task_type == 'regression':
    return loss, None, None

# --------------SMC Transformer train_step------------------------------------

@tf.function(input_signature=train_step_signature)
def train_step_SMC_T(inputs,
                     smc_transformer,
                     optimizer,
                     train_accuracy,
                     targets=None,
                     perplexity_metric=None,
                     SMC_loss=True,
                     classic_loss=True):
  '''
  compute a gradient descent step using categorical crossentropy loss by updating the trainable parameters.
  :param input: input data > shape (B,S) for nlp and univariate time_series.
  multivariate case needs to be implemented still.
  :param target: target data (sequential one) > shape (B,S).
  :param SMC_loss: boolean to compute SMC_loss or not. Default is False.
  :param classic_loss: boolean to compute classic cross-entropy loss, or not. Default is True.
  :return:
  The updated loss, the training accuracy (from average predictions and from max predictions).
  '''

  if targets is None:
    tar_inp = inputs[:, :-1]
    tar_real = inputs[:, 1:]
  else:
    tar_inp = inputs
    tar_real = targets # (B,S,F)

  seq_len = tf.shape(tar_inp)[1]
  mask_transformer = create_look_ahead_mask(seq_len)

  with tf.GradientTape() as tape:
    (predictions, trajectories, weights, ind_matrix), attn_weights = smc_transformer(inputs=tar_inp,
                                                                                             training=True,
                                                                                             mask=mask_transformer)
    # predictions: shape (B,P,S,C) > sequence of log_probas for the classification task.
    # trajectories: shape (B,P,S,D) = [z0,z1,z2,...,zT]
    # weights: shape (B,P,1) = w_T: used in the computation of the loss.

    #train_inf_pred_batch, train_avg_pred_batch, train_max_pred_batch = predictions_metric

    if smc_transformer.task_type == 'classification':
      assert tf.shape(predictions)[-1] > 2
      loss = loss_function_classification(real=tar_real,
                                          predictions=predictions,
                                          weights=weights,
                                          transformer=smc_transformer,
                                          SMC_loss=SMC_loss,
                                          classic_loss=classic_loss)

    elif smc_transformer.task_type == 'regression':
      loss, loss_mse, loss_mse_from_avg_pred, loss_mse_std = loss_function_regression(real=tar_real,
                                      predictions=predictions,
                                      weights=weights,
                                      transformer=smc_transformer,
                                      SMC_loss=SMC_loss,
                                      classic_loss=classic_loss)
    else:
      raise ValueError('task_type argument in Transformer class is not supported.'
                       'Please choose between "classification" or "regression"')

    trainable_variables = list(smc_transformer.trainable_variables)
    trainable_variables_names = [t.name for t in trainable_variables]
    gradients = tape.gradient(loss, smc_transformer.trainable_variables)
    #var_and_grad_dict = dict(zip(trainable_variables_names, gradients))
    #print('dict of variables and associated gradients', var_and_grad_dict)

  optimizer.apply_gradients(zip(gradients, smc_transformer.trainable_variables))

  # if smc_transformer.task_type == 'classification':
  #   train_inf_batch = train_accuracy(tar_real, train_inf_pred_batch)  # accuracy from average_predictions for now.
  #   train_avg_acc_batch = train_accuracy(tar_real, train_avg_pred_batch)  # average over logits instead of after softmax (inference case).
  #   train_max_acc_batch = train_accuracy(tar_real, train_max_pred_batch)
  #   train_metrics = (train_inf_batch, train_avg_acc_batch, train_max_acc_batch)
  # else:
  train_metrics = (loss_mse, loss_mse_from_avg_pred, loss_mse_std)

  if perplexity_metric is not None:
    train_perplexity = perplexity_metric(tar_real, predictions)
  else:
    train_perplexity = None

  return loss, train_metrics, train_perplexity


@tf.function
def train_step_rnn_regression(inp, target, model, optimizer):
  #inp = tf.expand_dims(inp, axis=-1)
  assert len(tf.shape(inp)) == 3
  with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = tf.keras.losses.MSE(target, predictions)
  loss = tf.reduce_mean(loss, axis=-1)
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss

@tf.function
def train_step_rnn_classif(inp, target, model, optimizer, accuracy_metric):
  with tf.GradientTape() as tape:
    predictions = model(inp)
    loss = tf.reduce_mean(
      tf.keras.losses.sparse_categorical_crossentropy(target, predictions, from_logits=True))
  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  train_acc_batch = accuracy_metric(target, predictions)
  return loss, train_acc_batch

# #------ old function not working----------------------------------------------------------------------------------------------------
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


#if __name__ == "__main__":











