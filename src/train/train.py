#TODO: add the Dummy Dataset
#TODO: look @ the train_loss
#TODO: look @ the tutorial of the text generation to add stuff for the dataset.

import tensorflow as tf
from models.SMC_Transformer.SMC_Transformer import Transformer as SMC_Transformer
from models.Baselines.Transformer_without_enc import Transformer
from
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from neural_toolbox.training_algos import loss_function_classification
from neural_toolbox.training_algos import loss_function_regression
import time
import sys
from preprocessing.time_series import df_to_dataset

#------------------UPLOAD the training dataset--------------------------------------------

file_path = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
fname = 'jena_climate_2009_2016.csv.zip'
col_name = 'T (degC)'
index_name = 'Date Time'
#TODO: put this as a fraction to work with number of samples.
TRAIN_SPLIT = 300000
min_value = -25
size_bin = 5
num_bins = 12
BATCH_SIZE = 256
buffer_size = 10000
seq_len_dataset =10 # one more than for the transformer.

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
                                                               buffer_size=buffer_size,
                                                               seq_len=seq_len_dataset)

# -------define hyperparameters-----------------------------------------------------------------

## Optimizer
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

EPOCHS = 5

# generate dummy dataset
num_particles = 5
num_heads = 2
d_model = 4
dff = 8
maximum_position_encoding = 12  # correspond to the number of classes: multi-class classification problem.
pe_target = None # no positional encoding for time_series dataset.
num_layers = 2
data_type = 'time_series'
task_type = 'classification'
seq_len=9

#----DEFINE THE MODEL------------------------------------------------------------------------------------------------

# SMC_Transformer
smc_transformer = SMC_Transformer(num_layers=num_layers,
                                  d_model=d_model,
                                  num_heads=num_heads,
                                  dff=dff,
                                  target_vocab_size=maximum_position_encoding,
                                  maximum_position_encoding=pe_target,
                                  num_particles=num_particles,
                                  sigma=1,
                                  seq_len=seq_len,
                                  data_type=data_type,
                                  task_type=task_type)

# Transformer - baseline.
transformer=Transformer(num_layers=num_layers,
                        d_model=d_model,
                        num_heads=num_heads,
                        dff=dff,
                        target_vocab_size=maximum_position_encoding,
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

@tf.function(input_signature=train_step_signature)
def train_step(inputs, targets=None, SMC_loss=True, classic_loss=True):
  '''
  compute a SGD step using categorical crossentropy loss by updating the trainable parameters.
  :param input: input data > shape (B,S) for nlp and univariate time_series.
  multivariate case needs to be implemented still.
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

  mask_transformer = create_look_ahead_mask(seq_len)

  with tf.GradientTape() as tape:
    predictions, trajectories, weights = smc_transformer(inputs=tar_inp,
                                                         training=True,
                                                         mask=mask_transformer)

    # predictions: shape (B,P,S,C) > sequence of log_probas for the classification task.
    # trajectories: shape (B,P,S,D) = [z0,z1,z2,...,zT]
    # weights: shape (B,P,1) = w_T: used in the computation of the loss.

    smc_transformer.summary()

    if smc_transformer.task_type== 'classification':
      # #print('shape of last dim of predictions', tf.shape(predictions)[-1])
      # if tf.shape(predictions)[-1]==2:
      #   print('binary classification task...')
      #   loss = loss_function_binary(real=tar_real,
      #                                       predictions=predictions,
      #                                       weights=weights,
      #                                       transformer=transformer,
      #                                       SMC_loss=SMC_loss,
      #                                       classic_loss=classic_loss)
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

    #print ('loss computed...')

    gradients = tape.gradient(loss, smc_transformer.trainable_variables)

    #print('gradients computed...')
  optimizer.apply_gradients(zip(gradients, smc_transformer.trainable_variables))

  #print('parameters updated...')

  train_loss(loss)
  #train_accuracy(tar_real, predictions)

#-----------------TRAINING---------------------------------------------------------------------------------

if __name__ == "__main__":

  # TF-2.0
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

  sys.setrecursionlimit(100000)

  tf.config.experimental_run_functions_eagerly(True)  # to remove TensorInacessibleError
  for epoch in range(EPOCHS):
    start = time.time()

    #TODO: ask Florian why we need reset_states
    train_loss.reset_states()
    #train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(train_dataset):
      if len(tf.shape(inp))==2:
        inp=tf.expand_dims(inp, axis=-1)
      if len(tf.shape(inp))==2:
        tar=tf.expand_dims(tar, axis=-1)
      loss=train_step(inp, tar)

      if batch % 10 == 0:
        print('epoch', epoch)
        print('loss', loss)

    # if (epoch + 1) % 5 == 0:
    #   ckpt_save_path = ckpt_manager.save()
    #   print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
    #                                                       ckpt_save_path))

    # print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
    #                                                     train_loss.result(),
    #                                                     train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
