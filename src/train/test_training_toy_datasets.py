#TODO: debug and add train_accuracy.
#TODO: add the testing on the loss on one batch.
#TODO: add callbacks and checkpoints.
#TODO: add logging (see Florian script to store all the loss values & train accuracy at each epoch / every certain number of batch...).

#TODO: add a validation dataset > compute the training and validation accuracy.
# basic logging tutorial: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial

#TODO: debug the mse_with_particles function for the regression case.
#TODO: debug the issue of the seq_len for the training of the classic Transformer in the NLP dataset (it seems that it always want to process input_data of seq length eqaul to 100...)

#TODO: for the nlp dataset, add a mask to the loss functions for padded sequences...

""""# to store:
# in a fichier .log: for each epoch, the average loss (train & val dataset), the training accuracy (train & val datasets
- 2 accuracies for the SMC Transformer), the time taken for each epoch, and the total training time
# - in a fichier .txt: the list of losses (train & val), accuracies (train & val) for plotting & comparing.
# - in files .npy: the output of the model (predictions, trajectories, attention_weights...).
# - checkpoints of the model in a file .ckpt.
# dictionary of hparams (cf Nicolas's script...).
"""

import tensorflow as tf
from models.Baselines.Transformer_without_enc import Transformer
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from train.loss_functions import train_step_classic_T
from train.loss_functions import train_step_SMC_T

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer

import time
import sys

from preprocessing.time_series.df_to_dataset import df_to_dataset
from preprocessing.time_series.df_to_dataset import df_continuous_to_dataset
from preprocessing.NLP.text_to_dataset import text_to_dataset

data_type = 'nlp'
task_type = 'classification'
resampling=True

#------------------UPLOAD the training dataset------------------------------------------------------------------------------------------------
if data_type=='time_series':

  file_path = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
  fname = 'jena_climate_2009_2016.csv.zip'
  col_name = 'T (degC)'
  index_name = 'Date Time'
  TRAIN_SPLIT = 0.8
  min_value = -25
  size_bin = 5
  num_bins = 12
  buffer_frac = 0.2 # fraction of total dataset to be taken in the buffer when shuffling.
  seq_len =10 # one more than for the transformer.
  BATCH_SIZE = 128 # small batch_size to avoid memory errors.
  num_bins = 12 # correspond to the number of classes for a classification task
  num_classes=num_bins
  reduce_for_test = 5000 # taking only 200,000 samples for testing.

  if task_type=='classification':

    reduce_for_test=None

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
                                                                      seq_len=seq_len,
                                                                      reduce_for_test=reduce_for_test)

    #TODO: save the data pre-processed in a .npy file so that the function is not called every-time.

    print('multi-class classification problem with {} classes...'.format(num_classes))
    num_samples_training=x_train.shape[0]
    print('number of training samples...', num_samples_training)
    df_train_samples=df_categorized[:num_samples_training]

    num_batches=int(num_samples_training/BATCH_SIZE)
    print('number of batches...', num_batches)

    #TODO: replace class numbers in original df_to_dataset function so that it works when reducing nulmber of smaples for testing.
    print('classes distributions for training data', df_train_samples.value_counts())

  elif task_type=='regression':

    resampling=False

    train_dataset, val_dataset, uni_data_df, x_train=df_continuous_to_dataset(file_path=file_path,
                                                                          fname=fname,
                                                                          col_name=col_name,
                                                                          index_name=index_name,
                                                                          train_split=TRAIN_SPLIT,
                                                                          batch_size=BATCH_SIZE,
                                                                          buffer_frac=buffer_frac,
                                                                          seq_len=seq_len,
                                                                          reduce_for_test=reduce_for_test)

    print('head of original dataset', uni_data_df.head())
    print('first samples of corresponding numpy array:', x_train[:10])

    num_samples_training=x_train.shape[0]
    print('number of training samples...', num_samples_training)
    num_batches=int(num_samples_training/BATCH_SIZE)
    print ('number of batches...', num_batches)

elif data_type=='nlp':
  file_path = tf.keras.utils.get_file('shakespeare.txt',
                                      'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
  file_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/data/shakespeare_reduced_for_testing.txt'

  BATCH_SIZE = 512
  BUFFER_SIZE = 10000
  seq_len = 50
  train_split=0.9
  train_dataset, val_dataset, num_classes, training_samples = text_to_dataset(file_path=file_path,
                                               seq_len=seq_len,
                                               buffer_size=BUFFER_SIZE,
                                               train_split=train_split,
                                               batch_size=BATCH_SIZE)

  steps_per_epochs=int(training_samples/BATCH_SIZE)

# -------define hyperparameters----------------------------------------------------------------------------------------------------------------
## Optimizer
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

# Model's hyper-parameters.
num_particles = 5
num_heads = 2
d_model = 64
dff = 1024
maximum_position_encoding_baseline=25
maximum_position_encoding_smc=None
target_vocab_size = num_classes if task_type=='classification' else 1 # correspond to the number of classes: multi-class classification problem.
num_layers = 1
sigma=1
noise_encoder=False
noise_SMC_layer=True


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
  EPOCHS = 20
  train_smc_transformer=True
  train_classic_transformer=False

  print('task type...', task_type)
  print('data type...', data_type)

  print('number of heads...', num_heads)
  print('depth model', d_model)
  print('num_classes', num_classes)

  print('steps per epochs', steps_per_epochs)

  print_loss=100

  #resampling=True if num_particles > 1 else False
  resampling=False

  print('resampling trajectories?', resampling)

  # -------------------------------------------TRAIN ON THE DATASET - CLASSIC TRANSFORMER -------------------------------------------
  if train_classic_transformer:
  #TODO: solve problem of sequence length & max_position_encoding in the classic model
  # (it seems that the output of the attention weights are of shape (B,H,S*E, S*E) instead of being of size (B,H,S,S).
    if maximum_position_encoding_baseline is not None:
      print('training a baseline transformer with positional encoding of size: {}'.format(maximum_position_encoding_baseline))

    # Transformer - baseline.
    transformer = Transformer(num_layers=num_layers,
                              d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,
                              target_vocab_size=target_vocab_size,
                              maximum_position_encoding=maximum_position_encoding_baseline,
                              data_type=data_type)

    # TEST THE LOSS ON A BATCH
    #TODO: adapt this for our case.
    # for input_example_batch, target_example_batch in dataset.take(1):
    #   example_batch_predictions = model(input_example_batch)
    #   print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    # example_batch_loss = loss(target_example_batch, example_batch_predictions)
    # print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    # print("scalar_loss:      ", example_batch_loss.numpy().mean())



    for epoch in range(EPOCHS):
      start = time.time()

      train_loss.reset_states()
      #TODO: in train_step_classic_T, add the option of the loss function for the regression_case.
      for (batch, (inp, tar)) in enumerate(dataset):
        loss_baseline, average_loss_batch, train_accuracy_batch = train_step_classic_T(inputs=inp,
                                             targets=tar,
                                             transformer=transformer,
                                             train_loss=train_loss,
                                             train_accuracy=train_accuracy,
                                             optimizer=optimizer,
                                             data_type=data_type)

        if batch % print_loss == 0:
          print('epoch', epoch)
          print('batch', batch)
          print('loss -  Baseline Transformer', loss_baseline.numpy())
          print('average loss', average_loss_batch.numpy())
          print('accuracy - Baseline Transformer', train_accuracy_batch.numpy())

      print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    print('training of a classic Transformer for weather dataset done...')


#-------------------TRAINING ON THE DATASET - SMC_TRANSFORMER-----------------------------------------------------------------------------------------------------------
  if train_smc_transformer:

    print('number of particles', num_particles)
    print ('noise in SMC_layer?', noise_SMC_layer)
    print('resampling?', resampling)

    smc_transformer=SMC_Transformer(num_layers=num_layers,
                          d_model=d_model,
                          num_heads=num_heads,
                          dff=dff,
                          target_vocab_size=target_vocab_size,
                          maximum_position_encoding=maximum_position_encoding_smc,
                          num_particles=num_particles,
                          sigma=sigma,
                          noise_encoder=noise_encoder,
                          noise_SMC_layer=noise_SMC_layer,
                          seq_len=seq_len,
                          data_type=data_type,
                          task_type=task_type,
                          resampling=resampling)

    #print('SMC transformer model summary...', smc_transformer.summary())

    # check the pass forward.
    for input_example_batch, target_example_batch in dataset.take(1):
      (example_batch_predictions, _, _), (average_predictions, max_predictions), _ = smc_transformer(inputs=input_example_batch,
                                              training=False,
                                              mask=create_look_ahead_mask(seq_len))
      print("predictions shape", example_batch_predictions.shape)

    for epoch in range(EPOCHS):
      start = time.time()

      train_loss.reset_states()
      train_accuracy.reset_states()

      for (batch, (inp, tar)) in enumerate(dataset):
        loss_smc, average_loss_batch, train_accuracy_average_pred, train_accuracy_max_pred=train_step_SMC_T(inputs=inp,
                                  targets=tar,
                                  smc_transformer=smc_transformer,
                                  optimizer=optimizer,
                                  train_loss=train_loss,
                                  train_accuracy=train_accuracy,
                                  classic_loss=True,
                                  SMC_loss=True)
        # if noise_SMC_layer:
        #   loss_smc_classic_part=train_step_SMC_T(inputs=inp,
        #                           targets=tar,
        #                           smc_transformer=smc_transformer,
        #                           optimizer=optimizer,
        #                           train_loss=train_loss,
        #                           train_accuracy=train_accuracy,
        #                           classic_loss=True,
        #                           SMC_loss=False)

        if batch % print_loss == 0:
          print('epoch', epoch+1)
          # if noise_SMC_layer:
          #   print('loss SMC Transformer - classic part', loss_smc_classic_part.numpy())
          print('average SMC loss - SMC Transformer', average_loss_batch.numpy())
          print('accuracy from average predictions - SMC Transformer', train_accuracy_average_pred.numpy())

      # if (epoch + 1) % 5 == 0:
      #   ckpt_save_path = ckpt_manager.save()
      #   print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
      #                                                       ckpt_save_path))
      # print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1,
      #                                                     train_loss.result(),
      #                                                     train_accuracy.result()))

      print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    print('training of SMC Transformer for weather dataset done...')
