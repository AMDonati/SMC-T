#TODO: add the testing on the loss on one batch.
#TODO: improve the logging display...

# basic logging tutorial: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial

#TODO: debug the issue of the seq_len for the training of the classic Transformer in the NLP dataset (it seems that it always want to process input_data of seq length eqaul to 100...)

""""# to store:
# in a fichier .log: for each epoch, the average loss (train & val dataset),
the training accuracy (train & val datasets),
- 2 accuracies for the SMC Transformer), the time taken for each epoch, and the total training time
# - in a fichier .txt: the list of losses (train & val), accuracies (train & val) for plotting & comparing.
# - in files .npy: the output of the model (predictions, trajectories, attention_weights...).
# - checkpoints of the model in a file .ckpt.
# dictionary of hparams (cf Nicolas's script...).
"""

import tensorflow as tf
from models.Baselines.Transformer_without_enc import Transformer
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from train.training_algos import train_step_classic_T
from train.training_algos import train_step_SMC_T
from train.Perplexity import PerplexityMetric

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer

import time
import sys
import os
import logging
import numpy as np

from preprocessing.NLP.text_to_dataset import text_to_dataset

from utils.utils_train import write_to_csv
from utils.utils_train import create_run_dir

data_type='nlp'
task_type='classification'


#------------------UPLOAD the training dataset------------------------------------------------------------------------------------------------
file_path = tf.keras.utils.get_file('shakespeare.txt',
                                    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
BATCH_SIZE = 64
BUFFER_SIZE = 10000
seq_len = 100
train_split = 0.9
train_dataset, val_dataset, num_classes, training_samples = text_to_dataset(file_path=file_path,
                                                          seq_len=seq_len,
                                                          train_split=train_split,
                                                          buffer_size=BUFFER_SIZE,
                                                          batch_size=BATCH_SIZE)

# -------define hyperparameters----------------------------------------------------------------------------------------------------------------
## Optimizer
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
perplexity=PerplexityMetric(name='perplexity')

# Model's hyper-parameters.
num_particles = 5
num_heads = 4
d_model = 4
dff = 8

maximum_position_encoding_smc = None
maximum_position_encoding_baseline=None

target_vocab_size = num_classes
num_layers = 1

sigma=1
noise_encoder=False
noise_SMC_layer=True
resampling=True

#-------------------- SIMPLE BASELINE FOR COMPARISON --------------------------------------------------------------------
#TODO: adapt this with the variables of this script.


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
  train_classic_transformer=True

  print('number of heads...', num_heads)
  print('depth model', d_model)

  # ------------- preparing the OUTPUT FOLDER------------------------------------------------------------------------

  # add utils to create directories. (cf Nicolas script.)

  output_path = '../../output'
  out_folder = '{}_{}_heads_{}_particles_{}_depth_{}_sigma_{}_noise_{}'.format(data_type,
                                                                               task_type,
                                                                               num_heads,
                                                                               num_particles,
                                                                               d_model,
                                                                               sigma,
                                                                               noise_SMC_layer)

  output_path = create_run_dir(path_dir=output_path, path_name=out_folder)

  # create the logging:
  out_file_log=os.path.join(output_path, 'training_log.log')
  logging.basicConfig(filename=out_file_log, level=logging.INFO)

  #  creating the checkpoint manager:
  checkpoint_path = create_run_dir(output_path, "checkpoints")

  # -------------------------------------------TRAIN ON THE DATASET - CLASSIC TRANSFORMER -------------------------------------------
  if train_classic_transformer:
    logging.info("training the baseline Transformer on the nlp dataset...")

    # storing the losses & accuracy in a list for each epoch
    average_losses_baseline=[]
    training_accuracies_baseline=[]
    val_accuracy=[]

    if maximum_position_encoding_baseline is not None:
      logging.info('training a baseline transformer with positional encoding...')

    # Transformer - baseline.
    transformer = Transformer(num_layers=num_layers,
                              d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,
                              target_vocab_size=target_vocab_size,
                              maximum_position_encoding=maximum_position_encoding_baseline,
                              data_type=data_type)

    # creating checkpoint manager
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    baseline_ckpt_path=os.path.join(checkpoint_path, "transformer_baseline")

    ckpt_manager = tf.train.CheckpointManager(ckpt, baseline_ckpt_path)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      print('Latest checkpoint restored!!')

    for epoch in range(EPOCHS):
      start = time.time()

      train_loss.reset_states()
      for (batch, (inp, tar)) in enumerate(dataset):
        _, average_loss_batch, train_accuracy_batch = train_step_classic_T(inputs=inp,
                                             targets=tar,
                                             transformer=transformer,
                                             train_loss=train_loss,
                                             train_accuracy=train_accuracy,
                                             optimizer=optimizer,
                                             data_type=data_type)

      logging.info('epoch {} - training_loss {} - training_accuracy {}'.format((epoch+1,average_loss_batch, train_accuracy_batch)))
      logging.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

      # saving loss and metrics information:
      average_losses_baseline.append(average_loss_batch.numpy())
      training_accuracies_baseline.append(train_accuracy_batch.numpy())

    logging.info('total training time for {} epochs:{}'.format((EPOCHS,time.time() - start)))

    # storing history of losses and accuracies in a csv file
    keys=['loss', 'accuracy']
    values=[average_losses_baseline, training_accuracies_baseline]
    history=dict(zip(keys,values))
    # save it on a .txt file:
    baseline_history_fn=os.path.join(output_path, 'baseline_history.csv') # use a create_directory function instead.
    write_to_csv(baseline_history_fn, history)
    #TODO: use function write_to_csv instead.
    #np.savetxt(history_path, history)
    logging.info('saving loss and metrics information...')

    # making predictions with the trained model and saving them on .npy files
    mask=create_look_ahead_mask(seq_len)
    predictions_bas, attn_weights_bas=transformer(input=val_dataset,
                                                  training=False,
                                                  mask=mask)
    model_output_path=create_run_dir(path_dir=output_path, path_name="model_outputs")
    predictions_fn=os.path.join(model_output_path, 'baseline_predictions.npy')
    attn_weights_fn=os.path.join(model_output_path, 'baseline_attn_weights.npy')
    np.save(predictions_fn, predictions_bas)
    np.save(attn_weights_fn, attn_weights_bas)
    logging.info("saving model output in .npy files...")

    logging.info('training of a classic Transformer for a nlp dataset done...')

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

    # creating checkpoint manager
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    smc_T_ckpt_path = os.path.join(checkpoint_path, "SMC_transformer")

    ckpt_manager = tf.train.CheckpointManager(ckpt, smc_T_ckpt_path)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      logging.info('Latest checkpoint restored!!')

    logging.info('SMC transformer model summary...', smc_transformer.summary())

    # check the pass forward.
    for input_example_batch, target_example_batch in dataset.take(1):
      (example_batch_predictions, _, _), (average_predictions, max_predictions), _ = smc_transformer(inputs=input_example_batch,
                                              training=False,
                                              mask=create_look_ahead_mask(seq_len))
      logging.info("predictions shape: {}", example_batch_predictions.shape)

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

        # print('epoch', epoch)
        # print('batch', batch)
        # print('loss - SMC Transformer', loss_smc.numpy())
        # print('average SMC loss - SMC Transformer', average_loss_batch.numpy())
        # print('accuracy from average predictions - SMC Transformer', train_accuracy_average_pred.numpy())

      logging.info('epoch {} - average training loss {} - training accuracy, average: {} - training accuracy, max: {}'.format((
        epoch+1, average_loss_batch.numpy(), train_accuracy_average_pred.numpy(), train_accuracy_max_pred.numpy())))

      ckpt_save_path = ckpt_manager.save()
      logging.info('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

      logging.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    print('training of SMC Transformer for nlp dataset done...')


    # TEST THE LOSS ON A BATCH
    #TODO: adapt this for our case.
    # for input_example_batch, target_example_batch in dataset.take(1):
    #   example_batch_predictions = model(input_example_batch)
    #   print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    # example_batch_loss = loss(target_example_batch, example_batch_predictions)
    # print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    # print("scalar_loss:      ", example_batch_loss.numpy().mean())

