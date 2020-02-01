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
task='char-LM'


#------------------UPLOAD the training dataset------------------------------------------------------------------------------------------------
file_path = tf.keras.utils.get_file('shakespeare.txt',
                                     'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

#file_path='/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/shakespeare_short.txt'

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
seq_len = 50
train_split = 0.9
train_dataset, val_dataset, num_classes, training_samples = text_to_dataset(file_path=file_path,
                                                          seq_len=seq_len,
                                                          train_split=train_split,
                                                          buffer_size=BUFFER_SIZE,
                                                          batch_size=BATCH_SIZE)

steps_per_epochs=int(training_samples / BATCH_SIZE)

# -------define hyperparameters----------------------------------------------------------------------------------------------------------------
## Optimizer
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
val_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
train_perplexity=PerplexityMetric(name='train_perplexity')
val_perplexity=PerplexityMetric(name='val_perplexity')

# Model's hyper-parameters.
num_particles = 1
num_heads = 1
d_model = 4
dff = 8

maximum_position_encoding_smc = None
maximum_position_encoding_baseline=None

target_vocab_size = num_classes
num_layers = 1

sigma=1
noise_encoder=False
noise_SMC_layer=False
resampling=True

#-------------------- SIMPLE BASELINE FOR COMPARISON --------------------------------------------------------------------
# experiments done on a notebook aside.

#-----------------TRAINING-----------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

  # TF-2.0
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  sys.setrecursionlimit(100000)
  tf.config.experimental_run_functions_eagerly(True)  # to remove TensorInacessibleError

  dataset = train_dataset
  EPOCHS = 2
  train_smc_transformer=True
  train_classic_transformer=True

  # ------------- preparing the OUTPUT FOLDER------------------------------------------------------------------------

  output_path = '../../output'
  out_folder = '{}_{}_heads_{}_particles_{}_depth_{}_dff_{}_sigma_{}_noise_{}'.format(data_type,
                                                                               task,
                                                                               num_heads,
                                                                               num_particles,
                                                                               d_model,
                                                                               dff,
                                                                               sigma,
                                                                               noise_SMC_layer)

  output_path = create_run_dir(path_dir=output_path, path_name=out_folder)

  # create the logging:
  out_file_log=output_path+ '/'+'training_log.log'
  logging.basicConfig(filename=out_file_log, level=logging.INFO)

  #  creating the checkpoint manager:
  checkpoint_path = create_run_dir(output_path, "checkpoints")

  # -------------------------------------------TRAIN ON THE DATASET - CLASSIC TRANSFORMER -------------------------------------------

  if train_classic_transformer:
    logging.info("training the baseline Transformer on the nlp dataset...")

    # storing the losses & accuracy in a list for each epoch
    average_losses_baseline=[]
    training_accuracies_baseline=[]
    val_accuracies_baseline=[]

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

    ckpt_manager = tf.train.CheckpointManager(ckpt, baseline_ckpt_path, max_to_keep=EPOCHS)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      print('Latest checkpoint restored!!')

    start_training=time.time()

    for epoch in range(EPOCHS):
      start = time.time()

      train_loss.reset_states()
      train_accuracy.reset_states()
      val_accuracy.reset_states()
      #train_perplexity.reset_states()
      #val_perplexity.reset_states()

      for (batch, (inp, tar)) in enumerate(dataset):
        #TODO: check the training boolean in the train_step_classic_T function.
        #TODO: add the computation of the perplexityMetric
        _, average_loss_batch, train_accuracy_batch = train_step_classic_T(inputs=inp,
                                             targets=tar,
                                             transformer=transformer,
                                             train_loss=train_loss,
                                             train_accuracy=train_accuracy,
                                             optimizer=optimizer,
                                             data_type=data_type)

      train_acc=train_accuracy.result()


      for (inp, tar) in val_dataset:
        predictions_val, attn_weights_val = transformer(inputs=inp,
                                                      training=False,
                                                      mask=create_look_ahead_mask(seq_len))
        # computing the validation accuracy for each batch...
        val_accuracy_batch=val_accuracy(tar, predictions_val)

      val_acc=val_accuracy.result()

      logging.info('epoch {} - training_loss {} - training_accuracy {} - validation accuracy {}'.format(epoch+1,
                                                                                                      average_loss_batch,
                                                                                                      train_acc,
                                                                                                        val_acc))
      logging.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

      # saving loss and metrics information:
      average_losses_baseline.append(average_loss_batch.numpy())
      training_accuracies_baseline.append(train_accuracy_batch.numpy())
      val_accuracies_baseline.append(val_accuracy_batch.numpy())

    logging.info('total training time for {} epochs:{}'.format(EPOCHS,time.time() - start_training))

    # storing history of losses and accuracies in a csv file
    keys=['loss', 'train_accuracy', 'val_accuracy']
    values=[average_losses_baseline, training_accuracies_baseline, val_accuracies_baseline]
    history=dict(zip(keys,values))
    baseline_history_fn=output_path+'/'+'baseline_history.csv' # use a create_directory function instead.
    write_to_csv(baseline_history_fn, history)
    logging.info('saving loss and metrics information...')

    # making predictions with the trained model and saving them on .npy files
    model_output_path=create_run_dir(path_dir=output_path, path_name="model_outputs")
    predictions_fn=model_output_path + '/' + 'baseline_predictions.npy'
    attn_weights_fn=model_output_path + '/' + 'baseline_attn_weights.npy'
    np.save(predictions_fn, predictions_val) # DO IT FOR A TEST DATASET INSTEAD?
    np.save(attn_weights_fn, attn_weights_val) # DO IT FOR A TEST DATASET INSTEAD?
    logging.info("saving model output in .npy files...")

    logging.info('training of a classic Transformer for a nlp dataset done...')

    #TODO: this for running the validation loss and metric.
    # Run a validation loop at the end of each epoch.
    # for x_batch_val, y_batch_val in val_dataset:
    #   val_logits = model(x_batch_val)
    #   # Update val metrics
    #   val_acc_metric(y_batch_val, val_logits)
    # val_acc = val_acc_metric.result()
    # val_acc_metric.reset_states()
    # print('Validation acc: %s' % (float(val_acc),))

#-------------------TRAINING ON THE DATASET - SMC_TRANSFORMER-----------------------------------------------------------------------------------------------------------
  if train_smc_transformer:
    print('number of particles', num_particles)
    print ('noise in SMC_layer?', noise_SMC_layer)
    print('resampling?', resampling)

    logging.info('starting the training of the smc transformer...')

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
    ckpt = tf.train.Checkpoint(transformer=smc_transformer,
                               optimizer=optimizer)
    smc_T_ckpt_path = os.path.join(checkpoint_path, "SMC_transformer")
    ckpt_manager = tf.train.CheckpointManager(ckpt, smc_T_ckpt_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
      ckpt.restore(ckpt_manager.latest_checkpoint)
      logging.info('Latest checkpoint restored!!')

    # check the pass forward.
    for input_example_batch, target_example_batch in dataset.take(1):
      (example_batch_predictions, _, _), (average_predictions, max_predictions), _ = smc_transformer(inputs=input_example_batch,
                                              training=False,
                                              mask=create_look_ahead_mask(seq_len))
      print("predictions shape: {}", example_batch_predictions.shape)

    print('SMC transformer model summary...', smc_transformer.summary())

    # preparing recording of loss and metrics information
    avg_loss_train=[]
    acc_from_avg_train=[]
    acc_from_max_train=[]
    acc_from_avg_val=[]
    acc_from_max_val=[]

    start_training=time.time()

    for epoch in range(EPOCHS):
      start = time.time()

      train_loss.reset_states()
      train_accuracy.reset_states()

      for (batch, (inp, tar)) in enumerate(dataset):
        #TODO: check the value of the training Boolean in the train_step function.
        #TODO: add the computation of the perplexity in the train_step.
        loss_smc, average_loss_batch, train_accuracy_average_pred, train_accuracy_max_pred=train_step_SMC_T(inputs=inp,
                                  targets=tar,
                                  smc_transformer=smc_transformer,
                                  optimizer=optimizer,
                                  train_loss=train_loss,
                                  train_accuracy=train_accuracy,
                                  classic_loss=True,
                                  SMC_loss=True)

      #compute the validation accuracy on the validation dataset:
      for (inp, tar) in val_dataset:
        (predictions_val,_,weights_val),(avg_pred_val, max_pred_val), attn_weights_val = smc_transformer(inputs=inp,
                                                      training=False,
                                                      mask=create_look_ahead_mask(seq_len))
        # computing the validation accuracy for each batch...
        val_accuracy_from_avg_pred=val_accuracy(tar, avg_pred_val)
        val_accuracy_from_max_pred=val_accuracy(tar, max_pred_val)

      template='epoch {} - average training loss {} - training accuracy, average: {} - validation accuracy, average: {}'
      logging.info(template.format(
        epoch+1, average_loss_batch.numpy(), train_accuracy_average_pred.numpy(), val_accuracy_from_avg_pred.numpy()))

      ckpt_save_path = ckpt_manager.save()

      logging.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

      # saving loss and metrics information:
      avg_loss_train.append(average_loss_batch.numpy())
      acc_from_avg_train.append(train_accuracy_average_pred.numpy())
      acc_from_max_train.append(train_accuracy_max_pred.numpy())
      acc_from_avg_val.append(val_accuracy_from_avg_pred.numpy())
      acc_from_max_val.append(val_accuracy_from_max_pred.numpy())

    logging.info('total training time for {} epochs:{}'.format(EPOCHS, time.time() - start_training))

    # storing history of losses and accuracies in a csv file
    keys = ['train loss', 'training accuracy, from avg', 'training accuracy, from max',
            'validation accuracy, from avg', 'validation accuracy, from max']
    values = [avg_loss_train, acc_from_avg_train, acc_from_max_train, acc_from_avg_val, acc_from_max_val]
    history = dict(zip(keys, values))
    baseline_history_fn = output_path + '/' + 'smc_transformer_history.csv'
    write_to_csv(baseline_history_fn, history)
    logging.info('saving loss and metrics information...')

    # making predictions with the trained model and saving them on .npy files
    mask = create_look_ahead_mask(seq_len)
    model_output_path = create_run_dir(path_dir=output_path, path_name="model_outputs")
    predictions_fn = model_output_path + '/' + 'smc_predictions.npy'
    weights_fn = model_output_path + '/' + 'smc_weights.npy'
    attn_weights_fn = model_output_path + '/' + 'smc_attn_weights.npy'
    np.save(predictions_fn, predictions_val)
    np.save(weights_fn, weights_val)
    np.save(attn_weights_fn, attn_weights_val)
    logging.info("saving model outputs in .npy files...")

    print('training of SMC Transformer for nlp dataset done...')


    # TEST THE LOSS ON A BATCH
    #TODO: adapt this for our case.
    # for input_example_batch, target_example_batch in dataset.take(1):
    #   example_batch_predictions = model(input_example_batch)
    #   print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    # example_batch_loss = loss(target_example_batch, example_batch_predictions)
    # print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    # print("scalar_loss:      ", example_batch_loss.numpy().mean())

