
#TODO: debug problem of positional encoding for baseline transformer (problem of shape being (seq_len * max_pos_enc) instead of (seq_len).
#TODO: keep the successive history_csv file from the different ckpts.
# basic logging tutorial: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial

#TODO: add on the logging into the number of trainable variables for each model.

""""# to store:
# in a fichier .log: for each epoch, the average loss (train & val dataset),
the training accuracy (train & val datasets),
- 2 accuracies for the SMC Transformer), the time taken for each epoch, and the total training time
# - in a fichier .txt: the list of losses (train & val), accuracies (train & val) for plotting & comparing.
# - in files .npy: the output of the model (predictions, trajectories, attention_weights...).
# - checkpoints of the model in a file .ckpt.
# dictionary of hparams (cf Nicolas's script...).
"""

"""Transformer model parameters (from the tensorflow tutorial):
d_model: 512
num_heads: 8
dff: 1048
num_layers: 2
max_positional_encoding=
"""

import tensorflow as tf
from models.Baselines.Transformer_without_enc import Transformer
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from train.train_step_functions import train_step_classic_T
from train.train_step_functions import train_step_SMC_T
from train.loss_functions import compute_accuracy_variance
from train.Perplexity import PerplexityMetric

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer

import time
import sys
import os
import logging
import numpy as np
import shutil
import json
import argparse

from preprocessing.time_series.df_to_dataset import data_to_dataset_uni_step
from preprocessing.time_series.df_to_dataset import split_input_target_uni_step

from utils.utils_train import write_to_csv
from utils.utils_train import create_run_dir

if __name__ == "__main__":

  # -------- parsing arguments ------------------------------------------------------------------------------------------------

  parser = argparse.ArgumentParser()

  parser.add_argument("-config", type=str, default='../../config/config_ts.json', help="path for the config file with hyperparameters")
  parser.add_argument("-out_folder", type=str, default='../../output', help="path for the outputs folder")
  parser.add_argument("-data_folder", type=str, default='/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/data',
                      help="path for the outputs folder")
  # parser.add_argument("-train_baseline", type=bool, required=True, help="Training a Baseline Transformer?")
  # parser.add_argument("-train_smc_T", type=bool, required=True, help="Training the SMC Transformer?")
  # parser.add_argument("-train_rnn", type=bool, required=True, help="Training the SMC Transformer?")

  parser.add_argument("-train_baseline", type=bool, default=False, help="Training a Baseline Transformer?")
  parser.add_argument("-train_smc_T", type=bool, default=True, help="Training the SMC Transformer?")
  parser.add_argument("-train_rnn", type=bool, default=False, help="Training the SMC Transformer?")


  parser.add_argument("-load_ckpt", type=bool, default=True, help="loading and restoring existing checkpoints?")

  args=parser.parse_args()
  config_path=args.config

  #------------------Uploading the hyperparameters info------------------------------------------------------------------------
  with open(config_path) as f:
    hparams = json.load(f)

  # model params
  num_layers=hparams["model"]["num_layers"]
  num_heads=hparams["model"]["num_heads"]
  d_model=hparams["model"]["d_model"]
  dff=hparams["model"]["dff"]
  rate=hparams["model"]["rate"] # p_dropout
  max_pos_enc_bas_str=hparams["model"]["maximum_position_encoding_baseline"]
  maximum_position_encoding_baseline=None if max_pos_enc_bas_str=="None" else max_pos_enc_bas_str
  max_pos_enc_smc_str=hparams["model"]["maximum_position_encoding_smc"]
  maximum_position_encoding_smc=None if max_pos_enc_smc_str=="None" else max_pos_enc_smc_str

  # smc params
  num_particles=hparams["smc"]["num_particles"]
  noise_encoder_str=hparams["smc"]["noise_encoder"]
  noise_encoder=True if noise_encoder_str=="True" else False
  noise_SMC_layer_str=hparams["smc"]["noise_SMC_layer"]
  noise_SMC_layer=True if noise_SMC_layer_str=="True" else False
  sigma=hparams["smc"]["sigma"]
  # computing manually resampling parameter
  resampling=False if num_particles==1 else True

  # optim params
  BATCH_SIZE=hparams["optim"]["BATCH_SIZE"]
  learning_rate=hparams["optim"]["learning_rate"]
  EPOCHS=hparams["optim"]["EPOCHS"]

  # task params
  data_type=hparams["task"]["data_type"]
  task_type=hparams["task"]["task_type"]
  task=hparams["task"]["task"]

  # adding RNN hyper-parameters
  #rnn_bs = hparams["RNN_hparams"]["rnn_bs"]
  rnn_bs=BATCH_SIZE
  rnn_emb_dim = hparams["RNN_hparams"]["rnn_emb_dim"]
  rnn_units = hparams["RNN_hparams"]["rnn_units"]

  #------------------UPLOAD the training dataset------------------------------------------------------------------------------------------------
  data_folder = args.data_folder
  train_data = np.load(data_folder + '/ts_weather_train_data.npy')
  val_data = np.load(data_folder + '/ts_weather_val_data.npy')
  test_data = np.load(data_folder + '/ts_weather_test_data.npy')

  print('train_data', train_data.shape)
  print('test_data', test_data.shape)

  BUFFER_SIZE=10000

  train_dataset, val_dataset = data_to_dataset_uni_step(train_data=train_data,
                                                        val_data=val_data,
                                                        split_fn=split_input_target_uni_step,
                                                        BUFFER_SIZE=BUFFER_SIZE,
                                                        BATCH_SIZE=BATCH_SIZE)

  for (inp, tar) in train_dataset.take(5):
    print('input example', inp[0])
    print('target example', tar[0])

  num_classes= 25
  target_vocab_size = num_classes # 25 bins
  seq_len=train_data.shape[1] - 1 # 24 observations
  training_samples=train_data.shape[0]
  steps_per_epochs=int(train_data.shape[0]/BATCH_SIZE)

  # -------define hyperparameters----------------------------------------------------------------------------------------------------------------
  optimizer = tf.keras.optimizers.Adam(learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.98,
                                       epsilon=1e-9)
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
  val_accuracy=tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

  # ------------- preparing the OUTPUT FOLDER------------------------------------------------------------------------
  output_path = args.out_folder
  folder_template='{}_{}_heads_{}_depth_{}_dff_{}_pos-enc_{}_pdrop_{}_b_{}'
  out_folder = folder_template.format(data_type,
                                      task,
                                      num_heads,
                                      d_model,
                                      dff,
                                      maximum_position_encoding_baseline,
                                      rate,
                                      BATCH_SIZE)

  if args.train_smc_T:
    out_folder=out_folder+'__particles_{}_noise_{}_sigma_{}_smc-pos-enc_{}'.format(num_particles,
                                                                        noise_SMC_layer,
                                                                        sigma,
                                                                        maximum_position_encoding_smc)

  if args.train_rnn:
    out_folder=out_folder+'__rnn-emb_{}_rnn-units_{}'.format(rnn_emb_dim, rnn_units)

  output_path = create_run_dir(path_dir=output_path, path_name=out_folder)

  # copying the config file in the output directory
  if not os.path.exists(output_path+'/config.json'):
    shutil.copyfile(config_path, output_path+'/config.json')
  else:
    shutil.copyfile(config_path, output_path + '/config_new.json')

  # ------------------ create the logging-----------------------------------------------------------------------------
  out_file_log = output_path + '/' + 'training_log.log'
  logging.basicConfig(filename=out_file_log, level=logging.INFO)
  # create logger
  logger = logging.getLogger('training log')
  logger.setLevel(logging.INFO)
  # create console handler and set level to debug
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  # create formatter
  formatter = logging.Formatter('%(asctime)s - %(message)s')
  # add formatter to ch
  ch.setFormatter(formatter)
  # add ch to logger
  logger.addHandler(ch)

  #  creating the checkpoint manager:
  checkpoint_path = os.path.join(output_path, "checkpoints")
  if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

  #-------------------- Building the RNN Baseline & the associated training algo --------------------------------------------------------------------
  def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]),
      tf.keras.layers.GRU(rnn_units,
                          return_sequences=True,
                          stateful=True,
                          recurrent_initializer='glorot_uniform'),
      tf.keras.layers.Dense(vocab_size)
    ])
    return model

  GRU_model=build_model(vocab_size=num_classes,
                        embedding_dim=rnn_emb_dim,
                        rnn_units=rnn_units,
                        batch_size=rnn_bs)


  @tf.function
  def train_step(inp, target, model, accuracy_metric):
    with tf.GradientTape() as tape:
      predictions = model(inp)
      loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
          target, predictions, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_acc_batch=accuracy_metric(target, predictions)
    return loss, train_acc_batch

  # Directory where the checkpoints will be saved
  checkpoint_dir = os.path.join(checkpoint_path, "RNN_baseline")
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  # Name of the checkpoint files
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

  #---------------------- if training all models : comparison of model's capacity (number of trainable variables)-------------------------------
  # Transformer - baseline.
  if args.train_rnn and args.train_baseline and args.train_smc_T:
    transformer = Transformer(num_layers=num_layers,
                                d_model=d_model,
                                num_heads=num_heads,
                                dff=dff,
                                target_vocab_size=target_vocab_size,
                                maximum_position_encoding=maximum_position_encoding_baseline,
                                data_type=data_type,
                              rate=rate)

    # SMC Transformer
    smc_transformer = SMC_Transformer(num_layers=num_layers,
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
                                      resampling=resampling,
                                      rate=rate)

    # forward pass on a batch of training examples:
    # GRU
    for input_example_batch, target_example_batch in train_dataset.take(1):
      prediction_GRU = GRU_model(input_example_batch)
      prediction_T, attn_weights_T = transformer(input_example_batch,
                                                 training=False,
                                                 mask=create_look_ahead_mask(seq_len))
      (prediction_smcT, _, _), predictions_metric, attn_weights_smcT = smc_transformer(inputs=input_example_batch,
                                                                                       training=False,
                                                                                       mask=create_look_ahead_mask(seq_len))

    logger.info('predictions from GRU shape:{}'.format(tf.shape(prediction_GRU)))
    logger.info('predictions from Baseline Transformer shape:{}'.format(tf.shape(prediction_T)))
    logger.info('predictions from SMC Transformer shape:{}'.format(tf.shape(prediction_smcT)))

    print('showing GRU model summary...')
    print(GRU_model.summary())
    print('showing Baseline Transformer model summary...')
    print(transformer.summary())
    print('showing SMC Transformer model summary...')
    print(smc_transformer.summary())
  #----------------------TRAINING OF A SIMPLE RNN BASELINE--------------------------------------------------------------------------------------
  if args.train_rnn:
    logger.info("training a RNN Baseline on the nlp dataset...")
    logger.info("number of training samples: {}".format(training_samples))
    logger.info("steps per epoch:{}".format(steps_per_epochs))
    #train_param_GRU = int(np.sum([tf.keras.backend.count_params(p) for p in set(GRU_model.trainable_weights)]))
    #logger.info("number of trainable parameters for the GRU: {}".format(train_param_GRU))

    start_training=time.time()

    for epoch in range(EPOCHS):
      start = time.time()
      # resarting metrics:
      train_accuracy.reset_states()
      val_accuracy.reset_states()
      # initializing the hidden state at the start of every epoch
      # initally hidden is None
      hidden = GRU_model.reset_states()

      for (inp, target) in train_dataset:
        loss, train_acc_batch = train_step(inp, target, model=GRU_model, accuracy_metric=train_accuracy)

      GRU_model.save_weights(checkpoint_prefix.format(epoch=epoch))

      # computing train and val acc for the current epoch:
      train_acc = train_accuracy.result()

      for (inp_val, tar_val) in val_dataset:
        predictions_val = GRU_model(inp_val)
        # computing the validation accuracy for each batch...
        val_accuracy_batch=val_accuracy(tar_val, predictions_val)
      val_acc=val_accuracy.result()

      logger.info('Epoch {} - Loss {:.4f} - train acc {:.4f} - val acc {:.4f}'.format(epoch + 1, loss, train_acc, val_acc))
      logger.info('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

      GRU_model.save_weights(checkpoint_prefix.format(epoch=epoch))

    logger.info('Training time for {} epochs: {}'.format(EPOCHS, time.time() - start_training))
    logger.info('training of a RNN Baseline (GRU) for a nlp dataset done...')
    logger.info(">>>---------------------------------------------------------------------------------------<<<")

  #-----------------TRAINING-----------------------------------------------------------------------------------------------------------------------------------------------------------

  logger.info("model hyperparameters from the config file: {}".format(hparams["model"]))
  logger.info("smc hyperparameters from the config file: {}".format(hparams["smc"]))
  if args.train_rnn:
    logger.info("GRU model hparams from the config file: {}".format(hparams["RNN_hparams"]))

  # TF-2.0
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  sys.setrecursionlimit(100000)
  tf.config.experimental_run_functions_eagerly(True)  # to remove TensorInacessibleError

  dataset = train_dataset
  # to choose which models to train.
  train_smc_transformer=args.train_smc_T
  train_classic_transformer=args.train_baseline

  # -------------------------------------------TRAIN ON THE DATASET - CLASSIC TRANSFORMER -------------------------------------------
  if train_classic_transformer:
    logger.info("training the baseline Transformer on the nlp dataset...")
    logger.info("number of training samples: {}".format(training_samples))
    logger.info("steps per epoch:{}".format(steps_per_epochs))

    #putting as default start_epoch=0
    start_epoch=0

    # storing the losses & accuracy in a list for each epoch
    average_losses_baseline=[]
    training_accuracies_baseline=[]
    val_accuracies_baseline=[]

    if maximum_position_encoding_baseline is not None:
      logger.info('training a baseline transformer with positional encoding...')

    # Transformer - baseline.
    transformer = Transformer(num_layers=num_layers,
                              d_model=d_model,
                              num_heads=num_heads,
                              dff=dff,
                              target_vocab_size=target_vocab_size,
                              maximum_position_encoding=maximum_position_encoding_baseline,
                              data_type=data_type)

    #train_param_T = int(np.sum([tf.keras.backend.count_params(p) for p in set(transformer.trainable_weights)]))
    #logger.info("number of trainable params for the baseline Transformer:{}".format(train_param_T))

    # creating checkpoint manager
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer)

    baseline_ckpt_path=os.path.join(checkpoint_path, "transformer_baseline")

    ckpt_manager = tf.train.CheckpointManager(ckpt, baseline_ckpt_path, max_to_keep=EPOCHS)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint and args.load_ckpt:
      ckpt_restored_path=ckpt_manager.latest_checkpoint
      ckpt_name = os.path.basename(ckpt_restored_path)
      _, ckpt_num = ckpt_name.split('-')
      start_epoch=int(ckpt_num)
      ckpt.restore(ckpt_manager.latest_checkpoint)
      print("checkpoint restored from {}".format(ckpt_manager.latest_checkpoint))
      print('Latest checkpoint restored!!')

    start_training=time.time()

    if start_epoch > 0:
      if start_epoch >= EPOCHS:
        print("adding {} more epochs to existing training".format(EPOCHS))
        start_epoch=0
      else:
        logger.info ("starting training after checkpoint restoring from epoch {}".format(start_epoch))

    for epoch in range(start_epoch, EPOCHS):
      start = time.time()
      logger.info("Epoch {}/{}".format(epoch+1, EPOCHS))

      train_loss.reset_states()
      train_accuracy.reset_states()
      val_accuracy.reset_states()
      # perplexity actually computed only for the test set.
      #train_perplexity.reset_states()
      #val_perplexity.reset_states()

      for (batch, (inp, tar)) in enumerate(dataset):
        _, avg_loss_batch, train_accuracy_batch, _ = train_step_classic_T(inputs=inp,
                                                                          targets=tar,
                                                                          transformer=transformer,
                                                                          train_loss=train_loss,
                                                                          train_accuracy=train_accuracy,
                                                                          optimizer=optimizer,
                                                                          data_type=data_type)

      train_acc=train_accuracy.result()
      #train_perplx=train_perplexity.result()

      for (inp, tar) in val_dataset:
        predictions_val, attn_weights_val = transformer(inputs=inp,
                                                      training=False,
                                                      mask=create_look_ahead_mask(seq_len))
        # computing the validation accuracy for each batch...
        val_accuracy_batch=val_accuracy(tar, predictions_val)
        #val_perplexity_batch=val_perplexity(tar, predictions_val)

      val_acc=val_accuracy.result()
      #val_perplx=val_perplexity.result()

      log_template='train loss {} - train acc {} - val acc {}'
      logger.info(log_template.format(avg_loss_batch,
                                      train_acc,
                                      val_acc))

      ckpt_save_path = ckpt_manager.save()
      logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

      # saving loss and metrics information:
      average_losses_baseline.append(avg_loss_batch.numpy())
      training_accuracies_baseline.append(train_accuracy_batch.numpy())
      val_accuracies_baseline.append(val_accuracy_batch.numpy())

    logger.info('total training time for {} epochs:{}'.format(EPOCHS,time.time() - start_training))

    # storing history of losses and accuracies in a csv file
    keys=['loss', 'train_accuracy', 'val_accuracy']
    values=[average_losses_baseline, training_accuracies_baseline, val_accuracies_baseline]
    history=dict(zip(keys,values))
    baseline_history_fn=output_path+'/'+'baseline_history.csv'
    if os.path.isdir(baseline_history_fn):
      logger.info("saving the history from the restored ckpt # {} in a new csv file...".format(start_epoch))
      baseline_history_fn=output_path+'/'+'baseline_history_from_ckpt{}.csv'.format(start_epoch)
    write_to_csv(baseline_history_fn, history)
    logger.info('saving loss and metrics information...')

    # making predictions with the trained model and saving them on .npy files
    model_output_path = os.path.join(output_path, "model_outputs")
    if not os.path.isdir(model_output_path):
      os.makedirs(model_output_path)
    predictions_fn=model_output_path + '/' + 'baseline_predictions.npy'
    attn_weights_fn=model_output_path + '/' + 'baseline_attn_weights.npy'
    np.save(predictions_fn, predictions_val) # DO IT FOR A TEST DATASET INSTEAD?
    np.save(attn_weights_fn, attn_weights_val) # DO IT FOR A TEST DATASET INSTEAD?
    logger.info("saving model output in .npy files...")

    logger.info('training of a classic Transformer for a time-series dataset done...')
    logger.info(">>>---------------------------------------------------------------------------------------<<<")

#-------------------TRAINING ON THE DATASET - SMC_TRANSFORMER-----------------------------------------------------------------------------------------------------------
  if train_smc_transformer:
    logger.info('starting the training of the smc transformer...')
    logger.info("number of training samples: {}".format(training_samples))
    logger.info("steps per epoch:{}".format(steps_per_epochs))
    if not resampling:
      logger.info("no resampling because only one particle is taken...")

    start_epoch=0

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
    ckpt_manager = tf.train.CheckpointManager(ckpt, smc_T_ckpt_path, max_to_keep=EPOCHS)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint and args.load_ckpt:
      ckpt_restored_path = ckpt_manager.latest_checkpoint
      ckpt_name = os.path.basename(ckpt_restored_path)
      _, ckpt_num = ckpt_name.split('-')
      start_epoch = int(ckpt_num)
      ckpt.restore(ckpt_manager.latest_checkpoint)
      print(" checkpoint restored from {}".format(ckpt_manager.latest_checkpoint))
      logger.info('Latest checkpoint restored!!')

    # check the pass forward.
    for input_example_batch, target_example_batch in dataset.take(1):
      (example_batch_predictions, traj, _), predictions_metric, _ = smc_transformer(inputs=input_example_batch,
                                              training=False,
                                              mask=create_look_ahead_mask(seq_len))
      print("predictions shape: {}", example_batch_predictions.shape)

    # train_param_smcT = int(np.sum([tf.keras.backend.count_params(p) for p in set(smc_transformer.trainable_weights)]))
    # logger.info("number of trainable params for the SMC Transformer:{}".format(train_param_smcT))

    # preparing recording of loss and metrics information
    train_loss_history=[]
    train_inf_acc_history=[]
    train_avg_acc_history=[]
    train_max_acc_history=[]

    val_avg_acc_history=[]
    val_max_acc_history=[]
    val_inf_acc_history=[]
    val_acc_variance_history=[]

    if start_epoch > 0:
      if start_epoch > EPOCHS:
        print("adding {} more epochs to existing training".format(EPOCHS))
        start_epoch=0
      else:
        logger.info ("starting training after checkpoint restoring from epoch {}".format(start_epoch))

    start_training=time.time()

    for epoch in range(start_epoch, EPOCHS):
      start = time.time()

      logger.info('Epoch {}/{}'.format(epoch+1,EPOCHS))

      train_loss.reset_states()
      train_accuracy.reset_states()

      for (batch, (inp, tar)) in enumerate(dataset):
        loss_smc, avg_loss_batch, train_accuracies, _ = train_step_SMC_T(inputs=inp,
                                                                         targets=tar,
                                                                         smc_transformer=smc_transformer,
                                                                         optimizer=optimizer,
                                                                         train_loss=train_loss,
                                                                         train_accuracy=train_accuracy,
                                                                         classic_loss=True,
                                                                         SMC_loss=True)
        train_inf_acc_batch, train_avg_acc_batch, train_avg_acc_old_batch, train_max_acc_batch=train_accuracies

      # compute the validation accuracy on the validation dataset:
      # TODO: here consider a validation set with a batch_size equal to the number of samples.
      for (inp, tar) in val_dataset:
        (predictions_val,_,weights_val),predictions_metric, attn_weights_val = smc_transformer(inputs=inp,
                                                      training=False,
                                                      mask=create_look_ahead_mask(seq_len))

        # computing the validation accuracy for each batch...
        val_inf_pred_batch, val_avg_pred_batch, val_avg_pred_old_batch, val_max_pred_batch=predictions_metric
        val_inf_acc_batch=val_accuracy(tar, val_inf_pred_batch)
        val_avg_acc_batch=val_accuracy(tar, val_avg_pred_batch)
        val_avg_acc_old_batch = val_accuracy(tar, val_avg_pred_old_batch)
        val_max_acc_batch=val_accuracy(tar, val_max_pred_batch)

      # computing the variance in accuracy for each 'prediction particle':
      val_acc_variance=compute_accuracy_variance(predictions_val=predictions_val, tar=tar, accuracy_metric=val_accuracy)

      template='train loss {} - train acc, inf: {} - train acc, avg: {} - train_acc, old avg:{} - train_acc, max: {},' \
               ' - val acc, inf: {} - val acc, avg: {} - val acc, old avg: {} - val acc, max: {}'
      logger.info(template.format(avg_loss_batch.numpy(),
                                  train_inf_acc_batch.numpy(),
                                  train_avg_acc_batch.numpy(),
                                  train_avg_acc_old_batch.numpy(),
                                  train_max_acc_batch.numpy(),
                                  val_inf_acc_batch.numpy(),
                                  val_avg_acc_batch.numpy(),
                                  val_avg_acc_old_batch.numpy(),
                                  val_max_acc_batch.numpy()))

      ckpt_save_path = ckpt_manager.save()

      logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

      # saving loss and metrics information:
      train_loss_history.append(avg_loss_batch.numpy())
      train_inf_acc_history.append(train_inf_acc_batch.numpy())
      train_avg_acc_history.append(train_avg_acc_batch.numpy())
      train_max_acc_history.append(train_max_acc_batch.numpy())

      val_inf_acc_history.append(val_inf_acc_batch.numpy())
      val_avg_acc_history.append(val_avg_acc_batch.numpy())
      val_max_acc_history.append(val_max_acc_batch.numpy())

      val_acc_variance_history.append(val_acc_variance)

    logger.info('total training time for {} epochs:{}'.format(EPOCHS, time.time() - start_training))

    # storing history of losses and accuracies in a csv file
    keys = ['train loss','train accuracy, inference' 'train accuracy, from avg', 'train accuracy, from max',
            'val accuracy - inference', 'val accuracy, from avg', 'val accuracy, from max', 'variance of validation accuracy']
    values = [train_loss_history, train_inf_acc_history, train_avg_acc_history, train_max_acc_history,
              val_inf_acc_history, val_avg_acc_history, val_max_acc_history, val_acc_variance_history]
    history = dict(zip(keys, values))
    baseline_history_fn = output_path + '/' + 'smc_transformer_history.csv'
    write_to_csv(baseline_history_fn, history)
    logger.info('saving loss and metrics information...')

    # making predictions with the trained model and saving them on .npy files
    mask = create_look_ahead_mask(seq_len)
    model_output_path = os.path.join(output_path, "model_outputs")
    if not os.path.exists(model_output_path):
      os.makedirs(model_output_path)
    predictions_fn = model_output_path + '/' + 'smc_predictions.npy'
    weights_fn = model_output_path + '/' + 'smc_weights.npy'
    attn_weights_fn = model_output_path + '/' + 'smc_attn_weights.npy'
    np.save(predictions_fn, predictions_val)
    np.save(weights_fn, weights_val)
    np.save(attn_weights_fn, attn_weights_val)
    logger.info("saving model outputs in .npy files...")

    logger.info('training of SMC Transformer for a time-series dataset done...')

    logger.info(">>>-------------------------------------------------------------------------------------------<<<")


    # TEST THE LOSS ON A BATCH
    #TODO: adapt this for our case.
    # for input_example_batch, target_example_batch in dataset.take(1):
    #   example_batch_predictions = model(input_example_batch)
    #   print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    # example_batch_loss = loss(target_example_batch, example_batch_predictions)
    # print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    # print("scalar_loss:      ", example_batch_loss.numpy().mean())


