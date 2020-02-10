#TODO: debug the case when the number of epochs is the same. (training is done...)

#TODO: debug problem of positional encoding for baseline transformer (problem of shape being (seq_len * max_pos_enc) instead of (seq_len).
# basic logging tutorial: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial
#TODO: add a loss function for the regression case.

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
from train.train_step_functions import train_step_rnn_regression
from train.train_step_functions import train_step_rnn_classif

from train.loss_functions import compute_accuracy_variance

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from models.Baselines.LSTMs import build_GRU_for_classification
from models.Baselines.LSTMs import build_LSTM_for_regression

import time
import sys
import os
import numpy as np
import shutil
import json
import argparse
import warnings

from preprocessing.time_series.df_to_dataset import df_to_data_regression
from preprocessing.time_series.df_to_dataset import data_to_dataset_uni_step
from preprocessing.time_series.df_to_dataset import split_input_target_uni_step

from utils.utils_train import create_run_dir
from utils.utils_train import create_logger
from utils.utils_train import saving_training_history
from utils.utils_train import saving_model_outputs
from utils.utils_train import restoring_checkpoint

if __name__ == "__main__":

  warnings.simplefilter("ignore")

  # -------- parsing arguments ------------------------------------------------------------------------------------------------

  parser = argparse.ArgumentParser()

  parser.add_argument("-config", type=str, default='../../config/config_ts_reg.json', help="path for the config file with hyperparameters")
  parser.add_argument("-out_folder", type=str, default='../../output/exp_reg', help="path for the outputs folder")
  parser.add_argument("-data_folder", type=str, default='/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/data/ts_10c_s24', help="path for the outputs folder")

  #TODO: ask Florian why when removing default value, it is not working...
  parser.add_argument("-train_baseline", type=bool, default=False, help="Training a Baseline Transformer?")
  parser.add_argument("-train_smc_T", type=bool, default=False, help="Training the SMC Transformer?")
  parser.add_argument("-train_rnn", type=bool, default=True, help="Training the SMC Transformer?")

  parser.add_argument("-load_ckpt", type=bool, default=True, help="loading and restoring existing checkpoints?")

  args=parser.parse_args()
  config_path=args.config

  #------------------Uploading the hyperparameters info------------------------------------------------------------------------

  with open(config_path) as f:
    hparams = json.load(f)

  # model params
  num_layers = hparams["model"]["num_layers"]
  num_heads = hparams["model"]["num_heads"]
  d_model = hparams["model"]["d_model"]
  dff = hparams["model"]["dff"]
  rate = hparams["model"]["rate"]  # p_dropout
  max_pos_enc_bas_str = hparams["model"]["maximum_position_encoding_baseline"]
  maximum_position_encoding_baseline = None if max_pos_enc_bas_str == "None" else max_pos_enc_bas_str
  max_pos_enc_smc_str = hparams["model"]["maximum_position_encoding_smc"]
  maximum_position_encoding_smc = None if max_pos_enc_smc_str == "None" else max_pos_enc_smc_str

  # smc params
  num_particles = hparams["smc"]["num_particles"]
  noise_encoder_str = hparams["smc"]["noise_encoder"]
  noise_encoder = True if noise_encoder_str == "True" else False
  noise_SMC_layer_str = hparams["smc"]["noise_SMC_layer"]
  noise_SMC_layer = True if noise_SMC_layer_str == "True" else False
  sigma = hparams["smc"]["sigma"]
  # computing manually resampling parameter
  resampling = False if num_particles == 1 else True

  # optim params
  BATCH_SIZE = hparams["optim"]["BATCH_SIZE"]
  learning_rate = hparams["optim"]["learning_rate"]
  EPOCHS = hparams["optim"]["EPOCHS"]

  # task params
  data_type = hparams["task"]["data_type"]
  task_type = hparams["task"]["task_type"]
  task = hparams["task"]["task"]

  # adding RNN hyper-parameters
  rnn_bs = BATCH_SIZE
  rnn_emb_dim = hparams["RNN_hparams"]["rnn_emb_dim"]
  rnn_units = hparams["RNN_hparams"]["rnn_units"]

  # loading data arguments for the regression case
  if task_type == 'regression':

    file_path = hparams["data"]["file_path"]
    fname = hparams["data"]["fname"]
    col_name = hparams["data"]["col_name"]
    index_name = hparams["data"]["index_name"]
    TRAIN_SPLIT = hparams["data"]["TRAIN_SPLIT"]
    history = hparams["data"]["history"]
    step = hparams["data"]["step"]

  test_loss=False

  #------------------UPLOAD the training dataset------------------------------------------------------------------------------------------------
  if task_type == 'classification':

    data_folder = args.data_folder
    train_data = np.load(data_folder + '/ts_weather_train_data.npy')
    val_data = np.load(data_folder + '/ts_weather_val_data.npy')
    test_data = np.load(data_folder + '/ts_weather_test_data.npy')

  elif task_type == 'regression':

    (train_data, val_data, test_data), original_df = df_to_data_regression(file_path=file_path,
                                                                           fname=fname,
                                                                           col_name=col_name,
                                                                           index_name=index_name,
                                                                           TRAIN_SPLIT=TRAIN_SPLIT,
                                                                           history=history,
                                                                           step=step)

  print('train_data', train_data.shape)
  print('test_data', test_data.shape)

  BUFFER_SIZE = 10000

  train_dataset, val_dataset = data_to_dataset_uni_step(train_data=train_data,
                                                        val_data=val_data,
                                                        split_fn=split_input_target_uni_step,
                                                        BUFFER_SIZE=BUFFER_SIZE,
                                                        BATCH_SIZE=BATCH_SIZE)

  for (inp, tar) in train_dataset.take(1):
    print('input example', inp[0])
    print('target example', tar[0])

  num_classes= 25 if data_type=='classification' else 1
  target_vocab_size = num_classes # 25 bins
  seq_len = train_data.shape[1] - 1 # 24 observations
  training_samples = train_data.shape[0]
  steps_per_epochs = int(train_data.shape[0]/BATCH_SIZE)

  # -------define hyperparameters----------------------------------------------------------------------------------------------------------------

  optimizer = tf.keras.optimizers.Adam(learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.98,
                                       epsilon=1e-9)
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
  val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

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
    out_folder = out_folder+'__particles_{}_noise_{}_sigma_{}_smc-pos-enc_{}'.format(num_particles,
                                                                        noise_SMC_layer,
                                                                        sigma,
                                                                        maximum_position_encoding_smc)

  if args.train_rnn:
    out_folder = out_folder+'__rnn-emb_{}_rnn-units_{}'.format(rnn_emb_dim, rnn_units)

  output_path = create_run_dir(path_dir=output_path, path_name=out_folder)

  # copying the config file in the output directory
  if not os.path.exists(output_path+'/config.json'):
    shutil.copyfile(config_path, output_path+'/config.json')
  else:
    print("creating a new config file...")
    shutil.copyfile(config_path, output_path + '/config_new.json')

  # ------------------ create the logging-----------------------------------------------------------------------------
  out_file_log = output_path + '/' + 'training_log.log'
  logger = create_logger(out_file_log=out_file_log)
  #  creating the checkpoint manager:
  checkpoint_path = os.path.join(output_path, "checkpoints")
  if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

  #-------------------- Building the RNN Baseline & the associated training algo --------------------------------------------------------------------
  if task_type == 'regression':
    model = build_LSTM_for_regression(rnn_units=rnn_units)
  elif task_type == 'classification':
    model = build_GRU_for_classification(vocab_size=target_vocab_size,
                                         embedding_dim=rnn_emb_dim,
                                         rnn_units=rnn_units,
                                         batch_size=BATCH_SIZE)

  # Directory where the checkpoints will be saved
  checkpoint_dir = os.path.join(checkpoint_path, "RNN_baseline")
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  # Name of the checkpoint files
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                           save_weights_only=True)

  #----------------------TRAINING OF A SIMPLE RNN BASELINE--------------------------------------------------------------------------------------

  # model.compile(optimizer=optimizer,
  #               loss='mse')
  # model.fit(train_dataset,
  #           epochs=EPOCHS,
  #           validation_data=val_dataset,
  #           verbose=2)

  if args.train_rnn:
    logger.info("training a RNN Baseline on the nlp dataset...")
    logger.info("number of training samples: {}".format(training_samples))
    logger.info("steps per epoch:{}".format(steps_per_epochs))

    start_training = time.time()

    for epoch in range(EPOCHS):
      start = time.time()
      train_accuracy.reset_states()
      val_accuracy.reset_states()
      # initializing the hidden state at the start of every epoch
      # initally hidden is None
      hidden = model.reset_states()


      for (inp, target) in train_dataset:
          if task_type == 'classification':
          # CAUTION: in a tf.keras.layers.LSTM, the input tensor needs to be of shape 3 (B,S,Features).
            loss, train_acc_batch = train_step_rnn_classif(inp,
                                                         target,
                                                         model=model,
                                                         optimizer=optimizer,
                                                         accuracy_metric=train_accuracy)
          elif task_type == 'regression':
            loss=train_step_rnn_regression(inp=inp,
                                         target=target,
                                         model=model,
                                         optimizer=optimizer)

      model.save_weights(checkpoint_prefix.format(epoch=epoch))

      # computing train and val acc for the current epoch:

      for (inp_val, tar_val) in val_dataset:
        # inp_val needs to be of shape (B,S,F) : length=3
        inp_val = tf.expand_dims(inp_val, axis=-1)
        predictions_val = model(inp_val)
        # computing the validation accuracy for each batch...
        if task_type == 'classification':
          val_accuracy_batch = val_accuracy(tar_val, predictions_val)

      if task_type == 'classification':
        train_acc = train_accuracy.result()
        val_acc=val_accuracy.result()
      elif task_type == 'regression':
        train_acc = 0
        val_acc = 0

      logger.info('Epoch {} - Loss {} - train acc {} - val acc {}'.format(epoch + 1, loss, train_acc, val_acc))
      logger.info('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

      model.save_weights(checkpoint_prefix.format(epoch=epoch))

    logger.info('Training time for {} epochs: {}'.format(EPOCHS, time.time() - start_training))
    logger.info('training of a RNN Baseline (GRU) for a nlp dataset done...')
    logger.info(">>>--------------------------------------------------------------------------------------------------------------------------------------------------------------<<<")

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
  train_smc_transformer = args.train_smc_T
  train_classic_transformer = args.train_baseline

  # -------------------------------------------TRAIN ON THE DATASET - CLASSIC TRANSFORMER -------------------------------------------

  if train_classic_transformer:

    logger.info("training the baseline Transformer on the nlp dataset...")
    logger.info("number of training samples: {}".format(training_samples))
    logger.info("steps per epoch:{}".format(steps_per_epochs))

    #putting as default start_epoch=0
    start_epoch = 0

    # storing the losses & accuracy in a list for each epoch
    average_losses_baseline = []
    training_accuracies_baseline = []
    val_accuracies_baseline = []

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

    # creating checkpoint manager
    #TODO: put this as a function
    ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    baseline_ckpt_path = os.path.join(checkpoint_path, "transformer_baseline")
    ckpt_manager = tf.train.CheckpointManager(ckpt, baseline_ckpt_path, max_to_keep=EPOCHS)

    # if a checkpoint exists, restore the latest checkpoint.
    start_epoch = restoring_checkpoint(ckpt_manager=ckpt_manager, args=args, ckpt=ckpt, logger=logger)

    start_training = time.time()

    if start_epoch > 0:
      if start_epoch >= EPOCHS:
        print("adding {} more epochs to existing training".format(EPOCHS))
        start_epoch = 0
      else:
        logger.info("starting training after checkpoint restoring from epoch {}".format(start_epoch))

    for epoch in range(start_epoch, EPOCHS):
      start = time.time()
      logger.info("Epoch {}/{}".format(epoch+1, EPOCHS))

      #TODO: try a simple model.compile, model.fit instead...

      train_loss.reset_states()
      train_accuracy.reset_states()
      val_accuracy.reset_states()

      for (batch, (inp, tar)) in enumerate(dataset):
        _, avg_loss_batch, train_accuracy_batch, _ = train_step_classic_T(inputs=inp,
                                                                          targets=tar,
                                                                          transformer=transformer,
                                                                          train_loss=train_loss,
                                                                          train_accuracy=train_accuracy,
                                                                          optimizer=optimizer,
                                                                          data_type=data_type,
                                                                          task_type=task_type)

      if task_type == 'classification':
        train_acc = train_accuracy.result()

        for (inp, tar) in val_dataset:
          predictions_val, attn_weights_val = transformer(inputs=inp,
                                                        training=False,
                                                        mask=create_look_ahead_mask(seq_len))
          # computing the validation accuracy for each batch...
          val_accuracy_batch = val_accuracy(tar, predictions_val)

        val_acc = val_accuracy.result()

        log_template='train loss {} - train acc {} - val acc {}'
        logger.info(log_template.format(avg_loss_batch, train_acc, val_acc))
        # saving loss and metrics information:
        average_losses_baseline.append(avg_loss_batch.numpy())
        training_accuracies_baseline.append(train_accuracy_batch.numpy())
        val_accuracies_baseline.append(val_accuracy_batch.numpy())

      elif task_type == 'regression':
        logger.info('train loss {}'.format(avg_loss_batch))
        average_losses_baseline.append(avg_loss_batch.numpy())

      ckpt_save_path = ckpt_manager.save()
      logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

    logger.info('total training time for {} epochs:{}'.format(EPOCHS,time.time() - start_training))

    # storing history of losses and accuracies in a csv file
    if task_type == 'classification':
      keys = ['loss', 'train_accuracy', 'val_accuracy']
      values = [average_losses_baseline, training_accuracies_baseline, val_accuracies_baseline]

    elif task_type=='regression':
      keys = ['loss']
      values = [average_losses_baseline]

    history = dict(zip(keys, values))
    csv_fname = 'baseline_history.csv'

    saving_training_history(keys=keys,
                            values=values,
                            output_path=output_path,
                            csv_fname=csv_fname,
                            logger=logger,
                            start_epoch=start_epoch)

    #TODO: put this as a function.
    saving_model_outputs(output_path=output_path,
                         predictions=predictions_val,
                         attn_weights=attn_weights_val,
                         pred_fname='baseline_predictions.npy',
                         attn_weights_fname='baseline_attn_weights.npy',
                         logger=logger)

    logger.info('training of a classic Transformer for a time-series dataset done...')
    logger.info(">>>-------------------------------------------------------------------------------------------------------------------------------------------------------------<<<")

#-------------------TRAINING ON THE DATASET - SMC_TRANSFORMER----------------------------------------------------------------------------------------------------------------------

  if train_smc_transformer:

    logger.info('starting the training of the smc transformer...')
    logger.info("number of training samples: {}".format(training_samples))
    logger.info("steps per epoch:{}".format(steps_per_epochs))
    if not resampling:
      logger.info("no resampling because only one particle is taken...")

    start_epoch = 0

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
                          resampling=resampling)

    # creating checkpoint manager
    ckpt = tf.train.Checkpoint(transformer=smc_transformer,
                               optimizer=optimizer)
    smc_T_ckpt_path = os.path.join(checkpoint_path, "SMC_transformer")
    ckpt_manager = tf.train.CheckpointManager(ckpt, smc_T_ckpt_path, max_to_keep=EPOCHS)

    # if a checkpoint exists, restore the latest checkpoint.
    start_epoch=restoring_checkpoint(ckpt_manager=ckpt_manager, ckpt=ckpt, args=args, logger=logger)

    # check the pass forward.
    for input_example_batch, target_example_batch in dataset.take(1):
      (example_batch_predictions, traj, _), predictions_metric, _ = smc_transformer(inputs=input_example_batch,
                                              training=False,
                                              mask=create_look_ahead_mask(seq_len))
      print("predictions shape: {}", example_batch_predictions.shape)

    # preparing recording of loss and metrics information
    train_loss_history, train_inf_acc_history, train_avg_acc_history, train_max_acc_history= [], [], [], []
    val_inf_acc_history, val_avg_acc_history, val_max_acc_history, val_acc_variance_history = [], [], [], []

    if start_epoch > 0:
      if start_epoch > EPOCHS:
        print("adding {} more epochs to existing training".format(EPOCHS))
        start_epoch = 0
      else:
        logger.info("starting training after checkpoint restoring from epoch {}".format(start_epoch))

    start_training=time.time()

    for epoch in range(start_epoch, EPOCHS):
      start = time.time()

      logger.info('Epoch {}/{}'.format(epoch+1,EPOCHS))

      train_loss.reset_states()
      train_accuracy.reset_states()

      if test_loss:
      # TEST LOSS SUR ONE BATCH:
      #TODO: check that the loss is the same for 1 particule, and 5 particules with no noise.
        for (inp_ex_batch, target_ex_batch) in dataset.take(1):
          _, loss_temp, accuracies_temp, _ = train_step_SMC_T(inputs=inp,
                                                            targets=tar,
                                                            smc_transformer=smc_transformer,
                                                            optimizer=optimizer,
                                                            train_loss=train_loss,
                                                            train_accuracy=train_accuracy,
                                                            classic_loss=True,
                                                            SMC_loss=True)


      for (batch, (inp, tar)) in enumerate(dataset):
        loss_smc, avg_loss_batch, train_accuracies, _ = train_step_SMC_T(inputs=inp,
                                                                         targets=tar,
                                                                         smc_transformer=smc_transformer,
                                                                         optimizer=optimizer,
                                                                         train_loss=train_loss,
                                                                         train_accuracy=train_accuracy,
                                                                         classic_loss=True,
                                                                         SMC_loss=True)
        train_inf_acc_batch, train_avg_acc_batch, train_max_acc_batch = train_accuracies

      # compute the validation accuracy on the validation dataset:
      # TODO: here consider a validation set with a batch_size equal to the number of samples.
      for (inp, tar) in val_dataset:
        (predictions_val, _, weights_val), predictions_metric, attn_weights_val = smc_transformer(inputs=inp,
                                                                                                  training=False,
                                                                                                  mask=create_look_ahead_mask(seq_len))

        # computing the validation accuracy for each batch...
        val_inf_pred_batch, val_avg_pred_batch, val_max_pred_batch = predictions_metric
        val_inf_acc_batch = val_accuracy(tar, val_inf_pred_batch)
        val_avg_acc_batch = val_accuracy(tar, val_avg_pred_batch)
        val_max_acc_batch = val_accuracy(tar, val_max_pred_batch)

      # computing the variance in accuracy for each 'prediction particle':
      val_acc_variance=compute_accuracy_variance(predictions_val=predictions_val,
                                                 tar=tar,
                                                 accuracy_metric=val_accuracy)

      #TODO: remove the computation of the 'old' average.
      template='train loss {} - train acc, inf: {} - train acc, avg: {} - train_acc, max: {},' \
               ' - val acc, inf: {} - val acc, avg: {} - val acc, max: {}'
      logger.info(template.format(avg_loss_batch.numpy(),
                                  train_inf_acc_batch.numpy(),
                                  train_avg_acc_batch.numpy(),
                                  train_max_acc_batch.numpy(),
                                  val_inf_acc_batch.numpy(),
                                  val_avg_acc_batch.numpy(),
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
    keys = ['train loss','train accuracy, inference', 'train accuracy, from avg', 'train accuracy, from max',
            'val accuracy - inference', 'val accuracy, from avg', 'val accuracy, from max', 'variance of validation accuracy']
    values = [train_loss_history, train_inf_acc_history, train_avg_acc_history, train_max_acc_history,
              val_inf_acc_history, val_avg_acc_history, val_max_acc_history, val_acc_variance_history]

    saving_training_history(keys=keys, values=values,
                            output_path=output_path,
                            csv_fname='smc_transformer_history.csv',
                            logger=logger,
                            start_epoch=start_epoch)

    # making predictions with the trained model and saving them on .npy files
    #mask = create_look_ahead_mask(seq_len)
    saving_model_outputs(output_path=output_path,
                         predictions=predictions_val,
                         attn_weights=attn_weights_val,
                         pred_fname='smc_predictions.npy',
                         attn_weights_fname='smc_attn_weights.npy')
    model_output_path = os.path.join(output_path, "model_outputs")
    # saving weights on top of it.
    weights_fn = model_output_path + '/' + 'smc_weights.npy'
    np.save(weights_fn, weights_val)

    logger.info('training of SMC Transformer for a time-series dataset done...')

    logger.info(">>>--------------------------------------------------------------------------------------------------------------------------------------------------------------<<<")


    # TEST THE LOSS ON A BATCH
    #TODO: adapt this for our case.
    # for input_example_batch, target_example_batch in dataset.take(1):
    #   example_batch_predictions = model(input_example_batch)
    #   print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    # example_batch_loss = loss(target_example_batch, example_batch_predictions)
    # print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
    # print("scalar_loss:      ", example_batch_loss.numpy().mean())


