#TODO: debug the case when the number of epochs is the same. (training is done...)
# basic logging tutorial: https://docs.python.org/3/howto/logging.html#logging-basic-tutorial

#TODO - to improve the transformer performance - label smoothing and change order of layers.dense / layers.norm.

#TODO: add on the config file for the reegression case, an additional hparams for omega (covariance of the gaussian noise.).

"""# to store:
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
max_positional_encoding = target_vocab_size. 
"""
"""
ALL INPUTS CAN BE OF SHAPE (B,S,F) (F=1 for NLP / univariate time_series case, F > 1 for multivariate case.)
"""

import tensorflow as tf
from models.Baselines.Transformer_without_enc import Transformer
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask

from train.train_step_functions import train_step_classic_T
from train.train_step_functions import train_step_SMC_T

from train.loss_functions import compute_accuracy_variance
from train.loss_functions import CustomSchedule
from train.loss_functions import loss_function_regression

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
import statistics

from preprocessing.time_series.df_to_dataset import df_to_data_regression
from preprocessing.time_series.df_to_dataset import data_to_dataset_uni_step
from preprocessing.time_series.df_to_dataset import split_input_target_uni_step
from preprocessing.time_series.df_to_dataset import split_synthetic_dataset

from utils.utils_train import create_run_dir
from utils.utils_train import create_logger
from utils.utils_train import saving_training_history
from utils.utils_train import saving_model_outputs
from utils.utils_train import restoring_checkpoint

from eval.inference_SMC_Transformer import evaluate_one_timestep

if __name__ == "__main__":

  warnings.simplefilter("ignore")
  #if type(tf.contrib) != type(tf): tf.contrib._warning = None

  # -------- parsing arguments ---------------------------------------------------------------------------------------------------------------------------------------

  out_folder_for_args ='/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/exp_162_grad_not_zero_azure/'
  config_path_after_training = out_folder_for_args + 'time_series_multi_unistep-forcst_heads_1_depth_3_dff_12_pos-enc_50_pdrop_0.1_b_1048_cs_True__particles_25_noise_True_sigma_0.1_smc-pos-enc_None/config.json'

  out_folder_default = '../../output/exp_192_back_to_beginning_pressure_TRAIN_SPLIT_0.8'
  config_folder = '../../config/config_ts_reg_multi.json'

  parser = argparse.ArgumentParser()

  parser.add_argument("-config", type=str, default=config_folder, help="path for the config file with hyperparameters")
  parser.add_argument("-out_folder", type=str, default=out_folder_default, help="path for the outputs folder")
  parser.add_argument("-data_folder", type=str, default='../../data/synthetic_dataset.npy', help="path for the data folder")

  #TODO: ask Florian why when removing default value, it is not working...
  parser.add_argument("-train_baseline", type=bool, default=False, help="Training a Baseline Transformer?")
  parser.add_argument("-train_smc_T", type=bool, default=True, help="Training the SMC Transformer?")
  parser.add_argument("-train_rnn", type=bool, default=False, help="Training a Baseline RNN?")
  parser.add_argument("-skip_training", type=bool, default=False, help="skip training and directly evaluate?")
  parser.add_argument("-eval", type=bool, default=False, help="evaluate after training?")

  parser.add_argument("-load_ckpt", type=bool, default=True, help="loading and restoring existing checkpoints?")
  args = parser.parse_args()
  config_path = args.config

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

  # task params
  data_type = hparams["task"]["data_type"]
  task_type = hparams["task"]["task_type"]
  task = hparams["task"]["task"]

  # smc params
  num_particles = hparams["smc"]["num_particles"]
  noise_encoder_str = hparams["smc"]["noise_encoder"]
  noise_encoder = True if noise_encoder_str == "True" else False
  noise_SMC_layer_str = hparams["smc"]["noise_SMC_layer"]
  noise_SMC_layer = True if noise_SMC_layer_str == "True" else False
  sigma = hparams["smc"]["sigma"]
  omega = 1
  if task == 'synthetic':
    omega = hparams["smc"]["omega"]
  # computing manually resampling parameter
  resampling = False if num_particles == 1 else True

  # optim params
  BATCH_SIZE = hparams["optim"]["BATCH_SIZE"]
  learning_rate = hparams["optim"]["learning_rate"]
  EPOCHS = hparams["optim"]["EPOCHS"]
  custom_schedule = hparams["optim"]["custom_schedule"]

  # adding RNN hyper-parameters
  rnn_bs = BATCH_SIZE
  rnn_emb_dim = hparams["RNN_hparams"]["rnn_emb_dim"]
  rnn_units = hparams["RNN_hparams"]["rnn_units"]

  # loading data arguments for the regression case
  if task_type == 'regression' and task =='unistep-forcst':

    file_path = hparams["data"]["file_path"]
    fname = hparams["data"]["fname"]
    col_name = hparams["data"]["col_name"]
    index_name = hparams["data"]["index_name"]
    TRAIN_SPLIT = hparams["data"]["TRAIN_SPLIT"]
    history = hparams["data"]["history"]
    step = hparams["data"]["step"]
    target_feature = hparams["data"]["target_feature"]
    if target_feature == "None":
      target_feature = None

  if task_type == 'regression' and task == 'synthetic':
    file_path = hparams["data"]["file_path"]
    TRAIN_SPLIT = hparams["data"]["TRAIN_SPLIT"]
    target_feature = hparams["data"]["target_feature"]
    if target_feature == "None":
      target_feature = None

  test_loss = False

  #------------------ UPLOAD the training dataset ----------------------------------------------------------------------------------------------------------------------

  if task_type == 'classification':

    data_folder = args.data_folder
    train_data = np.load(data_folder + '/ts_weather_train_data.npy')
    val_data = np.load(data_folder + '/ts_weather_val_data.npy')
    test_data = np.load(data_folder + '/ts_weather_test_data.npy')

  elif task_type == 'regression':

    if task =='unistep-forcst':

      (train_data, val_data, test_data), original_df, stats = df_to_data_regression(file_path=file_path,
                                                                           fname=fname,
                                                                           col_name=col_name,
                                                                           index_name=index_name,
                                                                           TRAIN_SPLIT=TRAIN_SPLIT,
                                                                           history=history,
                                                                           step=step)

      BUFFER_SIZE = 10000

    elif task == 'synthetic':
      X_data = np.load(file_path)
      train_data, val_data = split_synthetic_dataset(x_data=X_data, TRAIN_SPLIT=TRAIN_SPLIT)
      val_data_path = '../../data/val_data_synthetic.npy'
      train_data_path = '../../data/train_data_synthetic.npy'
      np.save(val_data_path, val_data)
      np.save(train_data_path, train_data)

      BUFFER_SIZE = 2000

  print('train_data', train_data.shape)
  print('val_data', val_data.shape)


  train_dataset, val_dataset, train_dataset_for_RNN, val_dataset_for_RNN = data_to_dataset_uni_step(train_data=train_data,
                                                        val_data=val_data,
                                                        split_fn=split_input_target_uni_step,
                                                        BUFFER_SIZE=BUFFER_SIZE,
                                                        BATCH_SIZE=BATCH_SIZE,
                                                        target_feature=target_feature)

  for (inp, tar) in train_dataset.take(1):
    print('input example', inp[0])
    print('target example', tar[0])

  num_classes= 25 if data_type == 'classification' else 1
  target_vocab_size = num_classes # 25 bins
  seq_len = train_data.shape[1] - 1 # 24 observations
  training_samples = train_data.shape[0]
  steps_per_epochs = int(train_data.shape[0]/BATCH_SIZE)
  print("steps per epochs", steps_per_epochs)

  # -------define hyperparameters----------------------------------------------------------------------------------------------------------------

  if custom_schedule == "True":
    print("learning rate with custom schedule...")
    learning_rate=CustomSchedule(d_model)

  optimizer = tf.keras.optimizers.Adam(learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.98,
                                       epsilon=1e-9)
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
  val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

  # ------------- preparing the OUTPUT FOLDER------------------------------------------------------------------------

  output_path = args.out_folder
  folder_template='{}_{}_heads_{}_depth_{}_dff_{}_pos-enc_{}_pdrop_{}_b_{}_cs_{}'
  out_folder = folder_template.format(data_type,
                                      task,
                                      num_heads,
                                      d_model,
                                      dff,
                                      maximum_position_encoding_baseline,
                                      rate,
                                      BATCH_SIZE,
                                      custom_schedule)

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

  # ------------------ create the logging -----------------------------------------------------------------------------

  out_file_log = output_path + '/' + 'training_log.log'
  logger = create_logger(out_file_log=out_file_log)
  #  creating the checkpoint manager:
  checkpoint_path = os.path.join(output_path, "checkpoints")
  if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

  #-------------------- Building the RNN Baseline & the associated training algo -----------------------------------------------------------

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

  #---------------------- TRAINING OF A SIMPLE RNN BASELINE --------------------------------------------------------------------------------------
  if args.skip_training:
    logger.info("skipping training...")
  else:

    if args.train_rnn:
      for (inp, _) in train_dataset.take(1):
        pred_temp = model(inp)
      print('LSTM summary', model.summary())

      start_epoch = 0
      model.compile(optimizer=optimizer,
                    loss='mse')
      start_training = time.time()
      rnn_history=model.fit(train_dataset_for_RNN,
                epochs=EPOCHS,
                validation_data=val_dataset_for_RNN,
                verbose=2)

      train_loss_history_rnn = rnn_history.history['loss']
      val_loss_history_rnn = rnn_history.history['val_loss']
      keys = ['train_loss', 'val_loss']
      values = [train_loss_history_rnn, val_loss_history_rnn]
      history = dict(zip(keys, values))
      csv_fname = 'rnn_history.csv'

      saving_training_history(keys=keys,
                              values=values,
                              output_path=output_path,
                              csv_fname=csv_fname,
                              logger=logger,
                              start_epoch=start_epoch)

      logger.info('Training time for {} epochs: {}'.format(EPOCHS, time.time() - start_training))
      logger.info('training of a RNN Baseline for a timeseries dataset done...')
      logger.info(">>>--------------------------------------------------------------------------------------------------------------------------------------------------------------<<<")

    #----------------- TRAINING -----------------------------------------------------------------------------------------------------------------------------------------------------------

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

      logger.info("training the baseline Transformer on a time-series dataset...")
      logger.info("number of training samples: {}".format(training_samples))
      logger.info("steps per epoch:{}".format(steps_per_epochs))


      # storing the losses & accuracy in a list for each epoch
      average_losses_baseline = []
      val_losses_baseline = []
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
      ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
      baseline_ckpt_path = os.path.join(checkpoint_path, "transformer_baseline")
      ckpt_manager = tf.train.CheckpointManager(ckpt, baseline_ckpt_path, max_to_keep=EPOCHS)

      # if a checkpoint exists, restore the latest checkpoint.
      start_epoch = restoring_checkpoint(ckpt_manager=ckpt_manager, args=args, ckpt=ckpt, logger=logger)
      if start_epoch is None:
        start_epoch = 0

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

        train_loss.reset_states()
        train_accuracy.reset_states()
        val_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(dataset):
          inp_model = inp [:,:-1, :]
          train_loss_batch, avg_loss_batch, train_accuracy_batch, _ = train_step_classic_T(inputs=inp_model,
                                                                            targets=tar,
                                                                            transformer=transformer,
                                                                            train_loss=train_loss,
                                                                            train_accuracy=train_accuracy,
                                                                            optimizer=optimizer,
                                                                            data_type=data_type,
                                                                            task_type=task_type)
          if batch == 0 and epoch==0:
            print('baseline transformer summary', transformer.summary())

        for (inp, tar) in val_dataset:
          inp_model = inp[:, :-1, :]
          predictions_val, attn_weights_val = transformer(inputs=inp_model,
                                                          training=False,
                                                          mask=create_look_ahead_mask(seq_len))
          val_loss = tf.keras.losses.MSE(tar, predictions_val)
          val_loss = tf.reduce_mean(val_loss, axis=-1)
          val_loss = tf.reduce_mean(val_loss, axis=-1)

        if task_type == 'classification':
          train_acc = train_accuracy.result()
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
          logger.info('train loss: {} - val loss: {}'.format(train_loss_batch.numpy(), val_loss.numpy()))
          average_losses_baseline.append(train_loss_batch.numpy())
          val_losses_baseline.append(val_loss.numpy())

        ckpt_save_path = ckpt_manager.save()
        logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

      logger.info('total training time for {} epochs:{}'.format(EPOCHS,time.time() - start_training))

      # storing history of losses and accuracies in a csv file
      if task_type == 'classification':
        keys = ['loss', 'train_accuracy', 'val_accuracy']
        values = [average_losses_baseline, training_accuracies_baseline, val_accuracies_baseline]

      elif task_type == 'regression':
        keys = ['train loss', 'val loss']
        values = [average_losses_baseline, val_losses_baseline]

      history = dict(zip(keys, values))
      csv_fname = 'baseline_history.csv'

      saving_training_history(keys=keys,
                              values=values,
                              output_path=output_path,
                              csv_fname=csv_fname,
                              logger=logger,
                              start_epoch=start_epoch)

      saving_model_outputs(output_path=output_path,
                           predictions=predictions_val,
                           attn_weights=attn_weights_val,
                           pred_fname='baseline_predictions.npy',
                           attn_weights_fname='baseline_attn_weights.npy',
                           logger=logger)

      logger.info('training of a classic Transformer for a time-series dataset done...')
      logger.info(">>>-------------------------------------------------------------------------------------------------------------------------------------------------------------<<<")

      #------- computing statistics at the end of training -------------------------------------------------------------------------------------------------------------------

      logger.info("computing metrics at the end of training...")
      train_loss_mse, val_loss_mse= [], []
      for batch_train, (inp, tar) in enumerate(train_dataset):
        inp_model = inp[:, :-1, :]
        predictions_train, _ = transformer(
          inputs=inp_model,
          training=False,
          mask=create_look_ahead_mask(seq_len))
        train_loss_mse_batch = tf.keras.losses.MSE(tar, predictions_train)
        train_loss_mse_batch = tf.reduce_mean(train_loss_mse_batch, axis=-1)
        train_loss_mse_batch = tf.reduce_mean(train_loss_mse_batch, axis=-1)

        train_loss_mse.append(train_loss_mse_batch.numpy())

      for batch_val, (inp, tar) in enumerate(val_dataset):
        inp_model = inp[:, :-1, :]
        predictions_val, _ = transformer(
          inputs=inp_model,
          training=False,
          mask=create_look_ahead_mask(seq_len))
        val_loss_mse_batch = tf.keras.losses.MSE(tar, predictions_val)
        val_loss_mse_batch = tf.reduce_mean(val_loss_mse_batch, axis=-1)
        val_loss_mse_batch = tf.reduce_mean(val_loss_mse_batch, axis=-1)

        val_loss_mse.append(val_loss_mse_batch.numpy())

      # computing as a metric the mean of losses & std losses over the number of batches
      mean_train_loss_mse = statistics.mean(train_loss_mse)
      mean_val_loss_mse = statistics.mean(val_loss_mse)

      logger.info(
        "average losses over batches: train mse loss:{}  - val mse loss:{}".format(
          mean_train_loss_mse,
          mean_val_loss_mse))


  #------------------- TRAINING ON THE DATASET - SMC_TRANSFORMER ----------------------------------------------------------------------------------------------------------------------

    if train_smc_transformer:

      logger.info('starting the training of the smc transformer...')
      logger.info("number of training samples: {}".format(training_samples))
      logger.info("steps per epoch: {}".format(steps_per_epochs))
      if not resampling:
        logger.info("no resampling because only one particle is taken...")

      start_epoch = 0
      #TODO: add omega here:
      smc_transformer = SMC_Transformer(num_layers=num_layers,
                            d_model=d_model,
                            num_heads=num_heads,
                            dff=dff,
                            target_vocab_size=target_vocab_size,
                            maximum_position_encoding=maximum_position_encoding_smc,
                            num_particles=num_particles,
                            sigma=sigma,
                            omega=omega,
                            noise_encoder=noise_encoder,
                            noise_SMC_layer=noise_SMC_layer,
                            seq_len=seq_len,
                            data_type=data_type,
                            task_type=task_type,
                            resampling=resampling,
                            target_feature=target_feature)

      # creating checkpoint manager
      ckpt = tf.train.Checkpoint(transformer=smc_transformer,
                                 optimizer=optimizer)
      smc_T_ckpt_path = os.path.join(checkpoint_path, "SMC_transformer")
      ckpt_manager = tf.train.CheckpointManager(ckpt, smc_T_ckpt_path, max_to_keep=EPOCHS)

      # if a checkpoint exists, restore the latest checkpoint.
      start_epoch=restoring_checkpoint(ckpt_manager=ckpt_manager, ckpt=ckpt, args=args, logger=logger)
      if start_epoch is None:
        start_epoch = 0

      # check the pass forward.
      for input_example_batch, target_example_batch in dataset.take(2):
        #input_model = tf.concat([input_example_batch, target_example_batch[:,-1,:]], axis = 1)
        (example_batch_predictions, traj, _, _), predictions_metric, _ = smc_transformer(inputs=input_example_batch,
                                                training=True,
                                                mask=create_look_ahead_mask(seq_len))
        print("predictions shape: {}".format(example_batch_predictions.shape))

      #print('summary of the SMC Transformer', smc_transformer.summary())

      if start_epoch > 0:
        if start_epoch > EPOCHS:
          print("adding {} more epochs to existing training".format(EPOCHS))
          start_epoch = 0
        else:
          logger.info("starting training after checkpoint restoring from epoch {}".format(start_epoch))

      start_training = time.time()

      # preparing recording of loss and metrics information
      if task_type == 'classification':
        train_loss_history, train_inf_acc_history, train_avg_acc_history, train_max_acc_history= [], [], [], []
        val_inf_acc_history, val_avg_acc_history, val_max_acc_history, val_acc_variance_history = [], [], [], []

      elif task_type == 'regression':
        train_loss_history, train_loss_mse_history = [], []
        val_loss_history, val_loss_mse_history = [], []

      for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        logger.info('Epoch {}/{}'.format(epoch+1,EPOCHS))
        train_loss.reset_states()
        train_accuracy.reset_states()

        if test_loss:
        #TODO: check that the loss is the same for 1 particule, and 5 particules with no noise.
          for (inp_ex_batch, target_ex_batch) in dataset.take(1):
            loss_temp, accuracies_temp, _ = train_step_SMC_T(inputs=inp,
                                                              targets=tar,
                                                              smc_transformer=smc_transformer,
                                                              optimizer=optimizer,
                                                              train_loss=train_loss,
                                                              train_accuracy=train_accuracy,
                                                              classic_loss=True,
                                                              SMC_loss=True)
            logger.info('testing the loss on a batch...{}'.format(loss_temp.numpy()))

        # training step:
        for (batch, (inp, tar)) in enumerate(dataset):
          #TODO: add the output of the weights and indices matrix every 100 steps_per_epochs.
          avg_loss_batch, train_accuracies, _ = train_step_SMC_T(inputs=inp,
                                                                targets=tar,
                                                                smc_transformer=smc_transformer,
                                                                optimizer=optimizer,
                                                                train_loss=train_loss,
                                                                train_accuracy=train_accuracy,
                                                                classic_loss=True,
                                                                SMC_loss=True)

          if task_type == 'classification':
            train_inf_acc_batch, train_avg_acc_batch, train_max_acc_batch = train_accuracies
          elif task_type == 'regression':
            mse_metric, mse_loss_std= train_accuracies

        # compute the validation accuracy on the validation dataset:
        # TODO: here consider a validation set with a batch_size equal to the number of samples.
        for batch, (inp, tar) in enumerate(val_dataset):
          (predictions_val, _, weights_val, ind_matrix_val), predictions_metric, attn_weights_val = smc_transformer(inputs=inp,
                                                                                                    training=False,
                                                                                                    mask=create_look_ahead_mask(seq_len))
          val_loss, val_loss_mse, val_loss_mse_std = loss_function_regression(real = tar,
                                                            predictions = predictions_val,
                                                            weights = weights_val,
                                                            transformer = smc_transformer)
          #TODO: add the classification case

        logger.info('final weights of first 3 elements of batch: {}, {}, {}'.format(weights_val[0,:], weights_val[1,:], weights_val[2,:]))


        #------------------------- computing and saving metrics (train set and validation set)----------------------------------------------------

        if task_type == 'classification':
          train_inf_acc_batch, train_avg_acc_batch, train_max_acc_batch = train_accuracies
          # computing the validation accuracy for each batch...
          val_inf_pred_batch, val_avg_pred_batch, val_max_pred_batch = predictions_metric
          val_inf_acc_batch = val_accuracy(tar, val_inf_pred_batch)
          val_avg_acc_batch = val_accuracy(tar, val_avg_pred_batch)
          val_max_acc_batch = val_accuracy(tar, val_max_pred_batch)
          # computing the variance in accuracy for each 'prediction particle':
          val_acc_variance=compute_accuracy_variance(predictions_val=predictions_val,
                                                   tar=tar,
                                                   accuracy_metric=val_accuracy)

          template='train loss {} - train acc, inf: {} - train acc, avg: {} - train_acc, max: {},' \
                 ' - val acc, inf: {} - val acc, avg: {} - val acc, max: {}'
          logger.info(template.format(avg_loss_batch.numpy(),
                                    train_inf_acc_batch.numpy(),
                                    train_avg_acc_batch.numpy(),
                                    train_max_acc_batch.numpy(),
                                    val_inf_acc_batch.numpy(),
                                    val_avg_acc_batch.numpy(),
                                    val_max_acc_batch.numpy()))

          # saving loss and metrics information:
          train_loss_history.append(avg_loss_batch.numpy())
          train_inf_acc_history.append(train_inf_acc_batch.numpy())
          train_avg_acc_history.append(train_avg_acc_batch.numpy())
          train_max_acc_history.append(train_max_acc_batch.numpy())

          val_inf_acc_history.append(val_inf_acc_batch.numpy())
          val_avg_acc_history.append(val_avg_acc_batch.numpy())
          val_max_acc_history.append(val_max_acc_batch.numpy())

          val_acc_variance_history.append(val_acc_variance)

        elif task_type == 'regression':
          mse_metric, mse_loss_std = train_accuracies
          template = 'train loss {} -  train mse loss: {} - train loss std (mse): {} - val loss: {} - val mse loss: {} - val loss std (mse): {}'
          logger.info(template.format(avg_loss_batch.numpy(),
                                      mse_metric.numpy(),
                                      mse_loss_std.numpy(),
                                      val_loss.numpy(),
                                      val_loss_mse.numpy(),
                                      val_loss_mse_std.numpy()))

          #TODO: add a tf.keras.metrics.Mean
          # saving loss and metrics information:
          train_loss_history.append(avg_loss_batch.numpy())
          train_loss_mse_history.append(mse_metric.numpy())
          val_loss_history.append(val_loss.numpy())
          val_loss_mse_history.append(val_loss_mse.numpy())

          #------------- end of saving metrics information -------------------------------------------------------------------------------

        ckpt_save_path = ckpt_manager.save()

        logger.info('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

      logger.info('total training time for {} epochs:{}'.format(EPOCHS, time.time() - start_training))

      if task_type == 'classification':
        # storing history of losses and accuracies in a csv file
        keys = ['train loss', 'train accuracy, inference', 'train accuracy, from avg', 'train accuracy, from max',
                'val accuracy - inference', 'val accuracy, from avg', 'val accuracy, from max',
                'variance of validation accuracy']
        values = [train_loss_history, train_inf_acc_history, train_avg_acc_history, train_max_acc_history,
                  val_inf_acc_history, val_avg_acc_history, val_max_acc_history, val_acc_variance_history]

        saving_training_history(keys=keys, values=values,
                                output_path=output_path,
                                csv_fname='smc_transformer_history.csv',
                                logger=logger,
                                start_epoch=start_epoch)

      elif task_type == 'regression':
        keys = ['train loss', 'train mse loss', 'val loss', 'val mse loss']
        values = [train_loss_history, train_loss_mse_history, val_loss_history, val_loss_mse_history]
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
                           attn_weights_fname='smc_attn_weights.npy',
                           logger=logger)

      model_output_path = os.path.join(output_path, "model_outputs")
      # saving weights on top of it.
      weights_fn = model_output_path + '/' + 'smc_weights.npy'
      np.save(weights_fn, weights_val)

      #----------------------  compute statistics at the end of training ----------------------------------------------------------------------------

      logger.info("computing metrics at the end of training...")
      train_loss_mse, train_loss_std, val_loss_mse, val_loss_std = [], [], [], []
      for batch_train, (inp, tar) in enumerate(train_dataset):
        (predictions_train, _, weights_train, _), predictions_metric, attn_weights_train = smc_transformer(
          inputs=inp,
          training=False,
          mask=create_look_ahead_mask(seq_len))
        _, train_loss_mse_batch, train_loss_mse_std_batch = loss_function_regression(real=tar,
                                                                                     predictions=predictions_train,
                                                                                     weights=weights_train,
                                                                                     transformer=smc_transformer)
        train_loss_mse.append(train_loss_mse_batch.numpy())
        train_loss_std.append(train_loss_mse_std_batch.numpy())

      for batch_val, (inp, tar) in enumerate(val_dataset):
        (predictions_val, _, weights_val, _), _, attn_weights_val = smc_transformer(
          inputs=inp,
          training=False,
          mask=create_look_ahead_mask(seq_len))
        _, val_loss_mse_batch, val_loss_mse_std_batch = loss_function_regression(real=tar,
                                                                                 predictions=predictions_val,
                                                                                 weights=weights_val,
                                                                                 transformer=smc_transformer)

        val_loss_mse.append(val_loss_mse_batch.numpy())
        val_loss_std.append(val_loss_mse_std_batch.numpy())

      # computing as a metric the mean of losses & std losses over the number of batches
      mean_train_loss_mse = statistics.mean(train_loss_mse)
      mean_train_loss_std = statistics.mean(train_loss_std)
      mean_val_loss_mse = statistics.mean(val_loss_mse)
      mean_val_loss_std = statistics.mean(val_loss_std)

      logger.info("average losses over batches: train mse loss:{} - train loss std (mse):{} - val mse loss:{} - val loss (mse) std: {}".format(
        mean_train_loss_mse,
        mean_train_loss_std,
        mean_val_loss_mse,
        mean_val_loss_std))

      logger.info('training of SMC Transformer for a time-series dataset done...')

      logger.info(">>>--------------------------------------------------------------------------------------------------------------------------------------------------------------<<<")

# ----------------------------------- EVALUATION -------------------------------------------------------------------------------------------------------------------------------
  if args.eval:
    logger.info("start evaluation...")
    # restoring latest checkpoint
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
                                      target_feature=target_feature)

    # creating checkpoint manager
    ckpt = tf.train.Checkpoint(transformer=smc_transformer,
                               optimizer=optimizer)
    smc_T_ckpt_path = os.path.join(checkpoint_path, "SMC_transformer")
    ckpt_manager = tf.train.CheckpointManager(ckpt, smc_T_ckpt_path, max_to_keep=EPOCHS)

    # if a checkpoint exists, restore the latest checkpoint.
    num_epochs = restoring_checkpoint(ckpt_manager=ckpt_manager, ckpt=ckpt, args=args, logger=logger)

    # --------------------------------------------- compute latest statistics -----------------------------------------

    logger.info("<------------------------computing latest statistics----------------------------------------------------------------------------------------->")

    # compute last mse train loss, std loss / val loss
    train_loss_mse, train_loss_std, val_loss_mse, val_loss_std= [], [], [], []
    for batch_train, (inp, tar) in enumerate(train_dataset.take(5)):
      (predictions_train, _, weights_train, _), predictions_metric, attn_weights_train = smc_transformer(
        inputs=inp,
        training=False,
        mask=create_look_ahead_mask(seq_len))
      _, train_loss_mse_batch, train_loss_mse_std_batch= loss_function_regression(real=tar,
                                                                          predictions=predictions_train,
                                                                          weights=weights_train,
                                                                          transformer=smc_transformer)
      train_loss_mse.append(train_loss_mse_batch.numpy())
      train_loss_std.append(train_loss_mse_batch.numpy())

    for batch_val, (inp, tar) in enumerate(val_dataset.take(5)):
      (predictions_val, _, weights_val, _), _, attn_weights_val = smc_transformer(
        inputs=inp,
        training=False,
        mask=create_look_ahead_mask(seq_len))
      _, val_loss_mse_batch, val_loss_mse_std_batch = loss_function_regression(real=tar,
                                                                          predictions=predictions_val,
                                                                          weights=weights_val,
                                                                          transformer=smc_transformer)

      val_loss_mse.append(val_loss_mse_batch.numpy())
      val_loss_std.append(val_loss_mse_std_batch.numpy())

    # computing as a metric the mean of losses & std losses over the number of batches
    mean_train_loss_mse = statistics.mean(train_loss_mse)
    mean_train_loss_std = statistics.mean(train_loss_std)
    mean_val_loss_mse = statistics.mean(val_loss_mse)
    mean_val_loss_std = statistics.mean(val_loss_std)

    logger.info("train mse loss:{} - train loss std (mse):{} - val mse loss:{} - val loss (mse) std: {}".format(
      mean_train_loss_mse,
      mean_train_loss_std,
      mean_val_loss_mse,
      mean_val_loss_std))


    # # test loss value
    # for (inp, tar) in val_dataset.take(1):
    #   (predictions_val, _, weights_val, ind_matrix_val), predictions_metric, attn_weights_val = smc_transformer(
    #     inputs=inp,
    #     training=False,
    #     mask=create_look_ahead_mask(seq_len))
    #   val_loss, val_loss_mse, val_loss_mse_std = loss_function_regression(real=tar,
    #                                                                       predictions=predictions_val,
    #                                                                       weights=weights_val,
    #                                                                       transformer=smc_transformer)
    # logger.info('testing loss... val mse loss: {}'.format(val_loss_mse))

    # ------------------------------------------------- preparing test dataset ------------------------------------------------------------------

    # transform test data (numpy array) into a tensor (use a tf.dataset instead.)
    test_data = test_data[:100,:,:]
    x_test, y_test = split_input_target_uni_step(test_data)
    if target_feature is not None:
      y_test = y_test[:, :, target_feature]
      y_test = np.reshape(y_test, newshape=(y_test.shape[0], y_test.shape[1], 1))
    BATCH_SIZE_test = test_data.shape[0]
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE_test)

    # -----unistep evaluation with N = 1 ---------------------------------------------------------------------------------------------
    logger.info("unistep evaluation with N=1...")
    for (inp,tar) in test_dataset:
      (predictions_test, _, weights_test, _), _, attn_weights_test = smc_transformer(
        inputs=inp,
        training=False,
        mask=create_look_ahead_mask(seq_len))

    #TODO: unnormalized predictions and targets.
      # unnormalized predictions & target:
    data_mean, data_std = stats
    predictions_unnormalized = predictions_test * data_std + data_mean
    targets_unnormalized = y_test * data_std + data_mean

    # save predictions & attention weights:
    eval_output_path = os.path.join(output_path, "eval_outputs")
    if not os.path.isdir(eval_output_path):
      os.makedirs(eval_output_path)
    pred_unistep_N_1_test = eval_output_path + '/' + 'pred_unistep_N_1_test.npy'
    attn_weights_unistep_N_1_test = eval_output_path + '/' + 'attn_weights_unistep_N_1_test.npy'
    targets_test = eval_output_path + '/' + 'targets_test.npy'
    pred_unnorm = eval_output_path + '/' + 'pred_unistep_N_1_test_unnorm.npy'
    targets_unnorm = eval_output_path + '/' + 'targets_test_unnorm.npy'
    np.save(pred_unistep_N_1_test, predictions_test)
    np.save(attn_weights_unistep_N_1_test, attn_weights_test)
    np.save(targets_test, y_test)
    np.save(pred_unnorm, predictions_unnormalized)
    np.save(targets_unnorm, targets_unnormalized)

    # ---- multistep evaluation --------------------------------------------------------------------------------------------------------------------
    # input_seq_length = 13
    # num_samples = 2
    #
    # for (inp,tar) in test_dataset:
    #   (pred_inp, attn_weights), (mean_pred, pred_P, pred_NP), inp_to_infer = evaluate_one_timestep(
    #   model=smc_transformer,
    #   inputs=inp,
    #   num_samples=num_samples,
    #   inp_seq_len=input_seq_length)
    #
    # # save output of evaluation function in .npy files.
    # eval_output_path = os.path.join(output_path, "eval_outputs")
    # mean_pred_test = eval_output_path + '/' + 'mean_pred_test.npy'
    # pred_P_test = eval_output_path + '/' + 'pred_P_test.npy'
    # pred_NP_test = eval_output_path + '/' + 'pred_NP_test.npy'
    # np.save(mean_pred_test, mean_pred)
    # np.save(pred_P_test, pred_P)
    # np.save(pred_NP_test, pred_NP)

    ##-----------------old code scripts --------------------------------------------------------------------------------------------------------------

      # TEST THE LOSS ON A BATCH
      #TODO: adapt this for our case.
      # for input_example_batch, target_example_batch in dataset.take(1):
      #   example_batch_predictions = model(input_example_batch)
      #   print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
      # example_batch_loss = loss(target_example_batch, example_batch_predictions)
      # print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
      # print("scalar_loss:      ", example_batch_loss.numpy().mean())

      # if args.train_rnn:
      #   logger.info("training a RNN Baseline on the nlp dataset...")
      #   logger.info("number of training samples: {}".format(training_samples))
      #   logger.info("steps per epoch:{}".format(steps_per_epochs))
      #
      #   start_training = time.time()
      #
      #   for epoch in range(EPOCHS):
      #     start = time.time()
      #     train_accuracy.reset_states()
      #     val_accuracy.reset_states()
      #     # initializing the hidden state at the start of every epoch
      #     # initally hidden is None
      #     hidden = model.reset_states()
      #
      #
      #     for (inp, target) in train_dataset:
      #         if task_type == 'classification':
      #         # CAUTION: in a tf.keras.layers.LSTM, the input tensor needs to be of shape 3 (B,S,Features).
      #           loss, train_acc_batch = train_step_rnn_classif(inp,
      #                                                        target,
      #                                                        model=model,
      #                                                        optimizer=optimizer,
      #                                                        accuracy_metric=train_accuracy)
      #         elif task_type == 'regression':
      #           predictions=model(inp)
      #           loss = train_step_rnn_regression(inp=inp,
      #                                        target=target,
      #                                        model=model,
      #                                        optimizer=optimizer)
      #
      #     model.save_weights(checkpoint_prefix.format(epoch=epoch))
      #
      #     # computing train and val acc for the current epoch:
      #
      #     for (inp_val, tar_val) in val_dataset:
      #       # inp_val needs to be of shape (B,S,F) : length=3
      #       inp_val = tf.expand_dims(inp_val, axis=-1)
      #       predictions_val = model(inp_val)
      #       # computing the validation accuracy for each batch...
      #       if task_type == 'classification':
      #         val_accuracy_batch = val_accuracy(tar_val, predictions_val)
      #
      #     if task_type == 'classification':
      #       train_acc = train_accuracy.result()
      #       val_acc=val_accuracy.result()
      #     elif task_type == 'regression':
      #       train_acc = 0
      #       val_acc = 0
      #
      #     logger.info('Epoch {} - Loss {} - train acc {} - val acc {}'.format(epoch + 1, loss, train_acc, val_acc))
      #     logger.info('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
      #
      #     model.save_weights(checkpoint_prefix.format(epoch=epoch))
      #
      #   logger.info('Training time for {} epochs: {}'.format(EPOCHS, time.time() - start_training))
      #   logger.info('training of a RNN Baseline (GRU) for a nlp dataset done...')
      #   logger.info(">>>--------------------------------------------------------------------------------------------------------------------------------------------------------------<<<")



