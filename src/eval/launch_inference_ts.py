import tensorflow as tf
import numpy as np

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer

from utils.utils_train import create_run_dir
from utils.utils_train import create_logger
from utils.utils_train import restoring_checkpoint
from utils.utils_train import saving_inference_results
from train.loss_functions import CustomSchedule
from eval.inference_functions import inference_function_multistep_1D
from eval.inference_functions import generate_empirical_distribution_1D
import statistics

import ot
from utils.KL_divergences_estimators import naive_estimator

import argparse
import json
import os

if __name__ == "__main__":

  #---- parsing arguments --------------------------------------------------------------

  parser = argparse.ArgumentParser()
  results_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/post_UAI_exp/results_ws155_632020'
  exp_path = 'time_series_multi_synthetic_heads_2_depth_6_dff_24_pos-enc_50_pdrop_0_b_256_target-feat_0_cv_False__particles_1_noise_False_sigma_0.05'
  default_data_folder = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/data/test_data_synthetic_3_feat.npy'

  #parser.add_argument("-config", type=str, help="path for the config file with hyperparameters")
  parser.add_argument("-out_folder", default=exp_path, type=str, help="path for the output folder with training result")
  parser.add_argument("-data_path", default=default_data_folder, type=str, help="path for the test data folder")
  parser.add_argument("-num_timesteps", default=1, type=int, help="number of timesteps for doing inference")
  #parser.add_argument("-p_inf", default=15, type=int, help="number of particles generated for inference")
  parser.add_argument("-N", default=10, type=int, help="number of samples for MC sampling")
  #TODO: remove this one and do a for loop instead.
  #parser.add_argument("-N_est", type=int, help="number of samples for estimating the empirical distribution")

  args=parser.parse_args()
  exp_path = args.out_folder
  output_path = os.path.join(results_path, exp_path)
  config_path = os.path.join(output_path, 'config.json')
  test_data_path = args.data_path

  # ------ uploading the hparams info -----------------------------------------------------------------------------------------------------------------------------------

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
  #mc_dropout_samples = hparams["model"]["mc_dropout_samples"]

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
  rnn_units = hparams["RNN_hparams"]["rnn_units"]
  rnn_dropout_rate = hparams["RNN_hparams"]["rnn_dropout_rate"]

  # loading data arguments for the regression case

  file_path = hparams["data"]["file_path"]
  TRAIN_SPLIT = hparams["data"]["TRAIN_SPLIT"]
  VAL_SPLIT = hparams["data"]["VAL_SPLIT"]
  VAL_SPLIT_cv = hparams["data"]["VAL_SPLIT_cv"]
  cv_str = hparams["data"]["cv"]
  cv = True if cv_str == "True" else False
  target_feature = hparams["data"]["target_feature"]
  if target_feature == "None":
    target_feature = None

  if task_type == 'regression' and task == 'unistep-forcst':
    history = hparams["data"]["history"]
    step = hparams["data"]["step"]
    fname = hparams["data"]["fname"]
    col_name = hparams["data"]["col_name"]
    index_name = hparams["data"]["index_name"]

  # -------------- uploading the test dataset --------------------------------------------------------------------------------------------------------------------------
  test_dataset = np.load(test_data_path)
  seq_len = test_dataset.shape[1] - 1
  # convert it into a tf.tensor
  test_dataset = tf.convert_to_tensor(test_dataset, dtype=tf.float32)

  # ---------------preparing the output path for inference -------------------------------------------------------------------------------------------------------------
  num_timesteps = args.num_timesteps
  #p_inf = args.p_inf
  N = args.N
  list_p_inf = [10,20,50]
  N_est = 10000

  output_path = args.out_folder
  checkpoint_path = os.path.join(output_path, "checkpoints")

  if not os.path.isdir(os.path.join(output_path, 'inference_results')):
    output_path = create_run_dir(path_dir=output_path, path_name='inference_results')
  output_path = os.path.join(output_path, 'inference_results')
  folder_template = 'num-timesteps_{}_p_inf_{}-{}-{}_N_{}_N-est_{}'
  out_folder=folder_template.format(num_timesteps, list_p_inf[0], list_p_inf[1], list_p_inf[2], N, N_est)
  output_path = create_run_dir(path_dir=output_path, path_name=out_folder)

  # -------------- create the logging -----------------------------------------------------------------------------------------------------------------------------------
  out_file_log = output_path + '/' + 'inference_log.log'
  logger = create_logger(out_file_log)

  # --------------- restoring checkpoint for smc transformer ------------------------------------------------------------------------------------------------------------
  # define optimizer
  if custom_schedule == "True":
    learning_rate = CustomSchedule(d_model)

  optimizer = tf.keras.optimizers.Adam(learning_rate,
                                       beta_1=0.9,
                                       beta_2=0.98,
                                       epsilon=1e-9)
  target_vocab_size = 1
  # create a SMC_Transformer
  smc_transformer = SMC_Transformer(num_layers=num_layers,
                                    d_model=d_model,
                                    num_heads=num_heads,
                                    dff=dff,
                                    target_vocab_size=target_vocab_size,
                                    maximum_position_encoding=maximum_position_encoding_smc,
                                    num_particles=num_particles,
                                    sigma=sigma,
                                    rate=rate,
                                    noise_encoder=noise_encoder,
                                    noise_SMC_layer=noise_SMC_layer,
                                    seq_len=seq_len,
                                    data_type=data_type,
                                    task_type=task_type,
                                    resampling=resampling,
                                    target_feature=target_feature)

  # get checkpoint path for SMC_Transformer
  smc_T_ckpt_path = os.path.join(checkpoint_path, "SMC_transformer")
  # create checkpoint manager
  smc_T_ckpt = tf.train.Checkpoint(transformer=smc_transformer,
                                   optimizer=optimizer)

  smc_T_ckpt_manager = tf.train.CheckpointManager(smc_T_ckpt, smc_T_ckpt_path, max_to_keep=EPOCHS)
  # restore latest checkpoint from out folder
  num_epochs_smc_T = restoring_checkpoint(ckpt_manager=smc_T_ckpt_manager, ckpt=smc_T_ckpt,
                                          args_load_ckpt=True, logger=logger)

  # ------------------------------------- compute latest statistics as a check -----------------------------------------------------------------------------------------
  # ------------------------------------- compute inference timesteps --------------------------------------------------------------------------------------------------
  #list_num_samples = [500, 1000, 5000, 10000, 50000]
  cov_matrix_3D = tf.constant([0.2, 0.3, 0.4], dtype=tf.float32)
  A_3D = tf.constant([[0.8, 0.1, 0], [0.2, 0.9, 0.2], [0, 0.1, 0.85]], dtype=tf.float32)

  list_KL_exp = []
  for p_inf in list_p_inf:
    logger.info('inference results for number of particles: {}'.format(p_inf))

    (list_r_NP, list_X_pred_NP), (list_preds_sampled, tensor_preds_sampled) = inference_function_multistep_1D(inputs=test_dataset,
                                                                                                       smc_transformer=smc_transformer,
                                                                                                       N_prop=N,
                                                                                                       N_est=N_est,
                                                                                                       num_particles=p_inf,
                                                                                                       num_timesteps=num_timesteps,
                                                                                                       sample_pred=True)


    list_empirical_dist, tensor_empirical_distrib = generate_empirical_distribution_1D(inputs=test_dataset,
                                                                                  matrix_A=A_3D,
                                                                                  cov_matrix=cov_matrix_3D,
                                                                                  N_est=N_est,
                                                                                  num_timesteps=num_timesteps)
    #KL_measure = tf.keras.losses.KLDivergence()
    # KL_distance = KL_measure(y_true=true_distrib, y_pred=pred_distrib)
    # KL_distance_norm = KL_distance / N_est
    # KL_timesteps.append(KL_distance_norm.numpy())

    KL_timesteps = []
    for t, (true_distrib, pred_distrib) in enumerate(zip(list_empirical_dist, list_preds_sampled)):
      true_distrib = tf.squeeze(true_distrib, axis=-1)
      true_distrib = true_distrib.numpy()
      pred_distrib = pred_distrib.numpy()
      batch_size = pred_distrib.shape[0]
      num_samples = pred_distrib.shape[1]

      #KL_dist = scipy.stats.entropy(pk=pred_distrib, qk=pred_distrib)
      wassertein_dist_list = [ot.emd2_1d(x_a=true_distrib[i,:], x_b=pred_distrib[i,:]) for i in range(batch_size)]
      wassertein_dist = statistics.mean(wassertein_dist_list)
      KL_distance_list = [naive_estimator(true_distrib[i,:].reshape(num_samples,1), pred_distrib[i,:].reshape(num_samples,1)) for i in range(batch_size)]
      KL_dist = statistics.mean(KL_distance_list)
      logger.info('KL distance for timestep {}: {}'.format(t, KL_dist))
      logger.info('wassertein distance for timestep {}: {}'.format(t, wassertein_dist))

    #list_KL_exp.append(KL_timesteps)
    logger.info("<--------------------------------------------------------------------------------------------------------------------------------------------------------->")

  # ------------------------------------------------------- saving results on a csv file ---------------------------------------------------------------

  # keys = ['N_est = {}'.format(n) for n in list_num_samples]
  # keys = ['inference_timestep'] + keys
  # values = ['t+{}'.format(i+1) for i in range(num_timesteps)]
  # values = [values] + list_KL_exp
  # csv_fname='inference_results_KL_norm.csv'
  #
  # saving_inference_results(keys=keys,
  #                           values=values,
  #                           output_path=output_path,
  #                           csv_fname=csv_fname)
