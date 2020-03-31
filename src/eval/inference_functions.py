import tensorflow as tf
from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from models.Baselines.Transformer_without_enc import Transformer
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
import numpy as np
import scipy.stats

import ot

from utils.KL_divergences_estimators import naive_estimator
from eval.Transformer_dropout import MC_Dropout_Transformer

#import scipy.stats.wasserstein_distance as wass_distance
#import scipy.stats.entropy as KL_distance
#https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.norm.html

#TODO: create an inference function for the Baseline Transformer with a MC-Dropout algorithm.

def inference_Baseline_T_MC_Dropout_1D(inputs, transformer, num_mc_samples, num_timesteps, output_path):
  '''
  :param inputs: (B,S,F)
  :param transformer:
  :param num_mc_samples:
  :param num_timsteps:
  :return:
  '''
  s = tf.shape(inputs)[1] - num_timesteps
  inp_model = inputs[:, :s, :]
  inp_inference = inputs[:, s:, :]
  # forward pass on the first s inputs:
  predictions, _ = transformer(inputs=inp_model,
                                      training=True,
                                      mask=create_look_ahead_mask(s)) # predictions (B,s,1)


  last_pred = predictions [:,-1,:] # (B,1)
  #last_pred = tf.expand_dims(last_pred, axis=1)
  for t in range(num_timesteps):
    obs_feat = inp_inference[:,t,1:]
    new_input = tf.concat([last_pred,obs_feat], axis=1) # (B,F)
    new_input = tf.expand_dims(new_input, axis=1)
    inp_model = tf.concat([inp_model, new_input], axis=1) # (B,s+1,F)
    seq_len = tf.shape(inp_model)[1]
    if t == num_timesteps - 1:
      MC_Dropout_predictions = MC_Dropout_Transformer(transformer=transformer,
                                                      test_dataset=inp_model,
                                                      seq_len=seq_len,
                                                      num_samples=num_mc_samples,
                                                      task='synthetic',
                                                      stats=None,
                                                      output_path=None,
                                                      logger=None)  # (B,N,S,1)
    else:
      predictions, _ = transformer(inputs=inp_model,
                                   training=True,
                                   mask=create_look_ahead_mask(seq_len))
      last_pred = predictions[:,-1,:]

  # select only the inference part (number of timesteps):
  MC_Dropout_predictions = MC_Dropout_predictions[:,:,s:,:].numpy()
  np.save(file=output_path, arr=MC_Dropout_predictions)

  return MC_Dropout_predictions



def inference_function_multistep(inputs, smc_transformer, N_prop, N_est, num_particles, num_timesteps, sample_pred=False):
  '''
  :param inputs: shape (B,S,F)
  :param smc_transformer:
  :param N_prop:
  :param N_est:
  :param num_particles:
  :param num_timesteps:
  :param sample_pred:
  :return:
  '''
  list_X_pred_NP, list_r_NP = [], []
  N = N_prop

  # call of the smc_transformer on inputs:
  s = tf.shape(inputs)[1]
  mask = create_look_ahead_mask(s)
  smc_transformer.noise_SMC_layer = True
  smc_transformer.num_particles = num_particles
  smc_transformer.cell.num_particles = num_particles
  outputs, _, _ = smc_transformer(inputs=inputs,
                                  training=False,
                                  mask=mask)
  _, _, w_s, (K0_s, V0_s) = outputs
  # preprocessing initial input:
  input = inputs[:, -1, :] # (B,1,F)
  input = tf.expand_dims(input, axis=1)  # (B,1,F)
  input = tf.expand_dims(input, axis=1) # (B,1,1,F)
  input = tf.tile(input, multiples=[1, num_particles, 1, 1])  # (B,P,1,F)

  # adding zeros to KO_s and V0_s
  shape_future = (tf.shape(K0_s)[0], num_particles, num_timesteps, tf.shape(K0_s)[-1])
  future_K = tf.zeros(shape=shape_future)
  future_V = tf.zeros(shape=shape_future)
  K_init = tf.concat([K0_s, future_K], axis=2)
  V_init = tf.concat([V0_s, future_V], axis=2)
  K = K_init
  V= V_init
  X_pred_NP = input # (B,P,1,F)
  num_init_features = tf.shape(input)[-1]
  for t in range(num_timesteps):
    if tf.shape(X_pred_NP)[-1] == 1:
      #TODO: remove this if loop for the multi-dim case.
      X_pred_NP = tf.tile(X_pred_NP, multiples = [1, 1, 1, num_init_features]) # mandatory for t >0 (last dim is 1 otherwise and issue pf shape
    X_pred_NP = smc_transformer.input_dense_projection(X_pred_NP) # (B,P,1,D) or # (B,N*P,1,D)
    X_pred_NP, r_N_P, (K, V) = smc_transformer.cell.inference_function(inputs=X_pred_NP, K=K, V=V, num_samples=N, t=t+s, inf_timestep=t)
    list_X_pred_NP.append(X_pred_NP)
    list_r_NP.append(r_N_P)

  # getting sampled predictions from the list of r_NP:
  list_preds_multistep = []
  if sample_pred:
    for r in list_r_NP:
      # reshape to have a tensor of shape (B,N,P,1,D)
      new_shape = (tf.shape(r)[0], -1, N, num_particles, tf.shape(r)[-2], tf.shape(r)[-1])
      r = tf.reshape(r, shape=new_shape) # (B,-1,N,P,1,D)
      r = tf.squeeze(r, axis=1) # (B,N,P,1,D)
      # select n* and p*:
      p_ = tf.random.categorical(logits=w_s, num_samples=1)
      uniform_logits = tf.constant([[1/N for _ in range(N)] for _ in range(tf.shape(r)[0])])
      n_ = tf.random.categorical(logits=uniform_logits, num_samples=1)
      r_=tf.gather(r, p_, axis=2, batch_dims=1)
      r_=tf.gather(r_, n_, axis=1, batch_dims=1) # (B,1,1,1,D)
      r_= tf.squeeze(tf.squeeze(r_, axis=1), axis=1) # (B,1,D)

      list_pred_t = []
      for _ in range(N_est):
        mean_r_=smc_transformer.final_layer(r_)
        X_pred = mean_r_ + tf.random.normal(shape=tf.shape(mean_r_), stddev=smc_transformer.omega) # (B,1,F)
        list_pred_t.append(X_pred)
      tensor_pred_t = tf.stack(list_pred_t, axis=1) # (B,N_est,1,1)
      tensor_pred_t = tf.squeeze(tf.squeeze(tensor_pred_t, axis=-1), axis=-1) # (B, N_est)
      list_preds_multistep.append(tensor_pred_t)
    tensor_preds_multistep = tf.stack(list_preds_multistep, axis=2)

  return (list_r_NP, list_X_pred_NP), (list_preds_multistep, tensor_preds_multistep)

def inference_function_multistep_1D(inputs, smc_transformer, N_prop, N_est, num_particles, num_timesteps, omega, sigma, output_path, sample_pred=False):
  '''
  :param inputs: shape (B,S,F)
  :param smc_transformer:
  :param N_prop:
  :param N_est:
  :param num_particles:
  :param num_timesteps:
  :param sample_pred:
  :return:
  '''

  # ------ inference function -------------------------------------------------------------------------------------------------------------

  list_X_pred_NP, list_r_NP = [], []
  N = N_prop

  # call of the smc_transformer on inputs:
  s = tf.shape(inputs)[1] - num_timesteps
  mask = create_look_ahead_mask(s)
  smc_transformer.noise_SMC_layer = True
  smc_transformer.cell.noise = True
  smc_transformer.cell.mha_smc.noise = True
  smc_transformer.num_particles = num_particles
  smc_transformer.cell.num_particles = num_particles
  smc_transformer.sigma = sigma
  smc_transformer.cell.mha_smc.sigma_scalar = sigma

  inp_model = inputs[:,:s,:]
  inp_inference = inputs[:,s:,:]
  outputs, _, _ = smc_transformer(inputs=inp_model,
                                  training=False,
                                  mask=mask)
  _, _, w_s, (K0_s, V0_s) = outputs
  # preprocessing initial input:
  input = inp_model[:, -1, :] # (B,1,F)
  input = tf.expand_dims(input, axis=1)  # (B,1,F)
  input = tf.expand_dims(input, axis=1) # (B,1,1,F)
  input = tf.tile(input, multiples=[1, num_particles, 1, 1])  # (B,P,1,F)

  # adding zeros to KO_s and V0_s
  shape_future = (tf.shape(K0_s)[0], num_particles, num_timesteps, tf.shape(K0_s)[-1])
  future_K = tf.zeros(shape=shape_future)
  future_V = tf.zeros(shape=shape_future)
  K_init = tf.concat([K0_s, future_K], axis=2)
  V_init = tf.concat([V0_s, future_V], axis=2)
  K = K_init
  V= V_init
  X_pred_NP = input # (B,P,1,F)

  for t in range(num_timesteps):
    if tf.shape(X_pred_NP)[-1] == 1:
      NP = tf.shape(X_pred_NP)[1]
      obs_features = inp_inference[:,t,1:] # (B,F=2)
      obs_features = tf.expand_dims(obs_features, axis=1) # (B,1,F=2)
      obs_features_NP = tf.expand_dims(obs_features, axis=1) # (B,1,1,F=2)
      obs_features_NP = tf.tile(obs_features_NP, multiples=[1,NP,1,1]) # (B,NP,1,F=2)
      X_pred_NP = tf.concat([X_pred_NP, obs_features_NP], axis=-1)
    X_pred_NP = smc_transformer.input_dense_projection(X_pred_NP) # (B,P,1,D) or # (B,N*P,1,D)
    X_pred_NP, r_N_P, (K, V) = smc_transformer.cell.inference_function(inputs=X_pred_NP, K=K, V=V, num_samples=N, t=t+s, inf_timestep=t)
    list_X_pred_NP.append(X_pred_NP)
    list_r_NP.append(r_N_P)

  # ---------------------------------------------------------sampling N_est predictions for each timestep -------------------------------#
  list_preds_multistep = []
  if sample_pred:
    for r in list_r_NP:
      # reshape to have a tensor of shape (B,N,P,1,D)
      new_shape = (tf.shape(r)[0], -1, N, num_particles, tf.shape(r)[-2], tf.shape(r)[-1])
      r = tf.reshape(r, shape=new_shape) # (B,-1,N,P,1,D)
      r = tf.squeeze(r, axis=1) # (B,N,P,1,D)
      # select n* and p*:
      p_ = tf.random.categorical(logits=w_s, num_samples=1)
      uniform_logits = tf.constant([[1/N for _ in range(N)] for _ in range(tf.shape(r)[0])])
      n_ = tf.random.categorical(logits=uniform_logits, num_samples=1)
      r_=tf.gather(r, p_, axis=2, batch_dims=1)
      r_=tf.gather(r_, n_, axis=1, batch_dims=1) # (B,1,1,1,D)
      r_= tf.squeeze(tf.squeeze(r_, axis=1), axis=1) # (B,1,D)

    # list_pred_t = []
    #   for _ in range(N_est):
    #     #TODO: use here numpy.random.normal, instead of tf.random.normal.
    #     mean_r_=smc_transformer.final_layer(r_) #TODO: transform this into a numpy_array to avoid the for loop.
    #     X_pred = mean_r_ + tf.random.normal(shape=tf.shape(mean_r_), stddev=smc_transformer.omega) # (B,1,F)
    #     list_pred_t.append(X_pred)
    #   tensor_pred_t = tf.stack(list_pred_t, axis=1) # (B,N_est,1,1)
    #   tensor_pred_t = tf.squeeze(tf.squeeze(tensor_pred_t, axis=-1), axis=-1) # (B, N_est)
    #   list_preds_multistep.append(tensor_pred_t)
    # tensor_preds_multistep = tf.stack(list_preds_multistep, axis=2)

      mean_r_ = smc_transformer.final_layer(r_) # (B,1,1)
      mean_r_ = tf.squeeze(mean_r_, axis=-1)
      mean_r_ = mean_r_.numpy() # (B,1)
      batch_size = mean_r_.shape[0]
      X_pred = np.random.normal(loc=mean_r_, scale=omega, size=(batch_size, N_est))
      list_preds_multistep.append(X_pred)

    # -------------------------------- computing mean for each N*P Gaussian distribution to plot the complete distribution -----------------
    # get the mean of the mix of gaussian distributions from r
    list_mean_NP = [smc_transformer.final_layer(r_NP) for r_NP in list_r_NP]
    list_mean_NP = [mean.numpy() for mean in list_mean_NP] # transform list_mean_NP in a numpy array
    list_mean_NP = [m.reshape(m.shape[0], N, num_particles, m.shape[-1]) for m in list_mean_NP] # reshape (B,NP,1,1) to (B,N,P,1)

    # ------------------------------- save arrays for plotting the predicted probability density function------------------------------------
    list_X_pred_NP = [tf.squeeze(x, axis=-2).numpy() for x in list_X_pred_NP]
    w_s = w_s.numpy()
    gaussian_means_path = output_path + '/' + 'pred_gaussian_means_per_timestep.npy'
    sampled_distrib_path = output_path + '/' + 'preds_sampled_per_timestep.npy'
    sampling_weights_path = output_path + '/' + 'sampling_weights.npy'
    all_preds_path = output_path + '/' + 'list_X_pred_NP.npy'

    np.save(file=gaussian_means_path, arr=list_mean_NP)
    np.save(file=sampled_distrib_path, arr=list_preds_multistep)
    np.save(file=sampling_weights_path, arr=w_s)
    np.save(file=all_preds_path, arr=list_X_pred_NP)

  #TODO: transform list_X_pred_NP into a numpy array as well for consistency.
  return (list_mean_NP, list_X_pred_NP), list_preds_multistep, w_s


def generate_empirical_distribution_1D(inputs, matrix_A, cov_matrix, N_est, num_timesteps, output_path):
  '''
  :param inputs: ts input data of shape (B,S,F)
  :param matrix_A: matrix of autogressive model > shape (F * F)
  :param cov_matrix: covariance matrix for the gaussian noise > shape (F,1)
  :param N_est: number of samples to draw for the empirical distribution
  :param num_timesteps: number of timesteps for the inference
  :return:
  '''
  s = tf.shape(inputs)[1] - num_timesteps
  inp_model = inputs[:, :s, :]
  inp_inference = inputs[:, s:, :]
  last_input = inp_model[:,-1,:] # (B,F)
  num_features = tf.shape(last_input)[-1]
  batch_size = tf.shape(last_input)[0]
  list_preds_sampled = []
  list_mean_per_timestep = []

  for t in range(num_timesteps):
    list_pred_t = []
    mean = tf.matmul(last_input, matrix_A)
    list_mean_per_timestep.append(mean)

    #TODO: debug this code.
    #mean = mean.numpy()
    # cov_matrix = cov_matrix.numpy()
    # sampled_new_inputs = np.random.normal(loc=mean, scale=cov_matrix, size=(batch_size, N_est, num_features)) # (B, N_est, F)
    # sampled_new_inputs = sampled_new_inputs[:,:,0] # (B, N_est)
    # list_preds_sampled.append(sampled_new_inputs)
    # # compute new_input for next timestep
    # sample_ind = np.random.randint(0, N_est)
    # last_input = sampled_new_inputs[:, sample_ind]  # (B)
    # last_input = tf.expand_dims(last_input, axis=-1)
    # obs_features = inp_inference[:, t, 1:]  # (B,F_obs=2)
    # last_input = tf.concat([last_input, obs_features], axis=-1)

    for n in range(N_est):
      new_input = mean + tf.random.normal(stddev=cov_matrix, shape=(1, num_features)) # (B,F)
      new_input = tf.expand_dims(new_input[:,0], axis=-1) # (B,1)
      list_pred_t.append(new_input)

    tensor_pred_t = tf.stack(list_pred_t, axis=1)  # (B,N_est,1)
    list_preds_sampled.append(tensor_pred_t)
    # compute new_input for next timestep
    sample_ind = np.random.randint(0, N_est)
    last_input = list_pred_t[sample_ind]  # (B,1)
    obs_features = inp_inference[:,t,1:] #(B,F_obs=2)
    last_input = tf.concat([last_input, obs_features], axis=-1)

  # transforming tensors into numpy arrays.
  list_preds_sampled = [tf.squeeze(x, axis=-1).numpy() for x in list_preds_sampled] # (B,N_est)
  list_mean_per_timestep = [m.numpy() for m in list_mean_per_timestep]
  # saving information in .npy files
  true_distrib_path = output_path + '/' + 'true_empirical_distrib.npy'
  true_means_path = output_path + '/' + 'true_gaussian_means.npy'
  np.save(file=true_distrib_path, arr=list_preds_sampled)
  np.save(file=true_means_path, arr=list_mean_per_timestep)

  return list_preds_sampled, list_mean_per_timestep

def generate_empirical_distribution(inputs, matrix_A, cov_matrix, N_est, num_timesteps, output_path):
  '''
  :param inputs: ts input data of shape (B,S,F)
  :param matrix_A: matrix of autogressive model > shape (F * F)
  :param cov_matrix: covariance matrix for the gaussian noise > shape (F,1)
  :param N_est: number of samples to draw for the empirical distribution
  :param num_timesteps: number of timesteps for the inference
  :return:
  '''
  last_input = inputs[:,-1,:] # (B,F)
  num_features = tf.shape(last_input)[-1]
  list_preds_sampled = []
  list_mean_per_timestep = []
  for t in range(num_timesteps):
    list_pred_t = []
    mean = tf.matmul(last_input, matrix_A)
    list_mean_per_timestep.append(mean)
    for n in range(N_est):
      new_input = mean + tf.random.normal(stddev=cov_matrix, shape=(1, num_features)) # (B,F)
      list_pred_t.append(new_input)
    sample_ind = np.random.randint(0, N_est)
    last_input = list_pred_t[sample_ind] # (B,F)
    tensor_pred_t = tf.stack(list_pred_t, axis=1)  # (B,N_est,F)
    list_preds_sampled.append(tensor_pred_t[:,:,0])
  #tensor_preds_sampled = tf.stack(list_preds_sampled, axis=2) # (B, N_est, S)

  list_preds_sampled = [x.numpy() for x in list_preds_sampled]
  list_mean_per_timestep = [m.numpy for m in list_mean_per_timestep]

  #----- saving true empirical distrib and gaussian mean for plotting needs------------------------
  true_distrib_path = output_path + '/' + 'true_empirical_distrib.npy'
  true_means_path = output_path + '/' + 'true_gaussian_means.npy'
  np.save(file=true_distrib_path, arr=list_preds_sampled)
  np.save(file=true_means_path, arr=list_mean_per_timestep)

  return list_preds_sampled, list_mean_per_timestep


if __name__ == "__main__":
  num_particles_training = 1
  seq_len = 24
  b = 8
  F = 3 # multivariate case.
  num_layers = 1
  d_model = 12
  num_heads = 4
  dff = 48
  maximum_position_encoding = seq_len
  sigma = 0.05
  data_type = 'time_series_multi'
  task_type = 'regression'
  C = 1 # vocabulary size or number of classes.
  noise_encoder = False
  noise_SMC_layer = False
  rate = 0
  omega = 0.3

  target_feature = 0 if data_type == 'time_series_multi' else None
  maximum_position_encoding = 50

  sample_transformer = SMC_Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    target_vocab_size=C,
    maximum_position_encoding=maximum_position_encoding,
    num_particles=num_particles_training,
    seq_len=seq_len,
    sigma=sigma,
    omega=omega,
    noise_encoder=noise_encoder,
    noise_SMC_layer=noise_SMC_layer,
    data_type=data_type,
    task_type=task_type,
    target_feature=target_feature,
    rate=rate)

  inputs = tf.random.uniform(shape=(b,seq_len+1,F))

  # ---------------------- test of multi-step inference function ----------------------------------------------------------------------
  num_samples = 5
  N_est = 25
  num_timesteps = 4
  num_particles_inference = 10

  output_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/temp'

  (list_mean_NP, list_X_pred_NP), list_preds_sampled, w_s = inference_function_multistep_1D(inputs=inputs,
                                                                                            smc_transformer=sample_transformer,
                                                                                            N_prop=num_samples,
                                                                                            N_est=N_est,
                                                                                            num_particles=num_particles_inference,
                                                                                            num_timesteps=num_timesteps,
                                                                                            output_path=output_path,
                                                                                            sample_pred=True,
                                                                                            sigma=0.1,
                                                                                            omega=omega)

  print('number of timesteps predicted', len(list_preds_sampled))
  print('example of preds', list_preds_sampled[0])
  print('number of examples per preds', (list_preds_sampled[0].shape[1]))

  list_gaussian_means = np.load(file=output_path + '/' + 'pred_gaussian_means_per_timestep.npy')

  #--------------- test of generate_empirical_distribution ----------------------------------------------------------------------------

  cov_matrix_3D = tf.constant([0.2, 0.3, 0.4], dtype=tf.float32)
  A_3D = tf.constant([[0.8, 0.1, 0], [0.2, 0.9, 0.2], [0, 0.1, 0.85]], dtype=tf.float32)

  list_empirical_dist, list_true_means = generate_empirical_distribution_1D(inputs=inputs,
                                                                                  matrix_A=A_3D,
                                                                                  cov_matrix=cov_matrix_3D,
                                                                                  N_est=N_est,
                                                                                  num_timesteps=num_timesteps,
                                                                                  output_path=output_path)


  print('number of timesteps predicted - empirical distribution', len(list_preds_sampled))
  print('example of preds - empirical distribution', list_preds_sampled[0])
  print('number of examples per preds', list_preds_sampled[0].shape[1])

  true_gaussian_means = np.load(output_path + '/' + 'true_gaussian_means.npy')

  #------ test of MC Dropout inference function -----------------------------------------------------
  B=8
  S=24
  num_feat = 3
  num_layers = 1
  d_model = 2
  dff = 8
  num_heads = 1
  target_vocab_size = 1
  maximum_position_encoding_baseline = 50
  rate = 0.1
  num_mc_samples = 25

  test_dataset = tf.random.uniform(shape=(B, S, num_feat))
  transformer = Transformer(num_layers=num_layers,
                            d_model=d_model,
                            num_heads=num_heads,
                            dff=dff,
                            target_vocab_size=target_vocab_size,
                            maximum_position_encoding=maximum_position_encoding_baseline,
                            data_type=data_type,
                            rate=rate)

  output_path_T = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/temp' + '/' + 'Baseline_T_MC_Dropout_preds.npy'

  MC_Dropout_predictions = inference_Baseline_T_MC_Dropout_1D(inputs=test_dataset,
                                                              transformer=transformer,
                                                              num_mc_samples=num_mc_samples,
                                                              num_timesteps=4,
                                                              output_path=output_path_T)


  # ----------- computation of the KL divergence ----------------------------------------------------------------------------------------
  #KL_measure = tf.keras.losses.KLDivergence()
  #KL_measure_2 = tf.keras.losses.KLDivergence(reduction=losses_utils.ReductionV2.NONE)

  for t, (true_distrib, pred_distrib) in enumerate(zip(list_empirical_dist, list_preds_sampled)):
    N_est = pred_distrib.shape[1]
    std_pred_distrib = np.std(pred_distrib, axis=1)
    std_pred_distrib = np.mean(std_pred_distrib, axis=0)
    #KL_dist = scipy.stats.entropy(pk=pred_distrib, qk=true_distrib, axis=1)
    wass_dist = ot.emd2_1d(x_a=true_distrib[0,:], x_b=pred_distrib[0,:])
    KL_dist = naive_estimator(true_distrib[0,:].reshape(N_est,1), pred_distrib[0,:].reshape(N_est,1))
    #wass_dist = scipy.stats.wasserstein_distance(true_distrib, pred_distrib)

    #print('KL distance for timestep {}: {}'.format(t, KL_distance))
