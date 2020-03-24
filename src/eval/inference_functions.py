import tensorflow as tf
from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
import numpy as np

# def evaluate_one_timestep(model, num_samples, inputs, inp_seq_len):
#   """
#   :param model:
#   :param num_samples:
#   :param inputs: shape (B, S, F)
#   :param inp_seq_len:
#   :param target_seq_len:
#   :return:
#   """
#   num_particles = model.num_particles
#   #total_seq_len = tf.shape(inputs)[1]
#
#   # forward pass on inp_seq_len of inputs
#   inp_to_predict = inputs[:,:inp_seq_len, :]
#   inp_to_infer = inputs[:,inp_seq_len:,:] # (B,S-S_inp,F)
#   target_seq_len = tf.shape(inp_to_infer)[1]
#
#   mask_inp = create_look_ahead_mask(inp_seq_len)
#
#   (pred, traj, w_T, (K,V)),_, attn_weights = model(inputs=inp_to_predict, mask=mask_inp, training = False)
#
#   # adding zeros to (K,V) to have tensors of shape (B,P,S,D)
#   shape_future = (tf.shape(K)[0], num_particles, target_seq_len, tf.shape(K)[-1])
#   future_K = tf.zeros(shape=shape_future)
#   future_V = tf.zeros(shape=shape_future)
#   K = tf.concat([K, future_K], axis=2)
#   V = tf.concat([V, future_V], axis=2)
#
#   # take the last pred:
#   #last_pred = pred[:,:,-1,:]
#
#   list_inf_pred, list_pred_P, list_pred_N_P = [], [], []
#
#   # preprocess inp_to_infer:
#   inputs_mha = tf.expand_dims(inp_to_infer, axis=1)  # (B,1,S-Sinp,F)
#   inputs_mha = tf.tile(inputs_mha, multiples=[1, num_particles, 1, 1])  # (B,P,S-Sinp,F)
#   inputs_mha = model.input_dense_projection(inputs_mha) # (B,P,S-Sinp,D)
#
#   for t in range(target_seq_len):
#     inp_t = inputs_mha[:, :, t, :] # shape (B,P,D)
#     inp_t = tf.expand_dims(inp_t, axis=2)
#     (inf_pred, pred_P, pred_N_P), (K, V) = model.cell.inference_function(inouts=inp_t,
#                                                                          K=K,
#                                                                          V=V,
#                                                                          num_samples=num_samples,
#                                                                          t=t,
#                                                                          inf)
#     #TODO: add resampling step here? (by adding as output w_t and as input of the function the target?)
#     list_inf_pred.append(inf_pred)  # (B,1,F=1)
#     list_pred_P.append(pred_P)  # (B,P,1,F=1)
#     list_pred_N_P.append(pred_N_P)  # (B,N*P,1,F=1)
#
#   # seq_inf_pred = tf.stack(list_inf_pred, axis=1)
#   # seq_pred_P = tf.stack(list_pred_P, axis=2)
#   # seq_pred_N_P = tf.stack(list_pred_N_P, axis=2)
#
#   return (pred, attn_weights), (list_inf_pred, list_pred_P, list_pred_N_P), inp_to_infer


def inference_function_multistep(inputs, smc_transformer, N_prop, N_est, num_particles, num_timesteps, sample_pred=False):
  '''
  :param inputs:
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

      #------------- OR... FOR SAMPLING PREDICTIONS...-----------------------------------------------------------------------------------------------
      # for r in list_r_NP:
      #   # reshape to have a tensor of shape (B,N,P,1,D)
      #   new_shape = (tf.shape(r)[0], -1, N, num_particles, tf.shape(r)[-2], tf.shape(r)[-1])
      #   r = tf.reshape(r, shape=new_shape)  # (B,-1,N,P,1,D)
      #   r = tf.squeeze(r, axis=1)  # (B,N,P,1,D)
      #   # select n* and p*:
      #   p_ = tf.random.categorical(logits=w_s, num_samples=num_particles)
      #   uniform_logits = tf.constant([[1 / N for _ in range(N)] for _ in range(tf.shape(r)[0])])
      #   n_ = tf.random.categorical(logits=uniform_logits, num_samples=N)
      #   r_ = tf.gather(r, p_, axis=2, batch_dims=1)
      #   r_ = tf.gather(r_, n_, axis=1, batch_dims=1)  # (B,N,P,1,D)
      #   mean_r_=smc_transformer.final_layer(r_) # (B,N,P,1,D)
      #   X_pred = mean_r_ + tf.random.normal(shape=tf.shape(mean_r_), stddev=smc_transformer.omega) # (B,N,P,1,D)
      #   list_preds_multistep.append(X_pred)

  return (list_r_NP, list_X_pred_NP), (list_preds_multistep, tensor_preds_multistep)

def generate_empirical_distribution(inputs, matrix_A, cov_matrix, N_est, num_timesteps):
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
  for t in range(num_timesteps):
    list_pred_t = []
    for n in range(N_est):
      new_input = tf.matmul(last_input, matrix_A) + tf.random.normal(stddev=cov_matrix, shape=(1, num_features)) # (B,F)
      list_pred_t.append(new_input)
    sample_ind = np.random.randint(0, N_est)
    last_input = list_pred_t[sample_ind] # (B,F)
    tensor_pred_t = tf.stack(list_pred_t, axis=1)  # (B,N_est,F)
    list_preds_sampled.append(tensor_pred_t[:,:,0])
  tensor_preds_sampled = tf.stack(list_preds_sampled, axis=2) # (B, N_est, S)

  return list_preds_sampled, tensor_preds_sampled

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
  sigma = 0.1
  data_type = 'time_series_multi'
  task_type = 'regression'
  C = 1 # vocabulary size or number of classes.
  noise_encoder = False
  noise_SMC_layer = False
  rate = 0

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

  (list_r_t, list_X_pred_t), (list_preds_sampled, tensor_preds_sampled) = inference_function_multistep(inputs=inputs,
                                                                               smc_transformer=sample_transformer,
                                                                               N_prop=num_samples,
                                                                               N_est=N_est,
                                                                               num_particles=num_particles_inference,
                                                                               num_timesteps=num_timesteps,
                                                                               sample_pred=True)
  print('number of timesteps predicted', len(list_preds_sampled))
  print('example of preds', list_preds_sampled[0])
  print('number of examples per preds', len(list_preds_sampled[0]))

  print('shape of tensor for multistep inference', tensor_preds_sampled.shape)

  #--------------- test of generate_empirical_distribution ----------------------------------------------------------------------------

  cov_matrix_3D = tf.constant([0.2, 0.3, 0.4], dtype=tf.float32)
  A_3D = tf.constant([[0.8, 0.1, 0], [0.2, 0.9, 0.2], [0, 0.1, 0.85]], dtype=tf.float32)

  list_empirical_dist, tensor_empirical_distrib = generate_empirical_distribution(inputs=inputs,
                                                       matrix_A=A_3D,
                                                       cov_matrix=cov_matrix_3D,
                                                       N_est=N_est,
                                                       num_timesteps=num_timesteps)
  print('number of timesteps predicted - empirical distribution', len(list_preds_sampled))
  print('example of preds - empirical distribution', list_preds_sampled[0])
  print('number of examples per preds', len(list_preds_sampled[0]))
  print('shape of tensor for multistep empirical distribution', tensor_preds_sampled.shape)

  # ----------- computation of the KL divergence ----------------------------------------------------------------------------------------
  KL_measure = tf.keras.losses.KLDivergence()
  for t, (true_distrib, pred_distrib) in enumerate(zip(list_empirical_dist, list_preds_sampled)):
    KL_distance = KL_measure(y_true=true_distrib, y_pred=pred_distrib)
    print('KL distance for timestep {}: {}'.format(t, KL_distance))