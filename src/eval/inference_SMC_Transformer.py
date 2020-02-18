import tensorflow as tf
from models.SMC_Transformer.SMC_Transformer import SMC_Transformer

def evaluate_one_timestep(model, mask, num_samples, inputs, inp_seq_len):
  """
  :param model:
  :param num_samples:
  :param inputs: shape (B, S, F)
  :param inp_seq_len:
  :param target_seq_len:
  :return:
  """
  num_particles = model.num_particles
  #total_seq_len = tf.shape(inputs)[1]

  # forward pass on inp_seq_len of inputs
  inp_to_predict = inputs[:,:inp_seq_len, :]
  inp_to_infer = inputs[:,inp_seq_len,:] # (B,S-S_inp,F)
  target_seq_len = tf.shape(inp_to_infer)[1]

  (pred, traj, w_T, (K,V)),_, attn_weights = model(inputs=inp_to_predict, mask=mask, training = False)

  # adding zeros to (K,V) to have tensors of shape (B,P,S,D)
  shape_future = (tf.shape(K)[0], num_particles, target_seq_len, tf.shape(K)[-1])
  future_K = tf.zeros(shape=shape_future)
  future_V = tf.zeros(shape=shape_future)
  K = tf.concat([K, future_K], axis=2)
  V = tf.concat([V, future_V], axis=2)

  # take the last pred:
  #last_pred = pred[:,:,-1,:]

  list_inf_pred, list_pred_P, list_pred_N_P = [], [], []

  # preprocess inp_to_infer:
  inputs_mha = tf.expand_dims(inp_to_infer, axis=1)  # (B,1,S-Sinp,F)
  inputs_mha = tf.tile(inputs_mha, multiples=[1, num_particles, 1, 1])  # (B,P,S-Sinp,F)
  inputs_mha = model.input_dense_projection(inputs_mha) # (B,P,S-Sinp,D)

  for t in range(target_seq_len):
    inp_t = inputs_mha[:, :, t, :] # shape (B,P,1,D)
    (inf_pred, pred_P, pred_N_P), (K, V) = model.cell.inference_function(inp_t, K, V, num_samples, t)
    list_inf_pred.append(inf_pred)  # (B,1,F=1)
    list_pred_P.append(pred_P)  # (B,P,1,F=1)
    list_pred_N_P.append(pred_N_P)  # (B,N*P,1,F=1)

  # seq_inf_pred = tf.stack(list_inf_pred, axis=1)
  # seq_pred_P = tf.stack(list_pred_P, axis=2)
  # seq_pred_N_P = tf.stack(list_pred_N_P, axis=2)

  return (pred, attn_weights), (list_inf_pred, list_pred_P, list_pred_N_P), inp_to_infer