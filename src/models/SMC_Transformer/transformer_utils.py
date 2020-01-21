import tensorflow as tf
import numpy as np


def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


def positional_encoding_SMC(position, d_model, num_particles):
  '''
  tiling operation added compared to the classical positional encoding.
  :param position:
  :param d_model:
  :param num_particles:
  :return:
  preprocessed word input of shape (B,P,S,D).
  '''
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]
  pos_encoding = pos_encoding[:, np.newaxis, :, :]

  pos_encoding = tf.tile(pos_encoding, [1, num_particles, 1, 1])

  return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq, num_particles):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  temp = seq[:, tf.newaxis, tf.newaxis, tf.newaxis, :]
  return tf.tile(temp, multiples=[1, num_particles, 1, 1, 1])  # (batch_size, num_particles 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_look_ahead_mask_v2(size):
  mask=1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return 1-mask


def create_masks(tar):
  # target is dim (B,P,S,D)
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.

  # TO CHANGE: mask need to be of shape (B,P,num_heads, S,S)
  look_ahead_mask = create_look_ahead_mask_v2(tf.shape(tar)[1]) #
  #dec_target_padding_mask = create_padding_mask(tar, num_particles) # NOT NEEDED???
  #combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return look_ahead_mask

def resample(params, indices):
  '''GOOD RESAMPLING FUNCTION!!!'''
  seq_length=tf.shape(params)[2]
  params_list=[]
  for t in range(seq_length):
    params_t=params[:,:,t,:]
    indices_t=indices[:,:,t]
    params_resampl_t=tf.gather(params_t, indices_t, batch_dims=1)
    params_resampl_t=tf.expand_dims(params_resampl_t, axis=2)
    params_list.append(params_resampl_t)
  params_resampl=tf.stack(params_list, axis=2)
  params_resampl=tf.squeeze(params_resampl, axis=-2)
  return params_resampl

def resample_z(z, indices, dec_timestep):
  curr_ind=indices[:,:,dec_timestep]
  z_resampl=tf.gather(z, curr_ind, batch_dims=1)
  return z_resampl

def sample_and_keep_indices(prev_sampling_weights, ind_matrix, num_particles, dec_timestep, indices=None):  # add a mask argument?
  '''samples the set of N indices for doing the weights resampling
  adds this set of indices to the matrix of ancestor indices
  Args:
  -prev_sampling_weights: w(k-1) > dim (B, P)
  -indice matrix: J0:k-1 > dim (B, P, S)
  Returns:
  -The current set of indices to do a forward pass on the Decoder Layer > dim (batch_size, num_particles)
  -The updated ancestor indices matrix > dim (batch_size, NUM_PARTICLES, seq_length)'''

  batch_size=tf.shape(ind_matrix)[0]

  # Sample current set of indices with proba proportional to prev_sampling_weights
  if indices is None:
    if len(tf.shape(prev_sampling_weights)) == 3:
      prev_sampling_weights = tf.squeeze(prev_sampling_weights, axis=-1)
    indices = tf.random.categorical(prev_sampling_weights, num_particles)  # shape (..., num_particles)
    indices=tf.expand_dims(indices, axis=-1)
    indices=tf.cast(indices, dtype=tf.int32)

  list_rows_J_t=[]
  # taking only ind_matrix until dec_timestep-1:
  J_prev_t=ind_matrix[:,:,:dec_timestep] # shape (B,P,dec_timestep-1)

  for m in range(num_particles):
    row_m_Jt=tf.gather(J_prev_t, indices[:,m], axis=1, batch_dims=1) # (B,dec_timestep-1)
    list_rows_J_t.append(row_m_Jt)
  J_t=tf.stack(list_rows_J_t, axis=1) # shape (B,P,1,dec_timestep-1)
  J_t=tf.squeeze(J_t, axis=2) # shape (B,P,dec_timestep-1)
  # adding the current set of indices as the last row.
  J_t=tf.concat([J_t, indices], axis=-1)  # shape (B,P,dec_timestep)

  # updating Jt in the total J of shape (B,P,S)
  ind_matrix=tf.concat([J_t, ind_matrix[:,:,dec_timestep+1:]], axis=-1)

  return indices, ind_matrix


  # # Add this set of indices to the indices matrix tensor:
  # indices = tf.cast(indices, tf.int32)
  # indices = tf.expand_dims(indices, axis=-1)
  # updated_ind_matrix = tf.concat(
  #   [ind_matrix[:, :, :dec_timestep], indices, ind_matrix[:, :, dec_timestep + 1:]], axis=-1)
  #
  # return indices, updated_ind_matrix

def initialize_indices_matrix(batch_size, seq_length, num_particles):
  # initialize it as the "identity transformation function"
  ind_matrix = tf.constant([l for l in range(num_particles)],
                           shape=(1, num_particles, 1), dtype=tf.int32,
                           name='indices_matrix')
  # tile to have the right shape
  ind_matrix = tf.tile(ind_matrix, [batch_size, 1, seq_length]) # shape (B,P,S)
  return ind_matrix  # tf.stop_gradient(ind_matrix)?


def compute_direct_update_cov_matrix(self):
    '''
    for each layer, update the sigma of the reparametrized noise with the optimal sigma of the current training step.
    '''
    # function that will be probably not needed.

    #TODO: change the implementation of the update operation following Florian's advice:
    #TODO https://stackoverflow.com/questions/48260394/whats-the-differences-between-tf-graphkeys-trainable-variables-and-tf-graphkeys

    list_stddev = self.decoder.list_stddev
    list_upd_sigmas = []
    # num_layers=len(list_stddev)
    seq_length = tf.shape(list_stddev[0])[2]
    for l, stddev in enumerate(list_stddev):
      sigma_optimal_l = []
      for t in range(seq_length):
        stddev_t = stddev[:, :, t, :]
        # permute P and D dims:
        stddev_t = tf.transpose(stddev_t, perm=[0, 2, 1])  # dim (B,D,P)
        # do double dot product to have on the particle and batch dims to have at the end a tensor of dim (D,D)
        sigma_optimal_l_t = tf.tensordot(stddev_t, tf.linalg.matrix_transpose(stddev_t),
                                         axes=[[0, 2], [0, 1]])  # dim (D,D)
        sigma_optimal_l.append(sigma_optimal_l_t)
      sigma_optimal_l = tf.stack(sigma_optimal_l, axis=0)  # dim (S,D,D)
      # sum over all the layers
      sigma_optimal_l = tf.reduce_sum(sigma_optimal_l, axis=0)  # dim (D,D)
      # multiply by 1/seq_len
      # sigma_optimal_l=tf.math.scalar_mul(1/seq_length, sigma_optimal_l) # to debug.
      # sigma_optimal_l=1/seq_length*sigma_optimal_l
      sigma_optimal_l = tf.cast(sigma_optimal_l, dtype=tf.float32)
      list_upd_sigmas.append(sigma_optimal_l)
      self.decoder.dec_layers[l].mha1.sigma = sigma_optimal_l  # to check with Florian and to change with new

    return list_upd_sigmas

# ### OLD VERSION NOT WORKING....
# def resample(ind_matrix, attn_parameter):
#   '''
#   -Args:
#     -ind_matrix: tensor storing the particle indices @ each timestep > shape (B,P,S)
#     -attn_parameter: K,V, or Z: shape (B,P,S,D)
#   -Return:
#     -the attn parameter resampled > shape (B,P,S,D)
#   '''
#   attn_all_p = []
#   for p in range(tf.shape(ind_matrix)[1]):
#     ind_matrix_p = ind_matrix[:, p, :]
#     attn_p = attn_parameter[:, p, :, :]
#     attn_p_resampl = tf.gather(attn_p, ind_matrix_p, axis=1, batch_dims=1)
#     attn_all_p.append(attn_p_resampl)
#   attn_resampl = tf.stack(attn_all_p, axis=1)
#
#   return attn_resampl
#
# def resample_v2(ind_matrix, attn_parameter):
#     '''
#     -Args:
#         -ind_matrix: tensor storing the particle indices @ each timestep > shape (B,P,S)
#         -attn_parameter: K,V, or Z: shape (B,P,S,D)
#     -Return:
#           -the attn parameter resampled > shape (B,P,S,D)
#     '''
#     attn_all_p = []
#     for p in range(tf.shape(ind_matrix)[1]):
#       ind_matrix_p = ind_matrix[:, p, :]
#       attn_p = attn_parameter[:, p, :, :]
#       attn_p_all_t = []
#       for t in range(tf.shape(ind_matrix)[2]):
#         ind_matrix_p_t = ind_matrix_p[:, t]  # (B,)
#         attn_p_t = attn_p[:, t, :]  # (B,D)
#         attn_p_t_resampl = tf.gather(attn_p_t, ind_matrix_p_t, axis=1)  # (B,D)
#         attn_p_all_t.append(attn_p_t_resampl)
#       # attn_p_resampl=tf.gather(attn_p, ind_matrix_p, axis=1, batch_dims=1)
#       attn_p_all_t = tf.stack(attn_p_all_t, axis=1)
#       attn_all_p.append(attn_p_all_t)
#     attn_resampl = tf.stack(attn_all_p, axis=1)
#     # attn_resampl=tf.concat([tf.expand_dims(attn_parameter[:,:,0,:], axis=2), attn_resampl], axis=2)
#     return attn_resampl
#
# def resample_v3(ind_matrix, attn_parameter):
#   '''
#     -Args:
#     -ind_matrix: tensor storing the particle indices @ each timestep > shape (B,P,S)
#     -attn_parameter: K,V, or Z: shape (B,P,S,D)
#     -Return:
#     -the attn parameter resampled > shape (B,P,S,D)
#     '''
#   attn_all_p = []
#   ind_matrix = ind_matrix[:, :, 1:]
#   for p in range(tf.shape(ind_matrix)[1]):
#     ind_matrix_p = ind_matrix[:, p, :]  # dim (B, S-1)
#     attn_p = attn_parameter[:, p, :, :]  # dim (B,P, S, D)
#     attn_p_resampl = tf.gather(attn_p, ind_matrix_p, axis=1, batch_dims=1)
#     attn_all_p.append(attn_p_resampl)
#   attn_resampl = tf.stack(attn_all_p, axis=1)  # dim (B, P, S-1, D)
#   attn_resampl = tf.concat([tf.expand_dims(attn_parameter[:, :, 0, :], axis=2), attn_resampl], axis=2)
#
#   return attn_resampl
#
# #def initialize_covariance_matrix(dim):
#   #spectrum=tf.Variable(tf.random.uniform(minval=0,shape=(int(dim/2), int(dim/2))))
#   #covariance_matrix=tf.linalg.LinearOperatorCirculant2D(spectrum, is_positive_definite=True, input_output_dtype=tf.float32)
#
#   #return covariance_matrix.to_dense()

if __name__ == "__main__":
  #mask=create_look_ahead_mask_v2(3)
  #print(mask)

  #Z=tf.random.uniform(shape=(8,10,8,3,6))
  #print('Z before masking', Z)
  #Z*=mask
  #print('Z after masking', Z)

  #---- resampling z test---------

  # z=tf.random.uniform(shape=(8,10,64))
  # indices=tf.ones(shape=(8,10,20), dtype=tf.int32)
  # z_resampl=resample_z(z, indices, 5)
  # print('z resampled', z_resampl.shape)

  #---- test of sample_and_keep_indices function-----
  B = 2
  P = 3
  S = 4
  t=2
  prev_sampling_weights = tf.random.uniform(shape=(B, P), maxval=1)
  ind_matrix_T = tf.constant([0 for _ in range(P)], shape=(1, P, 1), dtype=tf.int32)
  # S+1 is a trick to be able to update the last decoding timestep
  ind_matrix_T = tf.tile(ind_matrix_T, [B, 1, S+1])

  # ind_matrix=tf.constant([[1,2,3],[2,2,2]], shape=(1,P,2), dtype=tf.int32)
  # ind_matrix=tf.tile(ind_matrix, multiples=[B,1,1])
  # ind_matrix_right = tf.tile(ind_matrix_right, [B, 1, S-2])
  # ind_matrix_T=tf.concat([ind_matrix, ind_matrix_right],axis=-1)

  # FOR TESTING - to remove
  indices_t1 = tf.constant([2, 2, 2], shape=(1, P, 1), dtype=tf.int32)
  indices_t1 = tf.tile(indices_t1, multiples=[B, 1, 1])

  indices_t2 = tf.constant([1, 2, 2], shape=(1, P, 1), dtype=tf.int32)
  indices_t2= tf.tile(indices_t2, multiples=[B, 1, 1])

  indices_t3 = tf.constant([1, 1, 3], shape=(1, P, 1), dtype=tf.int32)
  indices_t3 = tf.tile(indices_t3, multiples=[B, 1, 1])

  curr_indices, matrix_updated_t1 = sample_and_keep_indices(prev_sampling_weights=prev_sampling_weights,
                                                            ind_matrix=ind_matrix_T,
                                                            num_particles=P,
                                                            dec_timestep=1,
                                                            indices=indices_t1)

  curr_indices, matrix_updated_t2=sample_and_keep_indices(prev_sampling_weights=prev_sampling_weights,
                                                  ind_matrix=matrix_updated_t1,
                                                  num_particles=P,
                                                  dec_timestep=2,
                                                  indices=indices_t2)

  #TODO: solve the bug happening at the last_timestep.
  # # does not work for the last time_step
  # curr_indices, matrix_updated_t3 = sample_and_keep_indices(prev_sampling_weights=prev_sampling_weights,
  #                                                           ind_matrix=matrix_updated_t2,
  #                                                           num_particles=P,
  #                                                           dec_timestep=3,
  #                                                           indices=indices_t3)


  print('indices matrix at time t1', matrix_updated_t1[0,:,:])
  print('indices matrix at time t2', matrix_updated_t2[0, :, :])
  #print('indices matrix at time t3', matrix_updated_t3[0, :, :])