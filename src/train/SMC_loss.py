import tensorflow as tf
import math

#TODO: ask Florian if I need to add a @tf.function to this function. cf https://www.tensorflow.org/tutorials/generative/cvae as an example.
def compute_SMC_log_likelihood(real, sampling_weights, list_stddev, list_sigmas):
#TODO: adapt this function to compute the 'scaled' mse loss.
  '''compute the SMC log likelihood: (write equations)
    -Args:
      -real: targets > tensor of shape (B,P,S) of dtype=tf.int32
      -sampling_weights: tensor of shape (B,P) > final resampling_weights for the last decoding timestep
      -list_stddev: list of stddev > shape (B,P,S,D)
      -list_sigmas:  list of covariance matrix shape (D,D)
  '''
  # get the reparametrised gaussian noise for each layer for the decoder
  # get the list of layers
  loss_by_layer = []
  list_sigma = []
  for stddev, sigma in zip(list_stddev, list_sigmas):
    # loop over the word sequence
    seq_length = tf.shape(real)[-1]
    loss_by_layer_timestep = []

    for t in range(seq_length):
      # inverse the covariance matrix
      sigma_inv = tf.linalg.inv(sigma)  # dim  (D,D)

      # take the stddev @ timestep t
      stddev_t = stddev[:, :, t, :]  # dim (B,P,D)

      # inv(covariance matrix) * stddev
      temp = tf.tensordot(sigma_inv, stddev_t, axes=[0, 2]) # dim (D,B,P)
      # reshape temp (D,B,P) to (B, D, P)
      temp = tf.transpose(temp, perm=[1, 0, 2])
      # idem for stddev_t (B,P,D) to (B,D,P)
      stddev_t = tf.transpose(stddev_t, perm=[0, 2, 1])

      # transpose(stddev)*(inv(covariance matrix) * stddev))
      #SMC_loss_element_2 = tf.linalg.matmul(stddev_t, temp, transpose_a=True)  # shape (B,P,P)

      SMC_loss_element_2=tf.reduce_sum(tf.multiply(stddev_t,temp), axis=1)

      # or tf.tensordot(tf.linalg.matrix_transpose(stddev_t),temp, axis=[2,1])
      #SMC_loss_element_2 = tf.reduce_mean(SMC_loss_element_2, axis=-1)  # trick to have the right shape (B,P)
      # do a loop over particles instead?

      # store the loss for each timestep
      loss_by_layer_timestep.append(SMC_loss_element_2)

    # store the list of timestep partial losses for each timestep
    loss_by_layer.append(loss_by_layer_timestep)
    list_sigma.append(sigma)

  # sum over the timestep: sum(0:j for j=1...seq_length)
  sum_losses = []
  #if len(tf.shape(real)) == 2:
    #real = tf.tile(tf.expand_dims(real, axis=-1), multiples=[1, 1, num_particles])
  #real = tf.transpose(real, perm=[0, 2, 1])
  for layer_loss in loss_by_layer:
    # stack all loss_by_layer_timestep
    temp_layer = tf.stack(layer_loss, axis=-1)  # dim (B,P,S) >> TO CHECK

    # sum(real*log) over all timesteps
    #real = tf.cast(real, dtype=temp_layer.dtype)
    sum_loss = tf.reduce_sum(temp_layer, axis=-1)
    # store the loss for each layer
    sum_losses.append(sum_loss)

  # sum the loss by layer over the # of layers.
  SMC_loss = tf.reduce_sum(tf.stack(sum_losses, axis=0), axis=0)


  # take the average of the sigma per layer for the 'global' sigma
  sigma_all_layers = tf.stack(list_sigma, axis=0)  # add a tf.stop_gradient on it?

  # add the log det (2pi*sigma)
  SMC_loss+=tf.reduce_sum(tf.linalg.logdet(2*math.pi*sigma_all_layers), axis=0) # 2math.pi can be removed...
  #SMC_loss += tf.math.log(tf.math.scalar_mul(2 * math.pi, tf.linalg.det(sigma_all_layers)))  # dim (B,P) # use tf.linalg.detlog instead

  # multiply the loss by -1/2
  SMC_loss=tf.math.scalar_mul(-1/2, SMC_loss)

  # weighted average over the number of particles
  if len(tf.shape(sampling_weights)) == 3:
    sampling_weights = tf.squeeze(sampling_weights, axis=-1)
  SMC_loss = tf.reduce_sum(sampling_weights * SMC_loss, axis=-1)  # dim (B,)

  # mean over the batch
  SMC_loss = tf.reduce_mean(SMC_loss, axis=0)

  return SMC_loss


if __name__ == "__main__":
  B=8
  P=5
  S=3
  D=12
  sigma_scal=10
  num_layers=2

  real=tf.ones(shape=(B,P,S), dtype=tf.int32)
  sampling_weights=tf.random.uniform(maxval=1,shape=(B,P))
  stddev = tf.random.normal(shape=(B,P,S,D))

  sigma_const=tf.constant(sigma_scal, shape=(D,), dtype=tf.float32)
  sigma = tf.linalg.diag(sigma_const)

  list_stddev=[tf.random.normal(shape=(B,P,S,D)) for _ in range(num_layers)]
  list_sigmas=[sigma for _ in range(num_layers)]

  loss=compute_SMC_log_likelihood(real, sampling_weights, list_stddev, list_sigmas)


# def compute_direct_update_cov_matrix(list_stddev, list_sigmas):
#     '''
#     to complete
#     :return:
#     '''
#     #list_stddev = self.decoder.list_stddev
#     seq_length=tf.shape(list_stddev[0])[2]
#     for l, stddev in enumerate(list_stddev):
#       sigma_optimal_l=[]
#       for t in range(seq_length):
#         stddev_t=stddev[:,:,t,:]
#         # permute P and D dims:
#         stddev_t=tf.transpose(stddev_t, perm=[0,2,1]) # dim (B,D,P)
#         sigma_optimal_l_t=tf.tensordot(stddev_t, tf.linalg.matrix_transpose(stddev_t), axes=[[0,2],[0,1]]) # double dot product
#         sigma_optimal_l.append(sigma_optimal_l_t)
#       sigma_optimal_l=tf.stack(sigma_optimal_l, axis=0) # dim (S,D,D)
#       sigma_optimal_l=tf.reduce_sum(sigma_optimal_l, axis=0) # dim (D,D)
#       list_sigmas[l].assign(sigma_optimal_l)
#       #self.decoder.dec_layers[l].mha1.sigma.assign(sigma_optimal_l)

# do a unit test

# def compute_SMC_loss():
#  '''compute the SMC loss > THIS IS THE OLD FALSE VERSION.
#     the good version is in the SMC_loss.py file
#       -Args:
#         -real: targets > tensor of shape (B,P,S)
#         sampling_weights: tensor of shape (B,P) > final resampling_weights for the last decoding timestep
#     '''
#     # get the reparametrised gaussian noise for each layer for the decoder
#     list_stddev = self.decoder.list_stddev
#     # get the list of layers
#     list_layers = self.decoder.dec_layers
#     loss_by_layer = []
#     list_sigma = []
#     for stddev, layer in zip(list_stddev, list_layers):
#       # loop over the word sequence
#       seq_length = tf.shape(real)[-1]
#       loss_by_layer_timestep = []
#       for t in range(seq_length):
#         sigma_inv = tf.linalg.inv(
#           layer.mha1.sigma)  # dim  (D,D) WARNING: TAKE AN INVERSIBLE PARAMETER (SYMETRIC MATRIX)
#         stddev_t = stddev[:, :, t, :]  # dim (B,P,D)
#         temp = tf.tensordot(sigma_inv, stddev_t, axes=[0, 2])
#         # reshape temp (D,B,P) to (B, D, P)
#         temp = tf.transpose(temp, perm=[1, 0, 2])
#         # idem for stddev_t (B,P,D) to (B,D,P)
#         stddev_t = tf.transpose(stddev_t, perm=[0, 2, 1])
#         SMC_loss_element_2 = tf.linalg.matmul(stddev_t, temp, transpose_a=True)  # shape (B,P,P)
#         SMC_loss_element_2 = tf.reduce_mean(SMC_loss_element_2, axis=-1)  # trick to have the right shape (B,P)
#         # do a loop over particles instead?
#         loss_by_layer_timestep.append(SMC_loss_element_2)
#
#       loss_by_layer.append(loss_by_layer_timestep)
#       list_sigma.append(layer.mha1.sigma)
#
#     # sum over the timestep: sum(0:j for j=1...seq_length)
#     sum_losses = []
#     if len(tf.shape(real)) == 2:
#       real = tf.tile(tf.expand_dims(real, axis=-1), multiples=[1, 1, self.num_particles])
#     real = tf.transpose(real, perm=[0, 2, 1])
#     for layer_loss in loss_by_layer:
#       # stack all loss_by_layer_timestep
#       temp_layer = tf.stack(layer_loss, axis=-1)  # dim (B,P,S) >> TO CHECK
#
#       # sum(real*log) over all timesteps
#       real = tf.cast(real, dtype=temp_layer.dtype)
#       sum_loss = tf.reduce_sum(real * temp_layer, axis=-1)
#       sum_losses.append(sum_loss)
#
#     # sum the loss by layer over the # of layers.
#     SMC_loss = tf.reduce_sum(tf.stack(sum_losses, axis=0), axis=0)
#     # add the log (2pi*sigma)
#     sigma_all_layers = tf.reduce_mean(tf.stack(list_sigma, axis=0), axis=0)  # add a tf.stop_gradient on it?
#     SMC_loss += tf.math.log(tf.math.scalar_mul(2 * math.pi, tf.linalg.det(sigma_all_layers)))  # dim (B,P)
#
#     # weighted average over the number of particles
#     if len(tf.shape(sampling_weights)) == 3:
#       sampling_weights = tf.squeeze(sampling_weights, axis=-1)
#     SMC_loss = tf.reduce_sum(sampling_weights * SMC_loss, axis=-1)  # dim (B,)
#
#     # mean over the batch
#     SMC_loss = tf.reduce_mean(SMC_loss, axis=0)
#     return SMC_loss
