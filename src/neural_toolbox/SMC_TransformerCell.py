import tensorflow as tf
import collections

# additional imports
from models.SMC_Transformer.self_attention_SMC import MultiHeadAttention_SMC
from neural_toolbox.classic_layers import point_wise_feed_forward_network

from models.SMC_Transformer.transformer_utils import positional_encoding_SMC
from models.SMC_Transformer.transformer_utils import positional_encoding
from models.SMC_Transformer.transformer_utils import resample
from models.SMC_Transformer.transformer_utils import resample_old
from models.SMC_Transformer.transformer_utils import resample_z
from models.SMC_Transformer.transformer_utils import sample_and_keep_indices

NestedInput = collections.namedtuple('NestedInput', ['r', 'x'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'w', 'I'])


class SMC_Transf_Cell(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, dff, target_vocab_size,
              num_particles, seq_len,
              num_layers, sigma, noise, task_type, rate, omega, target_feature, maximum_position_encoding=None, training=True, resampling=True, test=False,
      **kwargs):
    #TODO: remove default Value for omega and target feature.
    '''
    -Args:
      -d_model: model depth
      -num_heads: number of heads in the multi-head attention mechanism
      -dff: output dimension of the feed forward network
      -target_vocab_size (for computing the sampling weights)
      -maximum_position_encoding: number of positions for positional encodings. (None for time-series).
      -num_particles: number of simulated particles for the latent state space of the Transformer
      -layer_num: only used if resampling is done at layer-level (in the Decoder class)
      -seq_len: sequence length of the input data
      -rate: dropout rate for output layers
      -task_type: classification or regression (different weight computation)
    '''

    # store the decoding timestep
    self.dec_timestep = 0 # decoding timestep starts at 1 because we have the init step. Cell is called S times.
    self.mha_smc = MultiHeadAttention_SMC(d_model=d_model,
                                          num_heads=num_heads,
                                          num_particles=num_particles,
                                          dec_timestep=self.dec_timestep,
                                          sigma=sigma,
                                          noise=noise)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm1')
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='layer_norm2')
    self.dropout1 = tf.keras.layers.Dropout(rate, name='dropout1')
    self.dropout3 = tf.keras.layers.Dropout(rate, name='dropout2')

    self.num_heads = num_heads
    self.dff = dff
    self.num_particles = num_particles
    self.d_model = d_model
    self.target_vocab_size = target_vocab_size
    self.maximum_position_encoding = maximum_position_encoding
    self.seq_len = seq_len
    self.target_feature = target_feature
    # for multi-variate time-series case: select the target feature in the re-sampling weights computation.
    self.noise = noise
    self.omega = omega
    self.rate = rate

    self.training = training
    self.resampling = resampling

    self.layer_num = num_layers
    self.task_type=task_type

    # for unit tests of SMC_Transformer_cell & SMC_transformer
    self.test = test

    # output layer for computing the weights
    self.output_layer = tf.keras.layers.Dense(target_vocab_size, name='output_layer')

    #------------- state_size and output_size of the SMC Cell (without the batch dimension)-----------------------------------------

    # internal states: K,V,w,I.
    self.state_size = NestedState(K=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                  V=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                  w=tf.TensorShape([self.num_particles, 1]),
                                  I=tf.TensorShape([self.num_particles, self.seq_len]))

    # outputs: z, r, epsilon and attention_weights (output of the last SMC layer) before softmax.
    self.output_size = (tf.TensorShape([self.num_particles, 1, self.d_model]), # r^l
                        tf.TensorShape([self.num_particles, 1, self.d_model]), # z
                        tf.TensorShape([1, self.target_vocab_size]), #  inference prediction (after softmax).
                        tf.TensorShape([1, self.target_vocab_size]), # inference prediction (before softmax).
                        tf.TensorShape([1, self.target_vocab_size]), # max_prediction
                        tf.TensorShape([self.num_particles, 1, self.d_model]), # epsilon
                        tf.TensorShape([self.num_particles, 1, self.seq_len])) # attention_weights

    self.embedding = tf.keras.layers.Embedding(self.target_vocab_size, self.d_model, name='embedding_layer')
    if self.maximum_position_encoding is not None:
      self.pos_encoding = positional_encoding(self.maximum_position_encoding, self.d_model)
      self.pos_encoding_SMC = positional_encoding_SMC(self.maximum_position_encoding, self.d_model, self.num_particles)

    self.dropout = tf.keras.layers.Dropout(self.rate)

    super(SMC_Transf_Cell, self).__init__(**kwargs)

  def compute_w_classification(self, predictions, y):
    '''
    :param predictions: output of final layer (logits.)
    :param x: current sequence element (x_t) >
    :return:
    '''
    log_probas = tf.nn.softmax(predictions, axis=-1)  # shape (B,P,1,V)
    w = tf.gather(log_probas, y, axis=-1, batch_dims=1)
    w = tf.squeeze(w, axis=-1)  # shape (B,P,1)
    w_squeezed = tf.squeeze(w, axis=-1)  # shape (B,P)
    return w_squeezed  # shape (B,P)

  def compute_w_regression(self, predictions, y):
    '''
    # FORMULA
    # -0.5 * mu_t ^ T * mu_t / omega
    # w = exp(...)
    # logw = -0.5 * mu_t ^ T * mu_t / omega
    # logw = logw - min(logw)
    # w = exp(logw)
    :param predictions: output of final layer (logits.) > shape (B,P,F) or (B,P,1,F) (regression case)
    :param y: current sequence element (x_t) > shape (B,F,1); F > 1 for multivariate case.
    :return:
    '''
    # TODO: replace a the tf.cast by an assert (input data should be of dtype=tf.float32 for the regression case).
    y = tf.cast(y, dtype=tf.float32)  # y of shape (B,F) for classif case / shape (B,F) for time_series case.
    if len(tf.shape(predictions)) == 4:
      predictions = tf.squeeze(predictions, axis=-2)  # shape (B,P,F)
    # expanding and tiling x over the particle dimensions to have the right shape
    y = tf.expand_dims(y, axis=1) # (B,1,F,1)
    y = tf.tile(y, multiples=[1, self.num_particles, 1])

    # multivariate case: selecting the target feature as the ground truth (labels)
    if self.target_feature is not None:
      assert self.target_feature < tf.shape(y)[-1]
      y = y[:, :, self.target_feature] # (B, P)
      y = tf.expand_dims(y, axis=-1) # (B, P, 1)

    mu_t = y - predictions # (B,P,F)
    log_w = tf.matmul(mu_t, mu_t, transpose_b=True)  # (B,P,P)
    log_w = tf.scalar_mul(-1 / 2 * self.omega, log_w) #TODO: here, add the omega learned option.
    log_w = tf.linalg.diag_part(log_w)  # take the diagonal.
    log_w_min = tf.reduce_min(log_w, axis=-1, keepdims=True)
    log_w = log_w - log_w_min
    w = tf.math.exp(log_w)
    # normalization
    w = w / tf.reduce_sum(w, axis=1, keepdims=True)
    return w

  def inference_function(self, inputs, K, V, num_samples, t, inf_timestep):
    '''
    :param inputs preprocessed: input data > shape (B,P,1,D) of t=0 of (B,P*N,1,D) if t > 0
    :param K: key matrix in self_attention > shape (B,P,S,D)
    :param V: value matrix in self_attention > shape (B,P,S,D)
    :param num_samples:
    :param t: decoding timestep for inference from the beginning of the sequence
    :param inf_timstep: decoding timestep since the beginning of the inference
    :return:
    '''
    sampled_z, sampled_K, sampled_V = [], [], []
    N = num_samples

    inputs_mha = [inputs for _ in range(3)]  # trick to have an 'inputs' in the function call of the class MultiHeadAttention
    if inf_timestep == 0:
      for n in range(N):
        (z, K, V), _ = self.mha_smc(inputs=inputs_mha, timestep=t, K=K, V=V)
        sampled_z.append(z) # (B,P,1,D)
        sampled_K.append(K) # (B,P,S,D)
        sampled_V.append(V) # (B,P,S,D)

      sampled_z = tf.stack(sampled_z, axis=1) # (B,N,P,1,D)
      sampled_K = tf.stack(sampled_K, axis=1) # (B,N,P,S,D)
      sampled_V = tf.stack(sampled_V, axis=1) # (B,N,P,S,D)

      B, N, P, S, D = tf.shape(sampled_K)[0], tf.shape(sampled_K)[1], tf.shape(sampled_K)[2], tf.shape(sampled_K)[3], tf.shape(sampled_K)[4]

      shape_z = (B, N*P, -1, 1, D)
      shape_K = (B, N*P, -1, S, D)

      sampled_z = tf.reshape(sampled_z, shape=shape_z) # shape (B,N*P,1,1,D)
      sampled_z = tf.squeeze(sampled_z, axis=2) # shape (B,N*P,1,D) # $z^{m,i}$
      sampled_K = tf.reshape(sampled_K, shape=shape_K)
      sampled_K = tf.squeeze(sampled_K, axis=2)
      sampled_V = tf.reshape(sampled_V, shape=shape_K)
      sampled_V = tf.squeeze(sampled_V, axis=2)

    else:
      (sampled_z, sampled_K, sampled_V), _ = self.mha_smc(inputs=inputs_mha, timestep=t, K=K, V=V)

    # pass forward until output layer for sampled_z
    z = self.dropout1(sampled_z, training=False)
    if inf_timestep == 0:
      inputs_N = tf.tile(inputs, multiples=[1,num_samples,1,1]) # (B,N*P,1,D)
    else:
      inputs_N = inputs
    out1 = self.layernorm1(z + inputs_N) # (B, N*P, 1, D)
    ffn_output = self.ffn(out1)  # (B, N*P, 1, D)
    ffn_output = self.dropout3(ffn_output, training=False) # (B, N, P, 1, D)
    r_t_N_P = self.layernorm3(ffn_output + out1)  # (B, N*P, 1, D) # $r^{m,i}$
    mean_pred_N_P = self.output_layer(r_t_N_P)
    X_pred_N_P = mean_pred_N_P + tf.random.normal(shape=tf.shape(mean_pred_N_P), stddev=self.omega)# (B,NP,1,F)

    return X_pred_N_P, r_t_N_P, (sampled_K, sampled_V)

  def call(self, inputs, states):
    '''
    -args:
      - inputs: (r^{l-1}[0:T] > shape (B, P, S, D) , (X tiled. (X[0:T] input of the while model) > shape (B,P,S).
      - states: tuple (K,V,w,I)
        -K: attention vector > shape (B,P,S,D)
        -V : attention vector > shape (B,P,S,D)
        -w: re-sampling weights > shape (B,P,1)
        -I: indices matrix > shape (B,P,S)
    -returns:
       - outputs: tuple (r^l,z^l)
          - cell output for the current word > shape (B,P,1,D)
          - z: attention vector for the current word > shape (B,P,1,D)
          - predictions to compute training metrics (classification case).
          - epsilon: value of the reparametrized gaussian noise (for computation of the loss).
          - attention weights:
      - states for the last time-step: tuple (K,V,w,I)
    '''
    if self.test:
      print('decoding timestep', self.dec_timestep)

    # unnesting inputs
    x, y = tf.nest.flatten(inputs)  # r output prev transformer, y: label/target
    y = tf.cast(y, dtype=tf.float32)
    y = tf.squeeze(y, axis=-1)
    # getting x
    K, V, w, I = states
    I = tf.cast(I, dtype=tf.int32)

    if self.test:
      print('x', x[:,:,0])
      print('y', y)
      print('K before propagation', K[:,:,:,0])
    # resampling of (K,V) to compute the new set of (z,K,V) - what was done before (resampling before propagation.)
    #if self.resampling:
      #K = resample_old(K, I)
      #V = resample_old(V, I)

    # multi-head attention:
    input_mha = tf.expand_dims(x, axis=2)  # shape (B,P,1,D)
    inputs_mha = [input_mha for _ in range(3)]  # trick to have an 'inputs' in the function call of the class MultiHeadAttention
    if self.dec_timestep < self.seq_len:
      # store (K,V) only in that case.
      (z, K, V), attn_weights = self.mha_smc(inputs=inputs_mha, timestep=self.dec_timestep, K=K, V=V)
    else:
      # otherwise K,V is not updated.
      (z, KK, VV), attn_weights = self.mha_smc(inputs=inputs_mha, timestep=self.dec_timestep, K=K, V=V)

    if self.test:
      print('K after propagation', K[:,:,:,0])
      print('z', z[:,:,:,0])

    # TODO: demander à Florian s'il faut changer l'ordre des layernorm/FFN.
    # computing r from z:
    z = self.dropout1(z, training=self.training)
    x = tf.expand_dims(x, axis=2)
    # out1 = self.layernorm1(z + x) # r corresponds here to r^(l-1) (input of the cell).
    # ffn_output = self.ffn(out1)  # (B, P, 1, D)
    ffn_output = self.ffn(z) #TODO: remove this line to come back to normal.
    #ffn_output = self.dropout3(ffn_output, training=self.training)
    r_ = self.dropout3(ffn_output, training=self.training) #TODO: comment this line to come back to normal with layernorm.
    #r_ = self.layernorm3(ffn_output + out1)  # (B, P, 1, D) # r_ corresponds to r^l.
    # 3. FOR SMC: compute the new set of weights.
    predictions = self.output_layer(r_)  # (B,P,1,V)

    if self.test:
      print('predictions', predictions)

    # ----------- sampling_weights computation > for classification case or regression case... ----------------------------------------------------------------

    if self.task_type == 'classification':
      assert self.target_vocab_size > 1
      w_squeezed = self.compute_w_classification(predictions=predictions, x=y)
    elif self.task_type == 'regression':
      w_squeezed = self.compute_w_regression(predictions=predictions, y=y)

    # add a tf.stop_gradient on the weights to have backpropagation on these parameters:
    w_squeezed=tf.stop_gradient(w_squeezed)
    #TODO: add an assert that the sum over num of particles of w is equal to 1.
    predictions = tf.squeeze(predictions, axis=-2)  # (B,P,V)

    # compute the average prediction & max_prediction for the set of particles from predictions & w
    if len(tf.shape(w_squeezed)) == 2:
      w_for_pred = tf.expand_dims(w_squeezed, axis=-1)
    else:
      w_for_pred = w_squeezed
    good_avg_pred = tf.expand_dims(tf.reduce_mean(predictions, axis=1), axis=1)  # weights=1/M because of the resampling happening at the beginning of the cell.

    # predictions after softmax: inference formula for N=1
    log_probas = tf.nn.softmax(predictions, axis=-1)  # shape (B,P,1,V)
    avg_pred_after_softmax = tf.expand_dims(tf.reduce_mean(log_probas, axis=1), axis=1)  # shape (B,1,V)

    # max prediction:
    argmax_w = tf.argmax(w_for_pred, axis=1)  # (B, 1)
    max_prediction = tf.gather(predictions, argmax_w, axis=1, batch_dims=1)  # (B,1,V)

    #-----------------end of weights computation--------------------------------------------------------------------

    # update the genealogy indices matrix from the weights.
    # TODO: remove this function & consider only the current indice i_t.
    i_t, I = sample_and_keep_indices(w_squeezed, I, self.num_particles, self.dec_timestep)

    # adding a tf.stop_gradient on I to avoid backpropagation on this set of parameters
    I = tf.stop_gradient(I)

    # resample K, V, and z:
    if self.resampling:
      K = resample(params=K, i_t=tf.squeeze(i_t, axis=-1), t=self.dec_timestep)
      V = resample(params=K, i_t=tf.squeeze(i_t, axis=-1), t=self.dec_timestep)
      z = resample_z(z, I, self.dec_timestep)  # if z is of shape (B,P,D).

    if self.test:
      print('current indices', i_t)
      print('K resampled', K[:,:,:,0])
      print('z resampled', z[:,:,:,0])

    # get the normalized mean of z,k,q,v for the SMC loss.
    noise_z = self.mha_smc.noise_z_norm # shape (B,P,1,D)
    noise_k = self.mha_smc.noise_k_norm
    noise_v = self.mha_smc.noise_v_norm
    noise_q = self.mha_smc.noise_q_norm

    list_noise = [noise_z, noise_k, noise_v, noise_q]

    output = [r_, z, avg_pred_after_softmax, good_avg_pred, max_prediction, list_noise, attn_weights] # attn_weights > shape (B,P,H,1,D)

    if len(tf.shape(w_squeezed)) == 2:
      w = tf.expand_dims(w_squeezed, axis=-1)
    elif len(tf.shape(w_squeezed)) == 3:
      w = w_squeezed
    else:
      raise ValueError("w should be of shape (B,P) or shape (B,P,1)")

    new_states = NestedState(K=K, V=V, w=w, I=I)

    self.dec_timestep += 1

    if self.test:
      print('end of decoding timestep')
      print('<---------------------------------------------------------------------------------------------------------------------------------------------->')

    return output, new_states

if __name__ == "__main__":

  from models.SMC_Transformer.SMC_Transformer import SMC_Transformer

  batch_size = 8
  d_model = 12
  num_heads = 1
  dff = 32
  target_vocab_size = 1
  maximum_position_encoding = None
  num_particles = 5
  seq_len = 4
  layer_num = 1
  sigma = 1
  noise = False
  data_type = 'time_series_multi'
  task_type = 'regression'
  rate = 0.1

  cell = SMC_Transf_Cell(d_model=d_model, num_heads=num_heads, dff=dff, target_vocab_size=target_vocab_size,
                         num_particles=num_particles,
                         seq_len=seq_len, num_layers=layer_num,
                         training=True,
                         resampling=True,
                         sigma=sigma,
                         noise=noise,
                         task_type=task_type,
                         rate = rate)

  sample_transformer = SMC_Transformer(
    num_layers=layer_num, d_model=d_model, num_heads=num_heads,
    dff=dff, target_vocab_size=target_vocab_size,
    maximum_position_encoding=maximum_position_encoding,
    num_particles=num_particles,
  seq_len=seq_len,
  sigma=sigma,
  noise_encoder=noise,
  noise_SMC_layer=noise,
  data_type=data_type,
  task_type=task_type,
  target_feature=0,
  rate = rate)

  initial_word_tensor = tf.ones(shape=(batch_size, 1), dtype=tf.int32) # (B,1)

  (K0, V0), w0, I0 = sample_transformer.initialize_attn_SMC_parameters(batch_size,
                                                                       seq_len,
                                                                      initial_word_id=initial_word_tensor)

  initial_state = NestedState(K=K0, V=V0, w=w0, I=I0)


  def step_function(inputs, states):
    return cell(inputs, states)

  r = tf.random.uniform(
    shape=(batch_size, seq_len, num_particles, d_model))  # the dim 1 needs to be added. trick with nested inputs.

  F = 14
  x = tf.ones(shape=(batch_size, num_particles, seq_len, F)) # x needs to have at least of length of shape equal to 3.

  inputs = NestedInput(r=r, x=x)

  last_output, outputs, new_states = tf.keras.backend.rnn(step_function=step_function,
                                                          inputs=inputs,
                                                          initial_states=initial_state)

  cell_output = last_output[0]
  last_z = last_output[1]

  K = new_states[0]
  V = new_states[1]
  w = new_states[2]
  I = new_states[3]

  output_seq = outputs[0]
  z_seq = outputs[1]

  avg_pred_after_softmax = outputs[2] # (B,S,1,V)
  good_avg_pred = outputs[3] # (B,S,1,V))
  max_predictions = outputs[4] # (B,S,1,V)

  epsilon_seq = outputs[5]
  attn_w_seq = outputs[6]

  print('r_T', cell_output.shape)  # (B,P,1,D) # should be dff?
  print('z_T', last_z.shape)  # shape (B,P,1,D)
  print('r0_T', output_seq.shape)  # shape (B,S,P,1,D) > should be shape (B,S,P,D) instead
  print('z_0_T', z_seq.shape)  # shape (B,S,P,1,D)
  print('sequence of average predictions after softmax', avg_pred_after_softmax.shape)  # (B,S,1,V)
  print('sequence of good avg pred from logits', good_avg_pred.shape)  # (B,S,1,V)
  print('sequence of max_predictions', max_predictions.shape) # (B,S,1,V)
  print('Epsilon_0_T', epsilon_seq.shape)  # shape (B,S,P,1,D)
  print('attention weights', attn_w_seq.shape) # shape (B,S,P,H,1,S)
  print('w_T', w.shape)  # shape (B,P,1)
  print('K', K.shape)  # shape (B,P,S,D)
  print('I', I.shape)  # shape (B,P,S)

  #------- checking the aspect of the resampling weights ------------------------------------------------------
  for b in range(batch_size):
    print('w_{}'.format(b), w[b, :, :])

  # ------ draft of the code for inference function --------------------------------------------------

  #TODO: put this as an (external) function of call if possible... or at least an internal one.
  #TODO: needs as input: (Kt-1), V(t-1), r^(t-1)(l-1) (X(t-1))
  # def inference_pred(N):
  # TODO: add N as a parameter for the SMC Cell. requires (r_(t-1)^(l-1), K(t-1), V(t-1), w(t-1)
  # sampled_z = []
  # for n in range(N):
  # (z, K, V), attn_weights = self.mha_smc(inputs=inputs_mha, timestep=self.dec_timestep, K=K, V=V)
  # sampled_z.append(z)
  # sampled_z=tf.stack(z, axis=1) > (B,N,P,1,D)
  # PASS FORWARD UNTIL OUTPUT LAYER:
  # z = self.dropout1(sampled_z, training=self.training)
  # r = tf.expand_dims(r, axis=2)
  # out1 = self.layernorm1(z + r)
  # ffn_output = self.ffn(out1)  # (batch_size, NUM_PARTICLES, target_seq_len, d_model)
  # ffn_output = self.dropout3(ffn_output, training=self.training)
  # sampled_out3 = self.layernorm3(ffn_output + out1)  # (batch_size, N, P, 1, D)
  # sampled_pred=self.output_layer(sampled_out3) # (B,N,P,1,V) CAUTION logits before softmax!
  # inf_prediction=tf.scalar_mul(1/N, tf.reduce_sum(sampled_pred, axis=1)) # (B,P,1,V)
  # inf_prediction=tf.reduce_sum(w * inf_prediction, axis=1) # see if this needs reshaping. # (B,1,V)

  # return inf_prediction

  # if not self.training:
  # inf_pred=inference_pred(N)

  # print('r', r[:,:,0])
  # if self.dec_timestep == 1 or self.dec_timestep == self.seq_len - 1:
  # print('x', x[:,0])
  # print('K', K[:,:,:,0])
  # print('r', r)

  #print('K resampled', K[:,:,:,0])

  # print('r_ at decoding timestep {}: {}'.format(self.dec_timestep,r_))
  # print('z at decoding timestep {}: {}'.format(self.dec_timestep, z))

  # print('K propagated (after mha)', K[:,:,:,0])
  # print('z', z[:,:,:,0])

  # if self.dec_timestep == 1 or self.dec_timestep == self.seq_len - 1:
  # print('prediction at time {} : {}'.format(self.dec_timestep, predictions))


