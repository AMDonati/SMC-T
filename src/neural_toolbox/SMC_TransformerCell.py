import tensorflow as tf
import collections
#import tensorflow_probability as tfp

# additional imports
from models.SMC_Transformer.self_attention_SMC import MultiHeadAttention_SMC
from neural_toolbox.classic_layers import point_wise_feed_forward_network

from models.SMC_Transformer.transformer_utils import positional_encoding_SMC
from models.SMC_Transformer.transformer_utils import positional_encoding

from models.SMC_Transformer.transformer_utils import resample
from models.SMC_Transformer.transformer_utils import resample_z
from models.SMC_Transformer.transformer_utils import sample_and_keep_indices

NestedInput = collections.namedtuple('NestedInput', ['r', 'x'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'w', 'I'])


class SMC_Transf_Cell(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, dff, target_vocab_size,
              num_particles, seq_len,
              num_layers, sigma, noise, task_type, maximum_position_encoding=None, training=True, resampling=True,
               rate=0.1, **kwargs):
    #TODO: remove default Value for maximum_position_encoding, task_type (not essential).
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
    self.dec_timestep = 0
    self.mha_smc = MultiHeadAttention_SMC(d_model=d_model,
                                          num_heads=num_heads,
                                          num_particles=num_particles,
                                          dec_timestep=self.dec_timestep,
                                          sigma=sigma,
                                          noise=noise)
    self.ffn = point_wise_feed_forward_network(d_model, dff)
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

    self.num_heads = num_heads
    self.dff = dff
    self.num_particles = num_particles
    self.d_model = d_model
    self.target_vocab_size = target_vocab_size
    self.maximum_position_encoding = maximum_position_encoding
    self.seq_len = seq_len

    self.rate = rate

    self.training = training
    self.resampling = resampling

    self.layer_num = num_layers
    #self.decoder = decoder
    self.task_type=task_type

    # output layer for computing the weights
    self.output_layer = tf.keras.layers.Dense(target_vocab_size)

    #------------- state_size and output_size of the SMC Cell (without the batch dimension)-----------------------------------------

    # internal states: K,V,w,I.
    self.state_size = NestedState(K=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                  V=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                  w=tf.TensorShape([self.num_particles, 1]),
                                  I=tf.TensorShape([self.num_particles, self.seq_len]))

    # outputs: z, r, epsilon and attention_weights (output of the last SMC layer) before softmax.
    self.output_size = (tf.TensorShape([self.num_particles, 1, self.d_model]), # r^l
                        tf.TensorShape([self.num_particles, 1, self.d_model]), # z
                        tf.TensorShape([ 1, self.target_vocab_size]), # average_prediction
                        tf.TensorShape([1, self.target_vocab_size]), # max_prediction
                        tf.TensorShape([self.num_particles, 1, self.d_model]), # epsilon
                        tf.TensorShape([self.num_particles, 1, self.seq_len])) # attention_weights

    self.embedding = tf.keras.layers.Embedding(self.target_vocab_size, self.d_model)
    if self.maximum_position_encoding is not None:
      self.pos_encoding = positional_encoding(self.maximum_position_encoding, self.d_model)
      self.pos_encoding_SMC = positional_encoding_SMC(self.maximum_position_encoding, self.d_model, self.num_particles)
    self.dropout = tf.keras.layers.Dropout(self.rate)

    super(SMC_Transf_Cell, self).__init__(**kwargs)


  def call(self, inputs, states):
    # x -> r
    # w -> prev_sampling_weights
    # y -> target_word_id
    # put training and re-sampling as init parameters.
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
      - states for the last time-step: tuple (K,V,w,I)
    '''

    print('cell timestep', self.dec_timestep)

    # unnesting inputs
    r, x = tf.nest.flatten(inputs)  # r output prev trqnsformer, y: label/target
    x = tf.cast(x, dtype=tf.int32)

    # getting x
    K, V, w, I = states
    I = tf.cast(I, dtype=tf.int32)

    batch_size = tf.shape(K)[0]

    # resampling of (K,V) to compute the new set of (z,K,V)
    if self.resampling:
      K = resample(K, I)
      V = resample(V, I)

    # multi-head attention:
    input_mha = tf.expand_dims(r, axis=2)  # shape (B,P,1,D)
    inputs_mha = [input_mha for _ in
                  range(3)]  # trick to have an 'inputs' in the function call of the class MultiHeadAttention
    if self.dec_timestep < self.seq_len:
      # store (K,V) only in that case.
      (z, K, V), attn_weights = self.mha_smc(inputs=inputs_mha, timestep=self.dec_timestep, K=K, V=V)
    else:
      # otherwise K,V is not updated.
      (z, KK, VV), attn_weights = self.mha_smc(inputs=inputs_mha, timestep=self.dec_timestep, K=K, V=V)

    #TODO: demander à Florian s'il faut changer l'ordre des layernorm/FFN.
    # computing r from z:
    z = self.dropout1(z, training=self.training)
    r = tf.expand_dims(r, axis=2)
    out1 = self.layernorm1(z + r)

    ffn_output = self.ffn(out1)  # (batch_size, NUM_PARTICLES, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=self.training)
    out3 = self.layernorm3(ffn_output + out1)  # (batch_size, NUM_PARTICLES, target_seq_len, d_model)

    # 3. FOR SMC: compute the new set of weights.
    if len(tf.shape(x)) == 1:
      x = tf.expand_dims(x, axis=-1)  # shape (B,1)
    predictions = self.output_layer(out3)  # (B,P,1,V)

    # ----------- sampling_weights computation > for classification case or regression case... ----------------------------------------------------------------

    def compute_w_classification(predictions, x):
      # right now, the predictions corresponds to the logits. Adding a softmax layer to have the normalized log probas:
      log_probas=tf.nn.softmax(predictions, axis=-1) # shape (B,P,S,V)
      w = tf.gather(log_probas, x, axis=-1, batch_dims=1)
      w = tf.squeeze(w, axis=-1)  # shape (B,P,1)
      w_squeezed = tf.squeeze(w, axis=-1)  # shape (B,P)
      return w_squeezed  # shape (B,P)

    def compute_w_regression(predictions, x, omega=1):
      #TODO: replace a the tf.cast by an assert (input data should be of dtype=tf.float32 for the regression case).
      x=tf.cast(x, dtype=tf.float32)
      if len(tf.shape(predictions))==4:
        predictions=tf.squeeze(predictions, axis=-1) # shape (B,P,1)
      # expanding and tiling x over the particle dimensions to have the right shape
      x=tf.expand_dims(x, axis=1)
      x=tf.tile(x, multiples=[1, self.num_particles, 1]) # shape (B,P,1)
      mu_t = x - predictions
      w=tf.random.normal(shape=tf.shape(mu_t), mean=mu_t, stddev=omega)
      # normalization
      w=w/tf.reduce_sum(w, axis=1, keepdims=True)
      return w

    if self.task_type == 'classification':
      assert self.target_vocab_size > 1
      w_squeezed = compute_w_classification(predictions=predictions, x=x)
    elif self.task_type == 'regression':
      assert self.target_vocab_size == 1
      w_squeezed = compute_w_regression(predictions=predictions, x=x)

    # add a tf.stop_gradient on the weights to have backpropagation on these parameters:
    w_squeezed=tf.stop_gradient(w_squeezed)
    #TODO: add an assert that the sum over num of particles of w is equal to 1.

    # compute the average prediction & max_prediction for the set of particles from predictions & w
    predictions=tf.squeeze(predictions, axis=-2) # (B,P,V)
    w=w_squeezed
    average_prediction=tf.expand_dims(tf.reduce_sum(predictions*w, axis=1), axis=1) # (B,1,V)
    argmax_w=tf.argmax(w, axis=1)# (B, 1)
    max_prediction=tf.gather(predictions, argmax_w, axis=1, batch_dims=1) # (B,1,V)

    #-----------------end of weights computation--------------------------------------------------------------------

    # update the genealogy indices matrix from the weights.
    if self.dec_timestep < self.seq_len:
      # update it only until T-1
      i_t, I = sample_and_keep_indices(w_squeezed, I, self.num_particles, self.dec_timestep)

    # adding a tf.stop_gradient on I to avoid backpropagation on this set of parameters
    I=tf.stop_gradient(I)

    # resample z:
    if self.resampling:
      if self.dec_timestep < self.seq_len:
        z = resample_z(z, I, self.dec_timestep)  # if z is of shape (B,P,D).

    # get the output (r_t^l, z_t^l, epsilon_t^l, average prediction, prediction for largest w_t)
    epsilon=self.mha_smc.stddev # shape (B,P,1,D)

    output = [out3, z, average_prediction, max_prediction, epsilon, attn_weights] # attn_weights > shape (B,P,H,1,D)

    new_states = NestedState(K=K, V=V, w=w, I=I)

    self.dec_timestep += 1

    return output, new_states

if __name__ == "__main__":

  from models.SMC_Transformer.SMC_Transformer import SMC_Transformer

  batch_size = 8
  d_model = 64
  num_heads = 8
  dff = 32
  target_vocab_size = 20
  maximum_position_encoding = None
  num_particles = 5
  seq_len = 4
  layer_num = 2
  sigma=1
  noise=False
  data_type='time_series'
  task_type='classification'

  cell = SMC_Transf_Cell(d_model=d_model, num_heads=num_heads, dff=dff, target_vocab_size=target_vocab_size,
                         num_particles=num_particles,
                         seq_len=seq_len, num_layers=layer_num,
                         training=True,
                         resampling=True,
                         sigma=sigma,
                         noise=noise,
                         task_type=task_type)

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
  task_type=task_type)

  initial_word_tensor = tf.ones(shape=(batch_size, 1), dtype=tf.int32)

  (K0, V0), w0, I0 = sample_transformer.initialize_attn_SMC_parameters(batch_size,
                                                                       seq_len,
                                                                      initial_word_id=initial_word_tensor)

  initial_state = NestedState(K=K0, V=V0, w=w0, I=I0)


  def step_function(inputs, states):
    return cell(inputs, states)

  r = tf.random.uniform(
    shape=(batch_size, seq_len, num_particles, d_model))  # the dim 1 needs to be added. trick with nested inputs.
  x = tf.ones(shape=(batch_size, seq_len, 1)) # x needs to have at least of length of shape equal to 3.

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

  average_predictions = outputs[2] # (B,S,1,V)
  max_predictions = outputs[3] # (B,S,1,V)

  epsilon_seq=outputs[4]
  attn_w_seq=outputs[5]

  print('r_T', cell_output.shape)  # (B,P,1,D) # should be dff?
  print('z_T', last_z.shape)  # shape (B,P,1,D)
  print('r0_T', output_seq.shape)  # shape (B,S,P,1,D) > should be shape (B,S,P,D) instead
  print('z_0_T', z_seq.shape)  # shape (B,S,P,1,D)
  print ('sequence of average predictions', average_predictions.shape) # (B,S,1,V)
  print('sequence of max_predictions', max_predictions.shape) # (B,S,1,V)
  print('Epsilon_0_T', epsilon_seq.shape)  # shape (B,S,P,1,D)
  print('attention weights', attn_w_seq.shape) # shape (B,S,P,H,1,S)
  print('w_T', w.shape)  # shape (B,P,1)
  print('K', K.shape)  # shape (B,P,S,D)
  print('I', I.shape)  # shape (B,P,S)

  #-------checking the aspect of the resampling weights------------------------------------------------------
  for m in range(num_particles):
    print('w_{}'.format(m), w[0,m,:])


