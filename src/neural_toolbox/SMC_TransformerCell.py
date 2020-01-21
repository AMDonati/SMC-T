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
              num_layers, sigma, maximum_position_encoding=None, training=True, resampling=True,
               rate=0.1, task_type='classification', **kwargs):
    #TODO: see how to change the training parameter during inference (if needed.)
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
    self.mha1 = MultiHeadAttention_SMC(d_model=d_model,
                                       num_heads=num_heads,
                                       num_particles=num_particles,
                                       dec_timestep=self.dec_timestep,
                                       sigma=sigma)
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

    # internal states: K,V,w,I.
    # add the batch_size in the shapes?
    self.state_size = NestedState(K=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                  V=tf.TensorShape([self.num_particles, self.seq_len, self.d_model]),
                                  w=tf.TensorShape([self.num_particles, 1]),
                                  I=tf.TensorShape([self.num_particles, self.seq_len]))

    # outputs: z, r (output of the last SMC layer) before softmax.
    self.output_size = (tf.TensorShape([self.num_particles, 1, self.d_model]),
                        tf.TensorShape([self.num_particles, 1, self.d_model]))

    super(SMC_Transf_Cell, self).__init__(**kwargs)

  def build(self, input_shapes):
    # replace this by the beginning of the init function of the SMC_layer.
    # see if this is actually useful...
    self.embedding = tf.keras.layers.Embedding(self.target_vocab_size, self.d_model)
    if self.maximum_position_encoding is not None:
      self.pos_encoding = positional_encoding(self.maximum_position_encoding, self.d_model)
      self.pos_encoding_SMC = positional_encoding_SMC(self.maximum_position_encoding, self.d_model, self.num_particles)
    # to remove?
    self.dropout = tf.keras.layers.Dropout(self.rate)
    self.built = True

  def call(self, inputs, states):
    # x -> r
    # w -> prev_sampling_weights
    # y -> target_word_id
    # put training and re-sampling as init parameters.
    '''
    -args:
      - inputs: (r^{l-1}[0:T] > shape (B, P, S, D) , (X tiled. (X[0:T] input of the while model) > shape (B,P,S,D).
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
    #TODO: qdd error is y and I are not integer

    #print('cell timestep', self.dec_timestep)

    # unnesting inputs
    r, x = tf.nest.flatten(inputs)  # r output prev trqnsformer, y: label/target
    x = tf.cast(x, dtype=tf.int32)

    # getting x
    K, V, w, I = states
    I = tf.cast(I, dtype=tf.int32)

    batch_size = tf.shape(K)[0]

    # resampling of (K,V) to compute the new set of (z,K,V)
    if self.resampling:
      # print('resampling trajectories for timestep {}'.format(self.dec_timestep))
      # z = resample_z(z, I, self.dec_timestep)  # if z is of shape (B,P,D). use a different resample function for z.
      K = resample(K, I)
      V = resample(V, I)

    # multi-head attention:
    input_mha = tf.expand_dims(r, axis=2)  # shape (B,P,1,D)
    inputs_mha = [input_mha for _ in
                  range(3)]  # trick to have an 'inputs' in the function call of the class MultiHeadAttention
    if self.dec_timestep < self.seq_len:
      # store (K,V) only in that case.
      (z, K, V) = self.mha1(inputs=inputs_mha, timestep=self.dec_timestep, K=K, V=V)
    else:
      # otherwise K,V is not updated.
      (z, KK, VV) = self.mha1(inputs=inputs_mha, timestep=self.dec_timestep, K=K, V=V)

    # (batch_size, NUM_PARTICLES, target_seq_len, d_model)
    # demander à Florian s'il faut changer l'ordre des layernorm/FFN.
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
    predictions = self.output_layer(out3)  # (batch_size, NUM_PARTICLES, target_vocab_size)

    def compute_w_classification(predictions):
      w = tf.gather(predictions, x, axis=-1, batch_dims=1)
      w = tf.squeeze(w, axis=-1)  # shape (B,P,1)
      w_squeezed = tf.squeeze(w, axis=-1)  # shape (B,P)
      return w_squeezed  # shape (B,P)

    # TODO: uncomment lines and solve the problem of tensor probability and tf 2.1
    def compute_w_regression(batch_size, num_particles):
      # tfd = tfp.distributions
      # #TODO: the 1. by a sigma which is a parameter.
      # dist = tfd.Normal(loc=[0. for _ in range(batch_size)],
      #                   scale=[1. for _ in range(batch_size)]) #
      # TODO: evaluate the distribution in the diff (X_t - G_theta(z_t).
      cdf_value = inputs - self.output_layer(z)
      # dist.cdf(cdf_value)
      # w = dist.sample([num_particles])  # shape (P,B)
      # w = tf.transpose(w, perm=[1, 0])
      # w_sum = tf.reduce_sum(w, axis=-1)
      # w_sum = tf.expand_dims(w_sum, axis=-1)
      # w = w / w_sum  # find the right solution to do a diff.
      # return w # shape (B,P)
      return 0

    if self.task_type == 'classification':
      w_squeezed = compute_w_classification(predictions)
    elif self.task_type == 'regression':
      w_squeezed = compute_w_regression(batch_size, self.num_particles)

    # update the genealogy indices matrix from the weights.
    if self.dec_timestep < self.seq_len:
      # update it only T-1
      i_t, I = sample_and_keep_indices(w_squeezed, I, self.num_particles, self.dec_timestep)

    # resample z:
    if self.resampling:
      bool_resampl=self.dec_timestep < self.seq_len
      if bool_resampl:
      # TODO check with Sylvain if this works for the resampling of z.
        z = resample_z(z, I, self.dec_timestep)  # if z is of shape (B,P,D).

    # get the output (r^^, z^l)
    output = [out3, z]
    w = tf.expand_dims(w_squeezed, axis=-1)
    new_states = NestedState(K=K, V=V, w=w, I=I)

    #('cell computation done for timestep {}'.format(self.dec_timestep))
    self.dec_timestep += 1

    return output, new_states



if __name__ == "__main__":
  from models.SMC_Transformer.SMC_Transformer import Transformer
  batch_size = 8
  d_model = 64
  num_heads = 2
  dff = 32
  target_vocab_size = 50
  maximum_position_encoding = None
  num_particles = 2
  seq_len = 4
  layer_num = 2

  cell = SMC_Transf_Cell(d_model=d_model, num_heads=num_heads, dff=dff, target_vocab_size=target_vocab_size,
                         num_particles=num_particles,
                         seq_len=seq_len, num_layers=layer_num, training=True, resampling=True)

  sample_transformer = Transformer(
    num_layers=layer_num, d_model=d_model, num_heads=num_heads,
    dff=dff, target_vocab_size=target_vocab_size,
    maximum_position_encoding=maximum_position_encoding,
    num_particles=num_particles,
  seq_len=seq_len)

  initial_word_tensor = tf.ones(shape=(batch_size, 1), dtype=tf.int32)

  (K0, V0), w0, I0 = sample_transformer.initialize_attn_SMC_parameters(batch_size, seq_len,
                                                                           initial_word_id=initial_word_tensor)

  initial_state = NestedState(K=K0, V=V0, w=w0, I=I0)


  def step_function(inputs, states):
    return cell.call(inputs, states)

  # rnn=tf.keras.layers.RNN(cell)

  r = tf.random.uniform(
    shape=(batch_size, seq_len, num_particles, d_model))  # the dim 1 needs to be added. trick with nested inputs.
  x = tf.ones(shape=(batch_size, seq_len,
                     1))  # all the inputs need to have shape at least equal to 3... # see which trick we can apply here...

  inputs = NestedInput(r=r, x=x)

  last_output, outputs, new_states = tf.keras.backend.rnn(step_function=step_function,
                                                          inputs=inputs,
                                                          initial_states=initial_state)

  # outputs=rnn(NestedInput(r=r, y=y))
  # y=outputs[0] # shape (B,1,P,1,D) # to squeeze to remove second dimension.
  # z=outputs[1] # # shape (B,1,P,1,D) # to squeeze to remove second dimension
  # see how to get all sequence and complete internal state.

  cell_output = last_output[0]
  last_z = last_output[1]
  K = new_states[0]
  V = new_states[1]
  w = new_states[2]
  I = new_states[3]

  output_seq = outputs[0]
  z_seq = outputs[1]

  print('r_T', cell_output.shape)  # (B,P,1,D) # should be dff?
  print('z_T', last_z.shape)  # shape (B,P,1,D)
  print('r0_T', output_seq.shape)  # shape (B,1,P,1,D) > should be shape (B,S,P,D) instead
  print('z_0_T', z_seq.shape)  # idem
  print('w_T', w.shape)  # shape (B,P,1)
  print('K', K.shape)  # shape (B,P,S,D)
  print('I', I.shape)  # shape (B,P,S)

  # print('output', last_output)

# rnn = tf.keras.layers.RNN(cell)
