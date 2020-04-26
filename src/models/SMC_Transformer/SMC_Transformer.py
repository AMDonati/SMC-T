# imports
import tensorflow as tf
from models.SMC_Transformer.transformer_utils import positional_encoding_SMC
from models.SMC_Transformer.transformer_utils import positional_encoding
from neural_toolbox.SMC_layers import DecoderLayer
from neural_toolbox.SMC_TransformerCell import SMC_Transf_Cell
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from train.SMC_loss import compute_SMC_log_likelihood
from train.SMC_loss import compute_SMC_ll_one_layer
import collections

# for the sequential process in the Transformer class:
# use this instead: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN?version=stable
NestedInput = collections.namedtuple('NestedInput', ['x', 'y'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'R'])

# -------- Class Decoder and Class Transformer-------------------------------------------------------------------------------------------------------------
# class Decoder that takes a SMC Decoder Layers as input.
class Encoder(tf.keras.layers.Layer):
  '''Class Encoder with the Encoder architecture
  -args
    -num_layers: number of layers in the Decoder
    -d_model: model depth
    -num_heads: number of heads in the multi-attention mechanism
    -dff: output dim of the feedforward network
    -target_vocab_size (for computing the sampling weights for the last layer (or all layers))
    -num_particles
    -sigma: to inject noise in the Encoder.
    -data_type: 'nlp' or 'time_series'
    -maxixum_position_encoding: to preprocess the words sequence (addition of positional embeddings)
    -rate: dropout rate for feed-forward layers.
    '''

  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
              num_particles, sigma, noise, data_type, maximum_position_encoding,
              rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers  # num_layers - 1 compared to the Total Transformer.
    self.dff=dff
    self.maximum_position_encoding=maximum_position_encoding
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)

    if maximum_position_encoding is not None:
      self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
      self.pos_encoding_SMC = positional_encoding_SMC(maximum_position_encoding, d_model, num_particles) # used to pre-process input word.

    # build the decoder architecture
    self.dec_layers = [DecoderLayer(d_model=d_model,
                                    num_heads=num_heads,
                                    dff=dff, sigma=sigma,
                                    num_particles=num_particles,
                                    noise=noise) for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

    # additional parameters for SMC:
    self.data_type = data_type
    self.sigma = sigma
    self.num_particles = num_particles
    self.noise = noise

  def preprocess_words_input(self, x, training):
    '''pre_process sequence of words by adding embeddings + positional encoding
      -Args:
        -x: 4D tensor for the input sequence of words > dim (B, P, S, d_input) OR a 3D tensor of dim (B, P, S) (word_id instead of words...)
        -training: boolean for dropout
      -Returns:
        -A 3D tensor of the pre-processed sequence of words > dim (B, S, D)
    '''
    seq_len = tf.shape(x)[1]
    if len(tf.shape(x)) == 2:
      x = self.embedding(x)  # (batch_size, num_particles, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) # division by the root of the d_model
    x=tf.expand_dims(x, axis=1)
    x+=tf.tile(x, multiples=[1,self.num_particles,1,1])
    if maximum_position_encoding is not None:
      x += self.pos_encoding_SMC[:, :, :seq_len, :]  # addition of the positional encoding to the input x
    x = self.dropout(x, training=training)
    return x

  def preprocess_timeseries(self, x):
    '''preprocessing function for time-series data
    args:
     -x: input_data > shape (B,S,F)
    '''
    x=tf.expand_dims(x, axis=1)
    if len(tf.shape(x))==4:
      x=tf.tile(x, multiples=[1, self.num_particles,1,1])
    elif len(tf.shape(x))==3:
      x = tf.tile(x, multiples=[1, self.num_particles,1])
    else:
      raise ValueError('shape of x after expand_dims should be 3 or 4')
    return x

  def call(self, inputs, training, mask):
    # done without noise for now.
    '''
    -args:
      -inputs: input of the first decoder layer (X0:k-1) > shape (B,S) (nlp)
      or (B,S,F) (time-series)
      -training
      -look_ahead_mask > to mask the future.
    -returns:
      -r(0:T) > embedding 'particulaire' of the word input > shape (B,P,S,D)
      -attention_weights: for attention visualization. Not returned for now.
    '''

    self.list_stddev = []
    attention_weights = {}

    # do the pre_processing step for the input data.
    if len(tf.shape(inputs))<4:
      if self.data_type=='nlp':
        inputs = self.preprocess_words_input(inputs, training)
      elif self.data_type=='time_series':
        inputs=self.preprocess_timeseries(inputs)
      else:
        raise ValueError('data_type not supported; please choose either "nlp" or "time_series"')

    for i in range(self.num_layers):
      inputs, stddev, attn_weights = self.dec_layers[i](inputs=inputs, training=training, look_ahead_mask=mask)
      self.list_stddev.append(stddev)
      attention_weights['encoder_layer{}'.format(i + 1)] = attn_weights

    return inputs, attention_weights

#------------------------CREATE THE SMC TRANSFORMER MODEL ------------------------------------------------------------

class SMC_Transformer(tf.keras.Model):
  '''class for the Transformer Model
  -args
    -num_layers: number of decoder layers (before the final SMC_layer)
    -d_model: model_depth
    -num_heads: number of heads in the multi-head attention mechanism.
    -dff: output dimension of the feed-forward layer.
    -target_vocab_size:for computing the resampling weights # only used for nlp dataset
    -pe_target: maximum_positional_encoding # only used for nlp dataset.
    -num_particles: number of particles generated.
    -sigma:
    -noise:
    -rate: dropout rate for the feed-forward layer.
    '''

  def __init__(self, num_layers, d_model, num_heads, dff,
               target_vocab_size, num_particles, seq_len, sigma, noise_encoder, noise_SMC_layer, data_type, task_type, rate, omega,
               target_feature, maximum_position_encoding=None, resampling=True, layer_norm=True, test=False):
    super(SMC_Transformer, self).__init__()

    # add Encoder if num_layers > 1:
    if num_layers > 1:
      self.encoder = Encoder(num_layers=num_layers-1, # to have a total number of layers equal to num_layers.
                             d_model=d_model,
                             num_heads=num_heads,
                             dff=dff,
                             target_vocab_size=target_vocab_size,
                             maximum_position_encoding=maximum_position_encoding,
                             num_particles=num_particles,
                             sigma=sigma,
                             noise=noise_encoder,
                             rate=rate,
                             data_type=data_type)
    elif num_layers == 1:
      self.input_dense_projection=tf.keras.layers.Dense(d_model, name='projection_layer_ts')
    else:
      raise ValueError("num_layers should be superior or equal to 1.")


    self.cell = SMC_Transf_Cell(d_model=d_model,
                                dff=dff,
                                target_vocab_size=target_vocab_size,
                                maximum_position_encoding=maximum_position_encoding,
                                num_particles=num_particles,
                                seq_len=seq_len,
                                num_layers=num_layers,
                                num_heads=num_heads,
                                sigma=sigma,
                                noise=noise_SMC_layer,
                                omega=omega,
                                task_type=task_type,
                                resampling=resampling,
                                layer_norm=layer_norm,
                                rate=rate,
                                target_feature=target_feature,
                                test=test)  # put here the Transformer cell.

    # for pre_processing words in the one_layer case.
    if task_type == 'classification':
      self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    if maximum_position_encoding is not None:
      self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    self.dropout = tf.keras.layers.Dropout(rate)

    self.final_layer = self.cell.output_layer
    self.target_vocab_size = target_vocab_size
    self.num_particles = num_particles
    self.d_model = d_model
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.vocab_size = target_vocab_size
    self.dff = dff
    self.rate = rate

    self.data_type = data_type
    self.task_type = task_type

    self.seq_len = seq_len
    self.sigma = sigma
    self.omega = omega
    self.noise_encoder = noise_encoder
    self.noise_SMC_layer = noise_SMC_layer
    self.maximum_position_encoding = maximum_position_encoding
    self.target_feature = target_feature

    self.layer_norm = layer_norm

    # to test the class SMC_Transformer.
    self.test = test

    self.initialize = False
    self.pass_forward = False

  def preprocess_words(self, x, dec_timestep, training):
    '''add words embeddings and positional encodings:
        -Args:
          -x: 2D tensor of sequence of words id > dim (B, S)
          -training: boolean for dropout
        -Returns:
          - A 3D tensor of pre-processed words sequence > dim (B, S, D)
    '''
    if self.num_layers > 1:
      x = self.encoder.embedding(x)  # (batch_size, target_seq_len, d_model)
      x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # division by the root of the d_model
      # addition of the positional encoding to the input x for the current decoding step
      if self.maximum_position_encoding is not None:
        x += self.encoder.pos_encoding[:, dec_timestep, :]  # dim of positional encoding (1, num_positions, d_model)
      x = self.encoder.dropout(x, training=training)

    elif self.num_layers==1:
      x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
      x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # division by the root of the d_model
      # addition of the positional encoding to the input x for the current decoding step:
      if self.maximum_position_encoding is not None:
        x += self.pos_encoding[:, dec_timestep, :]  # dim of positional encoding (1, num_positions, d_model)
      x = self.dropout(x, training=training)

    return tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[2], tf.shape(x)[1], tf.shape(x)[-1]])

  def compute_SMC_log_likelihood(self, sampling_weights):
    '''
      -Args:
        -real: targets > tensor of shape (B,P,S)
        sampling_weights: tensor of shape (B,P) > final resampling_weights for the last decoding timestep
    '''

    # multi-layer case.
    if self.num_layers > 1:
    #TODO: modify this for the multilayer case (because of the addition of the noise in k,q,v.)
      # get epsilon for each layer for the Encoder
      list_epsilon = self.encoder.list_stddev
      # add the one from the last layer
      list_epsilon.append(self.noises_seq)
      # get the list of sigmas from the list of the Encoder's layer
      list_layers=self.encoder.dec_layers
      list_sigma = [l.mha1.sigma for l in list_layers]
      # append the sigma of the last SMC layer.
      sigma_last_layer=self.cell.mha_smc.sigma
      list_sigma.append(sigma_last_layer)

      SMC_loss = compute_SMC_log_likelihood(list_epsilon=list_epsilon,
                                          list_sigma=list_sigma,
                                          sampling_weights=sampling_weights)

    # one-layer case.
    elif self.num_layers == 1:
      list_noises=self.noises_seq

      SMC_loss_tensor = compute_SMC_ll_one_layer(list_means=list_noises)
      # multiply by -1/2 to get the right formula.
      #TODO: refactor this: put the -1/2 in the compute_SMC_ll_one_layer.
      SMC_loss = tf.scalar_mul(-1/2, SMC_loss_tensor) # shape (B,P,S)
      SMC_loss = tf.reduce_mean(SMC_loss, axis=-1) # mean over seq dim.
      SMC_loss = tf.reduce_mean(SMC_loss, axis=-1) # 'uniform' mean over particle dim. (w_final= 1/M)
      SMC_loss = tf.reduce_mean(SMC_loss, axis=-1) # mean over batch dim.

    return SMC_loss

  def call(self, inputs, training, mask):
    '''
    -args:
      -input tensor: transformer input data : sequence of words id. > shape (B,S+1,1) or (B,S,F)
      -targets: target tensor > shape (B,S). No need for that actually...
      -training: for dropout layers
      -look_ahead_mask:
    -returns
      -final_output: Y0:S > shape (?, P, S, V)
      -decoder output (before output layer): Z0:S > shape (B,P,S,D)
    '''
    # if necessary
    self.cell.training = training

    # check dimensionality of inputs (B,S,F)
    assert len(tf.shape(inputs)) == 3

    # initialize the attention parameters
    batch_size = tf.shape(inputs)[0]
    seq_len_total = tf.shape(inputs)[1] # S+1
    assert self.seq_len == seq_len_total - 1

    # splitting between input_data and targets(y)
    input_data = inputs[:,:-1,:] # (B,S,F)
    y = tf.expand_dims(inputs[:, 1:, :], axis=-1) # (B,S,F,1)

    if self.data_type=='nlp':
      # process input_tensor (embedding + positional_encoding + tile) to have a shape of (B,P,S,D)
      input_tensor_processed = tf.expand_dims(input_data, axis=-1)
      input_tensor_processed = self.preprocess_words(input_tensor_processed, 0, training=training)  # dim (B, S, D)
      input_tensor_processed = tf.tile(input_tensor_processed, multiples=[1, self.num_particles, 1, 1])  # dim (B,P,S,D)
      input_tensor_processed = tf.cast(input_tensor_processed, dtype=tf.float32)

    elif self.data_type=='time_series_uni' or "time_series_multi":
      input_tensor_processed = tf.expand_dims(input_data, axis=1) # (B,1,S,F)
      input_tensor_processed = tf.tile(input_tensor_processed, multiples=[1, self.num_particles, 1, 1]) # (B,P,S,F)

    else:
      raise ValueError('wrong data type: should be either "nlp", "time-series_uni", or "time_series_multi"')

    # First: 'Transformer embedding' for the first L-1 layers if num_layers > 1:
    if self.num_layers > 1:
      x, attn_weights_enc = self.encoder(inputs=input_tensor_processed,
                                         training=training,
                                         mask=mask)  # shape (B,P,S,D)
      x = tf.transpose(x, perm=[0, 2, 1, 3])  # shape (B,S,P,D) so that it can be processed by the RNN_cell & RNN_layer.

    elif self.num_layers == 1:
      input_tensor_processed = tf.cast(input_tensor_processed, dtype=tf.float32)
      x = self.input_dense_projection(input_tensor_processed) # one dense layer to have a tensor of shape (B,P,S,D)
      x = tf.transpose(x, perm=[0, 2, 1, 3])  # shape (B,S,P,D) so that it can be processed by the RNN_cell & RNN_layer.

    # 'dummy' initialization of cell's internal state for memory efficiency.
    shape = (batch_size, self.num_particles, self.seq_len+1, self.d_model)
    K0, V0, R0 = tf.zeros(shape=shape, dtype=tf.float32), tf.zeros(shape=shape, dtype=tf.float32), tf.zeros(shape=shape, dtype=tf.float32)
    initial_state = NestedState(K=K0,
                                V=V0,
                                R=R0)

    def step_function(inputs, states):
       return self.cell(inputs, states)

    inputs_for_rnn = NestedInput(x=x, y=y) # y > (B,S,F,1), #x > (B,S,P,D)

    if self.test:
      print('inputs(x)', input_data)
      print('y', y)

    last_output, outputs, new_states = tf.keras.backend.rnn(step_function=step_function,
                                                            inputs=inputs_for_rnn,
                                                            initial_states=initial_state)
    # reset decoding timestep of the cell to 1:
    self.cell.dec_timestep = 0

    # ------------------ EXTRACTING OUTPUTS OF THE RNN LAYER ------------------------------------------------------------------------
    # outputs
    indices_matrix = outputs[0] # (B,S,P)
    w0_T = outputs[1] # (B,S,P,D)
    list_noise_0_T = tf.squeeze(outputs[2], axis=-2) # (B,S,P,D)
    attn_weights_SMC_layer = tf.squeeze(outputs[3] , axis=-2) # shape (B,S,P,H,S)

    # states
    K = new_states[0] # (B,P,S+1,D)
    K = K[:,:,1:,:] # remove first timestep (dummy init.) # (B,P,S,D)
    V = new_states[1] # (B,P,S+1,D)
    V = V[:,:,1:,:] # (B,S,P,D)
    R = new_states[2] # (B,P,S+1,D)
    R = R[:,:,1:,:] # (B,P,S,D)

    Y0_T = self.final_layer(R) # (B,P,S,C) used to compute the categorical cross_entropy loss. # logits.

    indices_matrix = tf.transpose(indices_matrix, perm=[0,2,1]) # (B,P,S)
    list_noise_0_T = [tf.transpose(noise, perm=[0,2,1,3]) for noise in list_noise_0_T] # shape (B,P,S,D)
    self.noises_seq = list_noise_0_T # stocking epsilon as an internal parameter of the SMC_Transformer class to use it the computation of the loss.

    attn_weights_SMC_layer = tf.transpose(attn_weights_SMC_layer, perm=[0,2,3,1,4])

    if self.num_layers == 1:
      attn_weights = attn_weights_SMC_layer
    else:
      attn_weights = attn_weights_enc
      attn_weights['SMC_layer_{}'.format(self.num_layers)] = attn_weights_SMC_layer

    return (Y0_T, w0_T, indices_matrix, (K,V,R)), attn_weights

if __name__ == "__main__":
  num_particles = 10
  seq_len = 5
  b = 8
  F = 3 # multivariate case.
  num_layers = 1
  d_model = 12
  num_heads = 1
  dff = 128
  maximum_position_encoding = seq_len
  sigma = 'learned'
  omega = 0.5
  data_type = 'time_series_multi'
  task_type = 'regression'
  target_feature = 0
  C = F if target_feature is None else 1
  noise_encoder = False
  noise_SMC_layer = True
  rate = 0.1
  test = True
  ###----------Test of Encoder class-----------------------------------------------------------------------------------

  #TODO: debug the multivariate case for the case where num_layers > 1.

  # x = tf.random.uniform(shape=(b, seq_len ,F), dtype=tf.float32)
  #
  # encoder = Encoder(num_layers=num_layers,
  #                          d_model=d_model,
  #                          num_heads=num_heads,
  #                          dff=dff,
  #                          target_vocab_size=F,
  #                          maximum_position_encoding=maximum_position_encoding,
  #                          num_particles=num_particles,
  #                   sigma=sigma,
  #                   noise=noise_encoder,
  #                   data_type=data_type)
  #
  # r, attn_weights_enc = encoder(inputs=x, training=False, mask=None)

  ####---------test of Transformer class--------------------------------------------------------------------------------


  #target_feature = 0 if data_type == 'time_series_multi' else None
  maximum_position_encoding = None

  sample_transformer = SMC_Transformer(
    num_layers = num_layers,
    d_model = d_model,
    num_heads = num_heads,
    dff = dff,
    target_vocab_size = C,
    maximum_position_encoding = maximum_position_encoding,
    num_particles = num_particles,
    seq_len = seq_len-1,
    sigma = sigma,
    omega = omega,
    noise_encoder = noise_encoder,
    noise_SMC_layer = noise_SMC_layer,
    data_type = data_type,
    task_type = task_type,
    target_feature = target_feature,
    rate=rate,
    layer_norm=True,
    test=test)


  inputs = tf.constant([[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]], shape=(1, seq_len, F), dtype=tf.int32) # ok works with len(tf.shape(inputs)==3.
  inputs = tf.tile(inputs, multiples=[b,1,1])

  mask = create_look_ahead_mask(seq_len)

  (predictions, weights, ind_matrix, (K,V,R)), attn_weights = sample_transformer(inputs=inputs,
                                                                                    training=True,
                                                                                    mask=mask)
  print('final predictions - one sample', predictions[0,:,:,:])
  print('final K - one sample', K[0,:,:,0])
  print('w_T', weights)

  if num_layers > 1:
    print('attn weights first layer', attn_weights['encoder_layer1'].shape) # shape (B,P,H,S,S)

  #### test of compute_SMC_log_likelihood function-------------------------------------------------------------------------

  SMC_loss = sample_transformer.compute_SMC_log_likelihood(sampling_weights=weights)
  print('SMC_loss', SMC_loss.numpy())

  ### test of inference function.
  input = inputs[:,-1,:] # (B,F)
  input = tf.expand_dims(input, axis=1) # (B,1,F)
  input = tf.tile(input, multiples=[1, num_particles, 1]) # (B,P,F)
  input = tf.expand_dims(input, axis=2) # (B,P,1,F)
  input = tf.cast(input, dtype=tf.float32)
  input = sample_transformer.input_dense_projection(input) # (B,P,1,D)

  inference_dec_timestep = tf.shape(K)[2]
  num_samples = 2
