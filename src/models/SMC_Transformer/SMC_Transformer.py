# imports
import tensorflow as tf
from models.SMC_Transformer.transformer_utils import positional_encoding_SMC
from models.SMC_Transformer.transformer_utils import positional_encoding
from neural_toolbox.SMC_layers import DecoderLayer
from neural_toolbox.SMC_TransformerCell import SMC_Transf_Cell
from models.SMC_Transformer.transformer_utils import initialize_indices_matrix
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask
from models.SMC_Transformer.transformer_utils import sample_and_keep_indices
from train.SMC_loss import compute_SMC_log_likelihood
from train.SMC_loss import compute_SMC_ll_one_layer
import collections
from models.SMC_Transformer.transformer_utils import resample

# for the sequential process in the Transformer class:
# use this instead: https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN?version=stable
NestedInput = collections.namedtuple('NestedInput', ['r', 'x'])
NestedState = collections.namedtuple('NestedState', ['K', 'V', 'w', 'I'])

# -------- Class Decoder and Class Transformer--------------------
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
    self.data_type=data_type
    self.sigma=sigma
    self.num_particles = num_particles
    self.noise=noise

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
               target_vocab_size, num_particles, seq_len, sigma, noise_encoder, noise_SMC_layer, data_type, task_type,
               rate=0.1, target_feature=None, maximum_position_encoding=None, resampling=True):
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
    elif num_layers==1:
      self.input_embedding=tf.keras.layers.Dense(d_model)
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
                                task_type=task_type,
                                resampling=resampling,
                                rate=rate,
                                target_feature=target_feature)  # put here the Transformer cell.

    # for pre_processing words in the one_layer case.
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

    self.data_type = data_type
    if data_type=='time_series_multi':
      assert target_feature is not None
    self.task_type = task_type

    self.sigma = sigma
    self.noise_encoder = noise_encoder
    self.noise_SMC_layer = noise_SMC_layer
    self.maximum_position_encoding = maximum_position_encoding
    self.target_feature = target_feature

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

  def initialize_attn_SMC_parameters(self, batch_size, seq_length, initial_word_id):
    ''' initialize the attention parameters of the Transformer:
          -Args:
            -batch_size
            -seq_length: longueur of input sequence of words
            -initial_word_tensor: 1D tensor of dim (batch_size, 1) with the initial words for each element of the batch.
            Used to compute the initial set of weights
          -Returns
            -Z0, K0, V0 (dim (B,P,S,D)) w0 (dim (B,P,1)), initial indices matrix (dim (B, P, S))
    '''
    # initialize K0, V0, Z0 (=V0)
    K = tf.random.uniform(shape=(batch_size, self.num_particles, seq_length, self.d_model), maxval=1, name='K')
    V = tf.random.uniform(shape=(batch_size, self.num_particles, seq_length, self.d_model), maxval=1, name='V')
    Z = V
    # initialize w0
    #TODO: add the FFN layers after z before taking the final layer. (not essential. It is as if r was initializing equal to V finally).
    logits = self.final_layer(Z)  # shape (B, P, S, V)
    logits_initial = logits[:, :, 0, :]

    # computing w0
    if self.task_type == 'classification':
      initial_word_id = tf.expand_dims(initial_word_id, axis=-1)
      initial_word_id = tf.cast(initial_word_id, dtype=tf.int32)
      initial_weights = self.cell.compute_w_classification(predictions=logits_initial, x=initial_word_id) # shape (B,P).

    elif self.task_type == 'regression':
      assert self.target_vocab_size == 1
      # TODO replace the tf.cast by an assert.
      initial_word_id = tf.cast(initial_word_id, dtype=tf.float32) # shape (B,F)
      if len(tf.shape(initial_word_id)) == 1:
        initial_word_id = tf.expand_dims(initial_word_id, axis=-1)
      # tiling word_id to get the right shape:
      initial_word_id = tf.expand_dims(initial_word_id, axis=1)
      initial_word_id = tf.tile(initial_word_id, multiples=[1, self.num_particles, 1])  # shape (B,P,F)

      initial_weights = self.cell.compute_w_regression(predictions=logits_initial, x=initial_word_id) # shape (B,P)

    # call the initialization of the ancestor indices matrix - create an initial 'identity function' indices matrix.
    ind_matrix_init = initialize_indices_matrix(batch_size, seq_length, self.num_particles) # (B,P,S)
    # update it with i_o
    i0 = tf.random.categorical(initial_weights, self.num_particles)  # (B,P)
    i0 = tf.expand_dims(i0, axis=-1)
    i0=tf.cast(i0, dtype=tf.int32)
    ind_matrix_init = tf.concat([i0, ind_matrix_init[:,:,1:]], axis=-1)

    self.initialize = True

    initial_weights = tf.expand_dims(initial_weights, axis=-1)  # (B,P,1)

    # adding a tf.stop_gradient on the weights and the ind_matrix_init to avoid backpropagation on this set of parameters:
    initial_weights = tf.stop_gradient(initial_weights)
    ind_matrix_init = tf.stop_gradient(ind_matrix_init)

    # resample K & Z using i_0 (nned to be of shape (B,P)
    K = resample(params=K, i_t=tf.squeeze(i0, axis=-1), t=0)
    V = resample(params=V, i_t=tf.squeeze(i0, axis=-1), t=0)

    return (K, V), initial_weights, ind_matrix_init

  def compute_SMC_log_likelihood(self, sampling_weights):
    '''
      -Args:
        -real: targets > tensor of shape (B,P,S)
        sampling_weights: tensor of shape (B,P) > final resampling_weights for the last decoding timestep
    '''
    # check that the pass forward has been done when calling this function.
    assert self.pass_forward is True

    # multi-layer case.
    if self.num_layers > 1:

      # get epsilon for each layer for the Encoder
      list_epsilon = self.encoder.list_stddev
      # add the one from the last layer
      list_epsilon.append(self.epsilon_seq_last_layer)
      # get the list of sigmas from the list of the Encoder's layer
      list_layers=self.encoder.dec_layers
      list_sigma = [l.mha1.sigma for l in list_layers]
      # append the sigma of the last SMC layer.
      sigma_last_layer=self.cell.mha_smc.sigma
      list_sigma.append(sigma_last_layer)

      SMC_loss= compute_SMC_log_likelihood(list_epsilon=list_epsilon,
                                          list_sigma=list_sigma,
                                          sampling_weights=sampling_weights)

    # one-layer case.
    elif self.num_layers == 1:

      sigma=self.cell.mha_smc.sigma
      epsilon=self.epsilon_seq_last_layer

      SMC_loss_tensor=compute_SMC_ll_one_layer(epsilon=epsilon, sigma=sigma)
      # multiply by -1/2 to get the right formula.
      SMC_loss = tf.scalar_mul(-1/2, SMC_loss_tensor) # shape (B,P,S)

      SMC_loss = tf.reduce_mean(SMC_loss, axis=-1) # mean over seq dim.
      SMC_loss = tf.reduce_mean(SMC_loss, axis=-1) # 'uniform' mean over particle dim. (w_final= 1/M)
      SMC_loss = tf.reduce_mean(SMC_loss, axis=-1) # mean ober batch dim.

    return SMC_loss

  def call(self, inputs, training, mask):
    '''
    -args:
      -input tensor: transformer input data : sequence of words id. > shape (B,S,1) or (B,S,F)
      -targets: target tensor > shape (B,S). No need for that actually...
      -training: for dropout layers
      -look_ahead_mask:
      -eval timestep: to remove?
    -returns
      -final_output: Y0:S > shape (?, P, S, V)
      -decoder output (before output layer): Z0:S > shape (B,P,S,D)
    '''

    # if necessary
    self.cell.training = training

    # initialize the attention parameters
    batch_size = tf.shape(inputs)[0]
    seq_len = tf.shape(inputs)[1]

    if self.data_type=='nlp':
      # process input_tensor (embedding + positional_encoding + tile) to have a shape of (B,P,S,D)
      input_tensor_processed = tf.expand_dims(inputs, axis=-1)
      input_tensor_processed = self.preprocess_words(input_tensor_processed, 0, training=training)  # dim (B, S, D)
      input_tensor_processed = tf.tile(input_tensor_processed, multiples=[1, self.num_particles, 1, 1])  # dim (B,P,S,D)
      input_tensor_processed = tf.cast(input_tensor_processed, dtype=tf.float32)

    elif self.data_type=='time_series_uni' or "time_series_multi":
      if len(tf.shape(inputs)) == 2: # shape(B,S)
        input_tensor_processed = tf.expand_dims(inputs, axis=-1) # shape (B,S,F)
      else:
        input_tensor_processed = inputs
      # add the particle dimension
      input_tensor_processed = tf.expand_dims(input_tensor_processed, axis=1) # (B,1,S,F)
      input_tensor_processed = tf.tile(input_tensor_processed, multiples=[1, self.num_particles, 1, 1]) # (B,P,S,F)

    else:
      raise ValueError('wrong data type: should be either "nlp", "time-series_uni", or "time_series_multi"')

    # First: 'Transformer embedding' for the first L-1 layers if num_layers > 1:
    if self.num_layers > 1:
      r, attn_weights_enc = self.encoder(inputs=input_tensor_processed,
                                         training=training,
                                         mask=mask)  # shape (B,P,S,D)
      r = tf.transpose(r, perm=[0, 2, 1, 3])  # shape (B,S,P,D) so that it can be processed by the RNN_cell & RNN_layer.
    elif self.num_layers == 1:
      # casting input_tensor_processed to tf.float32 so that it can be processed by the input_layer.
      input_tensor_processed = tf.cast(input_tensor_processed, dtype=tf.float32)
      # one dense layer to have a tensor of shape (B,P,S,D)
      r = self.input_embedding(input_tensor_processed)
      r = tf.transpose(r, perm=[0, 2, 1, 3])  # shape (B,S,P,D) so that it can be processed by the RNN_cell & RNN_layer.

    # 'dummy' initialization of cell's internal state for memory efficiency.
    #TODO:  take only as initial_word_id the feature corresponding to the target (case of multivariate timeseries).
    if self.target_feature is not None:
      assert self.target_feature < tf.shape(inputs)[-1]
      initial_word_id = inputs[:, 0, self.target_feature]
    else:
      initial_word_id = inputs[:, 0, :]
    (K0, V0), w0, I0 = self.initialize_attn_SMC_parameters(batch_size=batch_size,
                                                           seq_length=seq_len,
                                                           initial_word_id=initial_word_id)
    initial_state = NestedState(K=K0,
                                V=V0,
                                w=w0,
                                I=I0)

    def step_function(inputs, states):
      return self.cell(inputs, states)

    inputs = NestedInput(x=tf.expand_dims(inputs, axis=-1), r=r) # x > (B,S,F,1), #r > (B,S,P,D)

    last_output, outputs, new_states = tf.keras.backend.rnn(step_function=step_function,
                                                            inputs=inputs,
                                                            initial_states=initial_state)

    #TODO: if not self.training: create another 'inference cell'?
    #def step_function_inference(inputs, states):
      #return self.cell.inf_fonction(inputs, states)

    #inf_predictions=tf.keras.backend.rnn(step_function=step_function_inference,
                                                            #inputs=inputs,
                                                            #initial_states=initial_state)

    # last_output > (B,P,1,D)
    last_output = [tf.squeeze(out, axis=-2) for out in last_output] # (B,P,D)
    outputs = [tf.squeeze(out, axis=-2) for out in outputs]  # (B,S,P,D) for r,z,epsilon, (B,S,P,H,S) for attn_weights

    r_T = last_output[0]
    z_T = last_output[1]
    r0_T = outputs[0] # shape (B,S,P,D)
    Z0_T = outputs[1] # shape (B,S,P,D)

    inference_pred = outputs[2]
    good_avg_predictions = outputs[3]
    max_predictions = outputs[4] # (B,S,V)

    Epsilon0_T = outputs[5] # shape (B,S,P,D)

    K = new_states[0]
    V = new_states[1]
    w_T = new_states[2]
    #TODO: output both w and the matrix of indices matrix. (to save).
    I = new_states[3]

    Y0_T = self.final_layer(r0_T) # (B,S,P,C) used to compute the categorical cross_entropy loss. # logits.
    Y0_T = tf.transpose(Y0_T, perm=[0,2,1,3]) # (B,P,S,C)

    w_T = tf.squeeze(w_T, axis=-1) # (B,P,1)
    Z0_T = tf.transpose(Z0_T, perm=[0,2,1,3]) # (B,P,S,D)

    Epsilon0_T = tf.transpose(Epsilon0_T, perm=[0,2,1,3]) # shape (B,P,S,D)
    # stocking epsilon as an internal parameter of the SMC_Transformer class to use it the computation of the loss.
    self.epsilon_seq_last_layer = Epsilon0_T

    attn_weights_SMC_layer = outputs[6] # shape (B,S,P,H,S)
    attn_weights_SMC_layer = tf.transpose(attn_weights_SMC_layer, perm=[0,2,3,1,4])

    if self.num_layers == 1:
      attn_weights = attn_weights_SMC_layer
    else:
      attn_weights = attn_weights_enc
      attn_weights['SMC_layer_{}'.format(self.num_layers)] = attn_weights_SMC_layer

    self.pass_forward = True

    return (Y0_T, Z0_T, w_T, I), (inference_pred, good_avg_predictions, max_predictions), attn_weights

if __name__ == "__main__":

  num_particles = 5
  seq_len = 4
  b = 8
  F = 1 # multivariate case.
  num_layers = 1
  d_model = 1
  num_heads = 1
  dff = 128
  maximum_position_encoding = seq_len
  sigma = 1
  data_type = 'time_series_uni'
  task_type = 'regression'
  C = 1 # vocabulary size or number of classes.
  noise_encoder = False
  noise_SMC_layer = True

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

  target_feature = 0 if data_type == 'time_series_multi' else None
  maximum_position_encoding = 2000

  sample_transformer = SMC_Transformer(
    num_layers = num_layers,
    d_model = d_model,
    num_heads = num_heads,
    dff = dff,
    target_vocab_size = C,
    maximum_position_encoding = maximum_position_encoding,
    num_particles = num_particles,
  seq_len = seq_len,
  sigma = sigma,
  noise_encoder = noise_encoder,
  noise_SMC_layer = noise_SMC_layer,
  data_type = data_type,
  task_type = task_type,
  target_feature = target_feature)

  inputs = tf.ones(shape=(b, seq_len, F), dtype=tf.int32) # ok works with len(tf.shape(inputs)==3.
  mask = create_look_ahead_mask(seq_len)

  (predictions, trajectories, weights, ind_matrix), predictions_metric, attn_weights = sample_transformer(inputs=inputs,
                                                                                              training=True,
                                                                                              mask=mask)

  inference_pred, good_avg_pred, max_pred = predictions_metric
  print('Transformer output', predictions.shape)  # (B,P,S,C)
  print('final z', trajectories.shape) # (B,P,S,D)
  print('weights', weights.shape) # (B,P,1)
  print('indices matrix for element 0', ind_matrix[0,:,:])
  print('inference predictions', inference_pred.shape)  # (B,P,V)
  print('good average predictions', good_avg_pred.shape)  # (B,P,V)
  print('max_predictions', max_pred.shape)

  print('w_T for element 0 of the batch', weights[0,:])
  print('indices matrix for element 0 of the batch', ind_matrix[0,:])

  if num_layers > 1:
    print('attn weights first layer', attn_weights['encoder_layer1'].shape) # shape (B,P,H,S,S)

  #### test of compute_SMC_log_likelihood function-------------------------------------------------------------------------

  SMC_loss = sample_transformer.compute_SMC_log_likelihood(sampling_weights=weights)
  print('SMC_loss', SMC_loss.numpy())
