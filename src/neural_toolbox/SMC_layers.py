import tensorflow as tf
import numpy as np

# additional imports
from models.SMC_Transformer.self_attention_classic import MultiHeadAttention_classic
from models.SMC_Transformer.self_attention_SMC import MultiHeadAttention_SMC
from neural_toolbox.classic_layers import point_wise_feed_forward_network
from models.SMC_Transformer.transformer_utils import create_look_ahead_mask

# original DecoderLayer from TF 2.0 tutorial on Tranformer
class DecoderLayer(tf.keras.layers.Layer):
  '''adaptated version of the original Decoder Layer of the Transformer.
  The only difference are the shapes of the input tensor (B, P, S, D) instead of (B, S, D)
  -args:
    -d_model: model depth
    -num_heads: number of heads in the multi-head attention mechanism
    -dff: output dimension of the feed forward network
    -num_particles: number of simulated particles for the latent state space of the Transformer
    -rate: dropout rate for output layers
  '''

  def __init__(self, d_model, num_heads, dff, num_particles, sigma, rate=0.1):
    super(DecoderLayer, self).__init__()

    #self.dec_timestep = 0 # to remove???
    self.num_particles = num_particles

    self.mha1 = MultiHeadAttention_classic(d_model=d_model,
                                           num_heads=num_heads,
                                           num_particles=num_particles,
                                           sigma=sigma)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, inputs, training, look_ahead_mask):
    '''
    -args:
        -inputs: input work or output of the previous layer > shape (B,P,S,D)
        -training: boolean to distinct between training and evaluation phase.
        -look_ahead_mask: for masking future decoding timestep
        -padding_mask: for fixed-size words sequence.
    -returns
        -r0:T output of the Decoder layer > dim (B, P, S, D)
        -reparametrized gaussian noise for the current layer (to compute the loss)
    '''
    # preparing inputs_mha[x,x,x (x float] for mha class.
    inputs_float=tf.cast(inputs, dtype=tf.float32)
    inputs_mha=[inputs_float for _ in range(3)]
    # computing multi-head attention.
    (Z, K, V) = self.mha1(inputs=inputs_mha, mask=look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    # put a None as the decoding timestep instead?
    attn1 = self.dropout1(Z, training=training)
    out1 = self.layernorm1(attn1 + inputs_float)

    ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

    return out3, self.mha1.stddev,  # attn_weights_block1


# Code for the Decoder Layer with SMC.
# class DecoderLayer_SMC(tf.keras.layers.Layer):
#   def __init__(self, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, num_particles, layer_num,
#                decoder=None, rate=0.1):
#     '''
#     -Args:
#       -d_model: model depth
#       -num_heads: number of heads in the multi-head attention mechanism
#       -dff: output dimension of the feed forward network
#       -target_vocab_size (for computing the sampling weights)
#       -maximum_position_encoding: number of positions for positional encodings.
#       -num_particles: number of simulated particles for the latent state space of the Transformer
#       -layer_num: only used if resampling is done at layer-level (in the Decoder class)
#       -rate: dropout rate for output layers
#     '''
#     super(DecoderLayer_SMC, self).__init__()
#
#     # store the decoding timestep
#     self.dec_timestep = 0
#     self.mha1 = MultiHeadAttention_SMC(d_model, num_heads, num_particles, self.dec_timestep, mode='self')
#     self.ffn = point_wise_feed_forward_network(d_model, dff)
#
#     self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#     self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
#
#     self.dropout1 = tf.keras.layers.Dropout(rate)
#     self.dropout3 = tf.keras.layers.Dropout(rate)
#
#     self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
#     self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
#     self.pos_encoding_SMC = positional_encoding_SMC(maximum_position_encoding, d_model, num_particles) # to remove?
#     self.dropout = tf.keras.layers.Dropout(rate)
#
#     self.num_particles = num_particles
#     self.d_model = d_model
#     self.target_vocab_size = target_vocab_size
#
#     self.layer_num = layer_num
#     self.decoder = decoder
#
#     # This layer needs to share weights with the Transformer.final_layer
#     if self.decoder is None:
#       print('SMC mechanism done at decoder-level...')
#       self.output_layer = tf.keras.layers.Dense(target_vocab_size)
#
#     self.initialize = False
#
#   def set_decoder(self, decoder):
#     '''only used is the resampling is done at layer level.'''
#     self.decoder = decoder
#
#   def forward_pass_from_layer(self, x):
#     '''compute the forward pass from the layer until the decoder output
#       only used if re-sampling is done at layer level.
#     '''
#     # TO ACTUALLY IMPLEMENT (not urgent...).
#     # if self.decoder is not None:
#     # forward_layers=self.decoder.dec_layers[self.layer_num:]
#     # for layer in forward_layers:
#     # forward_func=layer()
#     # else:
#     forward_func = self.output_layer
#     return forward_func(x)
#
#   def initialize_indices_matrix(self, batch_size, seq_length):
#     # initialize it as the "identity transformation function"
#     ind_matrix = tf.constant([l for l in range(self.num_particles)], shape=(1, self.num_particles, 1), dtype=tf.int32,
#                              name='indices_matrix')
#     # tile to have the right shape
#     ind_matrix = tf.tile(ind_matrix, [batch_size, 1, seq_length]) # shape (B,P,S)
#     self.ind_matrix = ind_matrix
#     return ind_matrix  # tf.stop_gradient(ind_matrix)?
#
#   def preprocess_words(self, x, training):
#     '''add words embeddings and positional encodings:
#         -Args:
#           -x: 3D tensor of words sequence > dim (B, S, dim_words)
#           -training: boolean for dropout
#         -Returns:
#           - A 3D tensor of pre-processed words sequence > dim (B, S, D)
#     '''
#     x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
#     x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # division by the root of the d_model
#     x += self.pos_encoding[:, self.dec_timestep, :]  # dim of positional encoding (1, num_positions, d_model)
#     x = self.dropout(x, training=training)
#
#     return x  # (batch_size, target_seq_len, d_model)
#
#   def sample_previous_word(self, indices, prev_Z, training):
#     '''sample Xk-1[Ik[l]] the embedding of a selected word at the previous position, with probabilities given by the last layer output
#     Args:
#       -prev_Z: Z0:k-1 > dim (B, P, S, D)
#       -indices: used to recompute Zk-1 > dim (B,P)
#       -training: boolean for dropout
#     Returns:
#       - the batch of sampled word of shape (B, P, 1, D)
#     '''
#     # TO BE CHECKED BY SYLVAIN.
#     # needs to be changed if z of shape (B,P,D).
#
#     # recompute Z0:k-1 with the set of indices
#     if len(indices.shape) == 3:
#       indices = tf.squeeze(indices, axis=-1)
#
#     # compute z(k-1) with resampling
#     Z_previousk = tf.gather(prev_Z[:, :, self.dec_timestep, :], indices, axis=1, batch_dims=1)
#
#     # compute the log probabilities associated to the prediction at Z0:k-1
#     sample_words_id = []
#     predictions_probas = self.forward_pass_from_layer(
#       Z_previousk) # PUT PREV_Z INSTEAD??
#     # dimensions (batch_size, num_particles, vocabulary_size)
#
#     # select a word id randomly with proba equal to predictions_probas.
#     for n in range(self.num_particles):
#       # TRY TO ELIMINATE THIS FOR LOOP
#       sample_words_id += [tf.random.categorical(predictions_probas[:, n, :], num_samples=1)]
#
#     sample_words_id = tf.concat(sample_words_id, axis=1)  # dimensions (B, P)
#     sample_words_id = tf.expand_dims(sample_words_id, axis=-1)  # adding the seq_length dimension
#     # > dim (B, P, 1)
#
#     # preprocess the words with the word embedding & positional encoding.
#     x = self.preprocess_words(sample_words_id, training)  # dim (B, P, 1)
#
#     return x
#
#   def sample_and_keep_indices(self, prev_sampling_weights, ind_matrix):  # add a mask argument?
#     '''samples the set of N indices for doing the weights resampling
#     adds this set of indices to the matrix of ancestor indices
#     Args:
#     -prev_sampling_weights: w(k-1) > dim (B, P)
#     -indice matrix: I0:k-1 > dim (B, P, S)
#     Returns:
#     -The current set of indices to do a forward pass on the Decoder Layer > dim (batch_size, num_particles)
#     -The updated ancestor indices matrix > dim (batch_size, NUM_PARTICLES, seq_length)'''
#
#     # FUNCTION THAT NEEDS TO BE TESTED... ok done.
#
#     # Sample current set of indices with proba proportional to prev_sampling_weights
#     if len(tf.shape(prev_sampling_weights)) == 3:
#       prev_sampling_weights = tf.squeeze(prev_sampling_weights, axis=-1)
#     indices = tf.random.categorical(prev_sampling_weights, self.num_particles)  # shape (..., num_particles)
#     # indices=tf.math.add(indices, tf.constant(1, dtype=indices.dtype))
#
#     # Add this set of indices to the indices matrix tensor:
#     indices = tf.cast(indices, tf.int32)
#     indices = tf.expand_dims(indices, axis=-1)
#     updated_ind_matrix = tf.concat(
#       [ind_matrix[:, :, :self.dec_timestep + 1], indices, ind_matrix[:, :, self.dec_timestep + 2:]], axis=-1)
#
#     return indices, updated_ind_matrix
#
#   def resample(self, params, indices):
#     '''
#     This is the correct resmapling function to use.
#     Has been tested.
#     :param params: past attention parameters > tensor of dim (B,P,S,D)
#     :param indices: indices genealogy > tensor of dim (B,P,S)
#     :return:
#     the resampled trajectories of attention parameters > tensor of dim (B,P,S,D)
#     '''
#     seq_length = tf.shape(params)[2]
#     params_list = []
#     for t in range(seq_length):
#       params_t = params[:, :, t, :] # dim (B,P,D)
#       indices_t = indices[:, :, t] # dim (B,P)
#       # resample trajectories using the indices genealogy
#       params_resampl_t = tf.gather(params_t, indices_t, batch_dims=1) # dim (B,P,D)
#       params_resampl_t = tf.expand_dims(params_resampl_t, axis=2)
#       params_list.append(params_resampl_t)
#     params_resampl = tf.stack(params_list, axis=2) # dim (B,P,S,1,D)
#     params_resampl = tf.squeeze(params_resampl, axis=-2)
#     return params_resampl
#
#   # create a resample function specific to resample only zt.
#
#   def call(self, x, PREV_SAMPL_WEIGHTS, K, V, TARGET_WORD_ID, training,
#            resampling=True):
#     '''
#     -args:
#         -x: input tensor of the multi-head attention mechanism (Zk-1) > (B,P,1,D) > r_k^(l-1)
#         -prev_sampling_weights:
#         -K: K0:k-1 > shape (B,P,S,D)
#         -V: VO:k-1 > shape (B,P,S,D)
#         -Target_word_ID: to compute the new set of sampling weights > dim (B,)
#         -training: for dropout
#         -previous_word: used only if are not sampling a word for computing attention but taking directly the previous word.
#     -returns:
#         -attention vectors (Zk, K0:k, V0:k) > shape (B,P,S,D)
#         -sampling_weights wk > shape (B,P)
#         -indices matrix I0:k > shape (B,P,S)
#         -attention_weights_block: for attention vizualisation
#         -stddev: reparametrized gaussian noise compute in the multi-head attention
#         (useful to compute the SMC_loss) > shape (B,P,S,D)
#     '''
#
#     # FOR SMC: 1. SAMPLE THE SET OF N INDICES USING THE sample_indices function
#     if self.dec_timestep == 0 and resampling==True:
#       self.initialize_indices_matrix(tf.shape(K)[0],
#                                      tf.shape(K)[2])
#
#     if resampling:
#       indices, self.ind_matrix = self.sample_and_keep_indices(PREV_SAMPL_WEIGHTS, self.ind_matrix)
#
#     # # 2. Using the sampled indices, get the set of N sampled previous embedded words using the function 'sample_previous_words'
#     # if previous_word is None:
#     #   # sampling method
#     #   #sample_word=0
#     #   indices=tf.ones(shape=(tf.shape(x)[0],tf.shape(x)[1]), dtype=tf.int32) # trick for when there is no resampling.
#     #   sample_word = self.sample_previous_word(indices, x, training)  # dim (B, P, 1, D)
#     # else:
#     #   # greedy method
#     #   sample_word = self.preprocess_words(previous_word, training)  # dim (B,1,D)
#     #   # adding the particle dimension with tf.tile
#     #   sample_word = tf.tile(tf.expand_dims(sample_word, axis=1), [1, self.num_particles, 1, 1])
#
#     # compute the self-attention vectors over x
#     # tensors of dim (B,P,D)
#     #(attn1, K, V) = self.mha1(sample_word, sample_word, sample_word, self.dec_timestep, K, V) # here, sample_word useless.
#     # should be x instead.
#     (attn1, K, V) = self.mha1(x, x, x, self.dec_timestep, K, V)
#
#     # RESAMPLE TRAJECTORIES OF ATTENTION PARAMETERS FROM THE INDICES MATRIX AND THE CURRENT SET OF WEIGHTS.
#     # attn1=tf.gather(attn1, self.ind_matrix, batch_dims=1, axis=2)
#     if resampling:
#       print('resampling trajectories...')
#
#       attn1 = self.resample(attn1, self.ind_matrix) # if z is of shape (B,P,D). use a different resample function.
#       K = self.resample(K, self.ind_matrix)
#       V = self.resample(V, self.ind_matrix)
#
#     # (batch_size, NUM_PARTICLES, target_seq_len, d_model)
#     # demander à Florian s'il faut changer l'ordre des layernorm/FFN.
#     attn1 = self.dropout1(attn1, training=training)
#     print('x', x.shape)
#     out1 = self.layernorm1(attn1 + x)
#
#     ffn_output = self.ffn(out1)  # (batch_size, NUM_PARTICLES, target_seq_len, d_model)
#     ffn_output = self.dropout3(ffn_output, training=training)
#     out3 = self.layernorm3(ffn_output + out1)  # (batch_size, NUM_PARTICLES, target_seq_len, d_model)
#
#     # 3. FOR SMC: compute the new set of weights.
#     if len(tf.shape(TARGET_WORD_ID)) == 1:
#       TARGET_WORD_ID = tf.expand_dims(TARGET_WORD_ID, axis=-1) # shape (B,1)
#     predictions_probas = self.output_layer(out3)  # (batch_size, NUM_PARTICLES, target_vocab_size)
#     sampling_weights = tf.gather(predictions_probas, TARGET_WORD_ID, axis=-1, batch_dims=1)
#
#
#     # FOR SMC: RETURN ADDITIONAL PARAMETERS, THE CURRENT SAMPLING WEIGHTS (wk[l]),
#     # THE ANCESTOR INDICES MATRIX, K, AND V, and the standard deviation (useful to compute the loss)
#     stddev = tf.tile(self.mha1.stddev, multiples=[1, 1, tf.shape(out3)[2], 1])  # to have a shape (B,P,S,D)
#
#     self.dec_timestep += 1  # right place to put it?
#
#     return (out3, K, V), sampling_weights, self.ind_matrix, stddev


###------ ADD A MAIN FUNCTION HERE--------------------------------

if __name__ == "__main__":
  d_model=512
  dff=2048
  num_heads=8
  num_particles=10

  sample_decoder_layer = DecoderLayer(d_model=d_model,
                                      dff=dff,
                                      num_heads=num_heads,
                                      num_particles=num_particles)

  inputs_layer=tf.ones((64, 10, 50, 512), dtype=tf.int32)
  seq_len=tf.shape(inputs_layer)[2]
  mask=create_look_ahead_mask(seq_len)
  sample_decoder_layer_output, stddev = sample_decoder_layer(inputs=inputs_layer, look_ahead_mask=mask, training=False)
  print('output of classic decoder layer', sample_decoder_layer_output.shape)  # (batch_size, target_seq_len, d_model)

  d_model = 512
  num_heads = 8
  dff = 2048
  target_vocab_size = 1000
  num_particles = 5
  max_positional_encoding = 5000

  # #sample_decoder_layer = DecoderLayer_SMC(d_model, num_heads, dff, target_vocab_size, max_positional_encoding,
  #                                         num_particles, layer_num=0)
  #
  # PREV_SAMPL_WEIGHTS = tf.random.uniform(shape=(64, num_particles), maxval=1)
  # K = tf.random.uniform((64, num_particles, 50, 512))
  # V = tf.random.uniform((64, num_particles, 50, 512))
  # TARGET_WORD_ID = tf.constant(34, shape=(64, 1))

  #(sample_decoder_layer_output, K, V), sampling_weights, ind_matrix, stddev = sample_decoder_layer(
    #tf.random.uniform((64, num_particles, 50, 512)), PREV_SAMPL_WEIGHTS, K, V, TARGET_WORD_ID, training=False,
    #look_ahead_mask=None, padding_mask=None)

  # (z, K, V), sampling_weights= sample_decoder_layer(
  # tf.random.uniform((64, num_particles, 1, 512)), PREV_SAMPL_WEIGHTS, K, V, TARGET_WORD_ID, training=False)
  #
  #
  # print('output of SMC decoder layer', z.shape)  # (batch_size, target_seq_len, d_model)
  # print('K',K.shape), print('V', V.shape)

