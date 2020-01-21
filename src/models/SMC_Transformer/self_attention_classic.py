import tensorflow as tf
#from utils.Transformer_utils import create_masks

# ----- scaled_dot_product_attention_function & mha function ------------

def self_attention_classic(Q, K, V, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.
  Args:
    q: query shape == (B,P,H,S,D)
    k: key shape == (B,P,H,S,D)
    v: value shape == (B,P,S,H,D)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(Q, K, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(K)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (B,P,H,S,S)

  output = tf.matmul(attention_weights, V)  # (B,P,H,S,D)

  return output

## ------ Multi-head attention ----------------------

class MultiHeadAttention_classic(tf.keras.layers.Layer):
  '''
  multi-head attention mechanism for each layer of the Transformer
  -args:
    -d_model: depth model
    -num_heads: number of heads for the multi-head attention mechanism
    -num_particles: number of particles to generate
    -sigma: constant, 'learned' (for learned noise)
    -
    '''

  def __init__(self, d_model, num_heads, num_particles, sigma):  # 2 arguments added: dec_timestep, mode.
    super(MultiHeadAttention_classic, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

    self.num_particles = num_particles

    self.sigma_scalar=sigma


  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    (batch_size, num_particle, seq_length, d_model) => (batch_size, num_particle, seq_length, num_heads, depth=d_model/num_heads)
    """
    x = tf.reshape(x, (batch_size, self.num_particles, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 1, 3, 2, 4])

  def concat_heads(self, x):
    '''concat attention parameters over all heads (and permute dimensions)
    -returns a tensor of shape (B, P, S, D)'''
    scaled_attention = tf.transpose(x, perm=[0, 1, 3, 2, 4])  # (batch_size, NUM_PARTICLES, seq_len_q, num_heads, depth)

    return tf.reshape(scaled_attention,
                      (tf.shape(scaled_attention)[0],
                       tf.shape(scaled_attention)[1],
                       -1,
                       self.d_model))  # (batch_size, NUM_PARTICLES, seq_len_q, d_model)

  def call(self, inputs, mask, seed=123):
    '''
    -Args:
      -v,k,q: v(k), k(k), q(k): attention parameters (over all heads) @ current decoding timestep. > shape (B,P,D)
      -mask: look_ahead mask.
      -K,V,Z: KO:k, V0:k, Z0:k: total length attention parameters (until decoding timestep) > shape (B, P, S, D)
    -Returns:
      -K:0:k+1, V0:k+1, Z0:k+1
      -attention_weights
    '''
    q=inputs[0]
    k=inputs[1]
    v=inputs[2]

    batch_size = tf.shape(v)[0]

    # computing the Q,K,V from the v,k,q parameters.
    #TODO: debug here issue of dtype for q.
    #q=tf.cast(q, dtype=tf.int32)
    Q = self.wq(q)  # (batch_size, NUM_PARTICLES, d_model) # this one does not work... strange.
    K = self.wk(k)  # (batch_size, NUM_PARTICLES, d_model) # ok works.
    V = self.wv(v)  # (batch_size, NUM_PARTICLES, d_model) # ok works.

    # splitting heads to do multi_head attention.
    Q = self.split_heads(Q, batch_size)  # (batch_size, NUM_PARTICLES, num_heads, depth)
    K = self.split_heads(K, batch_size)  # (batch_size, NUM_PARTICLES, num_heads, depth)
    V = self.split_heads(V, batch_size)  # (batch_size, NUM_PARTICLES, num_heads, depth)

    # FOR SMC: attention_weights.shape == (batch_size, NUM_PARTICLES, num_heads, seq_len_q, seq_len_k)
    #TODO: add a mask for the time-window considered.
    scaled_attention= self_attention_classic(Q, K, V, mask)
    # concat attention, K, V over all the heads
    concat_attention = self.concat_heads(scaled_attention) # shape (B,P,D) or (B,P,1,D)

    # COMPUTE THE REPARAMETRIZATION TRICK
    total_depth = tf.shape(concat_attention)[-1]

    # initialize sigma as a 'positive' diagonal matrix as a start
    if self.sigma_scalar=='learned':
      self.sigma=tf.Variable(tf.linalg.diag(tf.random.uniform(shape=(total_depth,), dtype=tf.float32)), dtype=tf.float32)
      # apply tf.stop_gradient on sigma to avoid backprop for this set of parameters
      self.sigma=tf.stop_gradient(self.sigma)
      self.sigma = tf.Variable(tf.linalg.diag(tf.random.uniform(shape=(total_depth,))), dtype=tf.float32)
    else:
      sigma_tensor=tf.constant(self.sigma_scalar, shape=(total_depth,), dtype=tf.float32)
      self.sigma = tf.Variable(tf.linalg.diag(sigma_tensor), dtype=tf.float32)

    self.stddev = tf.random.normal(shape=tf.shape(concat_attention), seed=seed, name='stddev')
    # tensordot multiplication for sigma and epsilon (fixed gaussian noise)
    stddev = tf.tensordot(self.sigma, self.stddev, axes=[0, 3])  # shape (D,B,1,D)
    # permuting dimensions to have a tensor of shape (B, P, 1, D)
    stddev = tf.transpose(stddev, perm=[1, 2, 3, 0])

    #TODO: check that if self.sigma_scalar=0, self.stddev is null.
    output = self.dense(concat_attention) + stddev

    # THE OUTPUT IS ALSO THE VARIABLE Z (CONCATENATION OF THE Z OF EACH HEAD)
    # FOR SMC: OUTPUT SHAPE (batch_size, NUM_PARTICLES, seq_len_q, d_model)
    return (output, K, V)  # attention_weights


## add a main function for testing here
if __name__ == "__main__":

  x = tf.ones(shape=(64, 10, 8, 1, 64))
  K = tf.random.uniform(shape=(64, 10, 8, 50, 64))
  V = tf.random.uniform(shape=(64, 10, 8, 50, 64))

  output = self_attention_classic(x, x, x,mask=None)
  print('temp_out', output.shape)

  """Create a `MultiHeadAttention` layer to try out. 
  At each location in the sequence, `y`, the `MultiHeadAttention` 
  runs all 8 attention heads across all other locations in the sequence, returning a new vector of the same length at each location."""

  temp_mha = MultiHeadAttention_classic(d_model=512, num_heads=8, num_particles=10, sigma=1)
  y = tf.ones((64, 10, 512), dtype=tf.float32)  # (batch_size, encoder_sequence, d_model)
  K = tf.random.normal((64, 10, 50, 512))
  V = tf.random.normal((64, 10, 50, 512))
  inputs_mha=[y for _ in range(3)]
  (z, K, V) = temp_mha(inputs=inputs_mha, mask=None)
  # out.shape, attn.shape, K.shape, V.shape
  print(z.shape)
  # print(attn.shape) # shape (B,P,num_heads,seq_length, seq_length)
  print(K.shape), print(V.shape)

#--------old code snippets--
# to remove.
# if K is not None:
# K = self.split_heads(K, batch_size)  # (batch_size, NUM_PARTICLES, num_heads, seq_len_k, depth)
# if V is not None:
# V = self.split_heads(V, batch_size)  # (batch_size, NUM_PARTICLES, num_heads, seq_len_v, depth)

# useless - to remove.
# if Z is not None:
# Z = self.split_heads(Z, batch_size)
