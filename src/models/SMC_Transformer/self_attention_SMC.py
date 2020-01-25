import tensorflow as tf

# ----- scaled_dot_product_attention_function & mha function ------------

def self_attention_SMC(q, k, v, dec_timestep, K=None, V=None):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.
  Args:
    q: query shape == (..., num_particles, depth) for sampled word
    k: key shape == (..., num_particles, depth) for sampled word
    v:value shape == (..., num_particles, depth_v) for sampled word
    K: key shape == (..., num_particles, seq_len_k, depth)
    V: value shape == (..., num_particles, seq_len_v, depth_v)
  Returns:
    output (new Z), attention_weights, K, V
  """

  # FOR SMC: SHAPE OF Q (..., NUM_PARTICLES, seq_len_q, depth)
  # SHAPE OF K (..., NUM_PARTICLES, seq_len_k, depth)
  # SHAPE OF V (..., NUM_PARTICLES, seq_len_v, depth_v)

  # FOR SMC: K[l]=K0:k[l], V[l]=v0:k[l], q[l]=q[Il]
  if K is not None:
    # compute K(0:k) from K(0:k-1) & k
    # dim of K in the case of multi-head attention: (batch_size, num_particles, num_heads, seq_length, Depth)
    if K.shape[3] == dec_timestep:
      K = tf.concat([K[:, :, :, :dec_timestep, :], k], axis=3)
    elif dec_timestep == 0:
      K = tf.concat([k, K[:, :, :, dec_timestep + 1:, :]], axis=3)
    else:
      K = tf.concat([K[:, :, :, :dec_timestep, :], k, K[:, :, :, dec_timestep + 1:, :]], axis=3)
  else:
    K = k
  if V is not None:
    # compute the V(0:k) with V(0:k-1) & v
    if dec_timestep == 0:
      V = tf.concat([v, V[:, :, :, dec_timestep + 1:, :]], axis=3)
    elif V.shape[3] == dec_timestep:
      V = tf.concat([V[:, :, :, :dec_timestep, :], v], axis=3)
    else:
      V = tf.concat([V[:, :, :, :dec_timestep, :], v, V[:, :, :, dec_timestep + 1:, :]], axis=3)
  else:
    V = v

  #TODO: add a mask for the time-window considered.
  matmul_qk = tf.matmul(q, K, transpose_b=True)  # (..., num_particles,H, 1, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(K)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) # (B,P,H,1,S)

  # softmax to get pi:
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., num_particles, 1, seq_len_k)

  z = tf.matmul(attention_weights, V) #(B,P,H,1,D)

  return (z, K, V)  # attention_weights

## ------'SMC' Multi-head attention class------------------------------------------------------------------

class MultiHeadAttention_SMC(tf.keras.layers.Layer):
  '''
  multi-head attention mechanism for each layer of the Transformer
  -args:
    -d_model: depth model
    -num_heads: number of heads for the multi-head attention mechanism
    -num_particles: number of particles to generate
    -dec_timestep: current decoding timestep (=k) for the sequential mechanism
    -sigma: constant, or 'learned', for learned noise.
    -noise: boolean: True if noise injected in the attention context vector z, False if no noise injected.
    '''

  def __init__(self, d_model, num_heads, num_particles, dec_timestep, sigma, noise):
    super(MultiHeadAttention_SMC, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

    self.num_particles = num_particles
    self.timestep = dec_timestep
    self.sigma_scalar=sigma
    self.noise=noise

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
                      (tf.shape(scaled_attention)[0], tf.shape(scaled_attention)[1], -1,
                       self.d_model))  # (batch_size, NUM_PARTICLES, seq_len_q, d_model)

  def call(self, inputs, timestep, K=None, V=None, seed=123):
    '''
    -Args:
      -v,k,q: v(k), k(k), q(k): attention parameters (over all heads) @ current decoding timestep. > shape (B,P,D)
      -K,V,Z: KO:k, V0:k: total length attention parameters (until decoding timestep) > shape (B, P, S, D)
    -Returns:
      -K:0:k+1, V0:k+1, Z0:k+1
    '''

    q=inputs[0] # (B,P,1,D)
    k=inputs[1] # (B,P,1,D)
    v=inputs[2] # (B,P,1,D)

    batch_size = tf.shape(v)[0]

    # > FOR SMC: q is only the query of the current word: shape (batch_size, num_particles, d_model)
    k = self.wk(k)  # (B,P,1,D)
    v = self.wv(v)  # (B,P,1,D)
    q = self.wq(q)  # (B,P,1,D)

    k = self.split_heads(k, batch_size)  # (B,P,H,1,D/H)
    v = self.split_heads(v, batch_size)  # (B,P,H,1,D/H)
    q = self.split_heads(q, batch_size)  # (B,P,H,1,D/H)

    if K is not None:
      K = self.split_heads(K, batch_size)  # (B,P,H,S,D/H)
    if V is not None:
      V = self.split_heads(V, batch_size)  # (B,P,H,S,D/H)

    # compute self_attention for every head:
    (z, K, V) = self_attention_SMC(q, k, v, timestep, K, V) # z (B,P,H,1,D/H), (K,V): (B,P,H,S,D/H)

    # concat attention, K, V over all the heads
    z = self.concat_heads(z) # shape (B,P,1,D)
    K = self.concat_heads(K) # shape (B,P,S,D)
    V = self.concat_heads(V) # shape (B,P,S,D)

    # --------------------------------------------------Add the noise using the reparametrization trick------------------------------------------------------------------------------

    total_depth = tf.shape(z)[-1]

    # adding a Gaussian noise using the reparametrization trick.

    #TODO: Show to Sylvain how this is done in the VAE tutorial to compare with my method. https://www.tensorflow.org/tutorials/generative/cvae
    # initialize sigma as a 'positive' diagonal matrix as a start
    if self.sigma_scalar=='learned':
      self.sigma=tf.Variable(tf.linalg.diag(tf.random.uniform(shape=(total_depth,), dtype=tf.float32)), dtype=tf.float32)
      # apply tf.stop_gradient on sigma to avoid backprop for this set of parameters
      self.sigma=tf.stop_gradient(self.sigma)
      self.sigma = tf.Variable(tf.linalg.diag(tf.random.uniform(shape=(total_depth,))), dtype=tf.float32)
    else:
      sigma_tensor=tf.constant(self.sigma_scalar, shape=(total_depth,), dtype=tf.float32)
      self.sigma = tf.Variable(tf.linalg.diag(sigma_tensor), dtype=tf.float32)


    #compute the $\epsilon$ of the reparametrized noise.
    if self.noise:
      self.stddev = tf.random.normal(shape=tf.shape(z), seed=seed, name='stddev') # shape (B,P,1,D)
    else:
      self.stddev=tf.zeros(shape=tf.shape(z), dtype=tf.float32)

    # tensordot multiplication for sigma and epsilon (fixed gaussian noise)
    stddev = tf.tensordot(self.sigma, self.stddev, axes=[0, 3]) # shape (D,B,1,D)
    # permuting dimensions to have a tensor of shape (B, P, 1, D)
    stddev = tf.transpose(stddev, perm=[1, 2, 3, 0])

    z = self.dense(z) + stddev

    return (z, K, V) # shapes: z (B,P,1,D), K (B,P,S,D), V (B,P,S,D)

if __name__ == "__main__":
  B=64
  num_particles=10
  num_heads=8
  S=50
  d_model=512
  dec_timestep=0
  sigma=1
  noise=False

  x = tf.ones(shape=(B, num_particles, num_heads, 1, int(d_model/num_heads)))
  K = tf.random.uniform(shape=(B, num_particles, num_heads, S, int(d_model/num_heads)))
  V = tf.random.uniform(shape=(B, num_particles, num_heads, S, int(d_model/num_heads)))

  (temp_out, temp_K, temp_V) = self_attention_SMC(x, x, x, dec_timestep, K, V)
  print('temp_out', temp_out.shape)
  print('temp_K', temp_K.shape)
  print('temp_V', temp_V.shape)

  """Create a `MultiHeadAttention` layer to try out. 
  At each location in the sequence, `y`, the `MultiHeadAttention` 
  runs all 8 attention heads across all other locations in the sequence, returning a new vector of the same length at each location."""

  temp_mha = MultiHeadAttention_SMC(d_model=d_model,
                                    num_heads=num_heads,
                                    num_particles=num_particles,
                                    dec_timestep=dec_timestep,
                                    sigma=sigma,
                                    noise=noise)

  X_mha = tf.ones(shape=(B, num_particles, 1, d_model), dtype=tf.float32)
  inputs_mha = [X_mha for _ in range(3)]
  K = tf.random.uniform(shape=(B, num_particles, S, d_model))
  V = tf.random.uniform(shape=(B, num_particles, S, d_model))
  z,K,V=temp_mha(inputs=inputs_mha, timestep=dec_timestep, K=K, V=V)

  print('z', z.shape)
  print('K', K.shape)
  print('V', K.shape)
