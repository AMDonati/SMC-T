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
    output (new Z), K, V, attention_weights
  """
  # FOR SMC: K[l]=K0:k[l], V[l]=v0:k[l], q[l]=q[Il]
  if K is not None:
    # compute K(0:k) from K(0:k-1) & k
    # dim of K in the case of multi-head attention: (batch_size, num_particles, num_heads, seq_length, Depth)
    if dec_timestep == tf.shape(K)[3]: # here sequence is the third dimension (because of the head dimension).
      K = tf.concat([K[:, :, :, :dec_timestep-1, :], k], axis=3)
    elif dec_timestep == 0:
      K = tf.concat([k, K[:, :, :, dec_timestep + 1:, :]], axis=3)
    else:
      K = tf.concat([K[:, :, :, :dec_timestep, :], k, K[:, :, :, dec_timestep+1:, :]], axis=3)
  else:
    K = k
  if V is not None:
    # compute the V(0:k) with V(0:k-1) & v
    if dec_timestep == 0:
      V = tf.concat([v, V[:, :, :, dec_timestep+1:, :]], axis=3)
    elif dec_timestep == tf.shape(V)[3]:
      V = tf.concat([V[:, :, :, :dec_timestep-1, :], v], axis=3)
    else:
      V = tf.concat([V[:, :, :, :dec_timestep, :], v, V[:, :, :, dec_timestep+1:, :]], axis=3)
  else:
    V = v

  #TODO (later on): add a mask for the time-window considered.
  matmul_qk = tf.matmul(q, K, transpose_b=True)  # (B, P, H, 1, S)

  # scale matmul_qk
  dk = tf.cast(tf.shape(K)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) # (B,P,H,1,S)

  # softmax to get pi:
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (B, P, 1, S)

  z = tf.matmul(attention_weights, V) #(B,P,H,1,D)

  return (z, K, V), attention_weights

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

    self.wq = tf.keras.layers.Dense(d_model, name='dense_projection_q')
    self.wk = tf.keras.layers.Dense(d_model, name='dense_projection_k')
    self.wv = tf.keras.layers.Dense(d_model, name='dense_projection_v')

    self.dense = tf.keras.layers.Dense(d_model, name='dense_projection_z')

    self.num_particles = num_particles
    self.timestep = dec_timestep
    self.sigma_scalar = sigma
    self.noise = noise

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    (batch_size, num_particle, seq_length, d_model) => (batch_size, num_particle, seq_length, num_heads, depth=d_model/num_heads)
    """
    #TODO: replace here self
    #x = tf.reshape(x, (batch_size, self.num_particles, -1, self.num_heads, self.depth))
    x = tf.reshape(x, (batch_size, tf.shape(x)[1], -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 1, 3, 2, 4])

  def concat_heads(self, x):
    '''concat attention parameters over all heads (and permute dimensions)
    -returns a tensor of shape (B, P, S, D)'''
    scaled_attention = tf.transpose(x, perm=[0, 1, 3, 2, 4])  # (batch_size, NUM_PARTICLES, seq_len_q, num_heads, depth)

    return tf.reshape(scaled_attention,
                      (tf.shape(scaled_attention)[0], tf.shape(scaled_attention)[1], -1,
                       self.d_model))  # (batch_size, NUM_PARTICLES, seq_len_q, d_model)

  def compute_k_q_v_noise(self, mean_k, mean_q, mean_v):
    total_depth = tf.shape(mean_k)[-1]

    if self.noise:
      gaussian_noise_k = tf.random.normal(shape=tf.shape(mean_k), name='gaussian_k') # (B,P,1,D)
      gaussian_noise_q = tf.random.normal(shape=tf.shape(mean_q), name='gaussian_q')
      gaussian_noise_v = tf.random.normal(shape=tf.shape(mean_v), name='gaussian_v')
    else:
      gaussian_noise_k = tf.zeros(shape=tf.shape(mean_k)) # (B,P,1,D)
      gaussian_noise_q = tf.zeros(shape=tf.shape(mean_q))
      gaussian_noise_v = tf.zeros(shape=tf.shape(mean_v))

    if self.sigma_scalar == 'learned':
      diag_k = tf.Variable(tf.linalg.diag(tf.random.uniform(shape=(total_depth,), dtype=tf.float32)), dtype=tf.float32, name='cov_diag_k') # (D,D)
      diag_q = tf.Variable(tf.linalg.diag(tf.random.uniform(shape=(total_depth,), dtype=tf.float32)), dtype=tf.float32, name='cov_diag_q') # (D,D)
      diag_v = tf.Variable(tf.linalg.diag(tf.random.uniform(shape=(total_depth,), dtype=tf.float32)), dtype=tf.float32,name='cov_diag_v')  # (D,D)
      self.sigma_k = tf.matmul(diag_k, diag_k, transpose_b=True)  # (D, D)
      self.sigma_q = tf.matmul(diag_q, diag_q, transpose_b=True)  # (D, D)
      self.sigma_v = tf.matmul(diag_v, diag_v, transpose_b=True)  # (D, D)
      noise_k = tf.tensordot(self.sigma_k, gaussian_noise_k, axes=[0,3])  # shape (D,B,P,1)
      noise_q = tf.tensordot(self.sigma_q, gaussian_noise_q, axes=[0, 3])  # shape (D,B,P,1)
      noise_v = tf.tensordot(self.sigma_v, gaussian_noise_v, axes=[0, 3])  # shape (D,B,P,1)
      noise_k = tf.transpose(noise_k, perm=[1, 2, 3, 0])  # (B,P,1,D)
      noise_v = tf.transpose(noise_v, perm=[1, 2, 3, 0])  # (B,P,1,D)
      noise_q = tf.transpose(noise_q, perm=[1, 2, 3, 0])  # (B,P,1,D)
    else:
      noise_k = tf.scalar_mul(self.sigma_scalar, gaussian_noise_k) # (B,P,1,D)
      noise_q = tf.scalar_mul(self.sigma_scalar, gaussian_noise_q)
      noise_v = tf.scalar_mul(self.sigma_scalar, gaussian_noise_v)

    k = mean_k + noise_k # (B,P,1,D)
    q = mean_q + noise_q
    v = mean_v + noise_v

    # outputting the normalized noise of k,q,v to add it in the computation of the loss.
    if self.sigma_scalar == 'learned':
      sigma_inv_k, sigma_inv_q, sigma_inv_v = tf.linalg.inv(self.sigma_k), tf.linalg.inv(self.sigma_q), tf.linalg.inv(self.sigma_v)
      self.noise_k_norm = tf.tensordot(sigma_inv_k, (k - mean_k), axes=[0, 3])  # shape (D,B,P,1)
      self.noise_q_norm = tf.tensordot(sigma_inv_q, (q - mean_q), axes=[0, 3])  # shape (D,B,P,1)
      self.noise_v_norm = tf.tensordot(sigma_inv_v, (v - mean_v), axes=[0, 3])  # shape (D,B,P,1)
      self.noise_k_norm = tf.transpose(self.noise_k_norm, perm=[1, 2, 3, 0])  # (B,P,1,D)
      self.noise_q_norm = tf.transpose(self.noise_q_norm, perm=[1, 2, 3, 0])  # (B,P,1,D)
      self.noise_v_norm = tf.transpose(self.noise_v_norm, perm=[1, 2, 3, 0])  # (B,P,1,D)
    else:
      self.noise_k_norm = tf.scalar_mul(1/self.sigma_scalar, k - mean_k) # (B,P,1,D)
      self.noise_q_norm = tf.scalar_mul(1/self.sigma_scalar, q - mean_q)
      self.noise_v_norm = tf.scalar_mul(1/self.sigma_scalar, v - mean_v)

    return k,q,v

  def compute_z_noise(self, mean_z):
    total_depth = tf.shape(mean_z)[-1]

    # compute the $\epsilon$ of the reparametrized noise.
    if self.noise:
      gaussian_noise_z = tf.random.normal(shape=tf.shape(mean_z), name='gaussian_noise_z')  # shape (B,P,1,D)
    else:
      gaussian_noise_z = tf.zeros(shape=tf.shape(mean_z), dtype=tf.float32)  # (B,P,1,D)

    # adding a Gaussian noise using the reparametrization trick.
    if self.sigma_scalar == 'learned':
      diag = tf.Variable(tf.linalg.diag(tf.random.uniform(shape=(total_depth,), dtype=tf.float32)), dtype=tf.float32, name='cov_diag_z') # (D,D) initialize sigma as a 'positive' diagonal matrix as a start
      self.sigma_z = tf.matmul(diag, diag, transpose_b=True)  # (D, D) imposing constraints on the form of the covariance matrix to have a semi-definite positive matrix
      noise_z = tf.tensordot(self.sigma_z, gaussian_noise_z, axes=[0,3])  # shape (D,B,P,1) - tensordot multiplication for sigma and epsilon (fixed gaussian noise)
      noise_z = tf.transpose(noise_z, perm=[1, 2, 3, 0])  # (B,P,1,D)
    else:
      noise_z = tf.scalar_mul(self.sigma_scalar, gaussian_noise_z)

    mu = self.dense(mean_z)
    z = mu + noise_z

    if self.sigma_scalar == 'learned':
      sigma_inv = tf.linalg.inv(self.sigma_z)
      self.noise_z_norm = tf.tensordot(sigma_inv, (z - mu), axes=[0, 3])  # shape (D,B,P,1)
      self.noise_z_norm = tf.transpose(self.noise_z_norm, perm=[1,2,3,0]) # (B,P,1,D)
    else:
      self.noise_z_norm = tf.scalar_mul(1/self.sigma_scalar, (z - mu))  # used in the computation of the loss.

    return z

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
    mean_k = self.wk(k)  # (B,P,1,D)
    mean_q = self.wq(q)  # (B,P,1,D)
    mean_v = self.wv(v)  # (B,P,1,D)

    # add noise with reparametrization trick to k,q,v.
    k,q,v = self.compute_k_q_v_noise(mean_k=mean_k, mean_q=mean_q, mean_v=mean_v)
    #k,q,v=mean_k,mean_q,mean_v

    k = self.split_heads(k, batch_size)  # (B,P,H,1,D/H)
    q = self.split_heads(q, batch_size)  # (B,P,H,1,D/H)
    v = self.split_heads(v, batch_size)  # (B,P,H,1,D/H)

    if K is not None:
      K = self.split_heads(K, batch_size)  # (B,P,H,S,D/H)
    if V is not None:
      V = self.split_heads(V, batch_size)  # (B,P,H,S,D/H)

    # compute self_attention for every head:
    (mean_z, K, V), attn_weights = self_attention_SMC(q, k, v, timestep, K, V) # z (B,P,H,1,D/H), (K,V): (B,P,H,S,D/H)

    # concat attention, K, V over all the heads
    mean_z = self.concat_heads(mean_z) # shape (B,P,1,D)
    K = self.concat_heads(K) # shape (B,P,S,D)
    V = self.concat_heads(V) # shape (B,P,S,D)

    # add a dense layer and noise (with reparametrization trick to mean_z
    z = self.compute_z_noise(mean_z=mean_z)

    return (z, K, V), attn_weights # shapes: z (B,P,1,D), K (B,P,S,D), V (B,P,S,D)

if __name__ == "__main__":
  B = 64
  num_particles = 10
  num_heads = 8
  S = 20
  d_model = 512
  dec_timestep = 20
  sigma = 'learned'
  noise = True

  x = tf.ones(shape=(B, num_particles, num_heads, 1, int(d_model/num_heads)))
  K = tf.random.uniform(shape=(B, num_particles, num_heads, S, int(d_model/num_heads)))
  V = tf.random.uniform(shape=(B, num_particles, num_heads, S, int(d_model/num_heads)))

  (temp_out, temp_K, temp_V), attn_weights = self_attention_SMC(x, x, x, dec_timestep, K, V)
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
  (z,K,V), attn_weights = temp_mha(inputs=inputs_mha, timestep=dec_timestep, K=K, V=V)

  print('z', z.shape)
  print('K', K.shape)
  print('V', K.shape)
  print('attention weights', attn_weights.shape)
