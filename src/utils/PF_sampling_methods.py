import tensorflow as tf

# to change.

def _stratified_sampling_V2(self, weights):
        N=self.num_samples
        indexes=[]
        j=tf.constant(0)
        cum_weights=tf.gather(weights, j)

        def update_j(j, cum_w, weights):
            j = j + 1
            cum_w = cum_w + tf.gather(weights, j)

            return j, cum_w, weights

        for i in range(N):
            subdivision=tf.random_uniform(minval=i/N, maxval=(i+1)/N, shape=())
            j, cum_weights, weights= tf.while_loop(cond=lambda l, *_: tf.less(cum_weights, subdivision),
                                             body=update_j,
                                             loop_vars=[j, cum_weights, weights])
            indexes.append(j)

        indexes=tf.stack(indexes)
        return indexes

def _stratified_sampling(self, weights):
    N=self.num_samples
    indexes = tf.constant(0, shape=[1])
    i = 0
    j = tf.constant(0)
    subdivisions = [tf.random_uniform(minval=i / N, maxval=(i + 1) / N, shape=()) for i in range(N)]
    subdivisions=tf.stack(subdivisions)
    cum_weights = [tf.reduce_sum(weights[:i]) for i in range(N)]
    cum_weights=tf.stack(cum_weights)

    def body(i, j, indexes):
        def indexes_f(i, j, indexes):
            indexes = tf.concat([indexes, tf.reshape(j, shape=[1])], axis=0)
            return i + 1, j, indexes

        def f_j(i, j, indexes):
            return i, j + tf.constant(1), indexes

        i, j, indexes = tf.cond(pred=tf.less(tf.gather(subdivisions,i), tf.gather(cum_weights,j)),
                                true_fn=lambda: indexes_f(i, j, indexes),
                                false_fn=lambda: f_j(i, j, indexes))
        return i, j, indexes

    i, _, indexes = tf.while_loop(cond=lambda i, *_: tf.less(i, N),
                                  body=body,
                                  loop_vars=[i, j, indexes],
                                  shape_invariants=[tf.TensorShape(dims=()), tf.TensorShape(dims=()),
                                                    tf.TensorShape([None])])

    indexes = indexes[1:]  # trick to remove the initialization of the indexes tensor
    return indexes

def _residual_sampling(self, weights):
    N=self.num_samples
    floor=tf.floor(tf.math.scalar_mul(N, weights))
    floor=tf.cast(floor, dtype=tf.int32)
    indexes=[]
    # takes the floor(Nwi) copies of ith indices
    for i in range(N):
        indexes=indexes + [tf.constant(i)]*floor[i]
    indexes_tensor=tf.stack(indexes)

    # residual samples and weights
    #Nr = tf.constant(N) - tf.reduce_sum(floor, axis=-1)
    Nr = tf.constant(N)-indexes.shape[0]
    res = tf.math.scalar_mul(N, weights) - tf.cast(floor, dtype=tf.float32)
    res = res / tf.reduce_sum(res, axis=-1)

    indexes_res = tf.multinomial(tf.expand_dims(res, axis=0), Nr, output_dtype=tf.int32)
    indexes_res = tf.squeeze(indexes_res, axis=0)
    indexes = tf.concat([indexes_tensor, indexes_res], axis=0)

    return indexes

def compute_ancestral_index(prev_sampling_weights, uniform_distribution):
  # FUNCTION ACTUALLY NOT USED....
  '''
  -args:
    -prev_sampling_weights: float tensor of shape (B,P)
    -uniform distribution: float tensor of dimension (B,P)
  -returns:
    -the current set of M indices > tensor of shape (B,P)
  '''
  batch_size = tf.shape(prev_sampling_weights)[0]
  num_particles = tf.shape(prev_sampling_weights)[1]

  # compute w_bar:
  W_0 = tf.expand_dims(tf.constant(0, dtype=tf.float32, shape=(batch_size,)), axis=-1)
  W_m = [tf.reduce_sum(prev_sampling_weights[:, :m], axis=-1) for m in range(num_particles)]
  W_m = tf.stack(W_m, axis=-1)
  W_m = tf.concat([W_0, W_m], axis=-1)

  indices_func = np.zeros(shape=tf.shape(prev_sampling_weights))

  # TRY TO REMOVE THIS DOUBLE FOR LOOP!!! > see this github repo as an example:
  # https://github.com/rlabbe/filterpy/blob/master/filterpy/monte_carlo/resampling.py
  for b in range(batch_size):
    for i in range(num_particles):
      unif = uniform_distribution[b, i]
      if unif >= W_m[b, i] and unif <= W_m[b, i + 1]:
        indices_func[b, i] = i + 1

  indices_func = tf.convert_to_tensor(indices_func, dtype=tf.int32)

  ancestral_index = tf.stack([tf.reduce_sum(indices_func[:, :i], axis=-1) for i in range(num_particles)], axis=-1)
  ancestral_index = tf.cast(ancestral_index, tf.int32)

  return ancestral_index  # tf.stop_gradient(ancestral_index)
