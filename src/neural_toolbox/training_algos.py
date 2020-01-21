import tensorflow as tf

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

  def crf_loss(logits, labels, mask, num_labels, mask2len):
    """
    :param logits:
    :param labels:
    :param mask2len:each sample's length
    :return:
    """
    # TODO
    with tf.variable_scope("crf_loss"):
      trans = tf.get_variable(
        "transition",
        shape=[num_labels, num_labels],
        initializer=tf.contrib.layers.xavier_initializer()
      )

    log_likelihood, transition = tf.contrib.crf.crf_log_likelihood(logits, labels, transition_params=trans,
                                                                   sequence_lengths=mask2len)
    loss = tf.math.reduce_mean(-log_likelihood)

    return loss, transition


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  return tf.reduce_mean(loss_)

def categorical_ce_with_particules(real, pred, sampling_weights):
  '''
  :param real: targets tensor > shape (B,S)
  :param pred: predictions (particules logits) > shape (B,P,S,V)
  :param sampling_weights: re-sampling weights for last timestep > shape (B,P)
  :return:
  '''
  # tiling the targets to have a shape (B,P,S)
  num_particles=tf.shape(pred)[1]
  if len(tf.shape(real))<3:
    real=tf.expand_dims(real, axis=1)
    real=tf.tile(real, multiples=[1, num_particles,1])
  # creating the mask for masking padded sequences
  mask = tf.math.logical_not(tf.math.equal(real, 0)) # shape (B,P,S)
  loss_=loss_object(real, pred) # shape (B,P,S)
  # masking over padded sequences
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  # mean over sequence elements
  loss_=tf.reduce_mean(loss_, axis=-1) # shape (B,P)
  # weighted sum over number of particles
  loss_=tf.reduce_sum(sampling_weights*loss_, axis=-1)
  # mean over batch elements
  loss=tf.reduce_mean(loss_, axis=0)
  return loss

def binary_ce_with_particules(real, pred, sampling_weights):
  '''
  :param real: targets tensor > shape (B,S)
  :param pred: predictions (particules logits) > shape (B,P,S,1)
  :param sampling_weights: re-sampling weights for last timestep > shape (B,P)
  :return:
  '''
  # tiling the targets to have a shape (B,P,S)
  num_particles=tf.shape(pred)[1]
  if len(tf.shape(real))<3:
    real=tf.expand_dims(real, axis=1)
    real=tf.tile(real, multiples=[1, num_particles,1])
  # creating the mask for masking padded sequences
  mask = tf.math.logical_not(tf.math.equal(real, 0)) # shape (B,P,S)
  loss_=loss_binary=tf.keras.losses.binary_crossentropy(
    y_true=real,
    y_pred=pred,
    from_logits=False,
    label_smoothing=0) # shape (B,P,S)
  # masking over padded sequences
  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  # mean over sequence elements
  loss_=tf.reduce_mean(loss_, axis=-1) # shape (B,P)
  # weighted sum over number of particles
  loss_=tf.reduce_sum(sampling_weights*loss_, axis=-1)
  # mean over batch elements
  loss=tf.reduce_mean(loss_, axis=0)
  return loss

def mse_with_particles(real, pred, sampling_weights):
  #TODO: correct the formula with a rescaled mse when sigma (of the computation weights) is not equal to 1. (see with Sylvain.)
  '''
  :param real: shape (B,P,S,F)
  :param pred: shape (B,P,S,F)
  :param sampling_weights: shape (B,P)
  :return:
  the average mse scalar loss (with a weighted average over the dim number of particles)
  '''
  mse=tf.keras.losses.mse
  loss=mse(y_true=real, y_pred=pred) # shape (B,P,S)
  loss = tf.reduce_mean(loss, axis=-1)  # shape (B,P)
  loss = tf.reduce_sum(sampling_weights * loss) # shape (B,)
  loss = tf.reduce_mean(loss)
  return loss


# #------ old function not working------
# def categorical_crossentropy(real, logits, sampling_weights):
#     '''formula: mean(over batch)[sum(w(m)*-sum(real*log pred))
#     -args:
#       -real: tensor of dim (B,P,S) or dim (B,S)
#       -pred (logits):tensor of dim (B,P,S,V) with V the vocabulary size.
#       -sampling_weights: tensor of dim (B,P)'''
#     num_particles = tf.shape(sampling_weights)[-1]
#     #if len(tf.shape(real)) < 3:
#       #real = tf.tile(real[:, tf.newaxis, :], multiples=[1, num_particles, 1])  # add particles dimension > dim (B,P,S)
#     pred = tf.reduce_max(logits, axis=-1)  # dim (B, P, S)
#     pred = tf.cast(pred, dtype=tf.float32)
#     real = tf.cast(real, dtype=tf.float32)
#     loss = -tf.reduce_sum(real * tf.math.log(pred), axis=-1)  # dim (B,P)
#     # weighted sum using sampling_weights.
#     loss = tf.reduce_sum(sampling_weights * loss, axis=-1)  # dim (B,)
#     # averaging over the batch
#     loss = tf.reduce_mean(loss, axis=0)
#     print(loss.shape)
#     return loss


if __name__ == "__main__":

  real=tf.ones(shape=(8,1,10))
  #real=tf.transpose(real, perm=[1,0,2])
  logits=tf.random.uniform(shape=(8,1,10,50))
  #logits=tf.transpose(logits, perm=[1,0,2,3])
  sampling_weights=tf.ones(shape=(8,1))

  #loss_tens=loss_object(real, logits)
  #print('loss_tens', loss_tens)

  loss=categorical_ce_with_particules(real, logits, sampling_weights)
  print('manual loss', loss)

  loss_=loss_function(tf.squeeze(real, axis=1), tf.squeeze(logits, axis=1))
  print('tf loss', loss_)






