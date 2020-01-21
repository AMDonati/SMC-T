import tensorflow as tf
from models.SMC_Transformer.self_attention_SMC import self_attention_SMC
#from models.Baselines.Transformer_without_enc import scaled_dot_product_attention
#from models.Baselines.Transformer_without_enc import create_look_ahead_mask

b=1
P=5
S=3
D=6
h=1

q=3*tf.ones(shape=(b,P,h,1,D))
k=5*tf.ones(shape=(b,P,h,1,D))
v=8*tf.ones(shape=(b,P,h,1,D))

Q=2*tf.ones(shape=(b,P,h,S,D))
K=tf.ones(shape=(b,P,h,S,D))
V=tf.random.uniform(shape=(b,P,h,S,D))
Z=3*tf.ones(shape=(b,P,h,S,D))

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(Q, K, V, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(Q, K, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, V)  # (..., seq_len_q, depth_v)

  return output, attention_weights

if __name__ == "__main__":

  ZZ, KK, VV=self_attention_SMC(q,k,v,1,K,V)
  print('output SMC', ZZ.shape)

  mask=create_look_ahead_mask(tf.shape(V)[3])

  output, _=scaled_dot_product_attention(Q,K,V,mask)
  print('output classic', output.shape)

# z seems to be V instead of being tf.matmul(attention_weights,V) > to check.