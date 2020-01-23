#TODO: add data_type & task_type for the parameters.
#TODO: add a if loop to choose between the preprocessing between time_series & nlp.
#TODO: test the model.

import tensorflow as tf
from models.SMC_Transformer.transformer_utils import positional_encoding
from neural_toolbox.classic_layers import point_wise_feed_forward_network

def scaled_dot_product_attention(q, k, v, mask):
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
  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, inputs, mask):

    q=inputs[0]
    k=inputs[1]
    v=inputs[2]

    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, inputs, training, look_ahead_mask):
    input=inputs
    inputs=[inputs for _ in range(3)]
    attn1, attn_weights = self.mha1(inputs=inputs, mask=look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training) # (B,S,D)
    # casting x to dtype=tf.float32
    input=tf.cast(input, dtype=tf.float32)
    out1 = self.layernorm1(attn1 + input)

    ffn_output = self.ffn(out1)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out1)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights

class Decoder(tf.keras.layers.Layer):
  '''Class Decoder with the Decoder architecture
  -args
    -num_layers: number of layers in the Decoder
    -d_model: model depth
    -num_heads: number of heads in the multi-attention mechanism
    -dff: output dim of the feedforward network
    -target_vocab_size (for computing the sampling weights for the last layer (or all layers))
    -maxixum_position_encoding: to preprocess the words sequence (addition of positional embeddings)
    -PF_algo: decoder-level (default) or layer-level
    -rate: dropout rate for feed-forward layers.
    '''

  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, data_type, rate=0.1):
    super(Decoder, self).__init__()
    self.d_model = d_model
    self.num_layers = num_layers
    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    if maximum_position_encoding is not None:
      self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

    self.data_type=data_type

  def call(self, inputs, training, look_ahead_mask):
    seq_len = tf.shape(inputs)[1]
    attention_weights = {}
    if self.data_type=='nlp':
      # adding an embedding and positional encoding only if x is a nlp dataset.
      inputs = self.embedding(inputs)  # (batch_size, target_seq_len, d_model)
      inputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
      inputs += self.pos_encoding[:, :seq_len, :]
      inputs = self.dropout(inputs, training=training)

    for i in range(self.num_layers):
      inputs, block = self.dec_layers[i](inputs=inputs, training=training,
                                         look_ahead_mask=look_ahead_mask)

      attention_weights['decoder_layer{}'.format(i + 1)] = block

    # x.shape == (batch_size, target_seq_len, d_model)
    return inputs, attention_weights

"""## Create the Transformer

The transTransformer consists of the decoder and a final linear layer. 
The output of the decoder is the input to the linear layer and its output is returned.
"""

class Transformer(tf.keras.Model):
  def __init__(self, num_layers, d_model, num_heads, dff,
               target_vocab_size, maximum_position_encoding, data_type, rate=0.1):
    super(Transformer, self).__init__()
    self.decoder = Decoder(num_layers=num_layers,
                           d_model=d_model, num_heads=num_heads,
                           dff=dff,
                           target_vocab_size=target_vocab_size,
                           maximum_position_encoding=maximum_position_encoding,
                           data_type=data_type, rate=rate)
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inp, training, look_ahead_mask):

    # dec_output.shape == (batch_size, tar_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
      inputs=inp, training=training, look_ahead_mask=look_ahead_mask)
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

    return final_output, attention_weights

if __name__ == "__main__":
  B= 8
  F = 1
  num_layers = 4
  d_model = 64
  num_heads = 2
  dff = 128
  maximum_position_encoding = None
  data_type = 'time_series'
  C = 12 # binary case
  S=10

  sample_transformer = Transformer(
    num_layers=num_layers, d_model=d_model, num_heads=num_heads, dff=dff,
    target_vocab_size=C,
    maximum_position_encoding=maximum_position_encoding,
    data_type=data_type)

  temp_input = tf.random.uniform((B, S, 1), dtype=tf.float32, minval=0, maxval=200)

  fn_out, _ = sample_transformer(temp_input,
                                 training=False,
                                 look_ahead_mask=None)

  print('model output', fn_out.shape) # (B,P,D)