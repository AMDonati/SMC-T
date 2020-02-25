import tensorflow as tf


def build_GRU_for_classification(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)])
  return model

def build_LSTM_for_regression(rnn_units, dropout_rate):
  simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(rnn_units,
                         return_sequences=True),
    tf.keras.layers.Dropout(rate=dropout_rate),
    tf.keras.layers.Dense(1)])

  return simple_lstm_model
