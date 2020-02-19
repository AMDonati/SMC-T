import tensorflow as tf
import numpy as np


def generate_onesample(A,cov_matrix,seq_len):
  X = tf.random.normal(shape=(1,2))
  list_X=[X]
  for s in range(seq_len):
    X = tf.matmul(X,A) + tf.random.normal(stddev=cov_matrix, shape=(1,2))
    list_X.append(X)
  X_obs = tf.stack(list_X, axis=1)
  return X_obs

if __name__ == "__main__":
  seq_len = 24
  BATCH_SIZE = 64
  num_samples = 50000
  #cov_matrix = tf.linalg.diag(tf.constant([0.2, 0.3], dtype=tf.float32))
  cov_matrix = tf.constant([0.2, 0.3], dtype=tf.float32)
  A = tf.constant([[0.8, 0.1], [0.2, 0.9]], dtype=tf.float32)

  X_temp = tf.ones(shape=(1,2))
  XX = tf.matmul(X_temp, A)
  noise = tf.random.normal(stddev=cov_matrix, shape=(1,2))
  XX = XX + noise
  list_samples=[]
  for N in range(num_samples):
    X_seq = generate_onesample(A=A, cov_matrix=cov_matrix, seq_len=seq_len)
    list_samples.append(X_seq)
  X_data = tf.stack(list_samples, axis=0)
  X_data=tf.squeeze(X_data, axis=1)
  data_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/data'
  file_path = data_path + '/synthetic_dataset.npy'
  np.save(file_path, X_data)
  print('X data', X_data.shape)
