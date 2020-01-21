import tensorflow as tf
import numpy as np

from models.SMC_Transformer.SMC_Transformer import Transformer

# --- ON GOING IMPLEMENTATION -----

def evaluate(transformer, input_sentence, eos_token, sample_number):
  '''
  ON-GOING IMPLEMENTATION
  :param transformer: transformer model (class)
  :param input_sentence: sequence of input works > tensor of dim (B, seq_length)
  :param eos_token: end of sentence token to pad the input sentence

  :return:
  '''
  seq_len=tf.shape(input_sentence)[-1]

  # input_sentence
  inp_sent=input_sentence
  sentence=input_sentence + [eos_token]
  # target sentence
  targ_sent=sentence[:,1:]

  # forward pass on the transformer model to get the prediction (log probas) and the final resampling weights
  final_output, dec_output, sampling_weights=transformer(inp_sent, targ_sent, look_ahead_mask=None, training=False)
  # how to get Z0?


  # keep the first seq_length -1 final output to get the K first associated predicted particles
  Z1_k=dec_output[:,:,:seq_len,:] # shape (B,P,seq_len-1,D)


  # get Zk+1
  #z_nextk=dec_output[:,:,seq_len,:] # dim (B, P, 1, V)
  # sample uniformely over the last dimension (vocabulary size)

  #---- OR ----------


  # takes the last word of the sequence for predicting Yk+1:
  next_word_pred=final_output[:,:,-1,:] # shape (B,P,V)

  # take N uniform samples over the vocabulary.
  voc_len=tf.shape(next_word_pred)[-1]
  batch_size=tf.shape(next_word_pred)[0]
  vocab_sample_arr =np.array([np.random.choice(np.arange(voc_len), size=sample_number) for _ in range(batch_size)])
  vocab_sample_tensor=tf.constant(vocab_sample_arr, dtype=tf.int64) # dim (B,V)
  next_word_sampl=tf.gather(next_word_pred, vocab_sample_tensor, batch_dims=1, axis=-1) # dim (B,P,N)

  # weighted sum over the particles
  pred=tf.reduce_sum(sampling_weights*next_word_sampl, axis=1) # dim (B,N)

  # average over the number of N samples
  pred=tf.reduce_mean(pred, axis=-1) # dim (B,)

  return pred

if __name__ == "__main__":
  num_particles = 5
  batch_size = 8
  num_heads = 2
  d_model = 12
  dff = 24
  target_vocab_size = 50
  pe_target = 50
  num_layers = 2

  # dummy_train_dataset=tf.cast(tf.random_uniform(shape=(batch_size,seq_length)), dtype=tf.int32)
  transformer = Transformer(
    num_layers, d_model, num_heads, dff, target_vocab_size,
    pe_target, num_particles)