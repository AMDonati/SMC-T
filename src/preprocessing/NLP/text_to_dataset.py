import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

# perplexity metric in Tensorflow: https://gist.github.com/Gregorgeous/dbad1ec22efc250c76354d949a13cec3
# https://stackoverflow.com/questions/44697318/how-to-implement-perplexity-in-keras
# https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3

#TODO: transform this in a word_based language model. (cf tuto from ODSC).
#TODO: add a validation dataset.


def text_to_dataset(file_path, seq_len, train_split, buffer_size, batch_size):

  # Read, then decode for py2 compat.
  text = open(file_path, 'rb').read().decode(encoding='utf-8')
  # length of text is the number of characters in it
  print('Length of text: {} characters'.format(len(text)))

  # Take a look at the first 250 characters in text
  print(" looking at the first 250 characters of the text...", text[:250])

  # The unique characters in the file
  vocab = sorted(set(text))
  print ('{} unique characters...'.format(len(vocab)))
  vocab_size = len(vocab)

  # Creating a mapping from unique characters to indices
  char2idx = {u:i for i, u in enumerate(vocab)}
  idx2char = np.array(vocab)

  text_as_int = np.array([char2idx[c] for c in text])

  print('{')
  for char,_ in zip(char2idx, range(20)):
      print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
  print('  ...\n}')

  # split between training and validation dataset
  text_train, text_val=train_test_split(text_as_int, train_size=train_split, shuffle=False)

  num_samples_train=text_train.shape[0]
  print ('number of training examples: {}'.format(num_samples_train))
  print('number of test examples: {}'.format(text_val.shape[0]))

  # Create training examples / targets
  char_train_dataset = tf.data.Dataset.from_tensor_slices(text_train)
  char_val_dataset=tf.data.Dataset.from_tensor_slices(text_val)
  train_sequences = char_train_dataset.batch(seq_len + 1, drop_remainder=True)
  val_sequences= char_val_dataset.batch(seq_len + 1, drop_remainder=True)

  for item in train_sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))

  def split_input_target(chunk):
      input_text = chunk[:-1]
      target_text = chunk[1:]
      return input_text, target_text

  train_dataset = train_sequences.map(split_input_target)

  val_dataset = val_sequences.map(split_input_target)

  train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
  val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

  return train_dataset, val_dataset, vocab_size, num_samples_train

if __name__ == "__main__":
  file_path = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

  #file_path='/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/shakespeare_short.txt'

  BATCH_SIZE = 50
  BUFFER_SIZE = 500
  TRAIN_SPLIT=0.9
  seq_len=50
  train_dataset, val_dataset, vocab_size, training_samples = text_to_dataset(file_path=file_path, seq_len=seq_len,
                                                           train_split=TRAIN_SPLIT,
                                                           buffer_size=BUFFER_SIZE,
                                                           batch_size=64)
  steps_per_epoch=int(training_samples)/BATCH_SIZE

  #TODO: solve this issue of enumerate here...
  for (batch, (inp, tar)) in enumerate(train_dataset):
    print(inp.shape)




