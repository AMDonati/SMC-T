from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import pyconll, keras, pickle, os, random, nltk, datetime, warnings, gc, urllib.request, zipfile, collections
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import recall_score, precision_score, classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.metrics.classification import UndefinedMetricWarning

from keras.models import Sequential, Model
from keras.layers import Embedding, Dense, Input, concatenate, Layer, Lambda, Dropout, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical

import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer

from tqdm import tqdm_notebook
from IPython.display import Image
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

# First install some extra packages
# ! pip install pyconll
# ! pip install pydot
# ! pip install graphiz
# ! pip install bert-tensorflow



# ----------MODEL BUILDING---------------------------------

def bert_labels(labels):
  train_label_bert = []
  train_label_bert.append('-PAD-')
  for i in labels:
    train_label_bert.append(i)
  train_label_bert.append('-PAD-')
  print('BERT labels:', train_label_bert)


class BertLayer(Layer):
  def __init__(self, output_representation='sequence_output', trainable=True, **kwargs):
    self.bert = None
    super(BertLayer, self).__init__(**kwargs)

    self.trainable = trainable
    self.output_representation = output_representation

  def build(self, input_shape):
    # SetUp tensorflow Hub module
    self.bert = hub.Module(bert_path,
                           trainable=self.trainable,
                           name="{}_module".format(self.name))

    # Assign module's trainable weights to model
    # Remove unused layers and set trainable parameters
    # s = ["/cls/", "/pooler/", 'layer_11', 'layer_10', 'layer_9', 'layer_8', 'layer_7', 'layer_6']
    s = ["/cls/", "/pooler/"]
    self.trainable_weights += [var for var in self.bert.variables[:] if not any(x in var.name for x in s)]

    for var in self.bert.variables:
      if var not in self._trainable_weights:
        self._non_trainable_weights.append(var)

    # See Trainable Variables
    # tf.logging.info("**** Trainable Variables ****")
    # for var in self.trainable_weights:
    #    init_string = ", *INIT_FROM_CKPT*"
    #    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)

    print('Trainable weights:', len(self.trainable_weights))
    super(BertLayer, self).build(input_shape)

  def call(self, inputs, mask=None):
    inputs = [K.cast(x, dtype="int32") for x in inputs]
    input_ids, input_mask, segment_ids = inputs
    bert_inputs = dict(
      input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
    )
    result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
      self.output_representation
    ]
    return result

  def compute_mask(self, inputs, mask=None):
    return K.not_equal(inputs[0], 0.0)

  def compute_output_shape(self, input_shape):
    if self.output_representation == 'pooled_output':
      return (None, 768)
    else:
      return (None, None, 768)

##----------MODEL TRAINING--------------------------

initialize_vars(sess)

t_ini = datetime.datetime.now()

cp = ModelCheckpoint(filepath="bert_tagger.h5",
                     monitor='val_acc',
                     save_best_only=True,
                     save_weights_only=True,
                     verbose=1)

early_stopping = EarlyStopping(monitor = 'val_acc', patience = 5)

history = model.fit([train_input_ids, train_input_masks, train_segment_ids],
                    train_labels,
                    validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_labels),
                    #validation_split=0.3,
                    epochs=EPOCHS,
                    batch_size=16,
                    shuffle=True,
                    verbose=1,
                    callbacks=[cp, early_stopping]
                   )

t_fin = datetime.datetime.now()
print('Training completed in {} seconds'.format((t_fin - t_ini).total_seconds()))

##-------CLASSIFICATION REPORT-----------

model = build_model(MAX_SEQUENCE_LENGTH+2)
model.load_weights('bert_tagger.h5')`


def y2label(zipped, mask=0):
  out_true = []
  out_pred = []
  for zip_i in zipped:
    a, b = tuple(zip_i)
    if a != mask:
      out_true.append(int2tag[a])
      out_pred.append(int2tag[b])
  return out_true, out_pred

##-----MAKE A PREDICTION FOR A TEST SAMPLE-----
y_pred = model.predict([test_input_ids, test_input_masks, test_segment_ids], batch_size=16).argmax(-1)
y_true = test_labels_ids

def make_prediction(i=16):
    note = ''
    sent = []
    print("{:10} {:5} : {:5}".format("Word", "True", "Predicted"))
    print(35*'-')
    for w, true, pred in zip(test_input_ids[i], y_true[i], y_pred[i]):
        if tokenizer.convert_ids_to_tokens([w])[0]!='[PAD]' and \
            tokenizer.convert_ids_to_tokens([w])[0]!='[CLS]' and \
            tokenizer.convert_ids_to_tokens([w])[0]!='[SEP]':
            if int2tag[true] != int2tag[pred]: note='<<--- Error!'
            print("{:10} {:5} : {:5} {:5}".format(tokenizer.convert_ids_to_tokens([w])[0], int2tag[true], int2tag[pred], note))
            note=''

##-----FREQUENT TYPES OF MISTAKES--------------
def find_errors(X, y):
  error_counter = collections.Counter()
  support = 0
  for i in range(test_input_ids.shape[0]):
    for w, true, pred in zip(test_input_ids[i], y_true[i], y_pred[i]):
      if int2tag[true] != '-PAD-':
        if true != pred:
          word = tokenizer.convert_ids_to_tokens([w])[0]
          error_counter[word] += 1
        support += 1
  return error_counter, support