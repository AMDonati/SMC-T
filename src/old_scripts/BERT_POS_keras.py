# from https://github.com/soutsios/pos-tagger-bert

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


# load dataset
UD_ENGLISH_TRAIN = 'en_partut-ud-train.conllu'
UD_ENGLISH_DEV = 'en_partut-ud-dev.conllu'
UD_ENGLISH_TEST = 'en_partut-ud-test.conllu'


def download_files():
  print('Downloading English treebank...')
  urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-dev.conllu', 'en_partut-ud-dev.conllu')
  urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-test.conllu', 'en_partut-ud-test.conllu')
  urllib.request.urlretrieve('http://archive.aueb.gr:8085/files/en_partut-ud-train.conllu', 'en_partut-ud-train.conllu')
  print('Treebank downloaded.')

download_files()

### Pre-processing
def read_conllu(path):
  data = pyconll.load_from_file(path)
  tagged_sentences = []
  t = 0
  for sentence in data:
    tagged_sentence = []
    for token in sentence:
      if token.upos and token.form:
        t += 1
        tagged_sentence.append((token.form.lower(), token.upos))
    tagged_sentences.append(tagged_sentence)
  return tagged_sentences

### some useful functions
# Some usefull functions
def tag_sequence(sentences):
    return [[t for w, t in sentence] for sentence in sentences]

def text_sequence(sentences):
    return [[w for w, t in sentence] for sentence in sentences]

## build dictionary with tag vocabulary
tags = set([item for sublist in train_sentences+test_sentences+val_sentences for _, item in sublist])
print('TOTAL TAGS: ', len(tags))

tag2int = {}
int2tag = {}

for i, tag in enumerate(sorted(tags)):
    tag2int[tag] = i+1
    int2tag[i+1] = tag

# Special character for the tags
tag2int['-PAD-'] = 0
int2tag[0] = '-PAD-'

n_tags = len(tag2int)
print('Total tags:', n_tags)

# parameters
MAX_SEQUENCE_LENGTH = 70
EPOCHS = 30

#-------------special processing for NNs-----------------------------------

train_sentences = read_conllu(UD_ENGLISH_TRAIN)
val_sentences = read_conllu(UD_ENGLISH_DEV)
test_sentences = read_conllu(UD_ENGLISH_TEST)


def split(sentences, max):
  new = []
  for data in sentences:
    new.append(([data[x:x + max] for x in range(0, len(data), max)]))
  new = [val for sublist in new for val in sublist]
  return new



