import matplotlib.pypplot as plt
import tensorflow as tf

#TODO: adapt this function for our datasets.
def plot_attention_weights(attention, sentence_tknzed, result, layer):
  fig = plt.figure(figsize=(16, 8))

  sentence = tokenizer_pt.encode(sentence)

  attention = tf.squeeze(attention[layer], axis=0)

  for head in range(attention.shape[0]):
    ax = fig.add_subplot(2, 4, head + 1)

    # plot the attention weights
    ax.matshow(attention[head][:-1, :], cmap='viridis')

    fontdict = {'fontsize': 10}

    ax.set_xticks(range(len(sentence) + 2))
    ax.set_yticks(range(len(result)))

    ax.set_ylim(len(result) - 1.5, -0.5)

    ax.set_xticklabels(
      ['<start>'] + [tokenizer_pt.decode([i]) for i in sentence] + ['<end>'],
      fontdict=fontdict, rotation=90)

    ax.set_yticklabels([tokenizer_en.decode([i]) for i in result
                        if i < tokenizer_en.vocab_size],
                       fontdict=fontdict)

    ax.set_xlabel('Head {}'.format(head + 1))

  plt.tight_layout()
  plt.show()