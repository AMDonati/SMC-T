import tensorflow as tf
import numpy as np

print(tf.__version__)

import utils.transformer_utils as trf_utils

import models.SMC_Transformer.self_attention_SMC as smha


### simple tests on the mask

seq_length=5

mask=trf_utils.create_look_ahead_mask(seq_length)

print(mask)

# test with scaled_dot_product_attention
k=tf.ones(shape=(1,2,1,1,1)) #(B,P, num_heads,D)
q=tf.ones(shape=(1,2,1,1,1))
v=tf.ones(shape=(1,2,1,1,1))

K=tf.ones(shape=(1,2,1,5,1))
V=tf.ones(shape=(1,2,1,5,1))
Z=tf.ones(shape=(1,2,1,5,1))

for t in range(seq_length):
  ZZ, KK, VV=smha.scaled_dot_product_attention(q,k,v,mask,t,K,V,Z)
  print(ZZ)

### simple test on the resampling with the ancestor matrix.
ind_matrix=tf.constant(np.random.choice(np.arange(10), size=10), shape=(1,10,1))

# actually, we only need to store one element for a batch...
ind_matrix=tf.tile(ind_matrix, multiples=[8,10,5])

attn_params=tf.random.uniform(shape=(8,10,5,3))

#attn_resampl_v1=trf_utils.resample(ind_matrix, attn_params)

#print('attn resampled with v1 resampling function', attn_resampl_v1)

attn_resampl_v2=trf_utils.resample_v2(ind_matrix, attn_params)

print('attn resampled with v2 resampling function', attn_resampl_v2)

### test the tensordot and the 3D tensor multiplication & the SMC_loss.py

indices = tf.constant([[0], [2]])
print('indices', indices.shape)
updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    shape = tf.constant([4, 4, 4])
    scatter = tf.scatter_nd(indices, updates, shape)

