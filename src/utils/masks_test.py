import tensorflow as tf
from utils.transformer_utils import create_masks
from utils.transformer_utils import create_look_ahead_mask
from utils.transformer_utils import create_look_ahead_mask_v2

decoder_output=tf.ones(shape=(1,3))

mask=create_look_ahead_mask(3)

print(mask)

scaled_attention_logits=tf.ones(shape=(1,5,3,3))
scaled_attention_logits += (mask * -1e9)

attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

print(attention_weights)

V=tf.random.uniform(shape=(1,5,3,3))

output = tf.matmul(attention_weights, V)

print(output)

### test for SMC_Transformer attention weights:
attention_weights_SMC=tf.random.uniform(shape=(1,5,2,1,3))
mask_SMC=create_look_ahead_mask_v2(3)
print('mask_SMC', mask_SMC)
attention_weights_SMC*=mask_SMC
print('attention weights SMC after masking', attention_weights_SMC)
V=tf.random.uniform(shape=(1,5,2,3,6))
output=tf.matmul(attention_weights_SMC, V)
print(output.shape)
#seq_len=tf.shape(decoder_output)[2]

# for t in range(seq_len):
#   decoder_t=decoder_output[:,:,:t,:]
#   mask_t=create_masks(decoder_t)
#   attn_t=scaled_attention_logits[:,:,:t,:]
#   print('attn at t', attn_t.shape)
#   attn_t+= (mask_t * -1e9)
#   print('attn at t after masking', attn_t.shape)
#   attn_w_t=tf.nn.softmax(scaled_attention_logits, axis=-1)
#   print('attn', attn_w_t)