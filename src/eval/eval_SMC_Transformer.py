import tensorflow as tf
import numpy as np
import json

from models.SMC_Transformer.SMC_Transformer import SMC_Transformer
from eval.inference_SMC_Transformer import evaluate_one_timestep
from preprocessing.time_series.df_to_dataset import df_to_data_regression

# -------- load config file and checkpoint --------------------------------------
out_path='/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/exp_reg_162_loss_modified_grad_not0/time_series_multi_unistep-forcst_heads_1_depth_12_dff_48_pos-enc_50_pdrop_0.1_b_1048_cs_True__particles_5_noise_True_sigma_0.1_smc-pos-enc_None/'
ckpt_path = out_path + 'checkpoints/SMC_transformer/ckpt-30'
config_path = out_path + 'config_nlp.json'

# get hparams from config file:
with open(config_path) as f:
  hparams = json.load(f)

# model params
num_layers = hparams["model"]["num_layers"]
num_heads = hparams["model"]["num_heads"]
d_model = hparams["model"]["d_model"]
dff = hparams["model"]["dff"]
rate = hparams["model"]["rate"]  # p_dropout
max_pos_enc_bas_str = hparams["model"]["maximum_position_encoding_baseline"]
maximum_position_encoding_baseline = None if max_pos_enc_bas_str == "None" else max_pos_enc_bas_str
max_pos_enc_smc_str = hparams["model"]["maximum_position_encoding_smc"]
maximum_position_encoding_smc = None if max_pos_enc_smc_str == "None" else max_pos_enc_smc_str

# smc params
num_particles = hparams["smc"]["num_particles"]
noise_encoder_str = hparams["smc"]["noise_encoder"]
noise_encoder = True if noise_encoder_str == "True" else False
noise_SMC_layer_str = hparams["smc"]["noise_SMC_layer"]
noise_SMC_layer = True if noise_SMC_layer_str == "True" else False
sigma = hparams["smc"]["sigma"]
# computing manually resampling parameter
resampling = False if num_particles == 1 else True

# optim params
BATCH_SIZE = hparams["optim"]["BATCH_SIZE"]
learning_rate = hparams["optim"]["learning_rate"]
EPOCHS = hparams["optim"]["EPOCHS"]
custom_schedule = hparams["optim"]["custom_schedule"]

# task params
data_type = hparams["task"]["data_type"]
task_type = hparams["task"]["task_type"]
task = hparams["task"]["task"]

# adding RNN hyper-parameters
rnn_bs = BATCH_SIZE
rnn_emb_dim = hparams["RNN_hparams"]["rnn_emb_dim"]
rnn_units = hparams["RNN_hparams"]["rnn_units"]

# loading data arguments for the regression case
if task_type == 'regression':
  file_path = hparams["data"]["file_path"]
  fname = hparams["data"]["fname"]
  col_name = hparams["data"]["col_name"]
  index_name = hparams["data"]["index_name"]
  TRAIN_SPLIT = hparams["data"]["TRAIN_SPLIT"]
  history = hparams["data"]["history"]
  step = hparams["data"]["step"]
  target_feature = hparams["data"]["target_feature"]
  if target_feature == "None":
    target_feature = None

seq_len = 12
target_vocab_size = 1

# ------ load test dataset -------------------------------------------------------------------------------------------
(train_data, val_data, test_data), original_df = df_to_data_regression(file_path=file_path,
                                                                       fname=fname,
                                                                       col_name=col_name,
                                                                       index_name=index_name,
                                                                       TRAIN_SPLIT=TRAIN_SPLIT,
                                                                       history=history,
                                                                       step=step)

print('test_data shape', test_data.shape)

# ------ re-create model and load weights -----------------------------------------------------------------------------------

smc_transformer = SMC_Transformer(num_layers=num_layers,
                          d_model=d_model,
                          num_heads=num_heads,
                          dff=dff,
                          target_vocab_size=target_vocab_size,
                          maximum_position_encoding=maximum_position_encoding_smc,
                          num_particles=num_particles,
                          sigma=sigma,
                          noise_encoder=noise_encoder,
                          noise_SMC_layer=noise_SMC_layer,
                          seq_len=seq_len,
                          data_type=data_type,
                          task_type=task_type,
                          resampling=resampling,
                          target_feature=target_feature)

smc_transformer.load_weights(ckpt_path)



