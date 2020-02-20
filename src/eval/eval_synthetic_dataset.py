import numpy as np
import tensorflow as tf

data_path = "../../data/synthetic_dataset.npy"
target_path = '../../data/val_data_synthetic.npy'
output_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/azure_synthetic_dataset_exp/time_series_multi_synthetic_heads_1_depth_2_dff_8_pos-enc_None_pdrop_0.1_b_256_cs_True__particles_10_noise_True_sigma_0.05_smc-pos-enc_None/model_outputs/smc_predictions.npy'

predictions_val = np.load(output_path)

targets_val = np.load(target_path)

print('predictions val shape', predictions_val.shape)
print('targets_val shape', targets_val.shape)