import numpy as np
import tensorflow as tf

if __name__ == "__main__":

  data_path = "../../data/synthetic_dataset.npy"
  target_path = '../../data/val_data_synthetic.npy'
  predictions_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/azure_synthetic_dataset_exp/time_series_multi_synthetic_heads_1_depth_2_dff_8_pos-enc_None_pdrop_0.1_b_256_cs_True__particles_10_noise_True_sigma_0.05_smc-pos-enc_None/predictions_val_end_of_training.npy'
  output_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/azure_synthetic_dataset_exp/time_series_multi_synthetic_heads_1_depth_2_dff_8_pos-enc_None_pdrop_0.1_b_256_cs_True__particles_10_noise_True_sigma_0.05_smc-pos-enc_None'
  predictions_val = np.load(predictions_path)

  targets_val = np.load(target_path)

  print('predictions val shape', predictions_val.shape)
  print('targets_val shape', targets_val.shape)

  # reshaping predictions_val:
  shape_0 = predictions_val.shape[0]
  shape_1 = predictions_val.shape[1]
  shape_2 = predictions_val.shape[2]
  shape_3 = predictions_val.shape[3]
  predictions_val = np.reshape(predictions_val, newshape=(shape_0 * shape_1, shape_2, shape_3))
  print('predictions val - new shape', predictions_val.shape)
  num_samples_pred = predictions_val.shape[0]

  targets_val_trunc = targets_val[:num_samples_pred,:,:]
  targets_val_trunc = targets_val_trunc[:,1:,0]
  print('targets val truncated shape', targets_val_trunc.shape)

  predictions_val_reshape_path = output_path + '/predictions_val_end_training_reshaped.npy'
  targets_val_truncated_path = output_path + '/targets_val_truncated.npy'

  np.save(predictions_val_reshape_path, predictions_val)
  np.save(targets_val_truncated_path, targets_val_trunc)
