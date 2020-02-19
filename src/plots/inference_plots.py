import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

  eval_output_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/exp_162_grad_not_zero_azure/time_series_multi_unistep-forcst_heads_1_depth_3_dff_12_pos-enc_50_pdrop_0.1_b_1048_cs_True__particles_25_noise_True_sigma_0.1_smc-pos-enc_None/eval_outputs'

  predictions_N_1_uni_step_path = eval_output_path + '/' + 'pred_unistep_N_1_test.npy'
  predictions_N_1_uni_step_path_unnorm = eval_output_path + '/' + 'pred_unistep_N_1_test_unnorm.npy'
  targets_test_path = eval_output_path + '/' + 'targets_test.npy'
  targets_test_path_unnorm = eval_output_path + '/' + 'targets_test_unnorm.npy'


  # de-normalisation:


  # load prediction file:
  predictions_P_unistep = np.load(predictions_N_1_uni_step_path)
  targets_test = np.load(targets_test_path)
  predictions_P_unistep_unnorm = np.load(predictions_N_1_uni_step_path_unnorm)
  targets_test_unnorm = np.load(targets_test_path_unnorm)

  # mean prediction
  mean_prediction_unistep = np.mean(predictions_P_unistep, axis=1)
  print('pred shape', predictions_P_unistep.shape)
  print('pred unnorm shape', predictions_P_unistep_unnorm.shape)
  print('mean pred shape', mean_prediction_unistep.shape)
  print('targets shape', targets_test.shape)
  print('targets unnorm shape', targets_test_unnorm.shape)

  # abs diff between targets & mean pred
  diff_pred_targ = np.absolute(targets_test - mean_prediction_unistep)
  print('diff shape', diff_pred_targ.shape)
  std_prediction_unistep = np.std(predictions_P_unistep, axis=1)
  print('std pred shape', std_prediction_unistep.shape)


