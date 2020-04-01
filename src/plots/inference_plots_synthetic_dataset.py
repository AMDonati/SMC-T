import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import os
import statistics
import math

#TODO: arrange plots formatting (legend: etc...)
#TODO: check form of function if adding 'manually' gaussian pdfs in tf. > ok works.


def compute_mixture_gaussian_pdf(x, pred_means, omega_preds, sampling_weights):
  '''
  x: numpy array of shape (num_samples,)
  pred_means: numpy array of shape (num MC samples, num particles, 1)
  omega_preds: scalar for the covariance value.
  -sampling weights: numpy array of shape (num_particles)
  '''
  N = tf.shape(pred_means)[0]
  P = tf.shape(pred_means)[1]
  dist = tfp.distributions.Normal(loc=pred_means, scale=omega_preds)
  # prepare input x: convert it into a tensor and tile it.
  x = tf.convert_to_tensor(x, dtype=tf.float32)
  x = tf.expand_dims(tf.expand_dims(x, axis=0), axis=0)
  x = tf.tile(x, multiples=[N, P, 1])
  pdf = dist.prob(x) # (N,P, num_samples)
  pdf = tf.reduce_mean(pdf, axis=0) # mean over N: (P, num_samples)
  sampling_weights = tf.expand_dims(sampling_weights, axis=0) # (1,P)
  pdf = tf.matmul(sampling_weights, pdf) # (1,num_samples)
  pdf = tf.squeeze(pdf, axis=0) # (num_samples)

  return pdf

def compute_mixture_gaussian_pdf_2(x, pred_means, omega_preds, sampling_weights):
  '''
    x: numpy array of shape (num_samples,)
    pred_means: numpy array of shape (num MC samples, num particles, 1)
    omega_preds: scalar for the covariance value.
    -sampling weights: numpy array of shape (num_particles)
  '''
  N = tf.shape(pred_means)[0]
  P = tf.shape(pred_means)[1]
  list_pdf_P = []
  for p in range(P):
    list_pdf_N = []
    for n in range(N):
      pred_mean = pred_means[n,p] # scalar.
      dist = tfp.distributions.Normal(loc=pred_mean, scale=omega_preds)
      pdf = dist.prob(x) # (num_samples)
      list_pdf_N.append(pdf)
    pdf_N = tf.stack(list_pdf_N) # (N, num_samples)
    pdf_p = tf.reduce_mean(pdf_N, axis=0) # (num_samples)
    list_pdf_P.append(pdf_p)
  pdf_P = tf.stack(list_pdf_P) # (P, num_samples)
  sampling_weights = tf.expand_dims(sampling_weights, axis=0) # (1, P)
  pdf = tf.matmul(sampling_weights, pdf_P) # (1, num_samples)
  pdf = tf.squeeze(pdf, axis=0)

  return pdf


def plot_one_timestep(pred_means, true_means, sampled_preds, sampling_weights, omega_preds, omega_true_distrib, output_path, mc_dropout_preds=None, baseline_T_preds=None):
    '''
    args:
    -pred_means: numpy array of shape (batch_size, num MC samples, num particles, 1)
    -true_means: numpy array of shape (batch_size)
    -sampled_preds: array of shape (batch_size, N_est)
    -sampling weights: numpy array of shape (batch_size, num_particle)
    - mc_dropout_preds: None or array of shape (batch_size, N_est)
    -baseline_T_preds: array of shape (batch_size)
    '''
    batch_size = pred_means.shape[0]

    # prepare subplots
    fig, axs = plt.subplots(2, 2)
    list_samples = []
    # loop over number of plots
    for i in range(4):
      # select ax for plotting:
      if i == 0:
        ax = axs[0, 0]
        color = 'tab:blue'
      elif i == 1:
        ax = axs[0, 1]
        color = 'tab:orange'
      elif i == 2:
        ax = axs[1, 0]
        color = 'tab:green'
      elif i == 3:
        ax = axs[1, 1]
        color = 'tab:red'

      # draw a sample among the samples of the test set:
      index = np.random.randint(low=0, high=batch_size)
      list_samples.append(index)
      pred_mean = pred_means[index, :, :] # (N,P,1)
      pred_mean_P = tf.reduce_mean(pred_mean, axis=0)
      pred_mean_P = tf.squeeze(pred_mean_P, axis=-1).numpy() #(P)
      true_mean = true_means[index] # scalar.
      sampled_pred = sampled_preds[index,:]
      sampling_weight = sampling_weights[index, :] #(P)
      if mc_dropout_preds is not None:
        mc_dropout_pred = mc_dropout_preds[index, :] # (N_est)
      if baseline_T_preds is not None:
        baseline_T_pred = baseline_T_preds[index] #scalar

      # plot the predicted probability density function.
      x = np.linspace(start=true_mean - 5 * omega_preds, stop=true_mean + 5 * omega_preds, num=100) # (100)
      pdf_predicted = compute_mixture_gaussian_pdf(x=x, pred_means=pred_mean, omega_preds=omega_preds, sampling_weights=sampling_weight)
      ax.plot(x, pdf_predicted, color, lw=2, alpha=0.6, label='predicted pdf for sample number: {}'.format(index))

      # plot the true probability density function.
      true_dist = tfp.distributions.Normal(loc=true_mean, scale=omega_true_distrib)
      ax.plot(x, true_dist.prob(x), color, lw=2, linestyle='dashed', label='true pdf for sample number: {}')

      # plot the predicted empirical distribution:
      ax.hist(sampled_pred, density=True, histtype='stepfilled', alpha=0.2)
      if mc_dropout_preds is not None:
        ax.hist(mc_dropout_pred, color='k', density=True, histtype='stepfilled', alpha=0.8)
      if baseline_T_preds is not None:
        ax.vlines(baseline_T_pred, ymin=0, ymax=2)

      # scatterplot of P means:
      y_mean_P = np.zeros(shape=pred_mean_P.shape)
      ax.scatter(x=pred_mean_P, y=y_mean_P, c=color, marker='x')

    #plt.legend(fontsize=14)
    #plt.title('True pdf versus predicted pdf per timestep for samplne # {}'.format(index), fontsize=16)
    #plt.show()
    fig_path = output_path + '/'+'true_pdf_vs_pred_pdf_one_timestep_samples_{}_{}_{}_{}.png'.format(list_samples[0],
                                                                                                    list_samples[1],
                                                                                                    list_samples[2],
                                                                                                    list_samples[3])
    plt.savefig(fig_path)


def plot_multiple_timesteps(pred_means, true_means, sampled_preds, sampling_weights, omega_preds, omega_true_distrib, output_path, mc_dropout_preds=None, baseline_T_preds=None):
  '''
  args:
  -pred_means: numpy array of shape (num_timesteps, batch_size, num MC samples, num particles, 1)
  -true_means: numpy array of shape (num_timesteps, batch_size)
  -sampled_preds: array of shape (num_timesteps, batch_size, N_est)
  -sampling weights: numpy array of shape (batch_size, num_particle)
  -mc_dropout_preds: array (B,N_est, num_timsteps)
  '''
  num_timesteps = tf.shape(pred_means)[0]

  # prepare plot with multiple subplots:
  fig, axs = plt.subplots(2, 2)
  axs[0, 0].set_title('t+1')
  axs[0, 1].set_title('t+2')
  axs[1, 0].set_title('t+3')
  axs[1, 1].set_title('t+4')

  # for ax in axs.flat:
  #   ax.set(xlabel='x-label', ylabel='y-label')
  # # Hide x labels and tick labels for top plots and y ticks for right plots.
  # for ax in axs.flat:
  #   ax.label_outer()

  # draw a sample among the samples of the test set:
  batch_size = pred_means.shape[1]
  index = np.random.randint(low=0, high=batch_size)
  sampling_weights = sampling_weights[index, :]  # (P) # associated sampling weights.

  for t in range(num_timesteps):
    # select ax for plotting:
    if t == 0:
      ax = axs[0,0]
      color = 'tab:blue'
    elif t == 1:
      ax = axs[0, 1]
      color = 'tab:orange'
    elif t == 2:
      ax = axs[1, 0]
      color = 'tab:green'
    elif t == 3:
      ax = axs[1, 1]
      color = 'tab:red'

    pred_means_t = pred_means[t,:,:,:,:] # (B,N,P,1)
    true_means_t = true_means[t,:] # (B)
    sampled_preds_t = sampled_preds[t,:,:] # (B, N_est)
    if mc_dropout_preds is not None:
      mc_dropout_preds_t=mc_dropout_preds[:,:,t] # (B,N_est)
    if baseline_T_preds is not None:
      baseline_T_preds_t = baseline_T_preds[t,:] # (B)
    pred_means_t = pred_means_t[index, :, :]  # (N,P,1)
    true_mean_t = true_means_t[index]  # scalar.
    sampled_preds_t = sampled_preds_t[index,:] #(N_est)
    if mc_dropout_preds is not None:
      mc_dropout_preds_t = mc_dropout_preds_t[index, :] # (N_est)
    if baseline_T_preds is not None:
      baseline_T_preds_t = baseline_T_preds_t[index]  # scalar.

    # plot the predicted probability density function.
    x = np.linspace(start=true_mean_t - 5 * omega_preds, stop=true_mean_t + 5 * omega_preds, num=100)  # (100)
    pdf_predicted = compute_mixture_gaussian_pdf(x=x, pred_means=pred_means_t, omega_preds=omega_preds, sampling_weights=sampling_weights)
    #ax.plot(x, pdf_predicted, color, lw=5, alpha=0.6, label='predicted pdf for sample #{}'.format(index))
    ax.plot(x, pdf_predicted, color, lw=5, alpha=0.6)

    # plot the true probability density function.
    true_dist = tfp.distributions.Normal(loc=true_mean_t, scale=omega_true_distrib)
    #ax.plot(x, true_dist.prob(x), color, lw=2, linestyle='dashed', label='true pdf for sample #{}'.format(index))
    ax.plot(x, true_dist.prob(x), color, lw=2, linestyle='dashed')

    # plot the predicted empirical distribution:
    ax.hist(sampled_preds_t, density=True, histtype='stepfilled', alpha=0.2)
    if mc_dropout_preds is not None:
      ax.hist(mc_dropout_preds_t, color='k', density=True, histtype='stepfilled', alpha=0.8)
    if baseline_T_preds is not None:
      ax.vlines(baseline_T_preds_t, ymin=0, ymax=2)

  plt.legend(fontsize=10)
  #plt.title('True pdf versus predicted pdf per timestep for sample # {}'.format(index), fontsize=16)
  #plt.show()
  fig_path = output_path + '/' + 'true_pdf_vs_pred_pdf_{}_timesteps_sample{}.png'.format(num_timesteps, index)
  plt.savefig(fig_path)

def plot_multiple_P_one_timestep(list_pred_means, true_means, list_sampled_preds, list_sampling_weights, omega_preds, omega_true_distrib, output_path):
    '''
    args:
    -list_pred_means: list of numpy array of shape (batch_size, num MC samples, num particles, 1)
    -true_means: numpy array of shape (batch_size)
    -list_sampled_preds: list of arrays of shape (batch_size, N_est)
    -list_sampling weights: list of numpy array of shape (batch_size, num_particle)
    '''
    length_list = len(list_pred_means)
    batch_size = list_pred_means[0].shape[0]
    # prepare subplots
    fig, ax = plt.subplots(1, 1)
    # draw a sample among the samples of the test set:
    index = np.random.randint(low=0, high=batch_size)
    true_mean = true_means[index]  # scalar.
    x = np.linspace(start=true_mean - 5 * omega_preds, stop=true_mean + 5 * omega_preds, num=100)  # (100)
    list_colors = ['b', 'r', 'g', 'y']

    list_num_particles = []
    for pred_means, sampled_preds, sampling_weights, color in zip(list_pred_means, list_sampled_preds, list_sampling_weights, list_colors):
      pred_mean = pred_means[index, :, :]  # (N,P,1)
      pred_mean_P = tf.reduce_mean(pred_mean, axis=0)
      pred_mean_P = tf.squeeze(pred_mean_P, axis=-1).numpy()  # (P)
      sampled_pred = sampled_preds[index, :]
      sampling_weight = sampling_weights[index, :]  # (P)
      num_particles = tf.shape(pred_mean)[1]
      list_num_particles.append(num_particles)

      # plot the predicted probability density function.
      pdf_predicted = compute_mixture_gaussian_pdf(x=x, pred_means=pred_mean, omega_preds=omega_preds,
                                                   sampling_weights=sampling_weight)
      ax.plot(x, pdf_predicted, color, lw=2, alpha=0.6, label='M={}'.format(num_particles))
      # plot the predicted empirical distribution:
      ax.hist(sampled_pred, color=color, density=True, histtype='stepfilled', alpha=0.2)
      # scatterplot of P means:
      y_mean_P = np.zeros(shape=pred_mean_P.shape)
      ax.scatter(x=pred_mean_P, y=y_mean_P, c=color, marker='x')

    # plot the true probability density function.
    true_dist = tfp.distributions.Normal(loc=true_mean, scale=omega_true_distrib)
    ax.plot(x, true_dist.prob(x), color='k', lw=2, linestyle='dashed', label='true pdf')

    plt.legend(fontsize=12)
    # plt.title('True pdf versus predicted pdf per timestep for samplne # {}'.format(index), fontsize=16)
    #plt.show()
    fig_path = output_path + '/' + 'true_pdf_vs_pred_pdf_one_timestep_multiples_P_sample_{}_.png'.format(index)
    plt.savefig(fig_path)


if __name__ == "__main__":

  file_path= '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/post_UAI_exp/results_ws155_632020/time_series_multi_synthetic_heads_2_depth_6_dff_24_pos-enc_50_pdrop_0_b_256_target-feat_0_cv_False__particles_1_noise_False_sigma_0.05/inference_results/num-timesteps_4_p_inf_10_N_10_N-est_5000_sigma_0.05_omega_0.2'
  Baseline_T_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/post_UAI_exp/time_series_multi_synthetic_heads_2_depth_6_dff_24_pos-enc_50_pdrop_0_b_256_target-feat_0_cv_False'

  #SMC_T inference results:
  preds_gaussian_means_path = os.path.join(file_path, 'pred_gaussian_means_per_timestep.npy')
  true_gaussian_mean_path = os.path.join(file_path, 'true_gaussian_means.npy')
  sampling_weights_path = os.path.join(file_path, 'sampling_weights.npy')
  sampled_pred_distrib_path = os.path.join(file_path, 'preds_sampled_per_timestep.npy')
  true_emp_distrib_path = os.path.join(file_path, 'true_empirical_distrib.npy')
  preds_gaussian_means = np.load(preds_gaussian_means_path)  # (num_timesteps, B, N, P, 1)
  true_gaussian_mean = np.load(true_gaussian_mean_path)  # (num_timesteps, B, F)
  sampling_weights = np.load(sampling_weights_path)
  sampled_pred_distrib = np.load(sampled_pred_distrib_path) # (num_timesteps, B, N_est)
  true_emp_distrib = np.load(true_emp_distrib_path) # num_timesteps, B, N_est)

  MC_dropout_T_path = os.path.join(Baseline_T_path, 'Baseline_T_MC_Dropout_preds_inference.npy')
  baseline_T_preds_path = os.path.join(Baseline_T_path, 'Baseline_T_true_preds.npy')
  MC_dropout_T_preds = np.load(MC_dropout_T_path) # (B, N_est, num_timesteps, 1)
  baseline_T_preds = np.load(baseline_T_preds_path) # (num_timesteps, B, 1)

  # take the first feature of the true gaussian mean
  true_gaussian_mean = true_gaussian_mean[:, :, 0]  # (num_timesteps, B)
  # convert numpy arrays to tensor:
  preds_gaussian_means = tf.convert_to_tensor(preds_gaussian_means) # (num_timesteps, B, N, P, 1)
  sampling_weights = tf.convert_to_tensor(sampling_weights) # (B,P)
  # reshaping mc_dropout_preds:
  MC_dropout_T_preds = MC_dropout_T_preds.reshape(MC_dropout_T_preds.shape[:-1]) # (B, N_est, num_timesteps)
  baseline_T_preds = baseline_T_preds.reshape(baseline_T_preds.shape[:-1]) # (num_timesteps, B)

  omega_preds = 0.2
  omega_true_distrib = 0.2

  # # test of compute_gaussian_mixture_function
  # true_means = true_gaussian_mean[0,:] # (B)
  # pred_means = preds_gaussian_means[0,:,:,:,:] #(B,N,P,1)
  # batch_size = pred_means.shape[0]
  # index = np.random.randint(low=0, high=batch_size)
  # pred_means = pred_means[index, :, :]  # (N,P,1)
  # true_mean = true_means[index]  # scalar.
  # sampling_weights = sampling_weights[index, :]  # (P)
  # x = np.linspace(start=true_mean - 5 * omega_preds, stop=true_mean + 5 * omega_preds, num=100)  # (100)
  #
  # pdf = compute_mixture_gaussian_pdf_2(x=x, pred_means=pred_means, omega_preds=omega_preds, sampling_weights=sampling_weights)

  # test of plotting for one timestep and one sample
  t=0
  true_mean = true_gaussian_mean[t,:]
  pred_means = preds_gaussian_means[t,:,:,:,:]
  sampled_preds = sampled_pred_distrib[t,:,:]
  mc_dropout_preds = MC_dropout_T_preds[:,:,t]
  baseline_T_preds_t = baseline_T_preds[t,:]
  # plot_one_timestep(pred_means=pred_means,
  #                   true_means=true_mean,
  #                   sampled_preds=sampled_preds,
  #                   omega_preds=omega_preds,
  #                   omega_true_distrib=omega_true_distrib,
  #                   sampling_weights=sampling_weights,
  #                   output_path=file_path,
  #                   mc_dropout_preds=mc_dropout_preds,
  #                   baseline_T_preds=baseline_T_preds_t)

  plot_one_timestep(pred_means=pred_means,
                    true_means=true_mean,
                    sampled_preds=sampled_preds,
                    omega_preds=omega_preds,
                    omega_true_distrib=omega_true_distrib,
                    sampling_weights=sampling_weights,
                    output_path=file_path,
                    mc_dropout_preds=None,
                    baseline_T_preds=None)

  plot_multiple_timesteps(pred_means=preds_gaussian_means, true_means=true_gaussian_mean,
                          sampled_preds=sampled_pred_distrib, sampling_weights=sampling_weights,
                          omega_preds=omega_preds, omega_true_distrib=omega_true_distrib, output_path=file_path,
                          mc_dropout_preds=MC_dropout_T_preds, baseline_T_preds=baseline_T_preds)

  # ----- plotting for multiple values of number of particules --------------------------------------------------------------------------------------
  file_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/post_UAI_exp/results_ws155_632020/time_series_multi_synthetic_heads_2_depth_6_dff_24_pos-enc_50_pdrop_0_b_256_target-feat_0_cv_False__particles_1_noise_False_sigma_0.05/inference_results/num-timesteps_4_p_inf_10-50-100-500-_N_10_N-est_10000_sigma_0.05_omega_0.2'
  preds_gaussian_means_path = os.path.join(file_path, 'pred_gaussian_means_per_timestep_P_10.npy')
  sampling_weights_path = os.path.join(file_path, 'sampling_weights_P_10.npy')
  sampled_pred_distrib_path = os.path.join(file_path, 'preds_sampled_per_timestep_P_10.npy')
  true_gaussian_mean_path = os.path.join(file_path, 'true_gaussian_means.npy')
  true_emp_distrib_path = os.path.join(file_path, 'true_empirical_distrib.npy')
  preds_gaussian_means_10 = np.load(preds_gaussian_means_path)  # (num_timesteps, B, N, P, 1)
  sampling_weights_10 = np.load(sampling_weights_path)
  sampled_pred_distrib_10 = np.load(sampled_pred_distrib_path) # (num_timesteps, B, N_est)
  true_gaussian_mean = np.load(true_gaussian_mean_path)  # (num_timesteps, B, F)
  true_emp_distrib = np.load(true_emp_distrib_path) # num_timesteps, B, N_est)

  true_gaussian_mean = true_gaussian_mean[:, :, 0]

  preds_gaussian_means_path = os.path.join(file_path, 'pred_gaussian_means_per_timestep_P_50.npy')
  sampling_weights_path = os.path.join(file_path, 'sampling_weights_P_50.npy')
  sampled_pred_distrib_path = os.path.join(file_path, 'preds_sampled_per_timestep_P_50.npy')
  preds_gaussian_means_50 = np.load(preds_gaussian_means_path)  # (num_timesteps, B, N, P, 1)
  sampling_weights_50 = np.load(sampling_weights_path)
  sampled_pred_distrib_50 = np.load(sampled_pred_distrib_path)  # (num_timesteps, B, N_est)

  preds_gaussian_means_path = os.path.join(file_path, 'pred_gaussian_means_per_timestep_P_100.npy')
  sampling_weights_path = os.path.join(file_path, 'sampling_weights_P_100.npy')
  sampled_pred_distrib_path = os.path.join(file_path, 'preds_sampled_per_timestep_P_100.npy')
  preds_gaussian_means_100 = np.load(preds_gaussian_means_path)  # (num_timesteps, B, N, P, 1)
  sampling_weights_100 = np.load(sampling_weights_path)
  sampled_pred_distrib_100 = np.load(sampled_pred_distrib_path)  # (num_timesteps, B, N_est)

  preds_gaussian_means_path = os.path.join(file_path, 'pred_gaussian_means_per_timestep_P_500.npy')
  sampling_weights_path = os.path.join(file_path, 'sampling_weights_P_500.npy')
  sampled_pred_distrib_path = os.path.join(file_path, 'preds_sampled_per_timestep_P_500.npy')
  preds_gaussian_means_500 = np.load(preds_gaussian_means_path)  # (num_timesteps, B, N, P, 1)
  sampling_weights_500 = np.load(sampling_weights_path)
  sampled_pred_distrib_500 = np.load(sampled_pred_distrib_path)  # (num_timesteps, B, N_est)

  preds_gaussian_means = [preds_gaussian_means_10, preds_gaussian_means_50, preds_gaussian_means_100, preds_gaussian_means_500]
  sampling_weights = [sampling_weights_10, sampling_weights_50, sampling_weights_100, sampling_weights_500]
  sampled_pred_distrib = [sampled_pred_distrib_10, sampled_pred_distrib_50, sampled_pred_distrib_100, sampled_pred_distrib_500]
  preds_gaussian_means = [tf.convert_to_tensor(t) for t in preds_gaussian_means]
  sampling_weights = [tf.convert_to_tensor(t) for t in sampling_weights]

  t = 0
  true_mean = true_gaussian_mean[t, :]
  pred_means = [X[t, :, :, :, :] for X in preds_gaussian_means]
  sampled_preds = [X[t, :, :] for X in sampled_pred_distrib]
  plot_multiple_P_one_timestep(list_pred_means=pred_means, true_means=true_mean,
                               list_sampled_preds=sampled_preds,
                               list_sampling_weights=sampling_weights,
                               omega_preds=omega_preds,
                               omega_true_distrib=omega_true_distrib,
                               output_path=file_path)