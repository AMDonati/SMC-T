import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import os
import statistics
import math

#TODO: arrange plots formatting (legend: etc...)
#TODO: check form of function if adding 'manually' gaussian pdfs in tf.


def compute_mixture_gaussian_pdf(x, pred_means, omega_preds, sampling_weights):
  '''
  x: numpy array of shape (num_samples,)
  pred_means: numpy array of shape (num MC samples, num particles, 1)
  omega_preds: scalar for the covariance value.
  -sampling weights: numpy array of shape (num_particles)
  '''
  dist = tfp.distributions.Normal(loc=pred_means, scale=omega_preds)
  # prepare input x: convert it into a tensor and tile it.
  x = tf.convert_to_tensor(x, dtype=tf.float32)
  x = tf.expand_dims(tf.expand_dims(x, axis=0), axis=0)
  x = tf.tile(x, multiples=[N, P, 1])
  pdf = dist.prob(x) # (N,P, num_samples)
  pdf = tf.reduce_mean(pdf, axis=1) # mean over N: (P, num_samples)
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
    pdf_p = tf.reduce_mean(pdf_N) # (num_samples)
    list_pdf_P.append(pdf_p)
  pdf_P = tf.stack(list_pdf_P) # (P, num_samples)
  sampling_weights = tf.expand_dims(sampling_weights, axis=0) # (1, P)
  pdf = tf.matmul(sampling_weights, pdf_P) # (1, num_samples)
  pdf = tf.squeeze(pdf, axis=0)

  return pdf


def plot_one_timestep(pred_means, true_means, sampled_preds, sampling_weights, omega_preds, omega_true_distrib, color, output_path):
    '''
    args:
    -pred_means: numpy array of shape (batch_size, num MC samples, num particles, 1)
    -true_means: numpy array of shape (batch_size)
    -sampled_preds: array of shape (batch_size, N_est)
    -sampling weights: numpy array of shape (batch_size, num_particle)
    '''
    #np.random.seed = seed

    # draw a sample among the samples of the test set:
    batch_size = pred_means.shape[0]
    index = np.random.randint(low=0, high=batch_size)
    pred_means = pred_means[index, :, :] # (N,P,1)
    true_mean = true_means[index] # scalar.
    sampled_preds = sampled_preds[index,:]
    sampling_weights = sampling_weights[index, :] #(P)

    # plot the predicted probability density function.
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(start=true_mean - 5 * omega_preds, stop=true_mean + 5 * omega_preds, num=100) # (100)
    pdf_predicted = compute_mixture_gaussian_pdf(x=x, pred_means=pred_means, omega_preds=omega_preds, sampling_weights=sampling_weights)
    ax.plot(x, pdf_predicted, color, lw=2, alpha=0.6, label='predicted pdf for sample number: {}'.format(index))

    # plot the true probability density function.
    true_dist = tfp.distributions.Normal(loc=true_mean, scale=omega_true_distrib)
    ax.plot(x, true_dist.prob(x), color, lw=2, linestyle='dashed', label='true pdf for sample number: {}')

    # plot the predicted empirical distribution:
    ax.hist(sampled_preds, density=True, histtype='stepfilled', alpha=0.2)

    plt.legend(fontsize=14)
    plt.title('True pdf versus predicted pdf per timestep for samplne # {}'.format(index), fontsize=16)
    plt.show()
    fig_path = output_path + '/'+'true_pdf_vs_pred_pdf_one_timestep_sample{}.png'.format(index)
    plt.savefig(fig_path)


def plot_multiple_timesteps(pred_means, true_means, sampled_preds, sampling_weights, omega_preds, omega_true_distrib, output_path):
  '''
  args:
  -pred_means: numpy array of shape (num_timesteps, batch_size, num MC samples, num particles, 1)
  -true_means: numpy array of shape (num_timesteps, batch_size)
  -sampled_preds: array of shape (num_timesteps, batch_size, N_est)
  -sampling weights: numpy array of shape (batch_size, num_particle)
  '''
  num_timesteps = tf.shape(pred_means)[0]

  # prepare plot with multiple subplots:
  fig, axs = plt.subplots(2, 2)
  #axs[0, 0].plot(x, y)
  axs[0, 0].set_title('t+1')
  #axs[0, 1].plot(x, y, 'tab:orange')
  axs[0, 1].set_title('t+2')
  #axs[1, 0].plot(x, -y, 'tab:green')
  axs[1, 0].set_title('t+3')
  #axs[1, 1].plot(x, -y, 'tab:red')
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
    sampled_preds_t = sampled_preds[t,:,:]
    pred_means_t = pred_means_t[index, :, :]  # (N,P,1)
    true_mean_t = true_means_t[index]  # scalar.
    sampled_preds_t = sampled_preds_t[index,:] #(N_est)


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

  plt.legend(fontsize=14)
  #plt.title('True pdf versus predicted pdf per timestep for sample # {}'.format(index), fontsize=16)
  plt.show()
  fig_path = output_path + '/' + 'true_pdf_vs_pred_pdf_{}_timesteps_sample{}.png'.format(num_timesteps, index)
  plt.savefig(fig_path)


if __name__ == "__main__":

  file_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-T/output/post_UAI_exp/results_ws155_632020/time_series_multi_synthetic_heads_2_depth_6_dff_24_pos-enc_50_pdrop_0_b_256_target-feat_0_cv_False__particles_1_noise_False_sigma_0.05/inference_results/num-timesteps_4_p_inf_10_N_10_N-est_5000_sigma_0.05_omega_0.2'

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

  # take the first feature of the true gaussian mean
  true_gaussian_mean = true_gaussian_mean[:, :, 0]  # (num_timesteps, B)

  # convert numpy arrays to tensor:
  preds_gaussian_means = tf.convert_to_tensor(preds_gaussian_means) # (num_timesteps, B, N, P, 1)
  #preds_gaussian_means = tf.squeeze(preds_gaussian_means, axis=-1) # (num_timesteps, B, N, P, 1)
  sampling_weights = tf.convert_to_tensor(sampling_weights) # (B,P)

  omega_preds = 0.3
  omega_true_distrib = 0.2

  # test of plotting for one timestep and one sample
  pred_means = preds_gaussian_means[0, 0, :, :] # (N, P)
  N = tf.shape(pred_means)[0]
  P = tf.shape(pred_means)[1]
  dist = tfp.distributions.Normal(loc=pred_means, scale=omega_preds)
  #true_gaussian_mean = true_gaussian_mean[0, 0]
  true_mean = true_gaussian_mean[0, 0]
  x = np.linspace(start=true_mean - 5 * omega_preds, stop=true_mean + 5 * omega_preds, num=100) # (100)
  x = tf.convert_to_tensor(x, dtype=tf.float32)
  x = tf.expand_dims(tf.expand_dims(x, axis=0), axis=0)
  x = tf.tile(x, multiples=[N,P,1])
  sampling_weight = sampling_weights[0, :] # (P)
  pdf = dist.prob(x)
  pdf = tf.reduce_mean(pdf, axis=1)
  sampling_weight = tf.expand_dims(sampling_weight, axis=0)
  pdf = tf.matmul(sampling_weight, pdf)
  pdf = tf.squeeze(pdf, axis=0)

  # t=0
  # true_mean = true_gaussian_mean[t,:]
  # pred_means = preds_gaussian_means[t,:,:,:,:]
  # sampled_preds = sampled_pred_distrib[t,:,:]
  # plot_one_timestep(pred_means=pred_means,
  #                   true_means=true_mean,
  #                   sampled_preds=sampled_preds,
  #                   omega_preds=omega_preds,
  #                   omega_true_distrib=omega_true_distrib,
  #                   sampling_weights=sampling_weights,
  #                   output_path=file_path,
  #                   color='r')

  plot_multiple_timesteps(pred_means=preds_gaussian_means, true_means=true_gaussian_mean,
                          sampled_preds=sampled_pred_distrib, sampling_weights=sampling_weights,
                          omega_preds=omega_preds, omega_true_distrib=omega_true_distrib, output_path=file_path)




