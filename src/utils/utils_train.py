import os
import csv
import shutil
import pickle as pkl
import numpy as np

def write_to_csv(output_dir, dic):
  """Write a python dic to csv."""
  with open(output_dir, 'w') as csv_file:
    writer = csv.writer(csv_file)
    for key, value in dic.items():
      writer.writerow([key, value])


def create_run_dir_from_hparams(path_dir, hparams):
  dataset = hparams['dataset']
  algorithm = hparams['algorithm']
  epochs = hparams['epochs']
  samples = hparams['samples']
  lr = hparams['lr']
  bs = hparams['batch_size']
  p_dropout = hparams['p_dropout']

  if algorithm == 'sgdsgld':
    path_name = '{}_{}_lr-{}_bs-{}_s-{}'.format(dataset, algorithm,
                                                lr, bs, samples)
  elif algorithm == 'bootstrap':
    path_name = '{}_{}_ep-{}_lr-{}_bs-{}_s-{}'.format(dataset, algorithm, epochs,
                                                      lr, bs, samples)
  elif algorithm == 'dropout':
    path_name = '{}_{}_ep-{}_lr-{}_bs-{}_s-{}'.format(dataset, algorithm, epochs,
                                                      lr, bs, samples)
    path_name = path_name + '_pdrop-{}'.format(p_dropout)
  else:
    raise ValueError('This algorithm is not supported')

  path = os.path.join(path_dir, path_name)

  if os.path.isdir(path):
    print('Suppression of old directory with same parameters')
    os.chmod(path, 0o777)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
  return path


def create_run_dir(path_dir, path_name):
  path = os.path.join(path_dir, path_name)
  if os.path.isdir(path):
    ckpt_path = os.path.join(path, "checkpoints")
    if os.path.exists(ckpt_path):
      print("output folder already existing with checkpoints saved. keeping it and restoring checkpoints if allowed.")
    else:
      print('Suppression of old directory with same parameters')
      os.chmod(path, 0o777)
      shutil.rmtree(path, ignore_errors=True)
      os.makedirs(path)
  else:
    os.makedirs(path)
  return path


def save_to_pickle(file_name, np_array):
  with open(file_name,'wb') as f:
    pkl.dump(np_array, f)

if __name__ == "__main__":
  path_dir='../../output'
  path_name='ckpt-1'
  temp_path=create_run_dir(path_dir=path_dir, path_name=path_name)
  ckpt_name=os.path.basename(temp_path)
  print('checkpt name', ckpt_name)
  _, ckpt_num=ckpt_name.split('-')
  print(int(ckpt_num))

  # file_temp=temp_path+'/temp.pkl'
  # array=np.zeros(shape=(10,10))
  # save_to_pickle(file_temp, array)
  #
  # csv_temp=path_dir+'/temp_table.csv'
  # l1=['key', 'value']
  # l2=[1,2]
  # write_to_csv(csv_temp, dict(zip(l1,l2)))