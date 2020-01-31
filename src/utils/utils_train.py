import os
import csv
import shutil
import pickle as pkl

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
    print('Suppression of old directory with same parameters')
    os.chmod(path, 0o777)
    shutil.rmtree(path, ignore_errors=True)
  os.makedirs(path)

  return path


def save_to_pickle(file_name, np_array):
  with open(file_name,'wb') as f:
    pkl.dump(np_array, f)

if __name__ == "__main__":
  path_dir='../../output'
  path_name='checkpoints'
  temp_path=create_run_dir(path_dir=path_dir, path_name=path_name)
  temp_npy_fn=os.path.join(temp_path, 'temp_file.npy')