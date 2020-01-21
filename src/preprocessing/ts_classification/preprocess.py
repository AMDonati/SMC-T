# try to solve the issue of ValueError: could not convert string to float.

from preprocessing.ts_classification.utils import read_dataset

root_dir = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/SMC-RNN---Transformers/data'

archive_name = 'mts_archive'
dataset_name = 'BasicMotions'
#classifier_name=sys.argv[3]

if __name__ == "__main__":
  datasets_dict=read_dataset(root_dir, archive_name, dataset_name)

