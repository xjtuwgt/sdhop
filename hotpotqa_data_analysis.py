from utils.data_utils import length_analysis
from envs import DATASET_FOLDER
from os.path import join


if __name__ == '__main__':
    folder_name = 'data_feat/train'
    train_file_name = 'cached_long_low_hotpotqa_tokenized_examples_electra.pkl.gz'
    data_file_name = join(DATASET_FOLDER, folder_name, train_file_name)
    length_analysis(data_file_name=data_file_name)