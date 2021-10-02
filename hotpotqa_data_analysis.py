from utils.data_utils import length_analysis, support_fact_analysis
from envs import DATASET_FOLDER
from os.path import join
import numpy as np

if __name__ == '__main__':
    folder_name = 'data_feat/train'
    train_file_name = 'cached_long_low_hotpotqa_tokenized_examples_electra.pkl.gz'
    data_file_name = join(DATASET_FOLDER, folder_name, train_file_name)
    # len_array = length_analysis(data_file_name)
    len_array = support_fact_analysis(data_file_name)

    print('Min: {}\nMax: {}\nMean: {}\nP25: {}\nP50: {}\nP75: {}'.format(np.min(len_array), np.max(len_array), np.mean(len_array),
                    np.percentile(len_array, 25), np.percentile(len_array, 50), np.percentile(len_array, 75)))