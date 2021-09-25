from utils.ioutils import load_gz_file
import numpy as np

def length_analysis(data_file_name: str):
    examples = load_gz_file(file_name=data_file_name)
    num_sent_list = []
    for example in examples:
        num_sent_list.append(len(example.sent_names))
    num_array = np.array(num_sent_list)
    return num_array