from time import time
import json
from pandas import DataFrame
import pandas as pd
import os
import gzip, pickle
from model_envs import MODEL_CLASSES
###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def json_loader(json_file_name: str):
    with open(json_file_name, 'r', encoding='utf-8') as reader:
        json_data = json.load(reader)
    return json_data

def loadWikiData(json_file_name: str)->DataFrame:
    start_time = time()
    data_frame = pd.read_json(json_file_name, orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame

def load_gz_file(file_name):
    start_time = time()
    with gzip.open(file_name, 'rb') as fin:
        print('loading', file_name)
        data = pickle.load(fin)
    print('Loading {} from {} in {:.4f} seconds'.format(len(data), file_name, time() - start_time))
    return data

def load_encoder_model(encoder_name_or_path, model_type):
    if encoder_name_or_path in [None, 'None', 'none']:
        raise ValueError('no checkpoint provided for model!')

    config_class, model_encoder, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(encoder_name_or_path)
    if config is None:
        raise ValueError(f'config.json is not found at {encoder_name_or_path}')

    # check if is a path
    if os.path.exists(encoder_name_or_path):
        if os.path.isfile(os.path.join(encoder_name_or_path, 'pytorch_model.bin')):
            encoder_file = os.path.join(encoder_name_or_path, 'pytorch_model.bin')
        else:
            encoder_file = os.path.join(encoder_name_or_path, 'encoder.pkl')
        encoder = model_encoder.from_pretrained(encoder_file, config=config)
    else:
        encoder = model_encoder.from_pretrained(encoder_name_or_path, config=config)

    return encoder, config