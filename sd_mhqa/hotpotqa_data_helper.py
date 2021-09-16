from sd_mhqa.hotpotqa_dataset import HotpotDataset
from os.path import join
from envs import DATASET_FOLDER
from sd_mhqa.hotpotqa_dump_features import get_cached_filename
from torch.utils.data import DataLoader
import pickle, gzip

class DataHelper:
    def __init__(self, sep_token_id, gz=True, config=None):
        self.Dataset = HotpotDataset
        self.gz = gz
        self.suffix = '.pkl.gz' if gz else '.pkl'

        self.data_dir = join(DATASET_FOLDER, 'data_feat')

        self.__train_examples__ = None
        self.__dev_examples__ = None

        self.__train_example_dict__ = None
        self.__dev_example_dict__ = None
        self.sep_token_id = sep_token_id

        self.config = config

    def get_example_file(self, tag, f_type=None):
        if self.config.large_model:
            cached_filename = get_cached_filename('{}_large_hotpotqa_tokenized_examples'.format(f_type), self.config)
        else:
            cached_filename = get_cached_filename('{}_hotpotqa_tokenized_examples'.format(f_type), self.config)
        return join(self.data_dir, tag, cached_filename)

    @property
    def train_example_file(self):
        return self.get_example_file('train', self.config.daug_type)

    @property
    def dev_example_file(self):
        return self.get_example_file('dev_distractor', self.config.devf_type)

    def get_pickle_file(self, file_name):
        if self.gz:
            return gzip.open(file_name, 'rb')
        else:
            return open(file_name, 'rb')

    def __get_or_load__(self, name, file):
        if getattr(self, name) is None:
            with self.get_pickle_file(file) as fin:
                print('loading', file)
                setattr(self, name, pickle.load(fin))

        return getattr(self, name)

    # Examples
    @property
    def train_examples(self):
        return self.__get_or_load__('__train_examples__', self.train_example_file)

    @property
    def dev_examples(self):
        return self.__get_or_load__('__dev_examples__', self.dev_example_file)

    # Example dict
    @property
    def train_example_dict(self):
        if self.__train_example_dict__ is None:
            self.__train_example_dict__ = {e.qas_id: e for e in self.train_examples}
        return self.__train_example_dict__

    @property
    def dev_example_dict(self):
        if self.__dev_example_dict__ is None:
            self.__dev_example_dict__ = {e.qas_id: e for e in self.dev_examples}
        return self.__dev_example_dict__

    # Load
    def load_dev(self):
        return self.dev_examples, self.dev_example_dict

    def load_train(self):
        return self.train_examples, self.train_example_dict

    @property
    def hotpot_train_dataloader(self) -> DataLoader:
        train_examples, _ = self.load_train()
        train_data = self.Dataset(examples=train_examples,
                                  max_para_num=self.config.max_para_num,
                                  max_sent_num=self.config.max_sent_num,
                                  max_seq_num=self.config.max_seq_length,
                                  sent_drop_ratio=self.config.sent_drop_ratio,
                                  drop_prob=self.config.drop_prob,
                                  beta_drop=self.config.beta_drop,
                                  sep_token_id=self.sep_token_id)
        ####++++++++++++
        dataloader = DataLoader(dataset=train_data, batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=max(1, self.config.cpu_num // 2),
            collate_fn=HotpotDataset.collate_fn)
        return dataloader

    @property
    def hotpot_val_dataloader(self) -> DataLoader:
        dev_examples, _ = self.load_dev()
        dev_data = self.Dataset(examples=dev_examples,
                                max_para_num=self.config.max_para_num,
                                max_sent_num=self.config.max_sent_num,
                                max_seq_num=self.config.max_seq_length,
                                sep_token_id=self.sep_token_id,
                                sent_drop_ratio=-1,
                                drop_prob=-1)
        dataloader = DataLoader(
            dataset=dev_data,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=max(1, self.config.cpu_num // 2),
            collate_fn=HotpotDataset.collate_fn
        )
        return dataloader