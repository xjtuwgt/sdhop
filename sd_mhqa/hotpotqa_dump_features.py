from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from sd_mhqa.hotpotqa_data_utils import hotpot_answer_tokenizer
from os.path import join
import gzip
import pickle
from model_envs import MODEL_CLASSES
from sd_mhqa.hotpotqa_argument_parser import boolean_string

def get_cached_filename(f_type, config):
    f_type_set = {'long_low_hotpotqa_tokenized_examples', 'hgn_low_hotpotqa_tokenized_examples',
                  'long_low_large_hotpotqa_tokenized_examples', 'hgn_low_large_hotpotqa_tokenized_examples'}
    assert f_type in f_type_set
    return f"cached_{f_type}_{config.model_type}.pkl.gz"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--para_path", type=str, required=True)
    parser.add_argument("--full_data", type=str, required=True)
    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help='define output directory')
    # Other parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--large_model", default="false", type=boolean_string)
    parser.add_argument("--tokenizer_name", default="google/electra-base-discriminator", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--model_name_or_path", default='google/electra-base-discriminator', type=str,
                        help="Path to pre-trained model")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_sent_num", default=40, type=int)
    parser.add_argument("--max_query_length", default=50, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                             "than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--filter_no_ans", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--ranker", default=None, type=str, required=True,
                        help="The ranker for paragraph ranking")
    args = parser.parse_args()
    if args.large_model:
        args.tokenizer_name = 'google/electra-large-discriminator'
    else:
        args.model_name_or_path = 'google/electra-large-discriminator'
    print('*' * 75)
    for key, value in vars(args).items():
        print('Hype-parameter: {}:\t{}'.format(key, value))
    print('*' * 75)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    ranker = args.ranker
    data_type = args.data_type
    if args.do_lower_case:
        ranker = ranker + '_low'
    data_source_name = "{}".format(ranker)
    if "train" in data_type:
        data_source_type = data_source_name
    else:
        data_source_type = None
    print('data_type = {} \n data_source_id = {} \n data_source_name = {}'.format(data_type, data_source_type, data_source_name))
    examples = hotpot_answer_tokenizer(para_file=args.para_path,
                                       full_file=args.full_data,
                                       tokenizer=tokenizer,
                                       cls_token=tokenizer.cls_token,
                                       sep_token=tokenizer.sep_token,
                                       is_roberta=bool(args.model_type in ['roberta']),
                                       data_source_type=data_source_type)
    if args.large_model:
        cached_examples_file = join(args.output_dir,
                                            get_cached_filename('{}_large_hotpotqa_tokenized_examples'.format(data_source_name), args))
    else:
        cached_examples_file = join(args.output_dir,
                                            get_cached_filename('{}_hotpotqa_tokenized_examples'.format(data_source_name), args))
    with gzip.open(cached_examples_file, 'wb') as fout:
        pickle.dump(examples, fout)
    print('Saving {} examples in {}'.format(len(examples), cached_examples_file))