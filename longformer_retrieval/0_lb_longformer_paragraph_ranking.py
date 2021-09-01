from __future__ import absolute_import, division, print_function
from longformer_retrieval.lb_LongformerIRModel import LongformerGraphRetrievalModel
import argparse
from longformer_retrieval.lb_FullHotpotQADataSet import HotpotTestDataset
from longformer_retrieval.lb_LongformerUtils import get_hotpotqa_longformer_tokenizer
from torch.utils.data import DataLoader
import torch
import os
from utils.gpu_utils import gpu_setting
from tqdm import tqdm
from pandas import DataFrame
from time import time
import pandas as pd
import json

def loadJSONData(json_fileName)->DataFrame:
    start_time = time()
    data_frame = pd.read_json(json_fileName, orient='records')
    print('Loading {} in {:.4f} seconds'.format(data_frame.shape, time() - start_time))
    return data_frame

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Evaluating Longformer based retrieval Model')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--raw_data', default=None, type=str, required=True)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--input_data', default=None, type=str, required=True)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--data_dir', default=None, type=str, required=True)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument("--eval_ckpt", default=None, type=str, required=True, help="evaluation checkpoint")
    parser.add_argument("--model_type", default='Longformer', type=str, help="Longformer retrieval model")
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--max_doc_num', default=10, type=int)
    parser.add_argument('--test_log_steps', default=10, type=int)
    parser.add_argument('--cpu_num', default=4, type=int)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return parser.parse_args(args)

def batch2device(batch, device):
    sample = dict()
    for key, value in batch.items():
        sample[key] = value.to(device)
    return sample

def evaluation_step(output_scores, batch):
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    doc_scores = output_scores['doc_score']
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    batch_eids = batch['id'].squeeze().detach().tolist()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    out_put = {'doc_score': doc_scores.detach().tolist(), 'ids': batch_eids}
    return out_put

def test_data_loader(args):
    tokenizer = get_hotpotqa_longformer_tokenizer()
    test_data_frame = loadJSONData(json_fileName=args.input_data)
    test_data_frame['e_id'] = range(0, test_data_frame.shape[0]) ## for alignment
    test_data = HotpotTestDataset(data_frame=test_data_frame, tokenizer=tokenizer, max_doc_num=10)
    dataloader = DataLoader(
        dataset=test_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=HotpotTestDataset.collate_fn
    )
    return dataloader
########################################################################################################################
def paragraph_retrieval_procedure(model, test_data_loader, args, device):
    model.freeze()
    out_puts = []
    start_time = time()
    total_steps = len(test_data_loader)
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_data_loader)):
            batch = batch2device(batch=batch, device=device)
            output_scores = model.score_computation(sample=batch)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            output = evaluation_step(output_scores=output_scores, batch=batch)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if (batch_idx + 1) % args.test_log_steps == 0:
                print('Evaluating the model... {}/{} in {:.4f} seconds'.format(batch_idx + 1, total_steps, time()-start_time))
            out_puts.append(output)
            del batch
            torch.cuda.empty_cache()
    example_ids = []
    doc_scores = []
    for output in out_puts:
        example_ids += output['ids']
        doc_scores += output['doc_score']
    result_dict = {'e_id': example_ids,##for alignment
                   'doc_score': doc_scores}  ## for detailed results checking
    res_data_frame = DataFrame(result_dict)
    return res_data_frame
########################################################################################################################
def device_setting(args):
    if torch.cuda.is_available():
        free_gpu_ids, used_memory = gpu_setting(num_gpu=args.gpus)
        print('{} gpus with used memory = {}, gpu ids = {}'.format(len(free_gpu_ids), used_memory, free_gpu_ids))
        if args.gpus > 0:
            gpu_ids = free_gpu_ids
            device = torch.device("cuda:%d" % gpu_ids[0])
            print('Single GPU setting')
        else:
            device = torch.device("cpu")
            print('Single cpu setting')
    else:
        device = torch.device("cpu")
        print('Single cpu setting')
    return device
########################################################################################################################
def ParagraphRanker(data: DataFrame):
    def row_process(row):
        ctx_titles = [x[0] for x in row['context']]
        para_num = len(ctx_titles)
        predicted_scores = row['doc_score']
        predicted_scores = predicted_scores[:(len(ctx_titles))]
        title_score_pair_list = list(zip(ctx_titles, predicted_scores))
        title_score_pair_list.sort(key=lambda x: x[1], reverse=True)
        return para_num, tuple(title_score_pair_list)
    data[['para_num', 'ti_s_pair']] = data.apply(lambda row: pd.Series(row_process(row)), axis=1)
    doc_ids, para_score_pair = data['_id'].tolist(), data['ti_s_pair'].tolist()
    para_score_dict = dict(zip(doc_ids, para_score_pair))
    return para_score_dict
########################################################################################################################
def main(args):
    device = device_setting(args=args)
    hotpotIR_model = LongformerGraphRetrievalModel.load_from_checkpoint(checkpoint_path=args.eval_ckpt)
    hotpotIR_model = hotpotIR_model.to(device)
    print('Model Parameter Configuration:')
    for name, param in hotpotIR_model.named_parameters():
        print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    print('*' * 75)
    print("Model hype-parameter information...")
    for key, value in vars(args).items():
        print('Hype-parameter\t{} = {}'.format(key, value))
    print('*' * 75)
    test_data = test_data_loader(args=args)
    res_df = paragraph_retrieval_procedure(model=hotpotIR_model, test_data_loader=test_data, args=args, device=device)
    ##################################################################################################################
    raw_data = loadJSONData(json_fileName=args.raw_data)
    raw_data['e_id'] = range(0, raw_data.shape[0])
    merge_data = pd.concat([res_df.set_index('e_id'), raw_data.set_index('e_id')], axis=1, join='inner')
    rank_paras_dict = ParagraphRanker(data=merge_data)
    ####################################################################################################################
    json.dump(rank_paras_dict, open(os.path.join(args.data_dir, 'long_para_ranking.json'), 'w'))
    print('Saving {} records into {}'.format(len(rank_paras_dict), os.path.join(args.data_dir, 'long_para_ranking.json')))

if __name__ == '__main__':
    args = parse_args()
    main(args)
