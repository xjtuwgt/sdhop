import json
import sys

from tqdm import tqdm
from collections import Counter

assert len(sys.argv) == 5

raw_data = json.load(open(sys.argv[1], 'r'))
para_data = json.load(open(sys.argv[2], 'r'))
output_file = sys.argv[3]
###################################################
num_selected_docs = int(sys.argv[4])
print('num of selected docs = {}, type of para {}'.format(num_selected_docs, type(num_selected_docs)))
###################################################

def build_dict(title_list):
    title_to_id, id_to_title = {}, {}
    for idx, title in enumerate(title_list):
        id_to_title[idx] = title
        title_to_id[title] = idx
    return title_to_id, id_to_title

para_num = []
selected_para_dict = {}
#####++++++++++++++++++++++++++++++++++++
selected_para_score_threshold_dict = {}
#####++++++++++++++++++++++++++++++++++++

for case in tqdm(raw_data):
    guid = case['_id']
    context = dict(case['context'])
    para_scores = para_data[guid]
    selected_para_dict[guid] = []
    #####++++++++++++++++++++++++++++++++++++++++++
    selected_para_score_threshold_dict[guid] = []
    #####++++++++++++++++++++++++++++++++++++++++++

    if len(para_scores) == 0:
        print(guid)
        continue

    title_to_id, id_to_title = build_dict(context.keys())
    sel_para_idx = [0] * len(context)

    # others, keep a high recall
    other_titles = []
    ######+++++++++++++++++++++++++++++++
    other_scores = []
    ######+++++++++++++++++++++++++++++++
    for para, score in para_scores:
        if para not in title_to_id:
            continue
        if sum(sel_para_idx) == num_selected_docs:
            break
        ind = title_to_id[para]
        if sel_para_idx[ind] == 0:
            sel_para_idx[ind] = 1
            other_titles.append(para)
            ######+++++++++++++++++++++++++++++++
            other_scores.append(score)
            ######+++++++++++++++++++++++++++++++
    ######+++++++++++++++++++++++++++++++
    selected_para_dict[guid].append(other_titles)
    ##############################################
    selected_para_score_threshold_dict[guid].append(other_scores)
    #++++++++++++++++++++++++++++++++++++++++++++
    para_num.append(sum(sel_para_idx))
para_num_counter = Counter(para_num)
print(para_num_counter)
json.dump(selected_para_dict, open(output_file, 'w'))
print('Saving {} into {}'.format(len(selected_para_dict), output_file))
