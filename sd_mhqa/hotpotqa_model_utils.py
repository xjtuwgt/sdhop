import torch
import json
import numpy as np
import os
import shutil
import logging
import torch.nn.functional as F
from model_envs import MODEL_CLASSES
from tqdm import tqdm
from sd_mhqa.hotpotqa_data_utils import case_to_features
from transformers import AdamW
from eval.hotpot_evaluate_v1 import eval as hotpot_eval
IGNORE_INDEX = -100

logger = logging.getLogger(__name__)
def get_optimizer(encoder, model, args, learning_rate, remove_pooler=False):
    """
    get BertAdam for encoder / classifier or BertModel
    :param model:
    :param classifier:
    :param args:
    :param remove_pooler:
    :return:
    """

    param_optimizer = list(encoder.named_parameters())
    param_optimizer += list(model.named_parameters())

    if remove_pooler:
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0} ]
    print('Learning rate = {}'.format(learning_rate))
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=args.adam_epsilon)

    return optimizer

def convert_to_tokens(tokenizer, examples, ids, y1, y2, q_type_prob):
    answer_dict, answer_type_dict = {}, {}
    answer_type_prob_dict = {}

    q_type = np.argmax(q_type_prob, 1)

    def get_ans_from_pos(qid, y1, y2):
        example = examples[qid]
        feature_list = case_to_features(case=example, train_dev=True)
        doc_input_ids = feature_list[0]

        final_text = " "
        if y1 < len(doc_input_ids) and y2 < len(doc_input_ids):
            answer_input_ids = doc_input_ids[y1:(y2+1)] ## y2+1
            final_text = tokenizer.decode(answer_input_ids)
        return final_text

    for i, qid in enumerate(ids):
        if q_type[i] == 2:
            answer_text = get_ans_from_pos(qid, y1[i], y2[i])
        elif q_type[i] == 0:
            answer_text = 'yes'
        elif q_type[i] == 1:
            answer_text = 'no'
        else:
            raise ValueError("question type error")
        answer_dict[qid] = answer_text
        answer_type_prob_dict[qid] = q_type_prob[i].tolist()
        answer_type_dict[qid] = q_type[i].item()

    return answer_dict, answer_type_dict, answer_type_prob_dict

def jd_hotpotqa_eval_model(args, model, dataloader, example_dict, prediction_file, eval_file, dev_gold_file, output_score_file=None):
    _, _, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.encoder_name_or_path,
                                                do_lower_case=args.do_lower_case)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    model.eval()

    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}

    # dataloader.refresh()

    thresholds = np.arange(0.1, 1.0, 0.05)
    N_thresh = len(thresholds)
    total_sp_dict = [{} for _ in range(N_thresh)]

    for batch in tqdm(dataloader):
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key not in {'ids'}:
                batch[key] = value.to(args.device)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with torch.no_grad():
            start, end, q_type, paras, sent, yp1, yp2 = model(batch, return_yp=True)


        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(tokenizer, example_dict, batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)
        answer_type_dict.update(answer_type_dict_)
        answer_type_prob_dict.update(answer_type_prob_dict_)
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()

        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = [[] for _ in range(N_thresh)]
            cur_id = batch['ids'][i]

            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break

                for thresh_i in range(N_thresh):
                    if predict_support_np[i, j] > thresholds[thresh_i]:
                        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[j])

            for thresh_i in range(N_thresh):
                if cur_id not in total_sp_dict[thresh_i]:
                    total_sp_dict[thresh_i][cur_id] = []

                total_sp_dict[thresh_i][cur_id].extend(cur_sp_pred[thresh_i])

    def choose_best_threshold(ans_dict, pred_file):
        best_joint_f1 = 0
        best_metrics = None
        best_threshold = 0
        for thresh_i in range(N_thresh):
            prediction = {'answer': ans_dict,
                          'sp': total_sp_dict[thresh_i],
                          'type': answer_type_dict,
                          'type_prob': answer_type_prob_dict}
            tmp_file = os.path.join(os.path.dirname(pred_file), 'tmp.json')
            with open(tmp_file, 'w') as f:
                json.dump(prediction, f)
            metrics = hotpot_eval(tmp_file, dev_gold_file)
            if metrics['joint_f1'] >= best_joint_f1:
                best_joint_f1 = metrics['joint_f1']
                best_threshold = thresholds[thresh_i]
                best_metrics = metrics
                shutil.move(tmp_file, pred_file)
        return best_metrics, best_threshold
    best_metrics, best_threshold = choose_best_threshold(answer_dict, prediction_file)
    json.dump(best_metrics, open(eval_file, 'w'))
    return best_metrics, best_threshold