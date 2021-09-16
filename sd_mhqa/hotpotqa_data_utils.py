from tqdm import tqdm
import itertools
from sd_mhqa.hotpotqa_data_structure import Example
import spacy
import numpy as np
from numpy import random
import torch
from utils.ioutils import json_loader
from torch import Tensor
import re

nlp = spacy.load("en_core_web_lg", disable=['tagger', 'parser', 'lemmatizer'])
infix_re = re.compile(r'''[-—–~]''')
nlp.tokenizer.infix_finditer = infix_re.finditer
###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def split_sent(sent: str):
    nlp_doc = nlp(sent)
    words = []
    for token in nlp_doc:
        words.append(token.text)
    return words
def tokenize_text(text: str, tokenizer, is_roberta):
    words = split_sent(sent=text)
    sub_tokens = []
    for word in words:
        if is_roberta:
            sub_toks = tokenizer.tokenize(word, add_prefix_space=True)
        else:
            sub_toks = tokenizer.tokenize(word)
        sub_tokens += sub_toks
    return words, sub_tokens
###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def normalize_question(question: str) -> str:
    question = question
    if question[-1] == '?':
        question = question[:-1]
    return question
def normalize_text(text: str) -> str:
    text = ' ' + text.lower().strip() ###adding the ' ' is important to make the consist encoder, for roberta tokenizer
    return text
def answer_span_checker(answer, sentence):
    find_idx = sentence.find(answer)
    if find_idx < 0:
        return False
    return True
def find_answer_span(norm_answer, sentence, tokenizer, is_roberta):
    _, ans_sub_tokens = tokenize_text(text=norm_answer, tokenizer=tokenizer, is_roberta=is_roberta)
    _, sent_sub_tokens = tokenize_text(text=sentence, tokenizer=tokenizer, is_roberta=is_roberta)
    idx = sub_list_match_idx(target=ans_sub_tokens, source=sent_sub_tokens)
    flag = idx >= 0
    return flag, ans_sub_tokens, sent_sub_tokens
def find_sub_list_fuzzy(target: list, source: list) -> int:
    if len(target) > len(source):
        return -1
    t_len = len(target)
    temp_idx = -1
    if t_len >=4:
        temp_target = target[1:]
        temp_idx = find_sub_list(temp_target, source)
        if temp_idx < 1:
            temp_target = target[:(t_len-1)]
            temp_idx = find_sub_list(temp_target, source)
        else:
            temp_idx = temp_idx - 1
    return temp_idx
def sub_list_match_idx(target: list, source: list) -> int:
    idx = find_sub_list(target, source)
    if idx < 0:
        idx = find_sub_list_fuzzy(target, source)
    return idx
def find_sub_list(target: list, source: list) -> int:
    if len(target) > len(source):
        return -1
    t_len = len(target)
    def equal_list(a_list, b_list):
        for j in range(len(a_list)):
            if a_list[j] != b_list[j]:
                return False
        return True
    for i in range(len(source) - len(target) + 1):
        temp = source[i:(i+t_len)]
        is_equal = equal_list(target, temp)
        if is_equal:
            return i
    return -1
########################################################################################################################
def ranked_context_processing(row, tokenizer, selected_para_titles, is_roberta):
    question, supporting_facts, contexts, answer = row['question'], row['supporting_facts'], row['context'], row['answer']
    doc_title2doc_len = dict([(title, len(text)) for title, text in contexts])
    supporting_facts_filtered = [(supp_title, supp_sent_idx) for supp_title, supp_sent_idx in supporting_facts
                                 if supp_sent_idx < doc_title2doc_len[supp_title]] ##some supporting facts are out of sentence index
    positive_titles = set([x[0] for x in supporting_facts_filtered]) ## get postive document titles
    ################################################################################################################
    norm_answer = normalize_text(text=answer) ## normalize the answer (add a space between the answer)
    norm_question = normalize_question(question.lower()) ## normalize the question by removing the question mark
    ################################################################################################################
    answer_found_flag = False ## some answer might be not founded in supporting sentence
    ################################################################################################################
    selected_contexts = []
    context_dict = dict(row['context'])
    for title in selected_para_titles:
        para_text = context_dict[title]
        para_text_lower = [normalize_text(text=sent) for sent in para_text]
        if title in positive_titles:
            count = 1
            supp_sent_flags = []
            for supp_title, supp_sent_idx in supporting_facts_filtered:
                if title == supp_title:
                    supp_sent = para_text_lower[supp_sent_idx]
                    if norm_answer.strip() not in ['yes', 'no', 'noanswer']:
                        has_answer = answer_span_checker(norm_answer.strip(), supp_sent)
                        if has_answer:
                            encode_has_answer, X, Y = find_answer_span(norm_answer.strip(), supp_sent, tokenizer, is_roberta)
                            if not encode_has_answer:
                                encode_has_answer, X, Y = find_answer_span(norm_answer, supp_sent, tokenizer, is_roberta)
                                if not encode_has_answer:
                                    supp_sent_flags.append((supp_sent_idx, False))
                                else:
                                    supp_sent_flags.append((supp_sent_idx, True))
                                    count = count + 1
                                    answer_found_flag = True
                            else:
                                supp_sent_flags.append((supp_sent_idx, True))
                                count = count + 1
                                answer_found_flag = True
                        else:
                            supp_sent_flags.append((supp_sent_idx, False))
                    else:
                        supp_sent_flags.append((supp_sent_idx, False))
            selected_contexts.append([title, para_text_lower, count, supp_sent_flags, True])  ## support para
        else:
            selected_contexts.append([title, para_text_lower, 0, [], False]) ## no support para
    yes_no_flag = norm_answer.strip() in ['yes', 'no', 'noanswer']
    if not answer_found_flag and (norm_answer.strip() not in ['yes', 'no']):
        norm_answer = 'noanswer'
    return norm_question, norm_answer, selected_contexts, supporting_facts_filtered, yes_no_flag, answer_found_flag
#=======================================================================================================================
def hotpot_answer_tokenizer(para_file: str,
                            full_file: str,
                            tokenizer,
                            cls_token='[CLS]',
                            sep_token='[SEP]',
                            is_roberta=False,
                            data_source_type=None):
    sel_para_data = json_loader(json_file_name=para_file)
    full_data = json_loader(json_file_name=full_file)
    examples = []
    answer_not_found_count = 0
    for row in tqdm(full_data):
        key = row['_id']
        qas_type = row['type']
        sent_names = []
        sup_facts_sent_id = []
        para_names = []
        sup_para_id = []
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sel_paras = sel_para_data[key]
        selected_para_titles = itertools.chain.from_iterable(sel_paras)
        norm_question, norm_answer, selected_contexts, supporting_facts_filtered, yes_no_flag, answer_found_flag = \
            ranked_context_processing(row=row, tokenizer=tokenizer, selected_para_titles=selected_para_titles, is_roberta=is_roberta)
        # print(yes_no_flag, answer_found_flag)
        if not answer_found_flag and not yes_no_flag:
            answer_not_found_count = answer_not_found_count + 1
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        query_tokens = [cls_token]
        query_words, query_sub_tokens = tokenize_text(text=norm_question, tokenizer=tokenizer, is_roberta=is_roberta)
        query_tokens += query_sub_tokens
        if is_roberta:
            query_tokens += [sep_token, sep_token]
        else:
            query_tokens += [sep_token]
        query_input_ids = tokenizer.convert_tokens_to_ids(query_tokens)
        assert len(query_tokens) == len(query_input_ids)
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        sent_to_id, sent_id = {}, 0
        ctx_token_list = []
        ctx_input_id_list = []
        sent_num = 0
        para_num = 0
        ctx_with_answer = False
        answer_positions = []  ## answer position
        ans_sub_tokens = []
        ans_input_ids = []
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for para_idx, para_tuple in enumerate(selected_contexts):
            para_num += 1
            title, sents, _, answer_sent_flags, supp_para_flag = para_tuple
            para_names.append(title)
            if supp_para_flag:
                sup_para_id.append(para_idx)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            sent_tokens_list = []
            sent_input_id_list = []
            for local_sent_id, sent_text in enumerate(sents):
                sent_num += 1
                local_sent_name = (title, local_sent_id)
                sent_to_id[local_sent_name] = sent_id
                sent_names.append(local_sent_name)
                if local_sent_name in supporting_facts_filtered:
                    sup_facts_sent_id.append(sent_id)
                # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                sent_words, sent_sub_tokens = tokenize_text(text=sent_text, tokenizer=tokenizer, is_roberta=is_roberta)
                if is_roberta:
                    sent_sub_tokens.append(sep_token)
                sub_input_ids = tokenizer.convert_tokens_to_ids(sent_sub_tokens)
                assert len(sub_input_ids) == len(sent_sub_tokens)
                sent_tokens_list.append(sent_sub_tokens)
                sent_input_id_list.append(sub_input_ids)
                assert len(sent_sub_tokens) == len(sub_input_ids)
                sent_id = sent_id + 1
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            ctx_token_list.append(sent_tokens_list)
            ctx_input_id_list.append(sent_input_id_list)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if (norm_answer.strip() not in ['yes', 'no', 'noanswer']) and answer_found_flag:
                ans_words, ans_sub_tokens = tokenize_text(text=norm_answer, tokenizer=tokenizer, is_roberta=is_roberta)
                ans_input_ids = tokenizer.convert_tokens_to_ids(ans_sub_tokens)
                for sup_sent_idx, supp_sent_flag in answer_sent_flags:
                    supp_sent_encode_ids = sent_input_id_list[sup_sent_idx]
                    if supp_sent_flag:
                        answer_start_idx = sub_list_match_idx(target=ans_input_ids, source=supp_sent_encode_ids)
                        if answer_start_idx < 0:
                            ans_words, ans_sub_tokens = tokenize_text(text=norm_answer.strip(), tokenizer=tokenizer,
                                                                      is_roberta=is_roberta)
                            ans_input_ids = tokenizer.convert_tokens_to_ids(ans_sub_tokens)
                            answer_start_idx = sub_list_match_idx(target=ans_input_ids, source=supp_sent_encode_ids)
                        answer_len = len(ans_input_ids)
                        assert answer_start_idx >= 0, "supp sent={} \n answer={} \n answer={} \n {} \n {}".format(tokenizer.decode(supp_sent_encode_ids),
                            tokenizer.decode(ans_input_ids), norm_answer, supp_sent_encode_ids, ans_sub_tokens)
                        ctx_with_answer = True
                        # answer_positions.append((para_idx, sup_sent_idx, answer_start_idx, answer_start_idx + answer_len))
                        answer_positions.append(
                            (title, sup_sent_idx, answer_start_idx, answer_start_idx + answer_len))
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            assert len(para_names) == para_num
            assert len(sent_names) == sent_num
            assert len(ctx_token_list) == para_num and len(ctx_input_id_list) == para_num
        ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++diff the rankers
        if data_source_type is not None:
            key = key + "_" + data_source_type
        ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        example = Example(qas_id=key,
                          qas_type=qas_type,
                          ctx_text=selected_contexts,
                          ctx_tokens=ctx_token_list,
                          ctx_input_ids=ctx_input_id_list,
                          para_names=para_names,
                          sup_para_id=sup_para_id,
                          sent_names=sent_names,
                          para_num=para_num,
                          sent_num=sent_num,
                          sup_fact_id=sup_facts_sent_id,
                          question_text=norm_question,
                          question_tokens=query_tokens,
                          question_input_ids=query_input_ids,
                          answer_text=norm_answer,
                          answer_tokens=ans_sub_tokens,
                          answer_input_ids=ans_input_ids,
                          answer_positions=answer_positions,
                          ctx_with_answer=ctx_with_answer)
        examples.append(example)
    print('Answer not found = {}'.format(answer_not_found_count))
    return examples

##+++++++++++++++++++++++++++++++++sentence DROP++++++++++++++++++++++++++++++++++++
#######################################################################
def case_to_features(case: Example, train_dev=True):
    question_input_ids = case.question_input_ids
    ctx_input_ids = case.ctx_input_ids
    sent_num = case.sent_num
    para_num = case.para_num
    para_names = case.para_names
    sent_names = case.sent_names
    assert len(ctx_input_ids) == para_num and sent_num == len(sent_names)
    doc_input_ids = [] ### ++++++++
    doc_input_ids += question_input_ids
    para_len_list = [len(question_input_ids)]
    sent_len_list = [len(question_input_ids)]
    query_len = len(question_input_ids)
    query_spans = [(1, query_len)]
    para_sent_pair_to_sent_id, sent_id = {}, 0
    for para_idx, para_name in enumerate(para_names):
        para_sent_ids = ctx_input_ids[para_idx]
        para_len_ = 0
        for sent_idx, sent_ids in enumerate(para_sent_ids):
            doc_input_ids += sent_ids
            sent_len_i = len(sent_ids)
            sent_len_list.append(sent_len_i)
            para_len_ = para_len_ + sent_len_i
            para_sent_pair_to_sent_id[(para_name, sent_idx)] = sent_id
            sent_id = sent_id + 1
        para_len_list.append(para_len_)
    # print('In here {}'.format(doc_input_ids))
    assert sent_num == len(sent_len_list) - 1 and para_num == len(para_len_list) - 1
    assert sent_id == sent_num
    sent_cum_sum_len_list = np.cumsum(sent_len_list).tolist()
    para_cum_sum_len_list = np.cumsum(para_len_list).tolist()
    sent_spans = [(sent_cum_sum_len_list[i], sent_cum_sum_len_list[i+1]) for i in range(sent_num)]
    para_spans = [(para_cum_sum_len_list[i], para_cum_sum_len_list[i+1]) for i in range(para_num)]
    assert len(sent_spans) == sent_num
    assert len(para_spans) == para_num
    if train_dev:
        answer_text = case.answer_text.strip()
        if answer_text in ['yes']:
            answer_type_label = [0]
        elif answer_text in ['no', 'noanswer']:
            answer_type_label = [1]
        else:
            answer_type_label = [2]
        answer_positions = case.answer_positions
        ans_spans = []
        for ans_position in answer_positions:
            doc_title, sent_id, ans_start, ans_end = ans_position
            sent_idx = para_sent_pair_to_sent_id[(doc_title, sent_id)]
            sent_start_idx = sent_spans[sent_idx][0]
            ans_spans.append((sent_start_idx + ans_start, sent_start_idx + ans_end))

        # print('in', len(doc_input_ids))
        return doc_input_ids, query_spans, para_spans, sent_spans, ans_spans, answer_type_label
    else:
        return doc_input_ids, query_spans, para_spans, sent_spans
#######################################################################
def largest_valid_index(spans, limit):
    for idx in range(len(spans)):
        if spans[idx][1] >= limit:
            return idx
    return len(spans)

def trim_input_span(doc_input_ids, query_spans, para_spans, sent_spans, limit, sep_token_id, ans_spans=None):
    if len(doc_input_ids) <= limit:
        if ans_spans is not None:
            return doc_input_ids, query_spans, para_spans, sent_spans, ans_spans
        else:
            return doc_input_ids, query_spans, para_spans, sent_spans
    else:
        trim_doc_input_ids = []
        trim_doc_input_ids += doc_input_ids[:(limit - 1)]
        trim_doc_input_ids += [sep_token_id]
        trim_seq_len = len(trim_doc_input_ids)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        largest_sent_idx = largest_valid_index(sent_spans, limit)
        trim_sent_spans = []
        trim_sent_spans += sent_spans[:largest_sent_idx]
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++
        largest_para_idx = largest_valid_index(para_spans, trim_seq_len)
        trim_para_spans = []
        trim_para_spans += para_spans[:largest_para_idx]
        if largest_para_idx < len(para_spans):
            if para_spans[largest_para_idx][0] < trim_seq_len:
                trim_para_spans += [(para_spans[largest_para_idx][0], trim_seq_len)]
        # trim_para_spans = [[_[0], _[1]] for _ in trim_para_spans]
        # trim_para_spans[largest_para_idx][1] = limit
        # trim_para_spans = [(_[0], _[1]) for _ in trim_para_spans]
        # largest_sent_idx = largest_valid_index(sent_spans, limit)
        # trim_sent_spans = []
        # trim_sent_spans += sent_spans[:largest_sent_idx]
        # largest_sent_start, largest_sent_end = sent_spans[largest_sent_idx]
        # trim_sent_spans = []
        # if (limit - largest_sent_start) < (largest_sent_end - largest_sent_start) * 0.8:
        #     trim_sent_spans += sent_spans[:(largest_sent_idx)]
        # else:
        #     trim_sent_spans += sent_spans[:(largest_sent_idx+1)]
        #     trim_sent_spans = [[_[0], _[1]] for _ in trim_sent_spans]
        #     trim_sent_spans[largest_sent_idx][1] = limit
        #     trim_sent_spans = [(_[0], _[1]) for _ in trim_sent_spans]

        if ans_spans is not None:
            largest_ans_idx = largest_valid_index(ans_spans, limit)
            trim_ans_spans = []
            trim_ans_spans += ans_spans[:largest_ans_idx]
            return trim_doc_input_ids, query_spans, trim_para_spans, trim_sent_spans, trim_ans_spans
        else:
            return trim_doc_input_ids, query_spans, trim_para_spans, trim_sent_spans

#######################################################################
def example_sent_drop(case: Example, drop_ratio:float = 0.1):
    qas_id = case.qas_id
    qas_type = case.qas_type
    question_tokens = case.question_tokens
    ctx_tokens = case.ctx_tokens
    question_text = case.question_text
    question_input_ids = case.question_input_ids
    ctx_input_ids = case.ctx_input_ids
    sent_names = case.sent_names
    para_names = case.para_names
    sup_fact_id = case.sup_fact_id
    sup_para_id = case.sup_para_id
    ctx_text = case.ctx_text
    answer_text = case.answer_text
    answer_tokens = case.answer_tokens
    answer_input_ids = case.answer_input_ids
    answer_positions = case.answer_positions
    ctx_with_answer = case.ctx_with_answer
    para_num = case.para_num
    sent_num = case.sent_num
    assert para_num == len(ctx_tokens) and para_num == len(ctx_input_ids) and para_num == len(para_names)
    assert sent_num == len(sent_names)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sent_name_to_id_dict = dict([((x[1][0], x[1][1]), x[0]) for x in enumerate(sent_names)])
    keep_sent_idxs = []
    drop_ctx_tokens = []
    drop_ctx_input_ids = []
    for para_idx, para_name in enumerate(para_names):
        drop_para_ctx_tokens, drop_para_input_ids = [], []
        for sent_idx, (sent_sub_token, sent_inp_ids) in enumerate(zip(ctx_tokens[para_idx], ctx_input_ids[para_idx])):
            abs_sent_idx = sent_name_to_id_dict[(para_name, sent_idx)]
            if abs_sent_idx not in sup_fact_id:
                rand_s_i = random.rand()
                if rand_s_i > drop_ratio:
                    drop_para_ctx_tokens.append(sent_sub_token)
                    drop_para_input_ids.append(sent_inp_ids)
                    keep_sent_idxs.append(abs_sent_idx)
            else:
                drop_para_ctx_tokens.append(sent_sub_token)
                drop_para_input_ids.append(sent_inp_ids)
                keep_sent_idxs.append(abs_sent_idx)
        drop_ctx_tokens.append(drop_para_ctx_tokens)
        drop_ctx_input_ids.append(drop_para_input_ids)
    ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    keep_sent_idx_remap_dict = dict([(x[1], x[0]) for x in enumerate(keep_sent_idxs)]) ## for answer map
    ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    drop_para_names = []
    for para_idx, para_name in enumerate(para_names):
        assert len(drop_ctx_input_ids[para_idx]) == len(drop_ctx_tokens[para_idx])
        if len(drop_ctx_tokens[para_idx]) > 0:
            drop_para_names.append(para_name)
    drop_ctx_tokens = [_ for _ in drop_ctx_tokens if len(_) > 0]
    drop_ctx_input_ids = [_ for _ in drop_ctx_input_ids if len(_) > 0]
    drop_para_fact_id = []
    supp_para_names = [para_names[_] for _ in sup_para_id]
    for para_idx, para_name in enumerate(drop_para_names):
        if para_name in supp_para_names:
            drop_para_fact_id.append(para_idx)
    drop_para_num = len(drop_para_names)
    ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    drop_sent_names = []
    for para_idx, para_name in enumerate(drop_para_names):
        for sent_idx in range(len(drop_ctx_input_ids[para_idx])):
            drop_sent_names.append((para_name, sent_idx))
    drop_supp_fact_ids = [keep_sent_idx_remap_dict[_] for _ in sup_fact_id]
    drop_sent_num = len(drop_sent_names)
    ###+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    drop_answer_positions = []
    for answer_position in answer_positions:
        title, sent_idx, start_pos, end_pos = answer_position
        orig_sent_name = (title, sent_idx)
        orig_abs_sent_idx = sent_name_to_id_dict[orig_sent_name]
        drop_abs_sent_idx = keep_sent_idx_remap_dict[orig_abs_sent_idx]
        drop_sent_name = drop_sent_names[drop_abs_sent_idx]
        assert drop_sent_name[0] == title
        drop_answer_positions.append((drop_sent_name[0], drop_sent_name[1], start_pos, end_pos))

    drop_example = Example(
        qas_id=qas_id,
        qas_type=qas_type,
        ctx_text=ctx_text,
        ctx_tokens=drop_ctx_tokens,
        ctx_input_ids=drop_ctx_input_ids,
        para_names=drop_para_names,
        sup_para_id=drop_para_fact_id,
        sent_names=drop_sent_names,
        para_num=drop_para_num,
        sent_num=drop_sent_num,
        sup_fact_id=drop_supp_fact_ids,
        question_text=question_text,
        question_tokens=question_tokens,
        question_input_ids=question_input_ids,
        answer_text=answer_text,
        answer_tokens=answer_tokens,
        answer_input_ids=answer_input_ids,
        answer_positions=drop_answer_positions,
        ctx_with_answer=ctx_with_answer)
    return drop_example
###++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def para_sent_state_feature_extractor(batch, input_state: Tensor):
    sent_start, sent_end = batch['sent_start'], batch['sent_end']
    para_start, para_end = batch['para_start'], batch['para_end']
    assert (sent_start.max() < input_state.shape[1]) \
           and (sent_end.max() < input_state.shape[1]), '{}\t{}\t{}'.format(sent_start, sent_end, input_state.shape[1])

    batch_size, para_num, sent_num = para_start.shape[0], para_start.shape[1], sent_start.shape[1]
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    sent_batch_idx = torch.arange(0, batch_size, device=input_state.device).view(batch_size, 1).repeat(1, sent_num)
    sent_start_output = input_state[sent_batch_idx, sent_start]
    sent_end_output = input_state[sent_batch_idx, sent_end]
    sent_state = torch.cat([sent_start_output, sent_end_output], dim=-1)
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    para_batch_idx = torch.arange(0, batch_size, device=input_state.device).view(batch_size, 1).repeat(1, para_num)
    para_start_output = input_state[para_batch_idx, para_start]
    para_end_output = input_state[para_batch_idx, para_end]
    para_state = torch.cat([para_start_output, para_end_output], dim=-1)
    state_dict = {'para_state': para_state, 'sent_state': sent_state}
    return state_dict