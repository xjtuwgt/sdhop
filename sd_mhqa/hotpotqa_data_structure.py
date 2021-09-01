class Example(object):
    def __init__(self,
                 qas_id,
                 qas_type,
                 question_text,
                 question_tokens,
                 question_input_ids,
                 ctx_text,
                 ctx_tokens,
                 ctx_input_ids,
                 sent_names,
                 sent_num,
                 para_names,
                 para_num,
                 sup_fact_id=None,
                 sup_para_id=None,
                 answer_text=None,
                 answer_tokens=None,
                 answer_input_ids=None,
                 answer_positions=None,
                 ctx_with_answer=None):
        self.qas_id = qas_id
        self.qas_type = qas_type
        self.question_tokens = question_tokens
        self.ctx_tokens = ctx_tokens
        self.question_text = question_text
        self.question_input_ids = question_input_ids
        self.ctx_input_ids = ctx_input_ids
        self.sent_names = sent_names
        self.para_names = para_names
        self.sup_fact_id = sup_fact_id
        self.sup_para_id = sup_para_id
        self.ctx_text = ctx_text
        self.answer_text = answer_text
        self.answer_tokens = answer_tokens
        self.answer_input_ids = answer_input_ids
        self.answer_positions = answer_positions
        self.ctx_with_answer = ctx_with_answer
        self.para_num = para_num
        self.sent_num = sent_num