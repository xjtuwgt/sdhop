import numpy as np
from torch import nn
from torch.autograd import Variable
import torch

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class OutputLayer(nn.Module):
    def __init__(self, hidden_dim, config, num_answer=1):
        super(OutputLayer, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            BertLayerNorm(hidden_dim*2, eps=1e-12),
            nn.Dropout(config.trans_drop),
            nn.Linear(hidden_dim*2, num_answer),
        )

    def forward(self, hidden_states):
        return self.output(hidden_states)

class ParaSentPredictionLayer(nn.Module):
    def __init__(self, config, hidden_dim):
        super(ParaSentPredictionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.para_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)
        self.sent_mlp = OutputLayer(self.hidden_dim, config, num_answer=1)

    def forward(self, state_dict):
        para_state = state_dict['para_state']
        sent_state = state_dict['sent_state']

        N, _, _ = para_state.size()
        sent_logit = self.sent_mlp(sent_state)
        para_logit = self.para_mlp(para_state)

        # print('sentpred', sent_state.shape, sent_logit.shape)

        para_logits_aux = Variable(para_logit.data.new(para_logit.size(0), para_logit.size(1), 1).zero_())
        para_prediction = torch.cat([para_logits_aux, para_logit], dim=-1).contiguous()

        sent_logits_aux = Variable(sent_logit.data.new(sent_logit.size(0), sent_logit.size(1), 1).zero_())
        sent_prediction = torch.cat([sent_logits_aux, sent_logit], dim=-1).contiguous()


        return (para_prediction, sent_prediction)

class PredictionLayer(nn.Module):
    """
    Identical to baseline prediction layer
    for answer span prediction
    """
    def __init__(self, config):
        super(PredictionLayer, self).__init__()
        self.config = config
        self.hidden = config.hidden_dim

        self.start_linear = OutputLayer(self.hidden, config, num_answer=1)
        self.end_linear = OutputLayer(self.hidden, config, num_answer=1)
        self.type_linear = OutputLayer(self.hidden, config, num_answer=3)

        self.cache_S = 0
        self.cache_mask = None

    def get_output_mask(self, outer):
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), 15)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)

    def forward(self, batch, context_input, packing_mask=None, return_yp=False):
        context_mask = batch['context_mask']
        start_prediction = self.start_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L
        end_prediction = self.end_linear(context_input).squeeze(2) - 1e30 * (1 - context_mask)  # N x L
        type_prediction = self.type_linear(context_input[:, 0, :])

        if not return_yp:
            return (start_prediction, end_prediction, type_prediction)

        outer = start_prediction[:, :, None] + end_prediction[:, None]
        # print('outer', outer.shape)
        outer_mask = self.get_output_mask(outer)
        # print('outer mask', outer_mask.shape)
        # print(outer_mask)
        outer = outer - 1e30 * (1 - outer_mask[None].expand_as(outer))
        if packing_mask is not None:
            outer = outer - 1e30 * packing_mask[:, :, None]
        # yp1: start
        # yp2: end
        yp1 = outer.max(dim=2)[0].max(dim=1)[1]
        yp2 = outer.max(dim=1)[0].max(dim=1)[1]
        # print(start_prediction.shape, end_prediction.shape, batch['y1'].shape, batch['y2'].shape)
        # print(yp1.shape, yp2.shape)
        return (start_prediction, end_prediction, type_prediction, yp1, yp2)