import torch
import torch.nn.functional as F
from torch import nn

import numpy as np


class GroundeR(nn.Module):
    def __init__(self, im_feature_size=4096, lm_emb_size=200, hidden_size=50, concat_size=128, output_size=100):

        super().__init__()

        self.im_feature_size = im_feature_size
        self.lm_emb_size = lm_emb_size
        self.hidden_size = hidden_size
        self.concat_size = concat_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=lm_emb_size, hidden_size=hidden_size, batch_first=True, dropout=0.5)
        self.ph_bn = nn.BatchNorm1d(hidden_size)
        self.im_bn = nn.BatchNorm1d(im_feature_size)

        self.ph_proj = nn.Linear(hidden_size, concat_size)
        self.im_proj = nn.Linear(im_feature_size, concat_size)
        self.attn = nn.Linear(concat_size, 1)

        self.init_params()

    def forward(self, im_input, h0c0, ph_input, batch_size):
        ph_out, (hn, cn) = self.lstm(ph_input, h0c0)
        hn = self.ph_bn(hn.permute(1,2,0)) 
        ph_concat = self.ph_proj(hn.permute(0,2,1))

        im_bn = self.im_bn(im_input.permute(0,2,1))
        im_concat = self.im_proj(im_bn.permute(0,2,1))

        out = F.relu((ph_concat + im_concat))

        attn_weights_raw = self.attn(out).squeeze(2) # [bs, 100, 1]
        attn_weights = F.softmax(attn_weights_raw, dim=1)

        return attn_weights


    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda()
    
    
    def initCell(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda()


    def init_params(self):
        nn.init.uniform_(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self.ph_proj.weight)
        nn.init.xavier_uniform_(self.im_proj.weight)
        nn.init.kaiming_uniform_(self.attn.weight)




