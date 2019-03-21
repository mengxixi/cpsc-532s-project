import torch
import torch.nn.functional as F
from torch import nn

import numpy as np


class GroundeR(nn.Module):
    def __init__(self, im_feature_size=4096, lm_emb_size=200, hidden_size=150, concat_size=128, output_size=100):

        super().__init__()

        self.im_feature_size = im_feature_size
        self.lm_emb_size = lm_emb_size
        self.hidden_size = hidden_size
        self.concat_size = concat_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=lm_emb_size, hidden_size=hidden_size, batch_first=True)
        self.ph_proj = nn.Linear(hidden_size, concat_size)
        self.im_proj = nn.Linear(im_feature_size, concat_size)
        self.attn = nn.Conv2d(concat_size, 1, 1)


    def forward(self, im_input, h0c0, ph_input, batch_size):
        ph_out, (hn, cn) = self.lstm(ph_input, h0c0)
        
        ph_concat = self.ph_proj(hn).permute(1,0,2)
        im_concat = self.im_proj(im_input)

        width = int(np.sqrt(self.output_size))
        out = F.relu((ph_concat + im_concat)).view(batch_size, self.concat_size, width, width)

        attn_weights_raw = self.attn(out)   # [bs, 1, 10, 10]
        attn_weights = attn_weights_raw.view(batch_size, self.output_size)
        attn_weights = F.softmax(attn_weights, dim=1)

        return attn_weights


    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda()
    
    
    def initCell(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda()