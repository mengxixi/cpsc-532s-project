import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import rnn


class GroundeR(nn.Module):
    def __init__(self, pretrained_embeddings, im_feature_size=4096, lm_emb_size=200, hidden_size=50, proj_size=128, output_size=100):

        super().__init__()

        self.im_feature_size = im_feature_size
        self.lm_emb_size = lm_emb_size
        self.hidden_size = hidden_size
        self.proj_size = proj_size
        self.output_size = output_size

        self.word_embeddings = nn.Embedding(len(pretrained_embeddings), lm_emb_size).from_pretrained(torch.from_numpy(pretrained_embeddings), freeze=False)

        self.dropout = nn.Dropout()
        self.lstm = nn.LSTM(input_size=lm_emb_size, hidden_size=hidden_size, batch_first=True)
        self.ph_bn = nn.BatchNorm1d(hidden_size)
        self.im_bn = nn.BatchNorm1d(im_feature_size)

        self.feat_proj = nn.Linear(hidden_size+im_feature_size, proj_size)
        self.attn = nn.Linear(proj_size, 1)

        self.init_params()


    def forward(self, im_input, h0c0, ph_input, batch_size):
        # ph_input is a packed sequence of word indices
        padded = rnn.pad_packed_sequence(ph_input, batch_first=True, padding_value=0)
        embedded = self.word_embeddings(padded[0])
        dropped_emb = self.dropout(embedded)
        packed_embedded = rnn.pack_padded_sequence(dropped_emb, padded[1], batch_first=True)

        _, (hn, cn) = self.lstm(packed_embedded, h0c0)

        hn = self.ph_bn(hn.permute(1,2,0))
        ph_bn = hn.repeat(1,1,self.output_size)
        dropped_im = self.dropout(im_input)
        im_bn = self.im_bn(im_input.permute(0,2,1))

        feat_concat = torch.cat([ph_bn, im_bn], dim=1)

        out = self.feat_proj(feat_concat.permute(0,2,1))
        out = self.dropout(F.relu(out))

        attn_weights_raw = self.attn(out).squeeze(2) # [bs, 100, 1]
        attn_weights = F.softmax(attn_weights_raw, dim=1)

        return attn_weights


    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda()
    
    
    def initCell(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda()


    def init_params(self):
        nn.init.kaiming_uniform_(self.attn.weight)
        nn.init.uniform_(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self.feat_proj.weight)




