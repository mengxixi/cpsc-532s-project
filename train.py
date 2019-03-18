import glob
import os

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from dataloader import Flickr30K_Entities
from language_model import GloVe
from grounding import GroundeR


BATCH_SIZE = 32 

flickr_mini = glob.glob('/home/siyi/flickr_mini/*.pkl')
train_ids = []
for file in flickr_mini:
    fid = file.split('/')[-1][:-4]
    train_ids.append(fid)


lm = GloVe(os.path.join('models', 'glove', 'glove.twitter.27B.200d.txt'), dim=200)

train_dataset = Flickr30K_Entities(train_ids, language_model=lm)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size = BATCH_SIZE, 
    shuffle=True,
    collate_fn=train_dataset.collate_fn)

grounder = GroundeR().cuda()

for epoch in range(10):
    for batch_idx, data in enumerate(train_loader):
        b_queries, b_pr_features, b_ph_features = data

        batch_size = len(b_queries)

        lstm_h0 = grounder.initHidden(batch_size)
        lstm_c0 = grounder.initCell(batch_size)
        attn_weights = grounder(b_pr_features, (lstm_h0, lstm_c0), b_ph_features, batch_size)
