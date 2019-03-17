import glob
import os

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from dataloader import Flickr30K_Entities
from language_model import GloVe


flickr_mini = glob.glob('/home/siyi/flickr_mini/*.pkl')
train_ids = []
for file in flickr_mini:
    fid = file.split('/')[-1][:-4]
    train_ids.append(fid)


lm = GloVe(os.path.join('models', 'glove.twitter.27B.50d.txt'), dim=50)

train_dataset = Flickr30K_Entities(train_ids, language_model=lm)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size = 32, 
    shuffle=True,
    collate_fn=train_dataset.collate_fn)

for epoch in range(10):
    for batch_idx, data in enumerate(train_loader):
        b_queries, b_pr_features, b_ph_features = data
