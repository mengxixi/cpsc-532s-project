import glob

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from dataloader import Flickr30K_Entities


flickr_mini = glob.glob('/home/siyi/flickr_mini/*.pkl')
train_ids = []
for file in flickr_mini:
    fid = file.split('/')[-1][:-4]
    train_ids.append(fid)

train_dataset = Flickr30K_Entities(train_ids)

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size = 32, 
    shuffle=True)

for epoch in range(10):
    for batch_idx, data in enumerate(train_loader):
        pass