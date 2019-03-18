import glob
import os

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm, tnrange

from dataloader import Flickr30K_Entities
from language_model import GloVe
from grounding import GroundeR


BATCH_SIZE = 32 
N_EPOCHS = 10
LR = 1e-3

def get_dataloader(im_ids, lm):
    dataset = Flickr30K_Entities(im_ids, language_model=lm)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = BATCH_SIZE, 
        shuffle=True,
        collate_fn=dataset.collate_fn)
    return loader


train_ids = [file.split('/')[-1][:-4] for file in glob.glob('/home/siyi/flickr_mini/*.pkl')]
lm = GloVe(os.path.join('models', 'glove', 'glove.twitter.27B.200d.txt'), dim=200)
train_loader = get_dataloader(train_ids, lm)


grounder = GroundeR().cuda()
optimizer = torch.optim.Adam(grounder.parameters(), lr=LR)
criterion = torch.nn.NLLLoss(ignore_index=-1)

# Train loop
for epoch in tqdm(range(N_EPOCHS), desc='epochs'):
    running_loss = 0

    for batch_idx, data in enumerate(tqdm(train_loader)):
        b_queries, b_pr_features, b_ph_features = data
        b_y = torch.tensor([query['gt_ppos_id'] for query in b_queries]).cuda()

        batch_size = len(b_queries)

        lstm_h0 = grounder.initHidden(batch_size)
        lstm_c0 = grounder.initCell(batch_size)
        attn_weights = grounder(b_pr_features, (lstm_h0, lstm_c0), b_ph_features, batch_size)

        loss = criterion(attn_weights, b_y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss 
        print_every = 50 # print every x examples
        print_batches = print_every//BATCH_SIZE
        if batch_idx % print_batches == print_batches-1:
            print("Epoch %d, example %d, loss: %.3f" % (epoch+1, (batch_idx+1)*BATCH_SIZE, running_loss/print_batches))
            running_loss = 0






