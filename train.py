import os
import sys
import glob
import logging
from datetime import datetime

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm, tnrange
from tensorboardX import SummaryWriter

import evaluate
from dataloader import Flickr30K_Entities, QuerySampler
from language_model import GloVe
from grounding import GroundeR


# logging configurations
LOG_FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")

# directories
FLICKR30K_ENTITIES = '/home/siyi/flickr30k_entities'

# TODO: Refactor constants later
BATCH_SIZE = 512
N_EPOCHS = 20
LR = 1e-2

PRINT_EVERY = 1000 # Every x iterations
EVALUATE_EVERY = 50000


def get_dataloader(im_ids, lm):
    dataset = Flickr30K_Entities(im_ids, language_model=lm)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = BATCH_SIZE, 
        collate_fn=dataset.collate_fn,
        sampler=QuerySampler(dataset))
    return loader


def train():

    with open(os.path.join(FLICKR30K_ENTITIES, 'train.txt')) as f1, open(os.path.join(FLICKR30K_ENTITIES, 'val.txt')) as f2, open(os.path.join(FLICKR30K_ENTITIES, 'nobbox.txt')) as f3:
        # TODO: Format this line nicely
        train_ids = f1.readlines()
        val_ids = f2.readlines()
        nobbox_ids = f3.readlines()

    train_ids = [x.strip() for x in train_ids if x not in nobbox_ids]
    val_ids = [x.strip() for x in val_ids if x not in nobbox_ids]

    lm = GloVe(os.path.join('models', 'glove', 'glove.twitter.27B.200d.txt'), dim=200)
    train_loader = get_dataloader(train_ids, lm)
    val_loader = get_dataloader(val_ids, lm)

    grounder = GroundeR().cuda()
    optimizer = torch.optim.Adam(grounder.parameters(), lr=LR)
    criterion = torch.nn.NLLLoss()

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    writer = SummaryWriter(os.path.join('logs', subdir))

    # Train loop
    best_acc = 0.0
    for epoch in tqdm(range(N_EPOCHS), file=sys.stdout):
        running_loss = 0

        for batch_idx, data in enumerate(train_loader):
            b_queries, b_pr_features, b_ph_features = data
            b_y = torch.tensor([query['gt_ppos_id'] for query in b_queries]).cuda()

            batch_size = len(b_queries)

            # Foward
            lstm_h0 = grounder.initHidden(batch_size)
            lstm_c0 = grounder.initCell(batch_size)
            attn_weights = grounder(b_pr_features, (lstm_h0, lstm_c0), b_ph_features, batch_size)

            loss = criterion(torch.log(attn_weights), b_y)

            # Backward and update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss 
            global_step = epoch*len(train_loader)+batch_idx

            # Log losses
            print_batches = PRINT_EVERY//BATCH_SIZE
            if batch_idx % print_batches == print_batches-1:
                writer.add_scalar('loss', loss.item(), global_step)
                logging.info("Epoch %d, query %d, loss: %.3f" % (epoch+1, (batch_idx+1)*BATCH_SIZE, running_loss/print_batches))
                running_loss = 0

            # Log evaluations
            evaluate_batches = EVALUATE_EVERY//BATCH_SIZE
            if batch_idx % evaluate_batches == evaluate_batches-1:
                acc = evaluate.evaluate(grounder, val_loader, summary_writer=writer, global_step=global_step)
                writer.add_scalar('val_acc', acc, global_step)
                logging.info("Validation accuracy: %.3f, best_acc: %.3f" % (acc, best_acc))

                # Improved on validation set
                if acc > best_acc:
                    torch.save(grounder.state_dict(), os.path.join('models', 'grounder.ckpt'))
                    best_acc = acc

                grounder.train()

    writer.close()



if __name__ == "__main__":
    train()


