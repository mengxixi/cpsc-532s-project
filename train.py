import os
import sys
import glob
import logging
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
from tqdm import tqdm

import evaluate
from dataloader import Flickr30K_Entities, QuerySampler
from language_model import GloVe
from grounding import GroundeR
from config import Config

Config.load_config()

# logging configurations
LOG_FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")

# directories
FLICKR30K_ENTITIES = Config.get('dirs.entities.root')
PRETRAINED_EMBEDDINGS = Config.get('dirs.tmp.pretrained_embeddings')
WORD2IDX = Config.get('dirs.tmp.word2idx')

BATCH_SIZE = Config.get('batch_size')
PRINT_EVERY = Config.get('print_every') # Every x iterations
EVALUATE_EVERY = Config.get('evaluate_every')


def get_dataloader(im_ids, word2idx=None):
    dataset = Flickr30K_Entities(im_ids, word2idx=word2idx)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = BATCH_SIZE, 
        collate_fn=dataset.collate_fn,
        sampler=QuerySampler(dataset))
    return loader


def train():

    # Load datasets
    with open(os.path.join(FLICKR30K_ENTITIES, Config.get('ids.train'))) as f1, open(os.path.join(FLICKR30K_ENTITIES, Config.get('ids.val'))) as f2,  open(os.path.join(FLICKR30K_ENTITIES, Config.get('ids.nobbox'))) as f3:
        train_ids = f1.read().splitlines()
        val_ids = f2.read().splitlines()
        nobbox_ids = f3.read().splitlines()

    train_ids = [x for x in train_ids if x not in nobbox_ids]
    val_ids = [x for x in val_ids if x not in nobbox_ids]

    train_loader = get_dataloader(train_ids)
    word2idx = train_loader.dataset.word2idx
    with open(WORD2IDX, 'wb') as f:
        pickle.dump(word2idx, f)
    val_loader = get_dataloader(val_ids, word2idx=word2idx)


    # Load pretrained embeddings
    if os.path.exists(PRETRAINED_EMBEDDINGS):
        pretrained_embeddings = np.load(PRETRAINED_EMBEDDINGS)
    else:
        lm = GloVe(Config.get('language_model'), dim=Config.get('word_emb_size'))
        pretrained_embeddings = np.array([lm.get_word_vector(w) for w in word2idx.keys()])
        np.save(PRETRAINED_EMBEDDINGS, pretrained_embeddings)


    # Model, optimizer, etc.
    grounder = GroundeR(pretrained_embeddings).cuda()
    optimizer = torch.optim.Adam(grounder.parameters(), lr=Config.get('learning_rate'), weight_decay=Config.get('weight_decay'))
    scheduler = MultiStepLR(optimizer, milestones=Config.get('sched_steps'))
    criterion = torch.nn.NLLLoss()

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    writer = SummaryWriter(os.path.join('logs', subdir))
    writer.add_text('config', str(Config.CONFIG_DICT))

    # Train loop
    best_acc = 0.0
    for epoch in tqdm(range(Config.get('n_epochs')), file=sys.stdout):
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        scheduler.step()

        running_loss = 0
        running_acc = 0

        for batch_idx, data in enumerate(train_loader):
            b_queries, b_pr_features, b_ph_indices = data
            b_y = torch.tensor([query['gt_ppos_id'] for query in b_queries]).cuda()

            batch_size = len(b_queries)

            # Foward
            lstm_h0 = grounder.initHidden(batch_size)
            lstm_c0 = grounder.initCell(batch_size)
            attn_weights = grounder(b_pr_features, (lstm_h0, lstm_c0), b_ph_indices, batch_size)

            topv, topi = attn_weights.topk(1)
            train_acc = 0.
            pred = topi.squeeze(1)
            for i, query in enumerate(b_queries):
                if pred[i] in query['gt_ppos_all']:
                    train_acc += 1

            train_acc = train_acc/batch_size

            loss = criterion(torch.log(attn_weights), b_y)

            # Backward and update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss 
            running_acc += train_acc
            global_step = epoch*len(train_loader)+batch_idx

            # Log losses
            print_batches = PRINT_EVERY//BATCH_SIZE
            if batch_idx % print_batches == print_batches-1:
                writer.add_scalar('loss', loss.item(), global_step)
                writer.add_scalar('train_acc', train_acc, global_step)
                logging.info("Epoch %d, query %d, loss: %.3f, acc: %.3f" % (epoch+1, (batch_idx+1)*BATCH_SIZE, running_loss/print_batches, running_acc/print_batches))
                running_loss = 0
                running_acc = 0

            # Log evaluations
            evaluate_batches = EVALUATE_EVERY//BATCH_SIZE
            if batch_idx % evaluate_batches == evaluate_batches-1:
                acc, val_loss = evaluate.evaluate(grounder, val_loader, summary_writer=writer, global_step=global_step)
                writer.add_scalar('val_acc', acc, global_step)
                writer.add_scalar('val_loss', val_loss, global_step)
                logging.info("Validation accuracy: %.3f, best_acc: %.3f" % (acc, best_acc))

                # Improved on validation set
                if acc > best_acc:
                    torch.save(grounder.state_dict(), Config.get('checkpoint'))
                    best_acc = acc

                grounder.train()

    writer.close()



if __name__ == "__main__":
    train()


