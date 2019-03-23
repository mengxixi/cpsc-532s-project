import os
import glob
import random
import pickle
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torchvision import transforms

import train
import util.misc as misc
from language_model import GloVe
from grounding import GroundeR



# TODO: Refactor constants
IMG_RAW_DIR = '/home/siyi/flickr30k-images'
FLICKR30K_ENTITIES = '/home/siyi/flickr30k_entities'


def evaluate(model, validation_loader, summary_writer=None, global_step=None, n_samples=5):
    model.eval()
    # TODO: Compute accuracy
    queries = []
    preds = []

    n_correct = 0
    criterion = torch.nn.NLLLoss(reduction='sum')
    val_loss = 0

    for batch_idx, data in enumerate(validation_loader):
        b_queries, b_pr_features, b_ph_features = data

        batch_size = len(b_queries)

        with torch.no_grad():
            # Forward
            lstm_h0 = model.initHidden(batch_size)
            lstm_c0 = model.initCell(batch_size)
            attn_weights = model(b_pr_features, (lstm_h0, lstm_c0), b_ph_features, batch_size)

            y =  torch.tensor([query['gt_ppos_id'] for query in b_queries]).cuda()
            val_loss += criterion(torch.log(attn_weights), y)

            # Get topk
            topv, topi = attn_weights.topk(1)
            # TODO: Log the probabilities as well
            pred = topi.squeeze(1)
            n_correct += sum(pred == y)

            # Save predictions for drawing
            queries.extend(b_queries)
            preds.extend(pred.cpu().numpy())


    n_queries = len(validation_loader.dataset)
    acc = n_correct.float()/n_queries
    val_loss = val_loss/n_queries

    # Drawing samples
    sample_queries, sample_preds = zip(*random.sample(list(zip(queries, preds)), n_samples))

    loader = transforms.ToTensor()
    for (query, pred) in zip(sample_queries, sample_preds):
        image_id = query['image_id']
        proposal_bboxes = validation_loader.dataset.proposals[image_id]
        filename = os.path.join(IMG_RAW_DIR, image_id+'.jpg')

        image = misc.inference_image(filename, query['gt_boxes'], [proposal_bboxes[query['gt_ppos_id']]], [proposal_bboxes[pred]], ' '.join(query['phrase']))

        # Saving
        if not global_step:
            image.save(os.path.join('tmp', '%s_%s.png' % (image_id, '_'.join(query['phrase']))), 'PNG')
        else:
            summary_writer.add_image('validation', loader(image), global_step)

    return acc, val_loss


if __name__ == "__main__":    

    with open(os.path.join(FLICKR30K_ENTITIES, 'val.txt')) as f1, open(os.path.join(FLICKR30K_ENTITIES, 'nobbox.txt')) as f2:
        # TODO: Format this line nicely
        val_ids = f1.read().splitlines()
        nobbox_ids = f2.read().splitlines()

    val_ids = [x for x in val_ids if x not in nobbox_ids]
    # lm = GloVe(os.path.join('models', 'glove', 'glove.twitter.27B.200d.txt'), dim=200)
    with open('tmp/vocabulary.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
    val_loader = train.get_dataloader(val_ids, vocabulary=vocabulary)

    grounder = GroundeR(lm_emb_size=len(vocabulary)).cuda()
    grounder.load_state_dict(torch.load(os.path.join("models", "grounder.ckpt")))

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    writer = SummaryWriter(os.path.join('logs', subdir))

    acc, loss = evaluate(grounder, val_loader, writer, n_samples=20, global_step=20)
    print("Accuracy: %.3f, Loss: %.3f" % (acc, loss))

