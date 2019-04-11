import os
import glob
import random
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision import transforms

import train
import util.misc as misc
from language_model import GloVe
from grounding import GroundeR
from config import Config
from util.iou import calc_iou_multiple, exact_group_union
from util.nms import nms


Config.load_config()

FLICKR30K_ENTITIES = Config.get('dirs.entities.root')
PRETRAINED_EMBEDDINGS = Config.get('dirs.tmp.pretrained_embeddings')
WORD2IDX = Config.get('dirs.tmp.word2idx')


def evaluate(model, validation_loader, summary_writer=None, global_step=None, n_samples=5):
    model.eval()
    queries = []
    preds = []
    corrects = []

    n_correct = 0.
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    val_loss = 0

    for batch_idx, data in enumerate(validation_loader):
        b_queries, b_pr_features, b_ph_features, b_y = data

        batch_size = len(b_queries)

        with torch.no_grad():
            # Forward
            lstm_h0 = model.initHidden(batch_size)
            lstm_c0 = model.initCell(batch_size)
            raw_attn_weights = model(b_pr_features, (lstm_h0, lstm_c0), b_ph_features, batch_size)

            val_loss += criterion(raw_attn_weights, b_y)

            # Get topk
            attn_weights = F.softmax(raw_attn_weights, dim=1)
            sorted_idx = torch.argsort(attn_weights, dim=1, descending=True)
            sorted_weights = torch.gather(attn_weights, 1, sorted_idx)

            topks = (sorted_weights.cumsum(dim=1) > Config.get('cumsum_cutoff')).sum(dim=1)

            # TODO: Log the probabilities as well?

            batch_pred = []
            for i, query in enumerate(b_queries):
                im_id = b_queries[i]['image_id']
                all_proposals = np.array(validation_loader.dataset.proposals[im_id])

                topv, topi = attn_weights[i,:].topk(topks[i])
                boxes_pred = all_proposals[topi.cpu().numpy()]
                pred_groups = exact_group_union(boxes_pred)
                boxes_pred = [box for g in pred_groups for box in nms(g)]

                boxes_true = all_proposals[b_queries[i]['gt_ppos_ids']]

                multi_iou = calc_iou_multiple(boxes_pred, boxes_true)
                if multi_iou >= Config.get('iou_threshold'):
                    n_correct += 1
                    corrects.append(1)
                else:
                    corrects.append(0)
                batch_pred.append(boxes_pred)

            # TODO: Remove debug prints
            # print(torch.log(attn_weights[0]))
            # print(y[0])
            # print(pred[0])

            # Save predictions for drawing
            queries.extend(b_queries)
            preds.extend(batch_pred)

    n_queries = len(validation_loader.dataset)
    acc = n_correct/n_queries
    val_loss = val_loss/n_queries

    # Drawing samples
    sample_queries, sample_preds, sample_corrects = zip(*random.sample(list(zip(queries, preds, corrects)), n_samples))

    loader = transforms.ToTensor()
    for (query, pred, correct) in zip(sample_queries, sample_preds, sample_corrects):
        image_id = query['image_id']
        proposal_bboxes = np.array(validation_loader.dataset.proposals[image_id])
        filename = os.path.join(Config.get('dirs.images.root'), image_id+'.jpg')

        if len(query['gt_ppos_ids']) == 0:
            # No gt positive proposals, don't bother drawing it
            continue

        image = misc.inference_image(filename, query['gt_boxes'], proposal_bboxes[query['gt_ppos_ids']], pred, correct, ' '.join(query['phrase']))

        # Saving
        if not global_step:
            image.save(os.path.join(Config.get('dirs.tmp.root'), '%s_%s.png' % (image_id, '_'.join(query['phrase']))), 'PNG')
        else:
            summary_writer.add_image('validation', loader(image), global_step)

    return acc, val_loss


if __name__ == "__main__":    
    pretrained_embeddings = np.load(PRETRAINED_EMBEDDINGS)
    with open(WORD2IDX, 'rb') as f:
        word2idx = pickle.load(f)

    with open(os.path.join(FLICKR30K_ENTITIES, Config.get('ids.test'))) as f1, open(os.path.join(FLICKR30K_ENTITIES, Config.get('ids.nobbox'))) as f2:
        val_ids = f1.read().splitlines()
        nobbox_ids = f2.read().splitlines()

    val_ids = [x for x in val_ids if x not in nobbox_ids]
    val_loader = train.get_dataloader(val_ids, word2idx=word2idx)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    writer = SummaryWriter(os.path.join('logs', subdir))

    grounder = GroundeR(pretrained_embeddings).cuda()
    grounder.load_state_dict(torch.load(Config.get('checkpoint')))
    acc, loss = evaluate(grounder, val_loader, writer, n_samples=20)
    print("Test Accuracy: %.3f, Loss: %.3f" % (acc, loss))

