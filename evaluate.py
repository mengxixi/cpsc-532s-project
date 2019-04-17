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
from config import Config


Config.load_config()

FLICKR30K_ENTITIES = Config.get('dirs.entities.root')
PRETRAINED_EMBEDDINGS = Config.get('dirs.tmp.pretrained_embeddings')
WORD2IDX = Config.get('dirs.tmp.word2idx')


def evaluate(model, validation_loader, sent_deps, summary_writer=None, global_step=None, n_samples=5):
    model.eval()
    queries = []
    preds = []
    corrects = []
    probs = []

    n_correct = 0.
    criterion = torch.nn.NLLLoss(reduction='sum', ignore_index=Config.get('n_proposals'))
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

            pred = topi.squeeze(1)
            prob = topv.squeeze(1)
            for i, query in enumerate(b_queries):
                probs.append(prob[i])
                if pred[i] in query['gt_ppos_all']:
                    n_correct += 1
                    corrects.append(1)
                else:
                    corrects.append(0)

            # print(torch.log(attn_weights[0]))
            # print(y[0])
            # print(pred[0])

            # Save predictions for drawing
            queries.extend(b_queries)
            preds.extend(pred.cpu().numpy())

    n_queries = len(validation_loader.dataset)
    acc = n_correct/n_queries
    val_loss = val_loss/n_queries

    # Drawing samples
    sample_queries, sample_preds, sample_probs, sample_corrects = zip(*random.sample(list(zip(queries, preds, probs, corrects)), n_samples))

    loader = transforms.ToTensor()
    for (query, pred, prob, correct) in zip(sample_queries, sample_preds, sample_probs, sample_corrects):
        image_id = query['image_id']
        sentence = ' '.join(sent_deps[query['sent_id']]['sent'])
        first_word_idx = query['first_word_idx']
        proposal_bboxes = validation_loader.dataset.proposals[image_id]
        filename = os.path.join(Config.get('dirs.images.root'), image_id+'.jpg')

        if query['gt_ppos_id'] == len(proposal_bboxes):
            continue

        query_acc = '1, prob:%.3f' % prob if correct==1 else '0'
            
        image = misc.inference_image(filename, query['gt_boxes'], [proposal_bboxes[query['gt_ppos_id']]], [proposal_bboxes[pred]], query_acc, ' '.join(query['phrase']), sentence, first_word_idx)

        # Saving
        if not global_step:
            image.save(os.path.join(Config.get('dirs.tmp.root'), '%s_%s.png' % (image_id, '_'.join(query['phrase']))), 'PNG')
        else:
            summary_writer.add_image('validation', loader(image), global_step)

    return acc, val_loss


if __name__ == "__main__":    
    with open(WORD2IDX, 'rb') as f:
        word2idx = pickle.load(f)

    # Load sentence dependencies
    with open(Config.get('dirs.tmp.sent_deps'), 'rb') as f:
        sent_deps = pickle.load(f)

    with open(os.path.join(FLICKR30K_ENTITIES, Config.get('ids.test'))) as f1, open(os.path.join(FLICKR30K_ENTITIES, Config.get('ids.nobbox'))) as f2:
        val_ids = f1.read().splitlines()
        nobbox_ids = f2.read().splitlines()

    val_ids = [x for x in val_ids if x not in nobbox_ids]
    val_loader = train.get_dataloader(val_ids, sent_deps, word2idx=word2idx)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    writer = SummaryWriter(os.path.join('logs', subdir))
    
    word_embedding_size = Config.get('word_emb_size')
    hidden_size = Config.get('hidden_size')
    concat_size = Config.get('concat_size')
    n_proposals = Config.get('n_proposals')
    im_feat_size = Config.get('im_feat_size')

    grounder = GroundeR(im_feature_size=im_feat_size, lm_emb_size=word_embedding_size, hidden_size=hidden_size, concat_size=concat_size, output_size=n_proposals).cuda()

    grounder.load_state_dict(torch.load(Config.get('checkpoint')))
    acc, loss = evaluate(grounder, val_loader, sent_deps, writer, n_samples=20)
    print("Test Accuracy: %.3f, Loss: %.3f" % (acc, loss))

