import os
import glob
import random
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter
from PIL import Image, ImageDraw

import train
from language_model import GloVe
from grounding import GroundeR


# TODO: Refactor constants
IMG_RAW_DIR = '/home/siyi/flickr30k-images'


def evaluate(model, validation_loader, summary_writer=None):
    model.eval()
    # TODO: Compute accuracy
    queries = []
    preds = []

    n_correct = 0
    for batch_idx, data in enumerate(validation_loader):
        b_queries, b_pr_features, b_ph_features = data

        batch_size = len(b_queries)

        # Forward
        lstm_h0 = model.initHidden(batch_size)
        lstm_c0 = model.initCell(batch_size)
        attn_weights = model(b_pr_features, (lstm_h0, lstm_c0), b_ph_features, batch_size)

        # Get topk
        topv, topi = attn_weights.topk(1)
        # TODO: Log the probabilities as well

        pred = topi.squeeze(1).cpu().numpy()
        gt = np.array([query['gt_ppos_id'] for query in b_queries])
        n_correct += sum(pred == gt)

        # Save predictions for drawing
        queries.extend(b_queries)
        preds.extend(pred)

    acc = n_correct/len(validation_loader.dataset)
    print("accuracy: %.2f" % acc)  # TODO: Get rid of this? Reformat it?

    # Drawing
    sample_queries, sample_preds = zip(*random.sample(list(zip(queries, preds)), 20))

    for (query, pred) in zip(sample_queries, sample_preds):
        image_id = query['image_id']
        filename = os.path.join(IMG_RAW_DIR, image_id+'.jpg')
        image = Image.open(filename).convert('RGB')

        draw = ImageDraw.Draw(image)

        # Draw predicted bbox
        proposal_bboxes = train_loader.dataset.proposals[image_id]
        drawrect(draw, proposal_bboxes[query['gt_ppos_id']], outline='red', width=3)

        for box in query['gt_boxes']:
            # Draw all groundtruth bboxes
            drawrect(draw, box, outline='green', width=3)

        image.save(os.path.join('tmp', '%s_%s.png' % (image_id, ' '.join(query['phrase']))), 'PNG')

    return acc


# TODO: Put into util
# https://stackoverflow.com/questions/34255938/is-there-a-way-to-specify-the-width-of-a-rectangle-in-pil
def drawrect(drawcontext, xy, outline=None, width=1):
    x1 = xy[0]; y1 = xy[1]; x2=xy[2]; y2=xy[3]
    points = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)
    drawcontext.line(points, fill=outline, width=width)


if __name__ == "__main__":
    grounder = GroundeR().cuda()
    grounder.load_state_dict(torch.load(os.path.join("models", "grounder.ckpt")))

    # TODO: Temp stuff, load validation instead
    train_ids = [file.split('/')[-1][:-4] for file in glob.glob('/home/siyi/flickr_mini/*.pkl')]
    lm = GloVe(os.path.join('models', 'glove', 'glove.twitter.27B.200d.txt'), dim=200)
    train_loader = train.get_dataloader(train_ids, lm)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    writer = SummaryWriter(os.path.join('logs', subdir))

    evaluate(grounder, train_loader, writer)
