import os
import glob
import shutil
import re
import pickle
import uuid

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image
from nltk import word_tokenize

from config import Config
import util.flickr30k_entities_utils as flickr30k 
from util.iou import calc_iou, calc_iou_multiple

Config.load_config()


ANNO_RAW_DIR = Config.get('dirs.entities.anno')
SENT_RAW_DIR = Config.get('dirs.entities.sent')
IMG_RAW_DIR = Config.get('dirs.images.root')

CROP_SIZE = Config.get('crop_size')
IOU_THRESHOLD = Config.get('iou_threshold')

with open(os.path.join(Config.get('dirs.entities.root'), Config.get('ids.all'))) as f:
    ALL_IDS = f.readlines()
ALL_IDS = [x.strip() for x in ALL_IDS]


def load_crop(filename, box):
    loader = transforms.Compose([
        transforms.Resize((CROP_SIZE,CROP_SIZE)),
        transforms.ToTensor(),
    ])

    image = Image.open(filename).convert('RGB')
    crop = transforms.functional.crop(image, box[0], box[1], box[3]-box[1], box[2]-box[0])

    image_tensor = loader(crop).float()
    image_var = Variable(image_tensor).unsqueeze(0)
    return image_var.cuda()


def generate_features(im_file, boxes, model):
    batch_size =32
    crops = [load_crop(im_file, box) for box in boxes]

    features = np.empty([len(crops), Config.get('im_feat_size')])

    for i in range(0, len(crops), batch_size):
        batch = crops[i:i+batch_size]
        batch_crops = torch.cat(batch)
        features[i:i+batch_size] = model(batch_crops).detach().cpu().numpy()

    return features


def preprocess_flickr30k_entities(get_features=True):
    vgg_model = models.vgg16(pretrained=True).cuda()
    vgg_model.classifier = nn.Sequential(*[vgg_model.classifier[i] for i in range(6)])
    vgg_model.eval()

    proposal_ub = 0
    n_queries = 0
    for fid in tqdm(ALL_IDS):
        file = os.path.join(IMG_RAW_DIR, Config.get('dirs.images.proposals'), fid+'.pkl')
        if not os.path.exists(file):
            continue

        with open(file, 'rb') as f:
            data = pickle.load(f)
            
        proposal_boxes = [list(map(int, box[:4])) for box in data['boxes']]

        sent_file = os.path.join(SENT_RAW_DIR, fid+'.txt')
        anno_file = os.path.join(ANNO_RAW_DIR, fid+'.xml')
        sent_data = flickr30k.get_sentence_data(sent_file)
        anno_data = flickr30k.get_annotations(anno_file)

        boxes = anno_data['boxes'] 

        phrases = []
        gt_boxes = []
        gt_ppos_all = []
        gt_ppos_ids = []

        for sent in sent_data:
            for phrase in sent['phrases']:
                phrase_id = phrase['phrase_id']

                if phrase_id not in boxes.keys():
                    # only care about phrases that actually has a corresponding box
                    continue

                clean_phrase = re.sub(u"(\u2018|\u2019)", "'", phrase['phrase'])
                phrases.append(word_tokenize(clean_phrase))
                gt_boxes.append(boxes[phrase_id])

                pos_proposals = set()

                good_ids = set()
                for gt in boxes[phrase_id]:
                    # TODO: Greedy way of finding the best set of proposals to
                    # be used as target labels for training, refine to make UB 
                    # higher?
                    best_match = -1
                    best_iou = 0.0
                    for i, proposal in enumerate(proposal_boxes):
                        if i in good_ids:
                            # Already matched with another gt box
                            continue
                        iou = calc_iou(proposal, gt)
                        if iou >= IOU_THRESHOLD:
                            pos_proposals.add(i)
                            if iou > best_iou:
                                best_iou = iou 
                                best_match = i

                    if best_match != -1:
                        # Found the best matching proposal for this gt box
                        good_ids.add(best_match)

                gt_ppos_all.append(list(pos_proposals))
                gt_ppos_ids.append(list(good_ids))

                # Proposal upper-bound stats
                n_queries += 1

                iou_multiple = calc_iou_multiple(boxes[phrase_id], [proposal_boxes[i] for i in good_ids])
                if iou_multiple >= IOU_THRESHOLD:
                    proposal_ub += 1

        if len(phrases) > 0:
            with open(os.path.join(Config.get('dirs.annotations'), fid+'.pkl'), 'wb') as f:
                fdata = {'phrases'      : phrases, 
                         'gt_boxes'     : gt_boxes,
                         'proposals'    : proposal_boxes,
                         'gt_ppos_all'  : gt_ppos_all,
                         'gt_ppos_ids'  : gt_ppos_ids,}

                pickle.dump(fdata, f)

            if get_features:
                features = generate_features(os.path.join(IMG_RAW_DIR, fid+'.jpg'), proposal_boxes, vgg_model)
                np.save(os.path.join(Config.get('dirs.features'), fid+'.npy'), features)

        else:
            print("No boxes annotated for %s.jpg" % fid)

    print("Number of queries: %d" % n_queries)
    print("Proposal upper-bound: %.3f" % (proposal_ub/n_queries))


if __name__ == "__main__":
    preprocess_flickr30k_entities(get_features=False)
