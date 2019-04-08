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
from stanfordcorenlp import StanfordCoreNLP

import util.flickr30k_entities_utils as flickr30k 
from util.iou import calc_iou, rec_convex_hull_union
from config import Config

Config.load_config()


ANNO_RAW_DIR = Config.get('dirs.entities.anno')
SENT_RAW_DIR = Config.get('dirs.entities.sent')
IMG_RAW_DIR = Config.get('dirs.images.root')

CROP_SIZE = Config.get('crop_size')

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

    nlp = StanfordCoreNLP(Config.get('dirs.corenlp'))

    proposal_ub = 0
    n_queries = 0

    sent_deps = {}

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

        for sidx, sent_dict in enumerate(sent_data):
            sent_id = '%s_%d' % (fid, sidx)
            sentence = sent_dict['sentence']
            tokens = nlp.word_tokenize(sentence)
            dependencies = nlp.dependency_parse(sentence)

            # build adjacency matrix over sentence, assume undirected
            G = np.zeros((len(tokens), len(tokens)))
            for dep in dependencies:
                if dep[0] != 'ROOT':
                    tok1 = dep[1]-1
                    tok2 = dep[2]-1
                    G[tok1, tok2] = 1
                    G[tok2, tok1] = 1

            sent_deps[sent_id] = {'sent'  : tokens, 
                                  'graph' : G}

            for phrase in sent_dict['phrases']:
                phrase_id = phrase['phrase_id']

                if phrase_id not in boxes.keys():
                    # only care about phrases that actually has a corresponding box
                    continue

                clean_phrase = nlp.word_tokenize(re.sub(u"(\u2018|\u2019)", "'", phrase['phrase']))

                phrase_data = {'sent_id'        : sent_id,
                               'first_word_idx' : phrase['first_word_index'],
                               'phrase'         : clean_phrase}
                phrases.append(phrase_data)

                # Union all the gt boxes with its rectangular convex hull
                # Later we can get rid of this for finer-grained
                # multi-instance grounding
                union = rec_convex_hull_union(boxes[phrase_id])
                boxes[phrase_id] = [union]

                gt_boxes.append(boxes[phrase_id])

                pos_proposals = set()
                gt_ppos_id = None
                best_iou = 0.0
                for i, proposal in enumerate(proposal_boxes):
                    for gt in boxes[phrase_id]:
                        iou = calc_iou(proposal, gt)
                        if iou > Config.get('iou_threshold'):
                            pos_proposals.add(i)
                            if iou > best_iou:
                                best_iou = iou
                                gt_ppos_id = i
                gt_ppos_all.append(list(pos_proposals))
                gt_ppos_ids.append(gt_ppos_id)

                n_queries += 1
                if gt_ppos_id != None:
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


    with open(Config.get('dirs.tmp.sent_deps'), 'wb') as f:
        pickle.dump(sent_deps, f)

    print("Number of queries: %d" % n_queries)
    print("Proposal upper-bound: %.3f" % (proposal_ub/n_queries))


if __name__ == "__main__":
    preprocess_flickr30k_entities(get_features=False)
