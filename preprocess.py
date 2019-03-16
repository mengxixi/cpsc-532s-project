import os
import glob
import shutil
import logging
import pickle
import uuid

import flickr30k_entities_utils as flickr30k 
from util.iou import calc_iou

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms
from tqdm import tqdm
from PIL import Image


# logging configurations
LOG_FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")


ANNO_RAW_DIR = '/home/siyi/flickr30k_entities/Annotations/'
SENT_RAW_DIR = '/home/siyi/flickr30k_entities/Sentences/'
IMG_RAW_DIR = '/home/siyi/flickr30k-images'


FEAT_DIR = 'features'
ANNO_DIR = 'annotations'

CROP_SIZE = 224
FEATURE_SIZE = 4096
IOU_THRESHOLD = 0.5


with open('/home/siyi/flickr30k_entities/all.txt') as f:
    ALL_IDS = f.readlines()
ALL_IDS = [x.strip() for x in ALL_IDS]


def load_crop(filename, box):
    loader = transforms.Compose([
        ## TODO: Cropped bbox isn't a square, but VGG requires squared input, is this ok?
        transforms.Resize((CROP_SIZE,CROP_SIZE)),
        transforms.ToTensor(),
    ])

    image = Image.open(filename).convert('RGB')
    crop = transforms.functional.crop(image, box[0], box[1], box[3]-box[1], box[2]-box[0])

    image_tensor = loader(crop).float()
    image_var = Variable(image_tensor).unsqueeze(0)
    return image_var.cuda()


def generate_features(im_file, boxes, model):

    crops = [load_crop(im_file, box) for box in boxes]

    features = torch.Tensor(len(crops), FEATURE_SIZE)

    for i in range(0, len(crops), 16):
        batch = crops[i:i+16]
        batch_crops = torch.cat(batch)
        features[i:i+16] = model(batch_crops)

    return features.detach().numpy()


def preprocess_flickr30k_entities():
    vgg_model = models.vgg16(pretrained=True).cuda()
    vgg_model.classifier = nn.Sequential(*[vgg_model.classifier[i] for i in range(6)])
    vgg_model.eval()

    ### TODO: get rid of this when ready
    flickr_mini = glob.glob('/home/siyi/flickr_mini/*.pkl')
    for file in tqdm(flickr_mini):
        fid = file.split('/')[-1][:-4]

    # for fid in tqdm(ALL_IDS):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            
        proposal_boxes = [list(map(int, box[:4])) for box in data['boxes']]
        ## TODO: remove this 
        proposal_boxes = list(filter(lambda box: not all(v==0 for v in box), proposal_boxes))

        sent_file = os.path.join(SENT_RAW_DIR, fid+'.txt')
        anno_file = os.path.join(ANNO_RAW_DIR, fid+'.xml')
        sent_data = flickr30k.get_sentence_data(sent_file)
        anno_data = flickr30k.get_annotations(anno_file)

        boxes = anno_data['boxes'] 

        phrase_ids = set()
        phrases = []
        gt_boxes = []
        gt_ppos_all = []

        for sent in sent_data:
            for phrase in sent['phrases']:
                phrase_id = phrase['phrase_id']

                if phrase_id in phrase_ids:
                    # same phrase already parsed in this file
                    continue

                if phrase_id not in boxes.keys():
                    # only care about phrases that actually has a corresponding box
                    continue

                phrase_ids.add(phrase_id)

                phrases.append(phrase['phrase'])
                gt_boxes.append(boxes[phrase_id])

                pos_proposals = []
                for i, proposal in enumerate(proposal_boxes):
                    ## TODO: instead of comparing with the union of the gt, i'm just checking if it overlaps with any of the gt boxes for now, may have to change to match with literature..
                    for gt in boxes[phrase_id]:
                        if calc_iou(proposal, gt) > IOU_THRESHOLD:
                            pos_proposals.append(i)
                            break

        if len(phrases) > 0:
            with open(os.path.join(ANNO_DIR, fid+'.pkl'), 'wb') as f:
                fdata = {'phrases'      : phrases, 
                         'gt_boxes'     : gt_boxes,
                         'proposals'    : proposal_boxes,
                         'gt_ppos_all'  : gt_ppos_all,}
                pickle.dump(fdata, f)

            features = generate_features(os.path.join(IMG_RAW_DIR, fid+'.jpg'), proposal_boxes, vgg_model)
            np.save(os.path.join(FEAT_DIR, fid+'.npy'), features)

        else:
            logging.info("No boxes annotated for %s.jpg" % fid)

        # TODO: only keep proposals that has a significant overlap with one of the GT boxes??
        # TODO: union the gt boxes for each phrase (if more than one gt box?)



if __name__ == "__main__":
    preprocess_flickr30k_entities()
