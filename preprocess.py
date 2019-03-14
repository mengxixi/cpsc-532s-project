import os
import glob
import shutil
import argparse
import logging
import pickle

import flickr30k_entities_utils as flickr30k 

import numpy as np
from tqdm import tqdm


# logging configurations
LOG_FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")


ANNO_RAW_DIR = '/home/siyi/flickr30k_entities/Annotations/'
SENT_RAW_DIR = '/home/siyi/flickr30k_entities/Sentences/'


FEAT_DIR = 'features'
ANNO_DIR = 'annotations'


with open('/home/siyi/flickr30k_entities/all.txt') as f:
    ALL_IDS = f.readlines()
ALL_IDS = [x.strip() for x in ALL_IDS]


def generate_features():
    ## 100 features per image i.e. 100x4096 tensor per .npy
    raise NotImplementedError


def generate_annotations():

    for fid in tqdm(ALL_IDS):
        sent_file = os.path.join(SENT_RAW_DIR, fid+'.txt')
        anno_file = os.path.join(ANNO_RAW_DIR, fid+'.xml')
        sent_data = flickr30k.get_sentence_data(sent_file)
        anno_data = flickr30k.get_annotations(anno_file)

        boxes = anno_data['boxes'] 

        phrase_ids = set()
        phrases = []
        gt_boxes = []

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

        with open(os.path.join(ANNO_DIR, fid+'.pkl'), 'wb') as f:
            fdata = {'phrases' : phrases, 'gt_boxes' : gt_boxes}
            pickle.dump(fdata, f)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, choices=["feat", "anno"],default="anno")

    args = parser.parse_args()

    if args.mode == "anno":
        generate_annotations()
    else:
        generate_features()
