import os
import pickle

import imageio
import selectivesearch
from tqdm import tqdm


IMG_RAW_DIR = '/home/siyi/flickr30k-images'
MIN_SIZE = 10

with open('/home/siyi/flickr30k_entities/all.txt') as f:
    ALL_IDS = f.readlines()
ALL_IDS = [x.strip() for x in ALL_IDS]

for im_id in tqdm(ALL_IDS):
    image = imageio.imread(os.path.join(IMG_RAW_DIR, im_id+'.jpg'))
    _, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=MIN_SIZE)
    proposals = []
    for box in regions:
        if box['size'] < MIN_SIZE:
            continue

        rect = box['rect']
        proposals.append([rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]])

    if len(proposals) < 100:
        print(im_id)
        continue

    with open(os.path.join('tmp/selective_search', im_id+'.pkl'), 'wb') as f:
        pickle.dump({'boxes' : proposals[:100]}, f)
