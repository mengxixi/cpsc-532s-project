import os
import glob
import shutil
import pickle
from functools import lru_cache

import torch
import torch.utils.data
import numpy as np

from language_model import GloVe


class Flickr30K_Entities(torch.utils.data.Dataset):
    def __init__(self, image_ids):
        self.queries = []
        self.proposals = {}

        for im_id in image_ids:
            with open(os.path.join('annotations', im_id+'.pkl'), 'rb') as f:
                anno = pickle.load(f)

            self.proposals[im_id] = anno['proposals']

            for i, ppos_id in enumerate(anno['gt_ppos_ids']):
                if ppos_id != -1:
                    # Ignore the queries that don't have a positive proposal
                    self.queries.append({'image_id'   : im_id, 
                                         'phrase'     : anno['phrases'][i],
                                         'gt_ppos_id' : ppos_id,
                                         'gt_ppos_all': anno['gt_ppos_all'][i],
                                         'gt_boxes'   : anno['gt_boxes']})
        # Init language model
        self.lm = GloVe(os.path.join('models', 'glove.twitter.27B.50d.txt'), dim=50)


    def __len__(self):
        return len(self.queries)


    def __getitem__(self, index):
        query = self.queries[index]

        proposal_features = self._get_features(query['image_id'])
        print(query['phrase'])
        phrase_features = None
        # print(self._get_features.cache_info())

        # TODO: put htings onto cuda
        return query, proposal_features#, phrase_features


    @lru_cache(maxsize=100) 
    def _get_features(self, im_id):
        features = np.load(os.path.join('features', im_id+'.npy'))
        print(features.shape)
        return features


