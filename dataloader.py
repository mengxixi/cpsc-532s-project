import os
import glob
import shutil
import pickle
from functools import lru_cache

import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pack_sequence


class Flickr30K_Entities(torch.utils.data.Dataset):
    def __init__(self, image_ids, language_model):
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
                                         'gt_boxes'   : anno['gt_boxes'][i]})
        self.lm = language_model


    def __len__(self):
        return len(self.queries)


    def __getitem__(self, index):
        query = self.queries[index]

        proposal_features = torch.FloatTensor(self._get_features(query['image_id'])).cuda()
        phrase_features = torch.FloatTensor([self.lm.get_word_vector(w) for w in query['phrase']]).cuda()
        # print(self._get_features.cache_info())

        return query, proposal_features, phrase_features


    @lru_cache(maxsize=100) 
    def _get_features(self, im_id):
        features = np.load(os.path.join('features', im_id+'.npy'))
        return features


    def collate_fn(self, data):
        sorted_data = zip(*sorted(data, key=lambda l:len(l[2]), reverse=True))
        queries, l_proposal_features, l_phrase_features = sorted_data        

        l_proposal_features = torch.stack(l_proposal_features, 0)
        l_phrase_features = pack_sequence(list(l_phrase_features))

        return list(queries), l_proposal_features, l_phrase_features

