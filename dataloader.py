import os
import glob
import shutil
import pickle
import random
from functools import lru_cache

import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pack_sequence


class Flickr30K_Entities(torch.utils.data.Dataset):
    def __init__(self, image_ids, word2idx=None):
        super().__init__()

        self.queries = []
        self.proposals = {}
        self.img2idx = {}
        self.word2idx = word2idx if word2idx else {'UNK' : 0}

        for im_id in image_ids:
            with open(os.path.join('annotations', im_id+'.pkl'), 'rb') as f:
                anno = pickle.load(f)

            self.proposals[im_id] = anno['proposals']

            query_count = 0
            for i, ppos_id in enumerate(anno['gt_ppos_ids']):
                ppos_id = ppos_id if ppos_id else len(self.proposals[im_id])
                query_data = {'image_id'   : im_id, 
                              'phrase'     : anno['phrases'][i],
                              'gt_ppos_id' : ppos_id,
                              'gt_ppos_all': anno['gt_ppos_all'][i],
                              'gt_boxes'   : anno['gt_boxes'][i]}

                for j, w in enumerate(anno['phrases'][i]):
                    if w not in self.word2idx:
                        if not word2idx:
                            # Train loader, building vocabulary
                            self.word2idx[w] = len(self.word2idx)
                        else:
                            # Validation/test loader, word unknown
                            query_data['phrase'][j] = 'UNK'

                self.queries.append(query_data)
                query_count += 1

            if query_count > 0:
                self.img2idx[im_id] = {'start' : len(self.queries)-query_count,
                                       'len'   : query_count}

        self.image_ids = list(self.img2idx.keys())
        self.idx2word = {idx : word for word, idx in self.word2idx.items()}


    def __len__(self):
        return len(self.queries)


    def __getitem__(self, index):
        query = self.queries[index]

        proposal_features = torch.FloatTensor(self._get_vis_features(query['image_id'])).cuda()
        phrase_indices = torch.LongTensor([self.word2idx[w] for w in query['phrase']]).cuda()
        # print(self._get_features.cache_info())

        return query, proposal_features, phrase_indices


    @lru_cache(maxsize=100) 
    def _get_vis_features(self, im_id):
        features = np.load(os.path.join('features', im_id+'.npy'))
        return features


    def collate_fn(self, data):
        sorted_data = zip(*sorted(data, key=lambda l:len(l[2]), reverse=True))
        queries, l_proposal_features, l_phrase_indices = sorted_data        

        l_proposal_features = torch.stack(l_proposal_features, 0)
        l_phrase_indices = pack_sequence(list(l_phrase_indices))

        return list(queries), l_proposal_features, l_phrase_indices


class QuerySampler(torch.utils.data.Sampler):
    """
    The idea is that we randomly shuffle the list of images first. Then we 
    construct the query pairs by grabbing phrases from each image in order 
    after shuffle. This way we reduce disk IO, since each image's proposal
    features now only needs to be loaded once per epoch (after caching).

    Example:

    Phrases (letter represents image id, then phrase index) -
    [a1, a2, a3, b1, b2, c1, c2, c3, c4, c5, d1, d2, d3]

    Shuflfed indices can be -
    [10, 11, 12, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]
    corresponding to shuffled images [d, b, c, a]

    """

    def __init__(self, data_source):
        super().__init__(data_source)

        self.data_source = data_source


    def __iter__(self):
        # Construct list of indices
        shuffled_images = random.shuffle(self.data_source.image_ids)
        indices = []
        for im_id in self.data_source.image_ids:
            start = self.data_source.img2idx[im_id]['start']
            n_queries = self.data_source.img2idx[im_id]['len']
            indices.extend(list(range(start, start+n_queries)))

        return iter(indices)


    def __len__(self):
        return len(self.data_source)



