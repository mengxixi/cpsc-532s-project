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
    def __init__(self, image_ids, language_model=None, vocabulary=None):
        super().__init__()

        self.queries = []
        self.proposals = {}
        self.img2idx = {}
        # TODO: Refactor vocabulary to preprocessing or somewhere outside
        # this class
        self.vocabulary = vocabulary if vocabulary else {'UNK' : 0} 

        for im_id in image_ids:
            with open(os.path.join('annotations', im_id+'.pkl'), 'rb') as f:
                anno = pickle.load(f)

            self.proposals[im_id] = anno['proposals']

            query_count = 0
            for i, ppos_id in enumerate(anno['gt_ppos_ids']):
                if ppos_id != -1:
                    # Ignore the queries that don't have a positive proposal
                    phrase = anno['phrases'][i]
                    self.queries.append({'image_id'   : im_id, 
                                         'phrase'     : phrase,
                                         'gt_ppos_id' : ppos_id,
                                         'gt_ppos_all': anno['gt_ppos_all'][i],
                                         'gt_boxes'   : anno['gt_boxes'][i]})
                    query_count += 1
                    if not language_model and not vocabulary:
                        for word in phrase:
                            if word not in self.vocabulary:
                                self.vocabulary[word] = len(self.vocabulary)

            if query_count > 0:
                self.img2idx[im_id] = {'start' : len(self.queries)-query_count,
                                       'len'   : query_count}

        # with open(os.path.join('tmp', 'vocabulary.pkl'), 'wb') as f:
        #     pickle.dump(self.vocabulary, f)
        # quit()
        self.image_ids = list(self.img2idx.keys())
        self.lm = language_model


    def __len__(self):
        return len(self.queries)


    def __getitem__(self, index):
        query = self.queries[index]

        proposal_features = torch.FloatTensor(self._get_features(query['image_id'])).cuda()
        if not self.lm:
            vocab_size = len(self.vocabulary)
            phrase_features = torch.zeros(vocab_size).repeat(len(query['phrase']),1).cuda()
            indices = []
            for w in query['phrase']:
                if w not in self.vocabulary:
                    indices.append(0)
                else:
                    indices.append(self.vocabulary[w])
            phrase_features.scatter_(1, torch.tensor(indices).unsqueeze(1).cuda(), 1)

        else:
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



