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
from torch import nn

from config import Config
from language_model import GloVe


class Flickr30K_Entities(torch.utils.data.Dataset):
    def __init__(self, image_ids, sent_deps, word2idx=None):
        super().__init__()

        self.queries = []
        self.proposals = {}
        self.img2idx = {}
        self.sent_deps = sent_deps

        if not word2idx:
            self.word2idx = {'UNK' : 0}

            # construct vocabulary based on all sentences in the training set
            for sent_id, sent_dict in sent_deps.items():
                for token in sent_dict['sent']:
                    if token not in self.word2idx:
                        self.word2idx[token] = len(self.word2idx)
        else:
            self.word2idx = word2idx

        # Load pretrained embeddings
        word_embedding_size = Config.get('word_emb_size')
        pretrained_path = Config.get('dirs.tmp.pretrained_embeddings')
        if os.path.exists(pretrained_path):
            pretrained_embeddings = np.load(pretrained_path)
        else:
            lm = GloVe(Config.get('language_model'), dim=word_embedding_size)
            pretrained_embeddings = np.array([lm.get_word_vector(w) for w in self.word2idx.keys()])
            np.save(pretrained_path, pretrained_embeddings)

        self.embeddings = nn.Embedding(len(pretrained_embeddings), word_embedding_size).from_pretrained(torch.from_numpy(pretrained_embeddings)).cuda()

        for im_id in image_ids:
            with open(os.path.join('annotations', im_id+'.pkl'), 'rb') as f:
                anno = pickle.load(f)

            self.proposals[im_id] = anno['proposals']

            query_count = 0
            for i, ppos_id in enumerate(anno['gt_ppos_ids']):
                ppos_id = ppos_id if ppos_id else len(self.proposals[im_id])
                phrase_dict = anno['phrases'][i]
                query_data = {'image_id'       : im_id, 
                              'sent_id'        : phrase_dict['sent_id'],
                              'first_word_idx' : phrase_dict['first_word_idx'],
                              'phrase'         : phrase_dict['phrase'],
                              'gt_ppos_id'     : ppos_id,
                              'gt_ppos_all'    : anno['gt_ppos_all'][i],
                              'gt_boxes'       : anno['gt_boxes'][i]}

                for j, w in enumerate(phrase_dict['phrase']):
                    if w not in self.word2idx:
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

        phrase = query['phrase']
        start = query['first_word_idx']
        end = start + len(phrase)
        phrase_indices_in_sent = torch.arange(start, end, dtype=torch.int64)

        X = self._get_sent_features(query['sent_id'])
        phrase_features = X[phrase_indices_in_sent]
        # print(self._get_features.cache_info())

        return query, proposal_features, phrase_features


    @lru_cache(maxsize=100) 
    def _get_vis_features(self, im_id):
        features = np.load(os.path.join(Config.get('dirs.features'), im_id+'.npy'))
        return features


    @lru_cache(maxsize=10) # cache size based on number of phrases per sentence
    def _get_sent_features(self, sent_id):
        sent_dict = self.sent_deps[sent_id]
        G = torch.FloatTensor(sent_dict['graph']).cuda()
        G = G + torch.eye(G.shape[0]).cuda()
        Dinv = torch.diag(1/torch.sum(G, axis=0))
        
        sent_indices = torch.LongTensor([self.word2idx[w] if w in self.word2idx else 0 for w in sent_dict['sent']])
        X = self.embeddings[sent_indices]
        X = Dinv@G@X
        return X


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



