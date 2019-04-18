import os
import sys
import pickle
from functools import lru_cache

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.autograd import Variable
from torchvision import models, transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
from PIL import Image

from language_model import GloVe
from config import Config

Config.load_config()

PRETRAINED_EMBEDDINGS = Config.get('dirs.tmp.pretrained_embeddings')
WORD2IDX = Config.get('dirs.tmp.word2idx')
FLICKR30K_ENTITIES = Config.get('dirs.entities.root')


class DecoderLSTM(nn.Module):
    def __init__(self, input_size=200, hidden_size=200, output_size=100000): 
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj = nn.Linear(4096, 100)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        output, (hn, cn) = self.lstm(x, (h0, c0)) 
        output = self.out(output)
        return output, (hn, cn)

    def initHidden(self, vis_features, sent_features):
        h0 = self.proj(vis_features)
        print(h0.shape)
        print(sent_features.shape)
        quit()
        return h0

    def initCell(self):
        return torch.zeros(1, 1, self.hidden_size).cuda()


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=100): 
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x) 
        return output, (hn, cn)


class sGCN(nn.Module):
    def __init__(self, input_size, l1_size, output_size):
        super(sGCN, self).__init__()
        self.input_size = input_size
        self.l1_size = l1_size
        self.output_size = output_size

        self.proj1 = nn.Linear(input_size, l1_size)
        self.proj2 = nn.Linear(l1_size, output_size)

    def forward(self, x, G):
        Ghat = G + torch.eye(G.shape[0]).cuda()
        D = torch.diag(1/(0.5*torch.sum(G, dim=0)))
        C = torch.mm(D, torch.mm(Ghat, D))
        Conv1 = self.proj1(torch.mm(C, x))
        Conv2 = self.proj2(torch.mm(C, Conv1))

        return Conv2


@lru_cache(maxsize=100) 
def get_raw_vis_features(im_id):
    features = np.load(os.path.join(Config.get('dirs.raw_img_features'), im_id+'.npy'))
    return features


def build_word2idx(sent_deps):
    word2idx = {'UNK' : 0}

    # construct vocabulary based on all sentences in the training set
    for sent_id, sent_dict in sent_deps.items():
        for token in sent_dict['sent']:
            if token not in word2idx:
                word2idx[token] = len(word2idx)

    return word2idx

def train():
    # Load datasets
    with open(os.path.join(FLICKR30K_ENTITIES, Config.get('ids.train'))) as f1, open(os.path.join(FLICKR30K_ENTITIES, Config.get('ids.val'))) as f2,  open(os.path.join(FLICKR30K_ENTITIES, Config.get('ids.nobbox'))) as f3:
        train_ids = f1.read().splitlines()
        val_ids = f2.read().splitlines()
        nobbox_ids = f3.read().splitlines()

    train_ids = [x for x in train_ids if x not in nobbox_ids]
    val_ids = [x for x in val_ids if x not in nobbox_ids]

    # Load sentence dependencies
    with open(Config.get('dirs.tmp.sent_deps'), 'rb') as f:
        sent_deps = pickle.load(f)

    word2idx = build_word2idx(sent_deps)
    with open(WORD2IDX, 'wb') as f:
        pickle.dump(word2idx, f)

    # Load pretrained embeddings
    word_embedding_size = Config.get('word_emb_size')
    if os.path.exists(PRETRAINED_EMBEDDINGS):
        embeddings = np.load(PRETRAINED_EMBEDDINGS)
    else:
        lm = GloVe(Config.get('language_model'), dim=word_embedding_size)
        embeddings = np.array([lm.get_word_vector(w) for w in word2idx.keys()])
        np.save(PRETRAINED_EMBEDDINGS, embeddings)


    pretrained_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(embeddings)).cuda()
    
    vocab_size = embeddings.shape[0]

    gcn = sGCN(word_embedding_size, 100, 50).cuda()
    encoder = EncoderLSTM(50).cuda()
    decoder = DecoderLSTM(output_size=vocab_size).cuda()
    params = list(encoder.parameters())+list(gcn.parameters())+list(decoder.parameters())

    optim = torch.optim.Adam(params, lr=Config.get('learning_rate'))
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in tqdm(range(10), file=sys.stdout):
        running_loss = 0
        train_sent_ids = [im_id+'_'+str(sent_idx) for im_id in train_ids for sent_idx in range(5)]

        for sent_id in train_sent_ids:
            sentence = sent_deps[sent_id]['sent']
            G = torch.FloatTensor(sent_deps[sent_id]['graph']).cuda()
            im_features = get_raw_vis_features(sent_id.split('_')[0])

            s_tensor = torch.LongTensor([word2idx[w] for w in sentence]).cuda()
            s_emb = pretrained_embeddings(s_tensor)
            s_conv = gcn(s_emb, G).unsqueeze(1)
            
            # Encoding
            _, (hn, cn) = encoder(s_conv)

            # Decoding





def generate_vgg_raw_features():
    with open(os.path.join(Config.get('dirs.entities.root'), Config.get('ids.all'))) as f:
        ALL_IDS = f.readlines()
    ALL_IDS = [x.strip() for x in ALL_IDS]

    img_size = Config.get('crop_size')
    loader = transforms.Compose([
      transforms.Resize(img_size),
      transforms.CenterCrop(img_size),
      transforms.ToTensor(),
    ])

    def load_image(filename):
        """
        Simple function to load and preprocess the image.
        1. Open the image.
        2. Scale/crop it and convert it to a float tensor.
        3. Convert it to a variable (all inputs to PyTorch models must be variables). 4. Add another dimension to the start of the Tensor (b/c VGG expects a batch). 5. Move the variable onto the GPU.
        """
        image = Image.open(filename).convert('RGB')
        image_tensor = loader(image).float()
        image_var = Variable(image_tensor, requires_grad=False).unsqueeze(0)
        return image_var.cuda()


    vgg_model = models.vgg16(pretrained=True).cuda()
    vgg_model.classifier = nn.Sequential(*[vgg_model.classifier[i] for i in range(6)])
    vgg_model.eval()

    batch_size = 32
    for i in tqdm(range(0, len(ALL_IDS), batch_size)):
        batch_id = ALL_IDS[i:i+batch_size]
        batch_im = torch.cat([load_image(os.path.join(Config.get('dirs.images.root'), file+'.jpg')) for file in batch_id])
        batch_features = vgg_model(batch_im).detach().cpu().numpy()

        for im_id, feature in zip(batch_id, batch_features):
            np.save(os.path.join(Config.get('dirs.raw_img_features'), im_id+'.npy'), feature)


if __name__ == "__main__":
    train()