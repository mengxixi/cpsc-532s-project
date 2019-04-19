import os
import sys
import pickle
import random
import logging
from functools import lru_cache
from datetime import datetime

import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence
from torch.autograd import Variable
from torchvision import models, transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
from PIL import Image

from language_model import GloVe
from config import Config

Config.load_config()

# logging configurations
LOG_FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")

PRETRAINED_EMBEDDINGS = Config.get('dirs.tmp.pretrained_embeddings')
WORD2IDX = Config.get('dirs.tmp.word2idx')
FLICKR30K_ENTITIES = Config.get('dirs.entities.root')


class DecoderLSTM(nn.Module):
    def __init__(self, input_size=200, hidden_size=200, output_size=100000): 
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj = nn.Linear(4096, 100)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0, c0):
        output, (hn, cn) = self.lstm(x, (h0, c0)) 
        output = self.out(output)
        return output, (hn, cn)

    def initHidden(self, vis_features, sent_features, batch_size):
        h0 = self.proj(vis_features)
        h0 = torch.cat((h0, sent_features.view(-1)), dim=0)
        return h0.view(1, batch_size, self.hidden_size)

    def initCell(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda()


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
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, G):        
        eyes = torch.eye(G.shape[1]).repeat(x.shape[0], 1, 1).cuda()
        Ghat = G + eyes
        D = torch.diag_embed(0.5/torch.sum(Ghat, dim=1), dim1=1, dim2=2)

        C = torch.bmm(torch.bmm(D, Ghat), D)
        Conv1 = self.dropout(torch.relu(self.proj1(torch.bmm(C, x))))
        Conv2 = self.dropout(torch.relu(self.proj2(torch.bmm(C, Conv1))))

        return Conv2


@lru_cache(maxsize=100) 
def get_raw_vis_features(im_id):
    features = np.load(os.path.join(Config.get('dirs.raw_img_features'), im_id+'.npy'))
    return features


def build_word2idx(sent_deps):
    word2idx = {'UNK' : 0, '<SOS>' : 1, '<EOS>' : 2}

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

    gcn = sGCN(word_embedding_size, 100, word_embedding_size).cuda()
    encoder = EncoderLSTM(word_embedding_size).cuda()
    decoder = DecoderLSTM(output_size=vocab_size).cuda()
    params = list(encoder.parameters())+list(gcn.parameters())+list(decoder.parameters())

    optim = torch.optim.Adam(params, lr=Config.get('learning_rate'))
    criterion = torch.nn.CrossEntropyLoss()

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    writer = SummaryWriter(os.path.join('logs', subdir))
    writer.add_text('config', str(Config.CONFIG_DICT))

    batch_size = Config.get('batch_size')
    for epoch in tqdm(range(10), file=sys.stdout):
        running_loss = 0
        random.shuffle(train_ids)

        train_sent_ids = [im_id+'_'+str(sent_idx) for im_id in train_ids for sent_idx in range(5)]

        for idx in range(0, len(train_sent_ids), batch_size):
            b_sent_ids = train_sent_ids[idx:idx+batch_size]
            b_sentences = [sent_deps[sid]['sent'] for sid in b_sent_ids]
            max_sent_len = max(len(s) for s in b_sentences)
            b_graphs = []
            b_im_features = []
            b_seq = []
            for i, sid in enumerate(b_sent_ids):
                graph = torch.zeros(max_sent_len, max_sent_len).cuda()
                sl = len(b_sentences[i])
                graph[:sl, :sl] = torch.FloatTensor(sent_deps[sid]['graph']).cuda()
                b_graphs.append(graph)
                b_im_features.append(torch.FloatTensor(get_raw_vis_features(sid.split('_')[0])).cuda())
                b_seq.append(torch.tensor([word2idx[word] for word in b_sentences[i]]).cuda()) 

            b_sentences, b_graphs, b_im_features, b_seq = zip(*sorted(zip(b_sentences, b_graphs, b_im_features, b_seq), key=lambda l:len(l[0]), reverse=True))
            b_graphs = torch.stack(b_graphs)
            b_im_features = torch.stack(b_im_features)
            b_padded_seq = pad_sequence(b_seq)

            b_emb = pretrained_embeddings(b_padded_seq).permute(1,0,2)
            b_conv = gcn(b_emb, b_graphs)

            b_conv_emb = []
            # TODO: Optimize this part for speed
            for i, sent in enumerate(b_sentences):
                sent_emb = b_conv[i,:len(sent),:]
                eos_emb = pretrained_embeddings(torch.LongTensor([2]).cuda())
                sent_emb = torch.cat((sent_emb, eos_emb), dim=0)
                b_conv_emb.append(sent_emb)
            
            b_conv_emb = pack_sequence(b_conv_emb)

            # Encoding
            _, (hn, cn) = encoder(b_conv_emb)

            # Decoding
            decoder_hidden = decoder.initHidden(im_features, hn, batch_size)
            decoder_cell = decoder.initCell(batch_size)

            # TODO: This need to be batched <SOS>
            # TODO: Also need to modify all_decoder_outputs, target_seqs, padded with -1, etc.
            decoder_input = pretrained_embeddings(torch.LongTensor([1]).cuda()).unsqueeze(1)

            all_decoder_outputs = torch.zeros(len(sentence)+1, vocab_size).cuda()

            for di in range(len(sentence)):
                decoder_output, (decoder_hidden, decoder_cell) = decoder(decoder_input, decoder_hidden, decoder_cell)
                all_decoder_outputs[di] = decoder_output.view(-1)

                if np.random.uniform() < 0.9:
                    decoder_input = pretrained_embeddings(s_tensor[di]).view(1, 1, word_embedding_size)
                else:
                    topv, topi = decoder_output.topk(1) 
                    decoder_input = pretrained_embeddings(topi.view(-1)).view(1, 1, word_embedding_size)
                    if topi.item() == 2:
                        break

            target_seq = torch.cat((s_tensor, torch.LongTensor([2]).cuda()))
            loss = criterion(all_decoder_outputs, target_seq)
            loss.backward()
            optim.step()
            optim.zero_grad()

            running_loss += loss.item()

            global_step = epoch*len(train_sent_ids)+idx
            # Log losses
            if idx % 500 == 499:
                writer.add_scalar('loss', loss.item(), global_step)
                logging.info("Epoch %d, query %d, loss: %.3f" % (epoch+1, idx+1, running_loss/500))
                running_loss = 0


# def evaluate():


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