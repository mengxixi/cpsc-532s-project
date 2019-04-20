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
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, pad_sequence, pack_padded_sequence
from torch.autograd import Variable
from torchvision import models, transforms
from torch.optim.lr_scheduler import MultiStepLR
from tensorboardX import SummaryWriter
from tqdm import tqdm
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu

from language_model import GloVe
from sGCN import sGCN
from config import Config

Config.load_config()

# logging configurations
LOG_FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")

PRETRAINED_EMBEDDINGS = Config.get('dirs.tmp.pretrained_embeddings')
WORD2IDX = Config.get('dirs.tmp.word2idx')
FLICKR30K_ENTITIES = Config.get('dirs.entities.root')
BATCH_SIZE = Config.get('batch_size')

PRINT_EVERY = Config.get('print_every') # Every x iterations
EVALUATE_EVERY = Config.get('evaluate_every')

class DecoderLSTM(nn.Module):
    def __init__(self, input_size=200, hidden_size=200, output_size=None): 
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.proj = nn.Linear(4096, 200)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, h0, c0, batch_size):
        output, (hn, cn) = self.lstm(x, (h0, c0)) 
        output = self.out(output)
        return output, (hn, cn)

    def initHidden(self, vis_features, sent_features, batch_size):
        h0 = self.dropout(self.proj(vis_features))
        h0 = h0.unsqueeze(0) + sent_features
        # h0 = torch.cat((h0, sent_features.squeeze(0)), dim=1).unsqueeze(0)
        return h0

    def initCell(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size).cuda()


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=200): 
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x) 
        return output, (hn, cn)


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

    vocabulary = list(word2idx.keys())
    return word2idx, vocabulary

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

    max_len = max(len(sdata['sent']) for sdata in sent_deps.values())

    word2idx, vocabulary = build_word2idx(sent_deps)
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

    optim = torch.optim.Adam(params, lr=Config.get('learning_rate'), weight_decay=Config.get('weight_decay'))
    scheduler = MultiStepLR(optim, milestones=[8, 15])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    writer = SummaryWriter(os.path.join('logs', subdir))
    writer.add_text('config', str(Config.CONFIG_DICT))

    best_loss = float("inf")
    for epoch in tqdm(range(20), file=sys.stdout):
        running_loss = 0
        writer.add_scalar('learning_rate', optim.param_groups[0]['lr'], epoch)
        scheduler.step()

        random.shuffle(train_ids)

        train_sent_ids = [im_id+'_'+str(sent_idx) for im_id in train_ids for sent_idx in range(5)]

        for batch_idx, idx in enumerate(range(0, len(train_sent_ids), BATCH_SIZE)):
            b_sent_ids = train_sent_ids[idx:idx+BATCH_SIZE]
            batch_size = len(b_sent_ids)
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
            seq_lengths = [len(s) for s in b_sentences]

            b_emb = pretrained_embeddings(b_padded_seq).permute(1,0,2)
            b_conv = gcn(b_emb, b_graphs, seq_lengths)

            b_conv_emb = []
            # TODO: Optimize this part for speed
            for i, sl in enumerate(seq_lengths):
                sent_emb = b_conv[i,:sl,:]
                eos_emb = pretrained_embeddings(torch.LongTensor([2]).cuda())
                sent_emb = torch.cat((sent_emb, eos_emb), dim=0)
                b_conv_emb.append(sent_emb)
            
            b_conv_emb = pack_sequence(b_conv_emb)

            # Encoding
            _, (hn, cn) = encoder(b_conv_emb)

            # Decoding
            decoder_hidden = decoder.initHidden(b_im_features, hn, batch_size)
            decoder_cell = cn

            decoder_input = pretrained_embeddings(torch.LongTensor([word2idx["<SOS>"]]).cuda()).repeat(batch_size, 1).unsqueeze(1)
            all_decoder_outputs = torch.zeros(max_sent_len+1, batch_size, vocab_size).cuda()

            b_input_seq = b_emb.permute(1,0,2)
            b_input_seq = torch.cat((b_input_seq, pretrained_embeddings(torch.LongTensor([word2idx["<EOS>"]]).cuda()).repeat(batch_size, 1).unsqueeze(0)), dim=0)

            b_target_seq = [torch.cat((seq, torch.tensor([word2idx['<EOS>']]).cuda())) for seq in b_seq]   
            b_target_seq = pad_sequence(b_target_seq, padding_value=-1)  

            EOS_mask = torch.ones(batch_size).type(torch.LongTensor).cuda()
            for di in range(max_sent_len+1):
                decoder_output, (decoder_hidden, decoder_cell) = decoder(decoder_input, decoder_hidden, decoder_cell, batch_size)
                all_decoder_outputs[di] = decoder_output.squeeze(1)
                if np.random.uniform() < 0.9:
                    decoder_input = b_input_seq[di].unsqueeze(1)
                else:
                    topv, topi = decoder_output.topk(1) 
                    topi = topi.view(-1)
                    decoder_input = pretrained_embeddings(topi).unsqueeze(1)

                    if di < max_sent_len-1:
                        EOS_mask[torch.nonzero((topi==word2idx["<EOS>"]))] = 0
                        b_target_seq[di+1, torch.nonzero(EOS_mask)] = -1

            loss = criterion(all_decoder_outputs.permute(1,2,0), b_target_seq.permute(1,0))
            loss.backward()
            optim.step()
            optim.zero_grad()

            running_loss += loss.item()
            global_step = epoch*len(train_sent_ids)+idx

            # Log losses
            print_batches = PRINT_EVERY//BATCH_SIZE
            if batch_idx % print_batches == print_batches-1:
                writer.add_scalar('loss', loss.item(), global_step)
                logging.info("Epoch %d, query %d, loss: %.3f" % (epoch+1, (batch_idx+1)*BATCH_SIZE, running_loss/print_batches))
                running_loss = 0

            # Log evaluations
            evaluate_batches = EVALUATE_EVERY//BATCH_SIZE
            if batch_idx % evaluate_batches == evaluate_batches-1:
                val_loss, val_bleu, sample_output_pairs = evaluate(val_ids, pretrained_embeddings, word2idx, vocabulary, max_len, gcn, encoder, decoder, sent_deps)
                writer.add_scalar('val_bleu', val_bleu, global_step)
                writer.add_scalar('val_loss', val_loss, global_step)
                logging.info("Validation loss: %.3f, best_loss: %.3f, bleu score: %.3f" % (val_loss, best_loss, val_bleu))

                # Improved on validation set
                if val_loss < best_loss:
                    torch.save(gcn.state_dict(), Config.get('sgcn_ckpt'))
                    torch.save(encoder.state_dict(), Config.get('sencoder_ckpt'))
                    torch.save(decoder.state_dict(), Config.get('sdecoder_ckpt'))
                    best_loss = val_loss

                gcn.train()
                encoder.train()
                decoder.train()
    writer.close()


def evaluate(ids, pretrained_embeddings, word2idx, vocabulary, max_length, gcn, encoder, decoder, sent_deps):
    gcn.eval()
    encoder.eval()
    decoder.eval()
    sent_ids = [im_id+'_'+str(sent_idx) for im_id in ids for sent_idx in range(5)]
    vocab_size = len(word2idx)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

    blue_total = 0.0
    loss_total = 0.0
    sample_output_pairs = []
    for batch_idx, idx in enumerate(range(0, len(sent_ids), BATCH_SIZE)):
        b_sent_ids = sent_ids[idx:idx+BATCH_SIZE]
        batch_size = len(b_sent_ids)
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
        seq_lengths = [len(s) for s in b_sentences]

        with torch.no_grad():
            b_emb = pretrained_embeddings(b_padded_seq).permute(1,0,2)
            b_conv = gcn(b_emb, b_graphs, seq_lengths)

            b_conv_emb = []
            # TODO: Optimize this part for speed
            for i, sl in enumerate(seq_lengths):
                sent_emb = b_conv[i,:sl,:]
                eos_emb = pretrained_embeddings(torch.LongTensor([2]).cuda())
                sent_emb = torch.cat((sent_emb, eos_emb), dim=0)
                b_conv_emb.append(sent_emb)
            
            
            b_conv_emb = pack_sequence(b_conv_emb)

            # Encoding
            _, (hn, cn) = encoder(b_conv_emb)

            # Decoding
            decoder_hidden = decoder.initHidden(b_im_features, hn, batch_size)
            decoder_cell = cn

            decoder_input = pretrained_embeddings(torch.LongTensor([word2idx["<SOS>"]]).cuda()).repeat(batch_size, 1).unsqueeze(1)
            all_decoder_outputs = torch.zeros(max_sent_len+1, batch_size, vocab_size).cuda()

            b_input_seq = b_emb.permute(1,0,2)
            b_input_seq = torch.cat((b_input_seq, pretrained_embeddings(torch.LongTensor([word2idx["<EOS>"]]).cuda()).repeat(batch_size, 1).unsqueeze(0)), dim=0)

            b_target_seq = [torch.cat((seq, torch.tensor([word2idx['<EOS>']]).cuda())) for seq in b_seq]   
            b_target_seq = pad_sequence(b_target_seq, padding_value=-1)  

            batch_output_idx = torch.zeros(max_sent_len+1, batch_size).cuda()

            for di in range(max_sent_len+1): 
                decoder_output, (decoder_hidden, decoder_cell) = decoder(decoder_input, decoder_hidden, decoder_cell, batch_size)
                all_decoder_outputs[di] = decoder_output.squeeze(1)
                topv, topi = decoder_output.topk(1)
                topi = topi.view(-1)
                batch_output_idx[di] = topi
                decoder_input = pretrained_embeddings(topi).unsqueeze(1)

            loss = criterion(all_decoder_outputs.permute(1,2,0), b_target_seq.permute(1,0))
            loss_total += (loss.item() * batch_size)

            batch_output_idx = batch_output_idx.transpose(0, 1).type(torch.LongTensor).cpu().numpy() 
            batch_output = []
            for i in range(batch_size):
                output_sent = []
                for ind in batch_output_idx[i]:
                    word = vocabulary[ind.item()] 
                    if word == "<EOS>":
                        break
                    output_sent.append(word)

                batch_output.append(output_sent)
                blue_total += compute_bleu(b_sentences[i], output_sent)
                if np.random.uniform() < 0.001:
                    sample_output_pairs.append((b_sentences[i], output_sent))
                    print("GT: %s" % ' '.join(b_sentences[i]))
                    print("RC: %s" % ' '.join(output_sent))

    return loss_total/len(sent_ids), blue_total/len(sent_ids), sample_output_pairs


def compute_bleu(reference_sentences, predicted_sentence): 
    """
        Given a list of reference sentences, and a predicted sentence, compute the BLEU similary between
        them.
    """
    return sentence_bleu(reference_sentences, predicted_sentence, weights=[1])


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