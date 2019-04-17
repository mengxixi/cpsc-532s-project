import pickle

import numpy as np
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from language_model import GloVe
from config import Config

Config.load_config()



def train():
    # Load sentence dependencies
    with open(Config.get('dirs.tmp.sent_deps'), 'rb') as f:
        sent_deps = pickle.load(f)

    for sent_id, data in sent_deps.items():
        print(data)
        quit()




if __name__ == "__main__":
    train()