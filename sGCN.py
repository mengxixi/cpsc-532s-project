import pickle

import numpy as np
import torch
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm

from language_model import GloVe
from config import Config

Config.load_config()


class sGCNDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, sent_deps, word2idx=None):
        super().__init__()

    def __len__(self):

    def __getitem__(self, index):

    # @lru_cache(maxsize=100) 



def train():
    # Load sentence dependencies
    with open(Config.get('dirs.tmp.sent_deps'), 'rb') as f:
        sent_deps = pickle.load(f)

    for sent_id, data in sent_deps.items():
        print(data)
        quit()


def generate_vgg_features():
    with open(os.path.join(Config.get('dirs.entities.root'), Config.get('ids.all'))) as f:
        ALL_IDS = f.readlines()
    ALL_IDS = [x.strip() for x in ALL_IDS]

    img_size = Config.get('crop_size')
    loader = transforms.Compose([
      transforms.Resize(img_size),
      transforms.CenterCrop(img_size),
      transforms.ToTensor(),
    ])
    def load_image(filename, volatile=False):
        """
        Simple function to load and preprocess the image.
    1. Open the image.
    2. Scale/crop it and convert it to a float tensor.
    3. Convert it to a variable (all inputs to PyTorch models must be variables). 4. Add another dimension to the start of the Tensor (b/c VGG expects a batch). 5. Move the variable onto the GPU.
    """
    image = Image.open(filename).convert('RGB')
    image_tensor = loader(image).float()
    image_var = Variable(image_tensor, volatile=volatile).unsqueeze(0)
    return image_var.cuda()


if __name__ == "__main__":
    train()