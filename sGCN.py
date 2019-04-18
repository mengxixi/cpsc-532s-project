import os
import pickle

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


def train():
    # Load sentence dependencies
    with open(Config.get('dirs.tmp.sent_deps'), 'rb') as f:
        sent_deps = pickle.load(f)

    for sent_id, data in sent_deps.items():
        print(sent_id)
        quit()


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