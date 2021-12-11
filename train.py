import utils
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary

import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage import color
from matplotlib import pyplot as plt
import eval
from unet import *
from sequential import *

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet



class BWDataset(Dataset):
    def __init__(self, paths, img_dir, transform=None):
        self.paths = paths
        self.transform = transform
        self.img_dir = img_dir

    def __getitem__(self, i):
        img = Image.open(self.img_dir + self.paths[i]).convert('RGB')
        img = self.transform(img)
        img = np.array(img)
        lab_img = color.rgb2lab(img).astype('float32')
        lab_img = transforms.ToTensor()(lab_img)
        lab_img[[0], ...] = lab_img[[0], ...] / 50. - 1
        lab_img[[1], ...] = lab_img[[1], ...] / 110.
        lab_img[[2], ...] = lab_img[[2], ...] / 110.
        return lab_img

    def __len__(self):
        return len(self.paths)


SEED = 420
SIZE = 256
batch_size = 16
dtype = torch.float
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
# device = torch.device('cuda:0')
print('using device:', device)

#
# Load images (currently 8000 total images, can increase up to 20k if needed)
#
img_path = './coco_sample/'
paths = os.listdir(img_path)
# paths = paths[:2000]

train_trans = transforms.Compose([transforms.Resize((SIZE, SIZE)),
                                  transforms.RandomHorizontalFlip()])  # TODO maybe rotate
train_paths, test_paths = train_test_split(paths, test_size=.2, random_state=SEED)
coco_train = BWDataset(train_paths, img_path, train_trans)
train_loader = DataLoader(coco_train, batch_size=batch_size, drop_last=True, pin_memory=True)

coco_test = BWDataset(test_paths, img_path, transforms.Resize((SIZE, SIZE)))
test_loader = DataLoader(coco_test, batch_size=batch_size, drop_last=True, pin_memory=True)


# Examine some images
# x = train_loader.__iter__().next()
# utils.show_batch(utils.batch_to_rgb(x))
# utils.show_bw_and_rgb(utils.batch_to_rgb(x, zero_lab=True), utils.batch_to_rgb(x), max_show=10)

def main():
    #
    # Create Models and optimizer
    #


    ###### FASTAI GEN #########
    # body = create_body(resnet18, pretrained=True, n_in=1, cut=-2)
    # gen = DynamicUnet(body, 2, (SIZE, SIZE)).to(device)


    ###### UNET/PAPER GEN ###########
    # gen = Unet_Gen(1, 2).to(device)
    # gen.apply(utils.initialize_weights)


    ####### Basic Residual Gen ########
    gen = Seq_Res_Gen(1, 2).to(device)
    gen.apply(utils.initialize_weights)

    ###### PAPER DISC (essentially patchgan) ##########
    # disc = Unet_Disc(3, full_size=False).to(device)
    # disc.apply(utils.initialize_weights)

    ###### SEQUENTIAL DISC ###########
    disc = Seq_Pool_Disc(3).to(device)
    disc.apply(utils.initialize_weights)

    gen_solver = utils.get_optimizer(gen)
    disc_solver = utils.get_optimizer(disc)

    # utils.pretrain(train_loader, gen, gen_solver, epochs=5, batch_size=batch_size, device=device)
    # torch.save(gen.state_dict(), "gen_pretrained_20_epochs.pt")

    # These give an overview of the networks
    # also, the gen summary has frozen my computer for a few seconds before so I will leave commented out for now
    # summary(gen, input_size=(batch_size, 1, 256, 256))
    # Note: the discriminator model in the paper says "the number of channels being doubled
    #  after each downsampling" but I haven't confirmed in the code if that's actually true
    #  as this gives a lot of parameters
    # summary(disc, input_size=(batch_size, 3, 256, 256))

    #
    # Train GAN
    #
    utils.run_a_gan(train_loader, disc, gen, disc_solver, gen_solver,
                    utils.discriminator_loss, utils.generator_loss,
                    device=device, size=SIZE, batch_size=batch_size, num_epochs=5,
                    show_every=25)
    torch.save(gen.state_dict(), './models/sequential_unet_gen_weights_5_epochs.pt')
    torch.save(disc.state_dict(), './models/sequential_disc_weights_5_epochs.pt')

    gen.eval()
    eval.evaluate_test(test_loader, gen)
    # t1 = test_loader.__iter__().next().to(device)
    # with torch.no_grad():
    #     L = t1[:, 0].view(batch_size, 1, SIZE, SIZE).to(device)
    #     x = gen(L)
    #     x = torch.cat([L, x.to(device)], dim=1)
    #     utils.show_bw_and_rgb(utils.batch_to_rgb(t1), utils.batch_to_rgb(x), max_show=10)


if __name__ == "__main__":
    main()

