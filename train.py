# import hw4
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage import color
from matplotlib import pyplot as plt

SIZE = 224
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')


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
        L = lab_img[[0], ...] / 50. - 1
        a = lab_img[[1], ...] / 110.
        b = lab_img[[2], ...] / 110.
        return L, a, b

    def __len__(self):
        return len(self.paths)

def batch_to_rgb(lab, zero_lab=False):
    print(len(lab), len(lab[0]), len(lab[0][0]), len(lab[0][0][0]), len(lab[0][0][0][0]))
    L = lab[0]
    a = lab[1]
    b = lab[2]
    L = (L + 1.) * 50
    if zero_lab:
        a = a * 0.
        b = b * 0.
    else:
        a = a * 110.
        b = b * 110.
    lab = torch.cat([L, a, b], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    print(lab.shape)
    rgb = color.lab2rgb(lab)
    print(type(rgb))
    return rgb

def show_batch(imgs, max_show=3):
    ct = min(max_show, len(imgs))
    for i in range(ct):
        ax = plt.subplot(1, 5, i+1)
        ax.imshow(imgs[i])
        ax.axis("off")
    plt.show()

def show_bw_and_rgb(bw_batch, rgb_batch, max_show=3):
    ct = min(max_show, len(bw_batch), len(rgb_batch))
    fig, axs = plt.subplots(ct, 2)
    for i in range(ct):
        axs[i, 0].imshow(bw_batch[i])
        axs[i, 1].imshow(rgb_batch[i])
        axs[i, 0].axis("off")
        axs[i, 1].axis("off")
    plt.show()


train_trans = transforms.Compose([transforms.Resize((SIZE, SIZE)),
                                  transforms.RandomHorizontalFlip()])  # TODO maybe rotate
# Load images (currently 8000 total images, can increase up to 20k if needed)
SEED = 420
batch_size = 128
img_path = './coco_sample/'
paths = os.listdir(img_path)

train_paths, test_paths = train_test_split(paths, test_size=.2, random_state=420)
coco_train = BWDataset(train_paths, img_path, train_trans)
train_loader = DataLoader(coco_train, batch_size=batch_size, drop_last=True)
print(len(train_loader.dataset))
coco_test = BWDataset(test_paths, img_path, transforms.Resize((SIZE, SIZE)))
test_loader = DataLoader(coco_test, batch_size=batch_size, drop_last=True)
print(len(test_loader.dataset))

for i in train_loader:
    # print(torch.tensor(i).shape)
    # show_batch(batch_to_rgb(i))
    show_bw_and_rgb(batch_to_rgb(i, zero_lab=True), batch_to_rgb(i))
    # print(i)
    break

