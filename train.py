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
        L = lab_img[[0], ...]
        a = lab_img[[1], ...]
        b = lab_img[[2], ...]
        return L, a, b

    def __len__(self):
        return len(self.paths)


train_trans = transforms.Compose([transforms.Resize((SIZE, SIZE)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomRotation(30)])
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
    print(i)
    break