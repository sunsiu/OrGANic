import math
import torch.nn as nn
import torch
from torch.nn import init
import torchvision.transforms as T
import torchvision.datasets as dset
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# mnist_train = dset.MNIST('./MNIST_data', train=True, download=True,
#                            transform=T.ToTensor())
# loader_train = DataLoader(mnist_train, batch_size=128,
#                           shuffle=True, drop_last=True, num_workers=2)
dtype = torch.float
if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')

print('using device:', device)

def show_images(images):
  images = torch.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
  sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
  sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))

  fig = plt.figure(figsize=(sqrtn, sqrtn))
  gs = gridspec.GridSpec(sqrtn, sqrtn)
  gs.update(wspace=0.05, hspace=0.05)

  for i, img in enumerate(images):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(img.reshape([sqrtimg,sqrtimg]))
  return

def count_params(model):
  """Count the number of parameters in the model"""
  param_count = sum([p.numel() for p in model.parameters()])
  return param_count


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)


def sample_noise(batch_size, dim, dtype=torch.float, device='cpu'):
    """
    Generate a PyTorch Tensor of uniform random noise.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, dim) containing uniform
      random noise in the range (-1, 1).
    """
    ##############################################################################
    # TODO: Implement sample_noise.                                              #
    ##############################################################################
    # Replace "pass" statement with your code
    r = torch.rand((batch_size, dim), dtype=dtype, device=device)
    return (r * 2) - 1
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

def discriminator():
  """
  Build and return a PyTorch model implementing the architecture above.
  """
  model = nn.Sequential(
    ############################################################################
    # TODO: Implement discriminator.                                           #
    ############################################################################
    # Replace "pass" statement with your code
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.LeakyReLU(negative_slope=0.01),
    nn.Linear(256, 256),
    nn.LeakyReLU(negative_slope=0.01),
    nn.Linear(256, 1),
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
  )
  return model


def generator(input_dim, output_dim, size):
    """
    Build and return U-Net with a resnet body
    """
    body = create_body(resnet18, pretrained=True, n_in=input_dim, cut=-2)  # Don't use last 2 activation layers
    model = DynamicUnet(body, output_dim, (size, size))
    print(model)
    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement discriminator_loss.                                        #
    ##############################################################################
    # Replace "pass" statement with your code
    dev = logits_real.device
    real_loss = bce_loss(logits_real, torch.ones(logits_real.shape, device=dev))
    fake_loss = bce_loss(logits_fake, torch.zeros(logits_fake.shape, device=dev))
    loss = real_loss + fake_loss
    loss = loss.to(dev)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # TODO: Implement generator_loss.                                            #
    ##############################################################################
    # Replace "pass" statement with your code
    dev = logits_fake.device
    loss = bce_loss(logits_fake, torch.ones(logits_fake.shape, device=dev))
    loss = loss.to(dev)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    # Replace "pass" statement with your code
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def run_a_gan(loader_train, D, G, D_solver, G_solver, discriminator_loss, generator_loss, show_every=250,
              batch_size=128, noise_size=96, num_epochs=10):
    """
    Train a GAN!

    Inputs:
    - D, G: PyTorch models for the discriminator and generator
    - D_solver, G_solver: torch.optim Optimizers to use for training the
      discriminator and generator.
    - discriminator_loss, generator_loss: Functions to use for computing the generator and
      discriminator loss, respectively.
    - show_every: Show samples after every show_every iterations.
    - batch_size: Batch size to use for training.
    - noise_size: Dimension of the noise to use as input to the generator.
    - num_epochs: Number of epochs over the training dataset to use for training.
    """
    iter_count = 0
    for epoch in range(num_epochs):
        for x, _ in loader_train:
            if len(x) != batch_size:
                continue
            D_solver.zero_grad()
            real_data = x.to(device)
            logits_real = D(2 * (real_data - 0.5))

            g_fake_seed = sample_noise(batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
            fake_images = G(g_fake_seed).detach()
            logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            D_solver.step()

            G_solver.zero_grad()
            g_fake_seed = sample_noise(batch_size, noise_size, dtype=real_data.dtype, device=real_data.device)
            fake_images = G(g_fake_seed)

            gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
            g_error = generator_loss(gen_logits_fake)
            g_error.backward()
            G_solver.step()

            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count, d_total_error.item(), g_error.item()))
                imgs_numpy = fake_images.data.cpu()  # .numpy()
                show_images(imgs_numpy[0:16])
                plt.show()
                print()
            iter_count += 1


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_discriminator_loss.                                     #
    ##############################################################################
    # Replace "pass" statement with your code
    real_loss = (scores_real - 1) ** 2
    fake_loss = scores_fake ** 2
    loss = torch.sum((real_loss + fake_loss) / (2 * scores_real.shape[0]))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = None
    ##############################################################################
    # TODO: Implement ls_generator_loss.                                         #
    ##############################################################################
    # Replace "pass" statement with your code
    loss = (scores_fake - 1) ** 2
    loss /= (2 * scores_fake.shape[0])
    loss = torch.sum(loss)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


# data = next(enumerate(loader_train))[-1][0].to(dtype=dtype, device=device)
# batch_size = data.size(0)
# # print(data.shape)
# c = data.size(1)
# h = data.size(2)
# w = data.size(3)
# b = build_dc_classifier().to(device)
# out = b(data)
# print(out.size())



# test_g_gan = build_dc_generator().to(device)
# test_g_gan.apply(initialize_weights)

# fake_seed = torch.randn(batch_size, NOISE_DIM, dtype=dtype, device=device)
# fake_images = test_g_gan.forward(fake_seed)
# fake_images.size()