import torch.nn as nn
import torch
from torch.nn import init
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from torch.nn.functional import l1_loss

from skimage import color
import matplotlib.pyplot as plt


def batch_to_rgb(lab, zero_lab=False):
    # Lab is of shape (batch_size=128, 3, size=224, size=224)
    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]
    L = (L + 1.) * 50.
    if zero_lab:
        a = a * 0.
        b = b * 0.
    else:
        a = a * 110.
        b = b * 110.
    lab = torch.stack([L, a, b]).permute(1, 2, 3, 0).cpu().numpy()
    rgb = color.lab2rgb(lab)
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
    fig, axs = plt.subplots(ct, 2, figsize=(4, 4*(int(ct/2))))

    for i in range(ct):
        axs[i, 0].imshow(bw_batch[i])
        axs[i, 1].imshow(rgb_batch[i])
        axs[i, 0].axis("off")
        axs[i, 1].axis("off")
    plt.show()


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


def initialize_weights(m, method='normal'):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        if method == 'normal':
            init.normal_(m.weight.data, mean=0.0, std=.02)
        elif method == 'xavier':
            init.xavier_uniform_(m.weight.data)
        else:
            raise Exception('Not a valid initialization method')


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    dev = logits_real.device
    real_loss = bce_loss(logits_real, torch.ones(logits_real.shape, device=dev))
    fake_loss = bce_loss(logits_fake, torch.zeros(logits_fake.shape, device=dev))
    loss = (real_loss + fake_loss) * .5  #TODO possibly better if multiplied by .5, test in colab
    loss = loss.to(dev)
    return loss


def generator_loss(logits_fake, fake_abs, real_abs, lambda_l1=100):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    dev = logits_fake.device
    loss = bce_loss(logits_fake, torch.ones(logits_fake.shape, device=dev))
    loss += l1_loss(fake_abs, real_abs) * lambda_l1  # Only compare ab channels
    loss = loss.to(dev)
    return loss


def get_optimizer(model, lr=.001):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    return optimizer


def run_a_gan(loader_train, D, G, D_solver, G_solver, discriminator_loss, generator_loss,
              device='cpu', size=256, show_every=250, batch_size=128, num_epochs=10):
    """
    Train a GAN!

    Inputs:
    - loader_train: Pytorch Dataloader with training data
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
        for x in loader_train:
            x = x.squeeze()
            if x.shape[0] != batch_size:
                continue
            D_solver.zero_grad()
            real_data = x.to(device)
            logits_real = D(real_data)

            L = real_data[:, 0].view(batch_size, 1, size, size)
            ab_preds = G(L).detach()
            fake_images = torch.cat([L, ab_preds], dim=1).detach()
            logits_fake = D(fake_images)

            d_total_error = discriminator_loss(logits_real, logits_fake)
            d_total_error.backward()
            D_solver.step()

            G_solver.zero_grad()
            gen_logits_fake = D(fake_images)
            g_error = generator_loss(gen_logits_fake, ab_preds, real_data[:, 1:])
            g_error.backward()
            G_solver.step()

            fake_images.cpu()
            if (iter_count % show_every == 0):
                print('Iter: {}, D: {:.4}, G:{:.4}\n'.format(iter_count, d_total_error.item(), g_error.item()))
                batch_to_rgb(fake_images)
                # show_bw_and_rgb(batch_to_rgb(x), batch_to_rgb(fake_images), max_show=2)
            iter_count += 1
            torch.cuda.empty_cache()


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.

    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    real_loss = (scores_real - 1) ** 2
    fake_loss = scores_fake ** 2
    loss = torch.sum((real_loss + fake_loss) / (2 * scores_real.shape[0]))
    return loss


def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.

    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    loss = (scores_fake - 1) ** 2
    loss /= (2 * scores_fake.shape[0])
    loss = torch.sum(loss)
    return loss
