import train
import torch
from torch.utils.data import Dataset, DataLoader
from unet import *
from utils import show_bw_and_rgb, batch_to_rgb
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import torchgeometry as tgm
import pytorch_ssim



def evaluate_test(test_loader, generator, window_size=5, max_show=10):
    with torch.no_grad():
        ct = 0
        mse_loss = 0
        ssim_loss = 0
        MSE = torch.nn.MSELoss(reduction='mean')
        SSIM = tgm.losses.SSIM(window_size, reduction='mean')
        for batch in test_loader:
            if len(batch) != train.batch_size:
                continue
            L = batch[:, 0].view(train.batch_size, 1, train.SIZE, train.SIZE).to(train.device)
            x = generator(L)
            x = torch.cat([L, x.to(train.device)], dim=1)
            rgb_batch = torch.tensor(batch_to_rgb(batch)).permute(0, 3, 1, 2)
            rgb_gen = torch.tensor(batch_to_rgb(x)).permute(0, 3, 1, 2)

            mse_loss += MSE(rgb_batch, rgb_gen)
            ssim_loss += 1 - (2*SSIM(rgb_batch, rgb_gen))
            ct += 1


        print("MSE_LOSS:", mse_loss/ct)
        print("SSIM_LOSS:", ssim_loss/ct)

        if max_show > 0:
            for batch in test_loader:
                L = batch[:, 0].view(train.batch_size, 1, train.SIZE, train.SIZE).to(train.device)
                x = generator(L)
                x = torch.cat([L, x.to(train.device)], dim=1)
                show_bw_and_rgb(batch_to_rgb(batch), batch_to_rgb(x), max_show=max_show)
                break


def main():

    gen = Unet_Gen(1, 2).to(train.device)
    disc = Unet_Disc(3, full_size=False).to(train.device)

    gen.load_state_dict(torch.load('./models/gen_full_unet_paper_20_epochs.pt'))
    disc.load_state_dict(torch.load('./models/disc_small_unet_paper_20_epochs.pt'))

    gen.eval()
    evaluate_test(train.test_loader, gen)

if __name__ == "__main__":
    main()