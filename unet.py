import torch
from torch import nn

class Unet_Gen(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Paper uses 4 x 4 kernel here but I can't get the padding to lineup
        # without using the padding='same' like they used in TF. Could guess
        # sides, but I think this is probably fine
        self.downconv1 = down_layer(in_channels, 64, kernel_size=(3, 3), stride=1, act=nn.LeakyReLU(negative_slope=0.2))

        self.downconv2 = down_layer(64, 64, act=nn.LeakyReLU(negative_slope=0.2))
        self.downconv3 = down_layer(64, 128, act=nn.LeakyReLU(negative_slope=0.2))
        self.downconv4 = down_layer(128, 256, act=nn.LeakyReLU(negative_slope=0.2))
        self.downconv5 = down_layer(256, 512, act=nn.LeakyReLU(negative_slope=0.2))
        self.downconv6 = down_layer(512, 512, act=nn.LeakyReLU(negative_slope=0.2))
        self.downconv7 = down_layer(512, 512, act=nn.LeakyReLU(negative_slope=0.2))
        self.downconv8 = down_layer(512, 512, act=nn.LeakyReLU(negative_slope=0.2))

        self.upconv8 = up_layer(512, 512, act=nn.ReLU())
        self.upconv7 = up_layer(512*2, 512, act=nn.ReLU())
        self.upconv6 = up_layer(512*2, 512, act=nn.ReLU())
        self.upconv5 = up_layer(512*2, 256, act=nn.ReLU())
        self.upconv4 = up_layer(256*2, 128, act=nn.ReLU())
        self.upconv3 = up_layer(128*2, 64, act=nn.ReLU())
        self.upconv2 = up_layer(64*2, 64, act=nn.ReLU())
        # self.upconv1 = up_layer(64*2, out_channels, act=nn.ReLU())

        self.final_layer = nn.Sequential(
            nn.Conv2d(64 * 2, out_channels, kernel_size=(1, 1), stride=1, padding=0),
            nn.Tanh())

    def __call__(self, x):

        # start 3x256x256
        downlayer_1 = self.downconv1(x)            # 64x256x256
        downlayer_2 = self.downconv2(downlayer_1)  # 64x128x128
        downlayer_3 = self.downconv3(downlayer_2)  # 128x64x64
        downlayer_4 = self.downconv4(downlayer_3)  # 256x32x32
        downlayer_5 = self.downconv5(downlayer_4)  # 512x16x16
        downlayer_6 = self.downconv6(downlayer_5)  # 512x8x8
        downlayer_7 = self.downconv7(downlayer_6)  # 512x4x4
        downlayer_8 = self.downconv8(downlayer_7)  # 512x2x2

        uplayer_8 = self.upconv8(downlayer_8)                                 # 512x4x4
        uplayer_7 = self.upconv7(torch.cat([uplayer_8, downlayer_7], dim=1))  # 512x8x8
        uplayer_6 = self.upconv6(torch.cat([uplayer_7, downlayer_6], dim=1))  # 512x16x16
        uplayer_5 = self.upconv5(torch.cat([uplayer_6, downlayer_5], dim=1))  # 256x32x32
        uplayer_4 = self.upconv4(torch.cat([uplayer_5, downlayer_4], dim=1))  # 128x64x64
        uplayer_3 = self.upconv3(torch.cat([uplayer_4, downlayer_3], dim=1))  # 64x128x128
        uplayer_2 = self.upconv2(torch.cat([uplayer_3, downlayer_2], dim=1))  # 64x256x256
        # uplayer_1 = self.upconv1(torch.cat([uplayer_2, downlayer_1], dim=1))  # 3x256x256

        final = self.final_layer(torch.cat([uplayer_2, downlayer_1], dim=1))

        return final



class Unet_Disc(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Note: the discriminator model in the paper says "the number of channels being doubled
        #  after each downsampling" but I haven't confirmed in the code if that's actually true
        #  as this gives a lot of parameters
        self.downconv1 = down_layer(in_channels, 64, kernel_size=(3, 3), stride=1, act=nn.LeakyReLU(negative_slope=0.2))
        self.downconv2 = down_layer(64, 128, act=nn.LeakyReLU(negative_slope=0.2))
        self.downconv3 = down_layer(128, 256, act=nn.LeakyReLU(negative_slope=0.2))
        self.downconv4 = down_layer(256, 512, act=nn.LeakyReLU(negative_slope=0.2))
        self.downconv5 = down_layer(512, 1024, act=nn.LeakyReLU(negative_slope=0.2))
        self.downconv6 = down_layer(1024, 2048, act=nn.LeakyReLU(negative_slope=0.2))
        self.downconv7 = down_layer(2048, 4096, act=nn.LeakyReLU(negative_slope=0.2))
        self.downconv8 = down_layer(4096, 8192, act=nn.LeakyReLU(negative_slope=0.2))

        self.final_layer = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(8192*4, 1),
            nn.Sigmoid())

    def __call__(self, x):

        downlayer_1 = self.downconv1(x)
        downlayer_2 = self.downconv2(downlayer_1)
        downlayer_3 = self.downconv3(downlayer_2)
        downlayer_4 = self.downconv4(downlayer_3)
        downlayer_5 = self.downconv5(downlayer_4)
        downlayer_6 = self.downconv6(downlayer_5)
        downlayer_7 = self.downconv7(downlayer_6)
        downlayer_8 = self.downconv8(downlayer_7)

        final = self.final_layer(downlayer_8)

        return final


def down_layer(in_channels, out_channels, kernel_size=(4, 4), stride=2, padding=1, act=None):
    if act is None:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels))
    else:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            act)
    return layer


def up_layer(in_channels, out_channels, kernel_size=(4, 4), stride=2, padding=1, act=None):
    if act is None:
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels))
    else:
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            act)
    return layer


