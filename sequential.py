import torch
from torch import nn

class Seq_Res_Gen(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.first = nn.Sequential(
            conv_block(in_channels, 2),
            conv_block(2, 2),

            nn.Tanh()
        )
        next_in = 2 + in_channels
        self.second = nn.Sequential(
            conv_block(next_in, 4),
            conv_block(4, 4),
        )
        next_in += 4
        self.third = nn.Sequential(
            conv_block(next_in, 8),
            conv_block(8, 8),
        )
        next_in += 8
        self.fourth = nn.Sequential(
            conv_block(next_in, 16),
            conv_block(16, 16),
        )
        next_in += 16
        self.final = nn.Sequential(
            conv_block(next_in, 8),
            conv_block(8, out_channels),
        )



    def __call__(self, x):
        first = self.first(x)
        x = torch.cat([x, first], dim=1)
        second = self.second(x)
        x = torch.cat([x, second], dim=1)
        third = self.third(x)
        x = torch.cat([x, third], dim=1)
        fourth = self.fourth(x)
        x = torch.cat([x, fourth], dim=1)
        final = self.final(x)

        return final

class Seq_Pool_Disc(nn.Module):
    def __init__(self, in_channels, drop=0.2, size=256):
        super().__init__()
        linear_nodes = (size / (2 ** 6)) ** 2
        self.linear_nodes = int(linear_nodes * 512)
        self.model = nn.Sequential(
            disc_block(in_channels, 64),
            disc_block(64, 128),
            nn.Dropout(p=drop),
            disc_block(128, 128),
            disc_block(128, 256),
            nn.Dropout(p=drop),
            disc_block(256, 512),
            disc_block(512, 512),
            nn.Flatten(),
            nn.Linear(self.linear_nodes, 1),
            nn.Tanh()
        )

    def __call__(self, x):
        # print(self.linear_nodes)
        final = self.model(x)
        # print(final.shape)

        return final


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )



def disc_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )
