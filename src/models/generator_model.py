import torch
import torch.nn as nn


# main block for down-sampling and up-sampling
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)  # kwargs are kernel size, stride ...
            if down  # we use the above nn.Conv2d if we are down-sampling
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()  # inplace=True helps the performances
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),  # stride=1 (default)
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)  # ?


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):  # either 9 residual blocks if the images are
        # 256x256 or larger, or 6 residual blocks if 128x128 or smaller # changed from num_features=32
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(  # down-sampling
            [
                ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
            ]
        )

        self.res_blocks = nn.Sequential(  # residual blocks that don't change the input or the number of channels
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]  # * is for unwrapping
        )

        self.up_blocks = nn.ModuleList(  # up-sampling
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),  # the output_padding adds an additional padding after the conv block
                ConvBlock(num_features * 2, num_features * 1, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),
            ]
        )

        self.last = nn.Conv2d(num_features * 1, img_channels, kernel_size=7, stride=1, padding=3,
                              padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        # print('Initial conv layer:', x.shape)
        for layer in self.down_blocks:
            x = layer(x)
            # print('Down-sampling block:', x.shape)
        x = self.res_blocks(x)
        # print('Residual block:', x.shape)
        for layer in self.up_blocks:
            x = layer(x)
            # print('Up-sampling block:', x.shape)

        return torch.tanh(self.last(x))  # [-1,1]


def test():
    img_channels = 1
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen(x).shape)


if __name__ == "__main__":
    test()
