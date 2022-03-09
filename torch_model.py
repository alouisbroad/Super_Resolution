"""
Super resolution using pytorch.
"""
import torch
from torch import nn
import torch.nn.functional as F


class SuperResolution(nn.Module):
    """
    Super resolution model using pytorch - including batch normalisation.
    """

    def __init__(self, upscale_factor, channels):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=channels, out_channels=64,
                            kernel_size=(5, 5), stride=1, padding="same")
        self.b1 = nn.BatchNorm2d(64)
        self.r1 = nn.ReLU()

        self.c2 = nn.Conv2d(in_channels=64, out_channels=64,
                            kernel_size=(5, 5), stride=1, padding="same")
        self.b2 = nn.BatchNorm2d(64)
        self.r2 = nn.ReLU()

        self.c3 = nn.Conv2d(in_channels=64, out_channels=32,
                            kernel_size=(5, 5), stride=1, padding="same")
        self.b3 = nn.BatchNorm2d(32)
        self.r3 = nn.ReLU()

        self.c4 = nn.Conv2d(in_channels=32, out_channels=channels * (upscale_factor ** 2),
                            kernel_size=(5, 5), stride=1, padding="same")
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        """
        Forward pass.
        """
        x = self.c1(inputs)
        nn.init.orthogonal_(x)
        x = self.b1(x)
        x = self.r1(x)

        x = self.c2(x)
        nn.init.orthogonal_(x)
        x = self.b2(x)
        x = self.r2(x)

        x = self.c3(x)
        nn.init.orthogonal_(x)
        x = self.b3(x)
        x = self.r3(x)

        output = F.relu(self.c4(x))
        nn.init.orthogonal_(output)
        return nn.functional.pixel_shuffle(output, self.upscale_factor)


class SimpleSuperResolution(nn.Module):
    """
    Simple super resolution model using pytorch - functional
    """

    def __init__(self, upscale_factor, channels):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels=channels, out_channels=64,
                            kernel_size=(5, 5), stride=1, padding="same")
        self.c2 = nn.Conv2d(in_channels=64, out_channels=64,
                            kernel_size=(5, 5), stride=1, padding="same")
        self.c3 = nn.Conv2d(in_channels=64, out_channels=32,
                            kernel_size=(5, 5), stride=1, padding="same")
        self.c4 = nn.Conv2d(in_channels=32, out_channels=channels * (upscale_factor ** 2),
                            kernel_size=(5, 5), stride=1, padding="same")
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        """
        Forward pass.
        """
        x = F.relu(self.c1(inputs))
        nn.init.orthogonal_(x)
        x = F.relu(self.c2(x))
        nn.init.orthogonal_(x)
        x = F.relu(self.c3(x))
        nn.init.orthogonal_(x)
        output = F.relu(self.c4(x))
        nn.init.orthogonal_(output)
        return nn.functional.pixel_shuffle(output, self.upscale_factor)


class SequentialSuperResolution(nn.Module):
    """
    Simple super resolution model using pytorch & sequential, weight initialisation
    is done below.
    """

    def __init__(self, upscale_factor, channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64,
                      kernel_size=(5, 5), stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(5, 5), stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=(5, 5), stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=channels * (upscale_factor ** 2),
                      kernel_size=(5, 5), stride=1, padding="same"),
            nn.ReLU(),
        )
        self.upscale_factor = upscale_factor

    def forward(self, inputs):
        """
        Forward pass.
        """
        output = self.main(inputs)
        return nn.functional.pixel_shuffle(output, self.upscale_factor)


def initialize_weights(m):
    """
    Apply orthogonal kernel initialiser to all
    Conv2D - https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
    :param m:
    """
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data)
    # elif isinstance(m, nn.Linear) ... etc.


model = SequentialSuperResolution(upscale_factor=16, channels=1)
model.apply(initialize_weights)
