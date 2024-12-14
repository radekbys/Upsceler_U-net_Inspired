import torch

class UnetUpscaler(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = torch.nn.Sequential(
            # initial upsampling
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            # downsampling
            torch.nn.Conv2d(3, 16, 5, 1, 4, 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 64, 5, 1, 4, 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 256, 5, 1, 4, 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(256, 64, 5, 1, 4, 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 16, 5, 1, 4, 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 3, 5, 1, 4, 2),
        )

    def forward(self, x):
        return self.layers(x)