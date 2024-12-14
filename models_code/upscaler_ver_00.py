import torch

class UnetUpscaler(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 5, 1, 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8, 32, 5, 1, 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 128, 5, 1, 2),
            torch.nn.LeakyReLU(),
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.Conv2d(128, 64, 9, 1, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 32, 9, 1, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 16, 9, 1, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 8, 9, 1, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8, 3, 9, 1, 4),
            torch.nn.LeakyReLU(),     
        )
        
    def forward(self, x):
        return self.layers(x);
    
model_00 = UnetUpscaler().to("cuda")
optimizer = torch.optim.Adam(params = model_00.parameters(), lr=0.00002)
cost_fn = torch.nn.SmoothL1Loss()