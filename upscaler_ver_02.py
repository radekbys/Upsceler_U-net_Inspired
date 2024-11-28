import torch

class UnetUpscaler(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.layers = torch.nn.Sequential(

            torch.nn.Conv2d(3, 16, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 32, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(256, 128, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.Conv2d(128, 64, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 32, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            torch.nn.Conv2d(32, 16, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 3, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.AdaptiveAvgPool2d((720,1280)),
            torch.nn.Conv2d(3, 16, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 16, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 3, 3, 1, 1),
            torch.nn.LeakyReLU(),
            
            
        )
        
    def forward(self, x):
        return self.layers(x);
    
model_01 = UnetUpscaler().to("cuda")

optimizer = torch.optim.Adam(params = model_01.parameters(), lr=0.00005)
cost_fn = torch.nn.SmoothL1Loss()