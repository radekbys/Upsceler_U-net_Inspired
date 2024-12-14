import torch

class UnetUpscaler(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
        
        self.layers = torch.nn.Sequential(

            # initial upsampling
            torch.nn.UpsamplingNearest2d(scale_factor=4),

            # downsampling
            torch.nn.Conv2d(3, 16, 5, 1, 8, 4),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 32, 5, 1, 8, 4),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0), 
            
            # downsampling
            torch.nn.Conv2d(32, 64, 5, 1, 4, 2),
            torch.nn.LeakyReLU(), 
            torch.nn.Conv2d(64, 128, 5, 1, 4, 2),
            torch.nn.LeakyReLU(),
            torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0),   
            
            # upsampling
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(256, 128, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 64, 3, 1, 1),
            torch.nn.LeakyReLU(),  
            torch.nn.UpsamplingNearest2d(scale_factor=2),
            
            #final layers
            torch.nn.Conv2d(64, 32, 5, 1, 4, 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 16, 5, 1, 4, 2),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(16, 8, 3, 1, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(8, 3, 3, 1, 1, 1),
        )
        
    def forward(self, x):
        return self.layers(x);
    
model_01 = UnetUpscaler().to("cuda")

optimizer = torch.optim.Adam(params = model_01.parameters(), lr=0.000008)
cost_fn = torch.nn.L1Loss()