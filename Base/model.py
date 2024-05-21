import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, stride=1, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.conv(x) 
        return x
