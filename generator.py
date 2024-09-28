import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()

        self.main = nn.Sequential(

            # Output size in each layer = (input_size − 1)*stride − 2*padding + kernel_size
            # out = out_channels x Output size in each layer x Output size in each layer

            # First deconvolutional layer
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # out: (64 * 8 =) 512 x 4 x 4

            # Layer 2: Upsample to 256 feature maps
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # out: (64 * 4 =) 256 x 8 x 8

            # Layer 3: Upsample to 128 feature maps
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # out: (64 * 2 =) 128 x 16 x 16

            # Layer 4: Upsample to 64 feature maps
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # out: 64 x 32 x 32

            # Output layer: Generate a gray color image
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 1 x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
