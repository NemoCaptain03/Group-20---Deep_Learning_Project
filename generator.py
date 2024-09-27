import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()

        # nz: size of the input noise vector(latent size)
        # ngf: Size of feature maps in the generator
        # nc: Number of output channels (=1 because Mednist dataset contain gray pictures only)

        self.main = nn.Sequential(

            # Output size = (Input_Size − 1)*Stride − 2*Padding + Kernel_Size

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
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # out: 128 x 16 x 16

            # Layer 4: Upsample to 64 feature maps
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # out: 64 x 32 x 32

            # Output layer: Generate a gray color image
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # out: 1 x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# parameter
nz = 100
ngf = 64
nc = 1

# Initialize Generator
generator = Generator(nz, ngf, nc)

# forward pass
noise = torch.randn(16, nz, 1, 1)  # Generate random noise
fake_images = generator(noise)  # Forward pass to generate images
