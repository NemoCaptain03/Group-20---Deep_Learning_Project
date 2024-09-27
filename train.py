import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from generator import Generator
from discriminator import Discriminator

# Parameter
batch_size = 128
image_size = 64
nc = 1  # Number of output channels (=1 because Mednist dataset contain gray pictures only)
nz = 128  # Input noise vector size(latent size)
ngf = 64  # Generator feature maps size
ndf = 64  # Discriminator feature maps size
epochs = 50  # Number of training epochs

# Initialize Generator
generator = Generator(nz, ngf, nc)

# Forward pass
noise = torch.randn(16, nz, 1, 1)  # Generate random noise
fake_images = generator(noise)  # Forward pass to generate images

# Initialize Discriminator
discriminator = Discriminator(nc, ndf)

# Forward pass
test_image = torch.randn(16, nc, 64, 64)
output = discriminator(test_image)

print(output.shape)