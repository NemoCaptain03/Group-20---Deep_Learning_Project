import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image, make_grid
from torchvision import datasets
from torchvision import transforms as trans
from torch.utils.data import DataLoader
import torch.nn.functional as func
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os

from generator import Generator
from discriminator import Discriminator

# Parameter
DATA_DIR = 'C:/Users/Admin/PycharmProjects/pythonProject/Group-20---Deep_Learning_Project/Mednist'
batch_size = 128
image_size = 64
nc = 1  # Number of output channels (=1 because Mednist dataset contain gray pictures only)
nz = 128  # Input noise vector size(latent size)
ngf = 64  # Generator feature maps size
ndf = 64  # Discriminator feature maps size
epochs = 50  # Number of training epochs
stats = (0.5,), (0.5,)  # Parameter for Normalize
lr = 0.0002  # Learning rate
betas = (0.5, 0.999)  # Decay rate


# Setup Device
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()

# Dataset, DataLoader and Transformations
train_ds = ImageFolder(DATA_DIR, transform=trans.Compose([
    trans.Resize(image_size),
    trans.CenterCrop(image_size),
    trans.ToTensor(),
    trans.Normalize(*stats)
]))

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=True)


# Helper functions to denormalize the image tensors and display some sample images from a training batch.
# Denormalization function
def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


# Show images function
def show_images(images, nmax=64, nrow=8):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=nrow).permute(1, 2, 0))


# Show batch function
def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break


show_batch(train_dl)

# Initialize Generator
generator = Generator(nz, ngf, nc).to(device)
# Initialize Discriminator
discriminator = Discriminator(nc, ndf).to(device)


def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = func.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images
    latent = torch.randn(batch_size, nz, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = func.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score


def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()

    # Generate fake images
    latent = torch.randn(batch_size, nz, 1, 1, device=device)
    fake_images = generator(latent)

    # Try to fool the discriminator
    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = func.binary_cross_entropy(preds, targets)

    # Update generator weights
    loss.backward()
    opt_g.step()

    return loss.item()


sample_dir = 'generated'
os.makedirs(sample_dir, exist_ok=True)


def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated-images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]);
        ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))


fixed_latent = torch.randn(64, nz, 1, 1, device=device)


def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)

    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Train generator
            loss_g = train_generator(opt_g)

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch + 1, epochs, loss_g, loss_d, real_score, fake_score))

        # Save generated images
        save_samples(epoch + start_idx, fixed_latent, show=False)

    return losses_g, losses_d, real_scores, fake_scores


if __name__ == '__main__':
    fit(epochs, lr)
