import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np


manualSeed = 69
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)  # Needed for reproducible results


def get_device():
    if torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


device = torch.device(get_device())

# HYPERPARAMETERS
batch_size = 64
learning_rate = 0.0001
beta1 = 0.5
num_epochs = 10
latent_dim = 100
num_classes = 10

# Image size (after resizing)
image_size = 32

# Load the dataset
transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# GAN


class Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_channels):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.init_size = image_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128 * self.init_size**2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_embedding(labels), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(num_classes, img_shape * img_shape)
        self.model = nn.Sequential(
            # not using torch.nn.utils.parametrizations.spectral_norm because of no mps support yet!
            nn.utils.spectral_norm(nn.Conv2d(2, 64, 3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1)),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1)),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 3, stride=2, padding=1)),
            nn.BatchNorm2d(512, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        ds_size = img_shape // 2**4
        self.adv_layer = nn.Sequential(nn.Linear(512 * ds_size**2, 1), nn.Sigmoid())

    def forward(self, img, labels):
        label_embedding = self.label_embedding(labels).view(
            labels.size(0), 1, img.size(2), img.size(3)
        )
        d_in = torch.cat((img, label_embedding), 1)
        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


# TRAINING


def train(device, dataloader):
    generator = Generator(latent_dim, num_classes, 1).to(device)
    discriminator = Discriminator(num_classes, image_size).to(device)

    adversarial_loss = nn.BCELoss()

    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=learning_rate, betas=(beta1, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=learning_rate, betas=(beta1, 0.999)
    )

    g_losses = []
    d_losses = []
    real_scores = []
    fake_scores = []

    fixed_noise = torch.randn(3, latent_dim).to(device)
    fixed_labels = torch.tensor([0, 1, 2]).to(device)

    for epoch in range(num_epochs):
        for imgs, labels in dataloader:
            batch_size = imgs.size(0)

            # Adversarial ground truths
            valid = torch.ones((batch_size, 1), requires_grad=False).to(device)
            fake = torch.zeros((batch_size, 1), requires_grad=False).to(device)

            # Configure input
            real_imgs = imgs.to(device)
            labels = labels.to(device)

            #  TRAIN GENERATOR

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = torch.randn(batch_size, latent_dim).to(device)
            gen_labels = torch.randint(0, num_classes, (batch_size,)).to(device)

            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            g_loss = adversarial_loss(validity, valid)

            g_loss.backward()
            optimizer_G.step()

            #  TRAIN DISCRIMINATOR

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(real_pred, valid)

            # Loss for fake images
            fake_pred = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(fake_pred, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # STATS
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            real_scores.append(real_pred.mean().item())
            fake_scores.append(fake_pred.mean().item())

        print(
            f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
        )
        with torch.no_grad():
            generated_imgs = generator(fixed_noise, fixed_labels)
            generated_imgs = generated_imgs * 0.5 + 0.5  # Unnormalize to [0, 1]
            grid = make_grid(generated_imgs, nrow=3)
            save_image(grid, f"images_epoch_{epoch}.png")

            # Display images
            np_grid = grid.cpu().numpy().transpose(1, 2, 0)
            plt.imshow(np_grid)
            plt.title(f"Generated images at epoch {epoch}")
            plt.show()

    # Plotting losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plotting discriminator scores
    plt.figure(figsize=(10, 5))
    plt.title("Discriminator Scores During Training")
    plt.plot(real_scores, label="Real")
    plt.plot(fake_scores, label="Fake")
    plt.xlabel("iterations")
    plt.ylabel("Score")
    plt.legend()
    plt.show()

    return generator, discriminator


if __name__ == "__main__":
    train(device, train_loader)
