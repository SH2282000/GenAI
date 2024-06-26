from typing import Optional
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Check for MPS device
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print(f"Using device: {device}")


class VAE(nn.Module):
    def __init__(self, num_channels=1, num_classes=10, latent_dim=2, embed_dim=16):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(
            num_embeddings=num_classes, embedding_dim=embed_dim
        )

        self.encoder = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=8,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.Conv2d(
                    in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1
                ),
                nn.Conv2d(
                    in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        self.decoder = nn.ModuleList(
            [
                nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
                nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3),
                nn.Conv2d(
                    in_channels=8, out_channels=num_channels, kernel_size=3, padding=1
                ),
            ]
        )
        self.fc_latent = nn.Linear(in_features=latent_dim + embed_dim, out_features=512)
        self.fc_mean = nn.Linear(in_features=512 + embed_dim, out_features=latent_dim)
        self.fc_var = nn.Linear(in_features=512 + embed_dim, out_features=latent_dim)
        self.leaky_relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        mean, log_var = self.encode(x, y)
        eps = torch.randn(log_var.shape, device=log_var.device)
        z = mean + torch.exp(log_var * 0.5) * eps
        x_recon = self.decode(z, y)
        return x_recon, mean, log_var

    def encode(self, x, y):
        for layer in self.encoder:
            x = layer(x)
            x = self.leaky_relu(x)
        x = torch.reshape(x, (x.shape[0], -1))
        class_embed = self.embedding(y)
        mean = self.fc_mean(torch.cat((x, class_embed), dim=1))
        log_var = self.fc_var(torch.cat((x, class_embed), dim=1))
        return mean, log_var

    def decode(self, z, y):
        class_embed = self.embedding(y)
        x = self.fc_latent(torch.cat((z, class_embed), dim=1))
        x = torch.reshape(x, (-1, 32, 4, 4))
        for layer in self.decoder:
            x = nn.functional.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=True
            )
            x = self.leaky_relu(x)
            x = layer(x)
        x = self.sigmoid(x)
        return x

    def sample(self, y):
        z = torch.randn((1, self.latent_dim), device=device)
        return self.decode(z, torch.tensor([y], device=device))

    def sample_latent(self, x, y):
        mean, log_var = self.encode(x, y)
        eps = torch.randn(log_var.shape, device=log_var.device)
        z = mean + torch.exp(log_var * 0.5) * eps
        return z


def loss_vae_wo(
    reconstructed_x, x, mean, log_var, kl_weight, without: Optional[str] = None
):
    reconstruction_loss = nn.functional.binary_cross_entropy(
        reconstructed_x, x, reduction="sum"
    )
    kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    if without == "reconstruction":
        return kl_loss * kl_weight
    elif without == "kl":
        return reconstruction_loss
    else:
        return (kl_loss * kl_weight) + reconstruction_loss


def reconstruction_loss(reconstructed_x, x):
    return nn.functional.binary_cross_entropy(reconstructed_x, x, reduction="sum")


def plot_samples(model, epoch):
    model.eval()
    num_classes = 10
    samples = []

    with torch.no_grad():
        for y in range(num_classes):
            sample = model.sample(y)
            samples.append(sample.cpu().view(28, 28))

    _, axs = plt.subplots(1, num_classes, figsize=(num_classes * 2, 2))
    for i in range(num_classes):
        axs[i].imshow(samples[i], cmap="gray")
        axs[i].axis("off")
    plt.suptitle(f"Samples at Epoch {epoch}")
    plt.show()


def plot_loss(training_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(
        range(len(training_losses)), training_losses, marker="o", label="Training Loss"
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.show()


def train(epochs: int, without: Optional[str] = None):
    learning_rate = 0.001
    batch_size = 128
    kl_weight = 0.0001

    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    training_losses = []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_index, (batch_data, labels) in enumerate(train_loader):
            batch_data, labels = batch_data.to(device), labels.to(device)
            optimizer.zero_grad()

            batch_recon, mean, log_var = model(batch_data, labels)

            loss = loss_vae_wo(
                batch_recon, batch_data, mean, log_var, kl_weight, without
            )
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_index % 100 == 0:
                print(
                    f"\tEpoch {epoch} [{batch_index * len(batch_data)}/{len(train_loader.dataset)}] "
                    f"\tLoss: {loss.item() / len(batch_data):.6f}"
                )
        avg_loss = train_loss / len(train_loader.dataset)
        training_losses.append(avg_loss)
        print(f"==> Epoch {epoch} Average loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"vae_epoch_{epoch}.pth")
        plot_samples(model, epoch)

    return model, training_losses


def embed_and_plot(model: VAE, title: str):
    model.eval()

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )

    embeddings = []
    labels = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            latent = model.sample_latent(data, target)
            embeddings.append(latent.cpu().numpy())
            labels.append(target.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    plt.figure(figsize=(8, 6))
    for i in range(10):
        indices = labels == i
        plt.scatter(
            embeddings[indices, 0], embeddings[indices, 1], label=str(i), alpha=0.6
        )

    plt.title(title)
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.legend(title="Digit Class")
    plt.grid(True)
    plt.show()


# Usage example:
# model, training_losses = train(epochs=10)
# embed_and_plot(model, "Latent Space")


# def plot_samples_and_loss(samples, ):
#     model.eval()
#     num_classes = 10

#     with torch.no_grad():
#         for y in range(num_classes):
#             sample = model.sample(y)
#             samples.append(sample.cpu().view(28, 28))

#     _, axs = plt.subplots(1, num_classes, figsize=(num_classes * 2, 2))
#     for i in range(num_classes):
#         axs[i].set_title(f"{reconstruction_loss())=}")
#         axs[i].imshow(samples[i], cmap="gray")
#         axs[i].axis("off")
#     plt.suptitle(f"Samples at Epoch {epoch}")
#     plt.show()

# def anomaly_detection(model: VAE):
#     model.eval()

#     # HYPERPARAMETERS
#     learning_rate = 0.001
#     batch_size = 128
#     kl_weight = 0.0001

#     # DATA
#     transform = transforms.Compose([transforms.ToTensor()])
#     fashion_test_dataset = datasets.FashionMNIST(
#         root="./data", train=False, download=True, transform=transform
#     )
#     fashion_test_loader = torch.utils.data.DataLoader(
#         fashion_test_dataset, batch_size=1, shuffle=True
#     )

#     plot_samples(model, epoch)

#     return model, training_losses
