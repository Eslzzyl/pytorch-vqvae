import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchvision.utils import make_grid
import argparse
import os
import multiprocessing as mp
from tqdm import tqdm

from modules import VectorQuantizedVAE
from data import TinyImagenet, MiniImagenet

from torch.utils.tensorboard import SummaryWriter


def train(data_loader, model, optimizer, args, writer):
    for images, _ in tqdm(data_loader, desc="Training"):
        images = images.cuda()

        optimizer.zero_grad()
        x_tilde, z_e_x, z_q_x = model(images)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, images)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss = loss_recons + loss_vq * 0.1 + args.beta * loss_commit
        loss.backward()

        # Logs
        writer.add_scalar("loss/train/reconstruction", loss_recons.item(), args.steps)
        writer.add_scalar("loss/train/quantization", loss_vq.item(), args.steps)

        optimizer.step()
        args.steps += 1


def test(data_loader, model, args, writer):
    with torch.no_grad():
        loss_recons, loss_vq = 0.0, 0.0
        for images, _ in data_loader:
            images = images.cuda()
            x_tilde, z_e_x, z_q_x = model(images)
            loss_recons += F.mse_loss(x_tilde, images)
            loss_vq += F.mse_loss(z_q_x, z_e_x)

        loss_recons /= len(data_loader)
        loss_vq /= len(data_loader)

    # Logs
    writer.add_scalar("loss/test/reconstruction", loss_recons.item(), args.steps)
    writer.add_scalar("loss/test/quantization", loss_vq.item(), args.steps)

    return loss_recons.item(), loss_vq.item()


def generate_samples(images, model, args):
    with torch.no_grad():
        images = images.cuda()
        x_tilde, _, _ = model(images)
    return x_tilde


def main(args):
    writer = SummaryWriter(f"{args.logs_folder}/{args.output_folder}")
    save_filename = f"./models/{args.output_folder}"

    if args.dataset in ["mnist", "fashion-mnist", "cifar10"]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        if args.dataset == "mnist":
            # Define the train & test datasets
            train_dataset = datasets.MNIST(
                args.data_folder, train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                args.data_folder, train=False, transform=transform
            )
            num_channels = 1
        elif args.dataset == "fashion-mnist":
            # Define the train & test datasets
            train_dataset = datasets.FashionMNIST(
                args.data_folder, train=True, download=True, transform=transform
            )
            test_dataset = datasets.FashionMNIST(
                args.data_folder, train=False, transform=transform
            )
            num_channels = 1
        elif args.dataset == "cifar10":
            # Define the train & test datasets
            train_dataset = datasets.CIFAR10(
                args.data_folder, train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR10(
                args.data_folder, train=False, transform=transform
            )
            num_channels = 3
        valid_dataset = test_dataset
    elif args.dataset == "miniimagenet":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(128),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(
            args.data_folder, train=True, download=True, transform=transform
        )
        valid_dataset = MiniImagenet(
            args.data_folder, valid=True, download=True, transform=transform
        )
        test_dataset = MiniImagenet(
            args.data_folder, test=True, download=True, transform=transform
        )
        num_channels = 3
    elif args.dataset == "tinyimagenet":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        # Define the train & test datasets
        train_dataset = TinyImagenet(
            args.data_folder,
            train=True,
            valid=False,
            download=True,
            transform=transform,
        )
        valid_dataset = TinyImagenet(
            args.data_folder,
            train=False,
            valid=True,
            download=True,
            transform=transform,
        )
        num_channels = 3

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Fixed images for Tensorboard
    fixed_images, _ = next(iter(valid_loader))
    fixed_images = fixed_images[:16]
    
    model = VectorQuantizedVAE(num_channels, args.hidden_size, args.k).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = -1.0
    for epoch in range(args.num_epochs):
        print("Epoch {0}/{1}".format(epoch + 1, args.num_epochs))
        train(train_loader, model, optimizer, args, writer)
        loss, _ = test(valid_loader, model, args, writer)

        # Generate reconstructions
        reconstruction = generate_samples(fixed_images, model, args)
        
        # Create alternating rows of original and reconstructed images
        batch_size = fixed_images.size(0)
        alternating_images = []
        for i in range(batch_size):
            alternating_images.append(fixed_images[i].cpu())  # Original image
            alternating_images.append(reconstruction[i].cpu())  # Reconstructed image
        
        grid = make_grid(alternating_images, nrow=2, value_range=(-1, 1), normalize=True)
        writer.add_image("original_vs_reconstruction", grid, epoch)

        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open("{0}/best.pt".format(save_filename), "wb") as f:
                torch.save(model.state_dict(), f)
        with open("{0}/model_{1}.pt".format(save_filename, epoch + 1), "wb") as f:
            torch.save(model.state_dict(), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VQ-VAE")

    # Dataset
    parser.add_argument(
        "--data-folder",
        type=str,
        help="path to the dataset folder",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="name of the dataset (mnist, fashion-mnist, cifar10, miniimagenet, tinyimagenet)",
    )

    # Latent space
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="size of the latent vectors (default: 256)",
    )
    parser.add_argument(
        "--k", type=int, default=512, help="number of latent vectors (default: 512)"
    )

    # Optimization
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size (default: 128)"
    )
    parser.add_argument(
        "--num-epochs", type=int, default=100, help="number of epochs (default: 100)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="learning rate for Adam optimizer (default: 2e-4)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)",
    )

    # Miscellaneous
    parser.add_argument(
        "--output-folder",
        type=str,
        default="vqvae",
        help="name of the output folder (default: vqvae)",
    )
    parser.add_argument(
        "--logs-folder",
        type=str,
        default="logs",
        help="name of the logs folder (default: logs)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=mp.cpu_count() - 1,
        help="number of workers for trajectories sampling (default: {0})".format(
            mp.cpu_count() - 1
        ),
    )

    args = parser.parse_args()

    # Create logs and models folder if they don't exist
    if not os.path.exists(args.logs_folder):
        os.makedirs(args.logs_folder)
    if not os.path.exists("./models"):
        os.makedirs("./models")
    # Check if CUDA is available
    assert torch.cuda.is_available(), "CUDA is not available"
    if not os.path.exists("./models/{0}".format(args.output_folder)):
        os.makedirs("./models/{0}".format(args.output_folder))
    args.steps = 0

    main(args)
