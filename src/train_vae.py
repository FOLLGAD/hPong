from vae import ViTVAE, train_vae
import torch
from torch.utils.data import DataLoader
from PongSim import pong_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE model.")
    parser.add_argument(
        "--continue",
        dest="continue_training",
        action="store_true",
        help="Continue training from the last checkpoint",
    )
    return parser.parse_args()


args = parse_args()


# Model params
img_size = (32, 64)
patch_size = 4
embed_dim = 96
depth = 6
num_heads = 8
latent_dim = 4

# Initialize models
vae = ViTVAE(
    img_size=img_size,
    patch_size=patch_size,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    latent_dim=latent_dim,
).to(device)

optimizer = torch.optim.AdamW(vae.parameters(), lr=1e-4)
train_loader = DataLoader(pong_dataset, batch_size=32, shuffle=True)

if args.continue_training:
    checkpoint = torch.load(
        "checkpoints/latest_model.pt", map_location=torch.device(device)
    )
    vae.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
else:
    start_epoch = 0
train_vae(
    vae, train_loader, optimizer, epochs=100, device=device, start_epoch=start_epoch
)
