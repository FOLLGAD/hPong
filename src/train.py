from vae import ViTVAE, train_vae
from generate_data import train_loader
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
device = "cpu"

# Model params
img_size = 32
patch_size = 4
embed_dim = 96
num_frames = 3
depth = 6
num_heads = 8
latent_dim = 4

# Initialize models
vae = ViTVAE(
    img_size=img_size,
    patch_size=patch_size,
    embed_dim=embed_dim,
    num_frames=num_frames,
    depth=depth,
    num_heads=num_heads,
    latent_dim=latent_dim,
).to(device)

# dit = DiT(latent_dim=latent_dim, hidden_dim=128, nhead=8, num_layers=4).to(device)

# Training
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
train_vae(vae, train_loader, optimizer, epochs=100, device=device)
