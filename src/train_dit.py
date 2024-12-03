import torch

from dit import DiT, train_dit
from vae import ViTVAE
from generate_data import train_loader

device = "cuda" if torch.cuda.is_available() else "mps"

vae_model = ViTVAE(
    img_size=32,
    patch_size=4,
    embed_dim=96,
    depth=6,
    num_heads=8,
    latent_dim=4,
).to(device)
checkpoint = torch.load("checkpoints/latest_model.pt")
vae_model.load_state_dict(checkpoint["model_state_dict"])

dit_model = DiT(latent_dim=4).to(device)
optimizer = torch.optim.Adam(dit_model.parameters(), lr=1e-4)

train_dit(
    dit_model=dit_model,
    vae_model=vae_model,
    train_loader=train_loader,
    optimizer=optimizer,
    epochs=100,
    beta=1.0,  # Adjust this weight to balance reconstruction vs KL loss
    device=device,
)
