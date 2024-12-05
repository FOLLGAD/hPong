import torch
from torch.utils.data import DataLoader

from dit import DiT, train_dit
from vae import ViTVAE
from PongSim import pong_dataset

device = "cuda" if torch.cuda.is_available() else "mps"

vae_model = ViTVAE(
    img_size=(32, 64),
    patch_size=4,
    embed_dim=96,
    depth=6,
    num_heads=8,
    latent_dim=4,
).to(device)
checkpoint = torch.load("best/best_vae_v2.pt", map_location=torch.device(device))
vae_model.load_state_dict(checkpoint["model_state_dict"])

dit_model = DiT(latent_dim=4).to(device)
optimizer = torch.optim.Adam(dit_model.parameters(), lr=1e-4)

train_loader = DataLoader(pong_dataset, batch_size=32, shuffle=True)

# import matplotlib.pyplot as plt

# def visualize_batch(train_loader, num_batches=1):
#     """Visualize a few batches of images and actions from the train_loader."""
#     for batch_idx, (x, left_action, _right_action) in enumerate(train_loader):
#         if batch_idx >= num_batches:
#             break

#         # Assuming x is of shape [batch_size, frames, channels, height, width]
#         batch_size, frames, channels, height, width = x.shape

#         fig, axes = plt.subplots(4, frames, figsize=(frames * 6, 4 * 6))
#         for i in range(4):
#             for j in range(frames):
#                 # Display each frame
#                 axes[i, j].imshow(x[i, j].permute(1, 2, 0).cpu().numpy())
#                 axes[i, j].axis('off')
#                 if j == 2:  # Display the action on the third frame
#                     axes[i, j].set_title(f"Action: {left_action[i, j].item()}")

#         plt.show()

# visualize_batch(train_loader, num_batches=1)


train_dit(
    dit_model=dit_model,
    vae_model=vae_model,
    train_loader=train_loader,
    optimizer=optimizer,
    epochs=100,
    beta=1e-9,  # Adjust this weight to balance reconstruction vs KL loss
    device=device,
)
