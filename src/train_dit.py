import math
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from dit import DiT, train_dit
from vae import ViTVAE
from PongSim import pong_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_scale: float = 0.0,
    num_cycles: float = 0.5,
):
    def lr_lambda(current_step):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        scale = (
            (1.0 - min_lr_scale)
            * 0.5
            * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        )
        return max(min_lr_scale, scale)

    return LambdaLR(optimizer, lr_lambda)


# Example usage
def create_optimizer_and_scheduler(model, num_training_steps):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * 0.02),
        num_training_steps=num_training_steps,
        min_lr_scale=0.1,
    )

    return optimizer, scheduler


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

num_training_steps = len(train_loader) * 10
optimizer, scheduler = create_optimizer_and_scheduler(dit_model, num_training_steps)


train_dit(
    dit_model=dit_model,
    vae_model=vae_model,
    train_loader=train_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=10,
    beta=1e-9,
    device=device,
)
