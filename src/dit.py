import os
from torch import nn
import torch
from tqdm import tqdm


class DiT(nn.Module):
    """Diffusion Transformer for predicting next latent state distribution"""

    def __init__(self, latent_dim=4, hidden_dim=128, nhead=8, num_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Separate projection heads for mean and logvar
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)
    
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights for input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Initialize weights for mu and logvar projections
        nn.init.xavier_uniform_(self.mu_proj.weight)
        nn.init.zeros_(self.mu_proj.bias)
        nn.init.xavier_uniform_(self.logvar_proj.weight)
        nn.init.zeros_(self.logvar_proj.bias)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x.unsqueeze(1))
        x = x.squeeze(1)

        # Predict both mean and variance
        mu = self.mu_proj(x)
        logvar = self.logvar_proj(x)
        return mu, logvar

    def predict_next_state(self, current_mu, current_logvar):
        """Predict the next latent state distribution given the current state."""
        with torch.no_grad():
            # Concatenate current mu and logvar as input
            model_input = torch.cat([current_mu, current_logvar], dim=-1)
            # Get predicted distribution parameters
            pred_mu, pred_logvar = self.forward(model_input)
            return pred_mu, pred_logvar


def train_dit(
    dit_model,
    vae_model,
    train_loader,
    optimizer,
    epochs=100,
    beta=1.0,  # Weight for KL divergence loss
    device="cuda",
    checkpoint_dir="dit_checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    dit_model.train()
    vae_model.eval()
    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, x in enumerate(pbar):
            x = x.to(device)
            B, F, C, H, W = x.shape

            # Get consecutive frame pairs
            input_frames = x[:, :3]  # First 3 frames
            target_frames = x[:, 1:4]  # Next 3 frames

            # import matplotlib.pyplot as plt
            # # Debug display for input and target frames using matplotlib
            # def plot_frames(frames, title):
            #     fig, axes = plt.subplots(1, frames.shape[1], figsize=(12, 4))
            #     fig.suptitle(title)
            #     for i, ax in enumerate(axes):
            #         ax.imshow(frames[0, i].permute(1, 2, 0).cpu().numpy())
            #         ax.axis("off")
            #     plt.show()

            # print(f"Batch {batch_idx}:")
            # plot_frames(input_frames, "Input Frames")
            # plot_frames(target_frames, "Target Frames")

            with torch.no_grad():
                # Get target distribution parameters
                target_mu, target_logvar = vae_model.encoder(target_frames)
                # Get input latents
                input_mu, input_logvar = vae_model.encoder(input_frames)

            # Train DiT to predict the next latent distribution
            optimizer.zero_grad()
            pred_mu, pred_logvar = dit_model(
                torch.cat([input_mu, input_logvar], dim=-1)
            )

            # Reconstruction loss (using KL divergence between predicted and target distributions)
            recon_loss = nn.functional.mse_loss(pred_mu, target_mu)

            # KL divergence between predicted and target distributions
            kl_loss = 0.5 * torch.sum(
                target_logvar
                - pred_logvar
                + (torch.exp(pred_logvar) + (pred_mu - target_mu).pow(2))
                / torch.exp(target_logvar)
                - 1
            )

            # Total loss
            loss = recon_loss + beta * kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            total_recon += recon_loss.item() * B
            total_kl += kl_loss.item() * B

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "recon": f"{recon_loss.item():.4f}",
                    "kl": f"{kl_loss.item():.4f}",
                }
            )

        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon / len(train_loader.dataset)
        avg_kl = total_kl / len(train_loader.dataset)

        print(
            f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f} "
            f"(Recon = {avg_recon:.4f}, KL = {avg_kl:.4f})"
        )

        # Save checkpoint if it's the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": dit_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"Saved new best model with loss: {best_loss:.4f}")

        # Save latest model
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": dit_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, "latest_model.pt"))
