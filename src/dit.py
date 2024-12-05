import os
from einops import rearrange
from torch import nn
import torch
from tqdm import tqdm


class DiT(nn.Module):
    """Diffusion Transformer for predicting next latent state distribution"""

    def __init__(self, latent_dim=4, hidden_dim=128, nhead=8, num_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(latent_dim * 2 + 1, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nhead,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

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

    def forward(self, input_mu, input_logvar, action):
        x = torch.cat((input_mu, input_logvar, action), dim=-1)
        x = self.input_proj(x)
        x = self.transformer(x)

        # Assuming you want the output corresponding to the last frame
        x = x[:, -1, :].unsqueeze(1)  # Select the last frame's output

        mu = self.mu_proj(x)
        logvar = self.logvar_proj(x)
        return mu, logvar


def train_dit(
    dit_model,
    vae_model,
    train_loader,
    optimizer,
    epochs=100,
    beta=1.0,
    device="cuda",
    checkpoint_dir="dit_checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    dit_model.train()
    vae_model.eval()
    best_loss = float("inf")

    num_frames = 3

    def process_frames(x, num_frames):
        # Assume x is of shape [batch_size, total_frames, C, H, W]
        input_frames = x[:, :num_frames]  # Select the first `num_frames` frames
        target_frame = x[:, num_frames : num_frames + 1]  # Select the next frame

        B, _, C, H, W = (
            input_frames.shape
        )  # Extract batch size, frames, channels, height, width
        input_frames = rearrange(input_frames, "b f c h w -> (b f) c h w")
        target_frame = rearrange(target_frame, "b f c h w -> (b f) c h w")

        input_mu, input_logvar = vae_model.encoder(input_frames)
        target_mu, target_logvar = vae_model.encoder(target_frame)

        input_mu = rearrange(input_mu, "(b f) l -> b f l", b=B, f=num_frames)
        input_logvar = rearrange(input_logvar, "(b f) l -> b f l", b=B, f=num_frames)
        target_mu = rearrange(target_mu, "(b f) l -> b f l", b=B, f=1)
        target_logvar = rearrange(target_logvar, "(b f) l -> b f l", b=B, f=1)

        return input_mu, input_logvar, target_mu, target_logvar

    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for _batch_idx, (x, left_action, _right_action) in enumerate(pbar):
            x = x.to(device)
            B, _, _, _, _ = x.shape
            left_action = left_action[:, :num_frames].unsqueeze(-1).to(device)

            input_mu, input_logvar, target_mu, target_logvar = process_frames(
                x, num_frames
            )

            pred_mu, pred_logvar = dit_model(input_mu, input_logvar, left_action)

            recon_loss = nn.functional.mse_loss(pred_mu, target_mu)

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
