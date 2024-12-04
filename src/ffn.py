import torch
import os
from torch import nn
from tqdm import tqdm


class FFN(nn.Module):
    def __init__(self, latent_dim=4, action_dim=1, hidden_dim=128):
        super(FFN, self).__init__()
        self.input_dim = latent_dim * 2 + action_dim  # input_mu, input_logvar, action
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Define the feedforward network layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc3_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3_mu.weight)
        nn.init.zeros_(self.fc3_mu.bias)
        nn.init.xavier_uniform_(self.fc3_logvar.weight)
        nn.init.zeros_(self.fc3_logvar.bias)

    def forward(self, input_mu, input_logvar, action):
        # Concatenate inputs
        x = torch.cat([input_mu, input_logvar, action], dim=-1)

        # Pass through the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Predict next latent dimensions
        pred_mu = self.fc3_mu(x)
        pred_logvar = self.fc3_logvar(x)

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
        for _batch_idx, (x, left_action, _right_action) in enumerate(pbar):
            x = x.to(device)
            # left_action: [batch_size, frame_number]
            left_action = left_action[:, 2].unsqueeze(-1)  # take the 3rd (active) frame
            left_action = left_action.to(device)

            B, F, C, H, W = x.shape

            # Get consecutive frame pairs
            input_frames = x[:, :3]  # First 3 frames
            target_frames = x[:, 1:4]  # Next 3 frames

            with torch.no_grad():
                # Get target distribution parameters
                target_mu, target_logvar = vae_model.encoder(target_frames)
                # Get input latents
                input_mu, input_logvar = vae_model.encoder(input_frames)

            # Train DiT to predict the next latent distribution
            optimizer.zero_grad()

            # input_mu: [batch_size, latent_dim]
            # input_logvar: [batch_size, latent_dim]

            pred_mu, pred_logvar = dit_model(input_mu, input_logvar, left_action)

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
