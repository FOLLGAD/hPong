import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os

class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=32, patch_size=4, in_chans=1, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x


class ViTEncoder(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=1,
        embed_dim=96,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        latent_dim=4,
    ):
        super().__init__()

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # VAE heads
        self.fc_mu = nn.Linear(embed_dim * num_patches, latent_dim)
        self.fc_var = nn.Linear(embed_dim * num_patches, latent_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize patch_embed like a linear layer
        w = self.patch_embed.proj.weight
        nn.init.kaiming_normal_(w.view([w.shape[0], -1]))

        # Initialize positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = x.flatten(1)

        mu = self.fc_mu(x)
        logvar = self.fc_var(x)
        return mu, logvar


class ViTDecoder(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=1,
        embed_dim=96,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        latent_dim=4,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Project latent to patches
        self.latent_proj = nn.Linear(latent_dim, embed_dim * self.num_patches)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)

        # Final projection to patches
        self.final_proj = nn.Linear(embed_dim, patch_size * patch_size * in_chans)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Project latent to patch tokens
        x = self.latent_proj(x)
        x = x.view(B, self.num_patches, -1)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transform
        x = self.transformer(x)

        # Project to patches
        x = self.final_proj(x)

        # Reshape to image
        x = x.view(
            B,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
            self.patch_size,
            self.patch_size,
            -1,
        )
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, -1, self.img_size, self.img_size)

        return torch.sigmoid(x)


class ViTVAE(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=1,
        embed_dim=96,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        latent_dim=4,
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            latent_dim,
        )
        self.decoder = ViTDecoder(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            latent_dim,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


class DiT(nn.Module):
    """Diffusion Transformer for predicting next latent state"""

    def __init__(self, latent_dim=4, hidden_dim=128, nhead=8, num_layers=4):
        super().__init__()

        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self, x, timesteps=None
    ):  # timesteps for future diffusion implementation
        x = self.input_proj(x)
        x = self.transformer(x.unsqueeze(1))
        x = self.output_proj(x.squeeze(1))
        return x


def train_step(model, optimizer, x, beta=0.0002):
    optimizer.zero_grad()

    # Forward pass
    recon_x, mu, logvar = model(x)

    # Reconstruction loss (binary cross entropy)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    loss = recon_loss + beta * kl_loss

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item(), recon_loss.item(), kl_loss.item()


def train_vae(
    model,
    train_loader,
    optimizer,
    epochs=100,
    device="cuda",
    checkpoint_dir="checkpoints",
):
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        # Add progress bar for batches
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, x in enumerate(pbar):
            x = x.to(device)
            loss, recon, kl = train_step(model, optimizer, x)
            
            total_loss += loss
            total_recon += recon
            total_kl += kl
            
            # Update progress bar with current batch metrics
            pbar.set_postfix({
                'loss': f'{loss/len(x):.4f}',
                'recon': f'{recon/len(x):.4f}',
                'kl': f'{kl/len(x):.4f}'
            })
            
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon / len(train_loader.dataset)
        avg_kl = total_kl / len(train_loader.dataset)
        
        print(f'Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f} '
              f'(Recon = {avg_recon:.4f}, KL = {avg_kl:.4f})')
        
        # Save checkpoint if it's the best model so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f'Saved new best model with loss: {best_loss:.4f}')
        
        # Save latest model
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest_model.pt'))
