import numpy as np
from data import SequentialBouncingBallDataset
from dit import DiT
from vae import ViTVAE
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

device = "cuda" if torch.cuda.is_available() else "mps"


def generate_from_latents(model, latent_values, device="cuda"):
    """Generate images from specific latent values"""
    model.eval()
    with torch.no_grad():
        z = torch.tensor(latent_values, dtype=torch.float32).to(device)
        if len(z.shape) == 1:
            z = z.unsqueeze(0)  # Add batch dimension if needed

        # Generate image through decoder
        generated = model.decoder(z)

        return generated


def main():
    vae = ViTVAE(
        img_size=32,
        patch_size=4,
        embed_dim=96,
        depth=6,
        num_heads=8,
        latent_dim=4,
    ).to(device)
    checkpoint = torch.load("best/best_model-VAE.pt", map_location=torch.device(device))
    vae.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = SequentialBouncingBallDataset(
        num_sequences=1000,
        sequence_length=4,
        img_size=32,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    vae.eval()

    dit_model = DiT(
        latent_dim=4,
    ).to(device)

    dit_checkpoint = torch.load(
        "best/best_model-DiT.pt", map_location=torch.device(device)
    )
    dit_model.load_state_dict(dit_checkpoint["model_state_dict"])

    # Run the data through the VAE encoder
    original_images = []
    reconstructed_images = []

    for batch in test_loader:
        x = batch.to(device)

        # Encode the first 3 frames to latent space using VAE
        mu, logvar = vae.encoder(x[:, :3])
        original_images.append(x[:, :3].cpu())  # Store the original 3 frames

        # Simulate 30 frames forward
        for _ in range(32):
            # Predict the next latent distribution using DiT
            pred_mu, pred_logvar = dit_model(torch.cat([mu, logvar], dim=-1))

            # Reparameterize to get the latent vector
            latent_z = vae.reparameterize(pred_mu, pred_logvar)

            # Decode the latent vector to generate the next frame
            recon_x = generate_from_latents(vae, latent_z, device)

            # Store the generated frame
            reconstructed_images.append(recon_x.cpu())

            # Update mu and logvar for the next iteration
            mu, logvar = pred_mu, pred_logvar

        break

    # Convert lists to tensors for easier manipulation
    original_images = torch.cat(original_images, dim=0)
    reconstructed_images = torch.cat(reconstructed_images, dim=0)

    # Number of frames to display
    num_display_frames = 32

    # Create figure with subplots
    fig, axes = plt.subplots(2, num_display_frames, figsize=(20, 2))

    axes[0, 16].set_title("Ground truth", fontsize=10)
    for i in range(num_display_frames):
        axes[0, i].imshow(original_images[i, 2].squeeze(), cmap="gray")
        axes[0, i].axis("off")

    axes[1, 16].set_title("Simulated by DiT", fontsize=10)
    for i in range(num_display_frames):
        # Check if the image is grayscale or RGB
        img = reconstructed_images[i]
        if img.shape[0] == 3:  # If the first dimension is 3, assume RGB
            img = img.permute(1, 2, 0)  # Change shape to (32, 32, 3)
        else:
            img = img.squeeze()  # For grayscale, ensure it's 2D

        axes[1, i].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[1, i].axis("off")

    plt.subplots_adjust(
        wspace=0.10, hspace=0.10, left=0.01, right=0.99, top=0.99, bottom=0.01
    )
    plt.savefig("simulation_comparison.png")
    print("\nSaved simulation comparison to simulation_comparison.png")


if __name__ == "__main__":
    main()
