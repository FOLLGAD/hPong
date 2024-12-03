import numpy as np
from data import SequentialBouncingBallDataset
from dit import DiT
from vae import ViTVAE
import torch
from torch.utils.data import DataLoader

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
    checkpoint = torch.load("checkpoints/latest_model.pt")
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

    dit_checkpoint = torch.load("dit_checkpoints/latest_model.pt")
    dit_model.load_state_dict(dit_checkpoint["model_state_dict"])

    # Run the data through the VAE encoder
    original_images = []
    reconstructed_images = []

    for batch in test_loader:
        x = batch.to(device)

        # Encode the images to latent space using VAE
        mu, logvar = vae.encoder(x[:, :3])  # only encode the first 3 frames

        pred_mu, pred_logvar = dit_model(torch.cat([mu, logvar], dim=-1))

        # Decode the predicted latent distribution to generate images
        latent_z = vae.reparameterize(pred_mu, pred_logvar)

        # Decode the DiT output back to image space using VAE decoder
        recon_x = generate_from_latents(vae, latent_z, device)

        # Store images for visualization
        original_images.append(x.cpu())
        reconstructed_images.append(recon_x.cpu())

    # Visualize the results
    import matplotlib.pyplot as plt

    orig_imgs = torch.cat(original_images, dim=0)
    recon_imgs = torch.cat(reconstructed_images, dim=0)

    fig, axes = plt.subplots(3, 8, figsize=(16, 6))

    # Plot original images on the first row
    for i in range(8):
        axes[0, i].imshow(orig_imgs[i, 2].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

    # Plot reconstructed images on the third row
    for i in range(8):
        axes[2, i].imshow(recon_imgs[i, 2].squeeze(), cmap="gray")
        axes[2, i].axis("off")
        axes[2, i].set_title("Reconstructed")

    plt.tight_layout()
    plt.savefig("dit_reconstruction_comparison.png")
    print("\nSaved DiT reconstruction comparison to dit_reconstruction_comparison.png")


if __name__ == "__main__":
    main()
