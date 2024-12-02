import torch
from tqdm import tqdm
from models import ViTVAE
from torchvision import transforms
from PIL import Image
import glob
import matplotlib.pyplot as plt


@torch.no_grad()  # Disable gradient computation during evaluation
def evaluate_vae(model, test_loader, device="cuda"):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    original_images = []
    reconstructed_images = []

    pbar = tqdm(test_loader, desc="Evaluating")
    for batch in pbar:
        # Unpack the batch - TensorDataset returns a tuple
        x = batch[0].to(device)  # Changed from x = x.to(device)

        # Rest of the function remains the same
        recon_x, mu, logvar = model(x)

        # Compute losses
        recon_loss = torch.nn.functional.binary_cross_entropy(
            recon_x, x, reduction="sum"
        )
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.0002 * kl_loss  # Using same beta as in training

        # Accumulate metrics
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

        # Store some images for visualization
        if len(original_images) < 8:  # Store first 8 images
            original_images.append(x.cpu())
            reconstructed_images.append(recon_x.cpu())

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss/len(x):.4f}",
                "recon": f"{recon_loss/len(x):.4f}",
                "kl": f"{kl_loss/len(x):.4f}",
            }
        )

    # Calculate average metrics
    n_samples = len(test_loader.dataset)
    avg_loss = total_loss / n_samples
    avg_recon = total_recon / n_samples
    avg_kl = total_kl / n_samples

    print("\nTest set results:")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Reconstruction loss: {avg_recon:.4f}")
    print(f"KL loss: {avg_kl:.4f}")

    # Return metrics and images for potential visualization
    metrics = {"loss": avg_loss, "recon_loss": avg_recon, "kl_loss": avg_kl}

    visualization_data = {
        "original": torch.cat(original_images, dim=0),
        "reconstructed": torch.cat(reconstructed_images, dim=0),
    }

    return metrics, visualization_data


def create_model(
    img_size=32,
    patch_size=4,
    embed_dim=96,
    depth=6,
    num_heads=8,
    latent_dim=4,
    device="cuda",
):
    """Create and initialize the ViTVAE model"""
    model = ViTVAE(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        latent_dim=latent_dim,
    ).to(device)

    return model


def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model and load checkpoint
    model = create_model(device=device)
    checkpoint = torch.load("checkpoints/latest_model.pt")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Setup image transforms
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
        ]
    )

    # Load all PNG images
    test_images = []
    for img_path in glob.glob("example_data/*.png"):
        img = Image.open(img_path)
        test_images.append(transform(img))

    test_data = torch.stack(test_images)

    test_dataset = torch.utils.data.TensorDataset(test_data)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False
    )

    # Evaluate
    model.eval()
    metrics, viz_data = evaluate_vae(model, test_loader, device)

    print("\nEvaluation complete!")

    # Get original and reconstructed images
    orig_imgs = viz_data["original"]
    recon_imgs = viz_data["reconstructed"]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 8, figsize=(16, 4))

    # Plot original images on top row
    for i in range(8):
        axes[0, i].imshow(orig_imgs[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

    # Plot reconstructed images on bottom row
    for i in range(8):
        axes[1, i].imshow(recon_imgs[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstructed")

    plt.tight_layout()
    plt.savefig("reconstruction_comparison.png")
    print("\nSaved reconstruction comparison to reconstruction_comparison.png")


if __name__ == "__main__":
    main()
