import torch
from tqdm import tqdm
from vae import ViTVAE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from PongSim import pong_test_dataset


@torch.no_grad()  # Disable gradient computation during evaluation
def evaluate_vae(model, test_loader, device="cuda"):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    original_images = []
    reconstructed_images = []

    pbar = tqdm(test_loader, desc="Evaluating")
    for i, (x, left_action, right_action) in enumerate(pbar):
        if i >= 2:
            break

        # Unpack the batch - TensorDataset returns a tuple
        x = x[:, :3].to(device)  # Changed from x = x.to(device)

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


def generate_from_latents(model, latent_values, device="cuda"):
    """Generate images from specific latent values"""
    model.eval()
    with torch.no_grad():
        # Convert input to tensor and ensure shape is correct
        z = torch.tensor(latent_values, dtype=torch.float32).to(device)
        if len(z.shape) == 1:
            z = z.unsqueeze(0)  # Add batch dimension if needed

        # Generate image through decoder
        generated = model.decoder(z)

        return generated


pong_test_dataset.frames_per_sample = 32
test_loader = DataLoader(pong_test_dataset, batch_size=32, shuffle=False)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ViTVAE(
        img_size=(32, 64),
        patch_size=4,
        embed_dim=96,
        depth=6,
        num_heads=8,
        latent_dim=4,
    ).to(device)
    checkpoint = torch.load("best/vae_pong_best.pt", map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    metrics, viz_data = evaluate_vae(model, test_loader, device)

    print("\nEvaluation complete!")

    # Get original and reconstructed images
    orig_imgs = viz_data["original"]
    recon_imgs = viz_data["reconstructed"]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 8, figsize=(32, 8))

    # Plot original images on top row
    for i in range(8):
        axes[0, i].imshow(orig_imgs[i, 2].squeeze(), cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title("Original")

    # Plot reconstructed images on bottom row
    for i in range(8):
        axes[1, i].imshow(recon_imgs[i, 2].squeeze(), cmap="gray")
        axes[1, i].axis("off")
        axes[1, i].set_title("Reconstructed")

    plt.tight_layout()
    plt.savefig("reconstruction_comparison.png")
    print("\nSaved reconstruction comparison to reconstruction_comparison.png")

    while True:
        try:
            user_input = input("\nEnter 4 comma-separated floats (or 'q' to quit): ")
            if user_input.lower() == "q":
                break

            latent = [float(x) for x in user_input.split(",")]
            if len(latent) != 4:
                print("Please enter exactly 4 values")
                continue

            generated = generate_from_latents(model, latent, device)

            plt.figure(figsize=(4, 4))
            plt.imshow(generated[0, 2].cpu().squeeze(), cmap="gray")
            plt.axis("off")
            plt.title(f"Latent: {latent}")
            plt.savefig("custom_latent.png")
            plt.close()
            print("Saved result to custom_latent.png")

        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
