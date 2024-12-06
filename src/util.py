from matplotlib import pyplot as plt


def visualize_batch(train_loader, num_batches=1):
    """Visualize a few batches of images and actions from the train_loader."""
    for batch_idx, (x, left_action, _right_action) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        # Assuming x is of shape [batch_size, frames, channels, height, width]
        batch_size, frames, channels, height, width = x.shape

        fig, axes = plt.subplots(4, frames, figsize=(frames * 6, 4 * 6))
        for i in range(4):
            for j in range(frames):
                # Display each frame
                axes[i, j].imshow(x[i, j].permute(1, 2, 0).cpu().numpy())
                axes[i, j].axis("off")
                if j == 2:
                    axes[i, j].set_title(f"Action: {left_action[i, j].item()}")

        plt.show()
