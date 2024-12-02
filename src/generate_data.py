# For VAE training (single frames)
from data import SequentialBouncingBallDataset
from torch.utils.data import DataLoader


dataset = SequentialBouncingBallDataset(num_sequences=10000, sequence_length=3)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# os.makedirs('generated_data', exist_ok=True)
# # Get a batch of images
# images = next(iter(train_loader))
# # Save individual images
# for i in range(32):
#     torchvision.utils.save_image(
#         images[i],
#         f'generated_data/bouncing_ball_{i:03d}.png',
#         normalize=True
#     )


# # For DiT training (sequential frames)
# seq_dataset = SequentialBouncingBallDataset(
#     num_sequences=10000,
#     sequence_length=2,  # pairs of frames for next-frame prediction
#     img_size=32
# )
# seq_loader = DataLoader(seq_dataset, batch_size=32, shuffle=True)
