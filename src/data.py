import torch
from torch.utils.data import Dataset
import numpy as np

class BouncingBallDataset(Dataset):
    def __init__(self, num_frames=10000, img_size=32, ball_radius=2, velocity_range=(-0.7, 0.7)):
        super().__init__()
        self.img_size = img_size
        self.ball_radius = ball_radius
        self.speed = 3
        
        # Generate random initial positions and velocities
        self.positions = []
        self.velocities = []
        
        # Start with one sequence
        pos = torch.tensor([img_size/2, img_size/2])  # Start in middle
        vel = torch.tensor([
            np.random.uniform(*velocity_range),
            np.random.uniform(*velocity_range)
        ])
        
        # Generate frames using physics
        for _ in range(num_frames):
            # Update position
            pos = pos + vel * self.speed
            
            # Bounce off walls
            for i in range(2):
                if pos[i] <= ball_radius:
                    pos[i] = ball_radius
                    vel[i] = abs(vel[i])  # Reverse velocity
                elif pos[i] >= img_size - ball_radius:
                    pos[i] = img_size - ball_radius
                    vel[i] = -abs(vel[i])  # Reverse velocity
            
            self.positions.append(pos.clone())
            self.velocities.append(vel.clone())
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        # Create empty image
        img = torch.zeros((1, self.img_size, self.img_size))
        
        # Get ball position
        pos = self.positions[idx]
        
        # Create ball (simple circle approximation)
        y, x = torch.meshgrid(
            torch.arange(self.img_size),
            torch.arange(self.img_size),
            indexing='ij'
        )
        
        # Calculate distances from ball center
        distances = torch.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        
        # Set ball pixels
        img[0, distances < self.ball_radius] = 1.0
        
        return img

class SequentialBouncingBallDataset(Dataset):
    def __init__(self, num_sequences=1000, sequence_length=2, **kwargs):
        self.base_dataset = BouncingBallDataset(num_sequences + sequence_length - 1, **kwargs)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.base_dataset) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Return sequence of frames
        frames = [self.base_dataset[idx + i] for i in range(self.sequence_length)]
        return torch.stack(frames)
