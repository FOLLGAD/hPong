import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import pickle


def set_seed(seed):
    """Set the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(1337)


class PongEnv:
    def __init__(
        self, height=32, width=64, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.height = height
        self.width = width
        self.device = device

        # Game parameters adjusted for 32x64 resolution
        self.paddle_height = 8
        self.paddle_width = 1
        self.ball_size = 3
        self.paddle_speed = 1
        self.ball_speed = 1

        # Initialize state
        self.reset()

    def reset(self):
        """Reset the game state and return initial observation."""
        # Paddle positions (vertical center)
        self.left_paddle = self.height // 2
        self.right_paddle = self.height // 2

        # Ball position (center of the screen)
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2

        # Random initial ball direction
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        if np.random.random() < 0.5:
            angle += np.pi

        self.ball_dx = self.ball_speed * np.cos(angle)
        self.ball_dy = self.ball_speed * np.sin(angle)

        return self._get_state()

    def _get_state(self):
        """Convert game state to tensor observation."""
        state = torch.zeros((1, self.height, self.width), device=self.device)

        # Draw paddles (single pixel width)
        left_top = max(0, int(self.left_paddle - self.paddle_height // 2))
        left_bottom = min(self.height, int(self.left_paddle + self.paddle_height // 2))
        right_top = max(0, int(self.right_paddle - self.paddle_height // 2))
        right_bottom = min(
            self.height, int(self.right_paddle + self.paddle_height // 2)
        )

        state[0, left_top:left_bottom, 0] = 1.0  # Left paddle
        state[0, right_top:right_bottom, -1] = 1.0  # Right paddle

        ball_x = int(round(self.ball_x))
        ball_y = int(round(self.ball_y))
        for i in range(self.ball_size):
            for j in range(self.ball_size):
                if 0 <= ball_x + i < self.width and 0 <= ball_y + j < self.height:
                    state[0, ball_y + j, ball_x + i] = 1.0

        return state

    def step(self, left_action, right_action):
        """
        Take one game step.
        Actions: -1 (up), 0 (stay), 1 (down)
        Returns: (next_state, reward, done)
        """
        # Move paddles
        self.left_paddle += left_action * self.paddle_speed
        self.right_paddle += right_action * self.paddle_speed

        # Clamp paddle positions
        self.left_paddle = np.clip(
            self.left_paddle,
            self.paddle_height // 2,
            self.height - self.paddle_height // 2,
        )
        self.right_paddle = np.clip(
            self.right_paddle,
            self.paddle_height // 2,
            self.height - self.paddle_height // 2,
        )

        # Store previous ball position for collision detection
        prev_ball_x = self.ball_x
        prev_ball_y = self.ball_y

        # Move ball
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Ball collision with top/bottom
        if self.ball_y <= 0 or self.ball_y >= self.height - self.ball_size:
            self.ball_dy *= -1
            self.ball_y = np.clip(self.ball_y, 0, self.height - self.ball_size)

        reward = 0
        done = False

        if (
            self.ball_x < self.ball_size // 2
            and abs(self.ball_y - self.left_paddle) <= self.paddle_height // 2
        ):
            hit_pos = (self.ball_y - self.left_paddle) / (self.paddle_height // 2)
            self.ball_dy += hit_pos * 0.5
            self.ball_dx = (
                abs(self.ball_dx) * 1.1
            )  # Bounce right, slight speed increase
            self.ball_x = 1
            reward = 1

        elif (
            prev_ball_x < self.width - self.ball_size
            and self.ball_x >= self.width - self.ball_size - 1
            and abs(self.ball_y - self.right_paddle) <= self.paddle_height // 2
        ):
            # Calculate the hit position on the paddle
            hit_pos = (self.ball_y - self.right_paddle) / (self.paddle_height // 2)
            # Adjust ball direction based on hit position
            self.ball_dy += hit_pos * 0.5
            self.ball_dx = (
                -abs(self.ball_dx) * 1.1
            )  # Bounce left, slight speed increase
            self.ball_x = self.width - self.ball_size - 1
            reward = 1

        # Ball out of bounds
        elif self.ball_x < 0:
            reward = -1
            done = True
        elif self.ball_x >= self.width:
            reward = -1
            done = True

        return self._get_state(), reward, done

    def render(self):
        """Return game state as a numpy array for visualization."""
        return self._get_state().cpu().numpy()[0]


class PongAgent:
    side: str

    def __init__(self, side="right"):
        self.side = side  # left or right

    def choose_action(self, env: PongEnv):
        """Choose an action based on current state."""
        current_paddle_y = env.right_paddle

        # Calculate the ball's future y position
        future_ball_y = env.ball_y

        # If future_ball_y is None, stay in place
        if future_ball_y is None:
            return 0

        # Move the paddle towards the future ball position
        if current_paddle_y < future_ball_y:
            return 1  # Move down
        elif current_paddle_y > future_ball_y:
            return -1  # Move up
        else:
            return 0  # Stay


class PongDataset(Dataset):
    frames_per_sample: int = 4

    def __init__(self, num_episodes=250, num_frames=500, frames_per_sample=4):
        self.data = self._generate_pong_dataset(num_episodes, num_frames)
        self.frames_per_sample = frames_per_sample

    def _generate_pong_dataset(self, num_episodes, num_frames):
        dataset = []
        env = PongEnv()
        right_agent = PongAgent("right")
        left_agent = PongAgent("left")

        for _ in range(num_episodes):
            state = env.reset()

            is_random_player = np.random.random() < 0.5

            for _ in range(num_frames):
                if is_random_player:
                    left_action = np.random.choice([-1, 0, 1])
                    right_action = right_agent.choose_action(env)
                else:
                    left_action = left_agent.choose_action(env)
                    right_action = right_agent.choose_action(env)

                dataset.append(
                    (
                        state.clone(),  # Image of the current state
                        torch.tensor(
                            left_action, dtype=torch.int64
                        ),  # Left action, save before stepping
                        torch.tensor(right_action, dtype=torch.int64),
                    )  # Right action
                )

                next_state, reward, done = env.step(
                    left_action, right_action
                )  # Step the environment
                state = next_state

                if done:
                    state = env.reset()

        return dataset

    def __len__(self):
        return len(self.data) - self.frames_per_sample + 1

    def __getitem__(self, idx):
        if idx + self.frames_per_sample > len(self.data):
            raise IndexError("Index out of range for available data samples.")

        samples = self.data[idx : idx + self.frames_per_sample]

        states = [sample[0] for sample in samples]
        left_actions = [sample[1] for sample in samples]
        right_actions = [sample[2] for sample in samples]

        stacked_states = torch.stack(states)

        return stacked_states, torch.stack(left_actions), torch.stack(right_actions)


pong_test_dataset = PongDataset(num_episodes=10, num_frames=32)

dataset_path = "pong_dataset.pkl"

print("Generating dataset...")
pong_dataset = PongDataset()
print("Finished generating")
