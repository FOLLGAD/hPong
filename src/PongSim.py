import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import pickle


class PongEnv:
    def __init__(
        self, height=32, width=64, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.height = height
        self.width = width
        self.device = device

        # Game parameters adjusted for 32x64 resolution
        self.paddle_height = 8  # 1/4 of height
        self.paddle_width = 1  # Single pixel width
        self.ball_size = 1  # Single pixel ball
        self.paddle_speed = 2  # Move 2 pixels per step
        self.ball_speed = 1  # Move 1 pixel per step

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

        # Draw ball (single pixel)
        ball_x = int(round(self.ball_x))
        ball_y = int(round(self.ball_y))
        if 0 <= ball_x < self.width and 0 <= ball_y < self.height:
            state[0, ball_y, ball_x] = 1.0

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
        if self.ball_y <= 0 or self.ball_y >= self.height - 1:
            self.ball_dy *= -1
            self.ball_y = np.clip(self.ball_y, 0, self.height - 1)

        # Ball collision with paddles
        reward = 0
        done = False

        # Left paddle collision
        if (
            prev_ball_x > 0
            and self.ball_x <= 1
            and abs(self.ball_y - self.left_paddle) <= self.paddle_height // 2
        ):
            # Calculate the hit position on the paddle
            hit_pos = (self.ball_y - self.left_paddle) / (self.paddle_height // 2)
            # Adjust ball direction based on hit position
            self.ball_dy += hit_pos * 0.5
            self.ball_dx = (
                abs(self.ball_dx) * 1.1
            )  # Bounce right, slight speed increase
            self.ball_x = 1
            reward = 1

        # Right paddle collision
        elif (
            prev_ball_x < self.width - 1
            and self.ball_x >= self.width - 2
            and abs(self.ball_y - self.right_paddle) <= self.paddle_height // 2
        ):
            # Calculate the hit position on the paddle
            hit_pos = (self.ball_y - self.right_paddle) / (self.paddle_height // 2)
            # Adjust ball direction based on hit position
            self.ball_dy += hit_pos * 0.5
            self.ball_dx = (
                -abs(self.ball_dx) * 1.1
            )  # Bounce left, slight speed increase
            self.ball_x = self.width - 2
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

    def predict_ball_trajectory(self, state, ball_pos, ball_vel):
        """Predict where the ball will intersect with the agent's paddle plane."""
        if ball_pos is None or ball_vel is None:
            return None

        ball_x, ball_y = ball_pos
        vel_x, vel_y = ball_vel

        # Only predict if ball is moving towards our side
        if (self.side == "left" and vel_x > 0) or (self.side == "right" and vel_x < 0):
            return None

        # Calculate time to reach paddle plane
        target_x = 0 if self.side == "left" else state.shape[2] - 1
        time_to_intersect = abs((target_x - ball_x) / vel_x)

        # Calculate y-intersection
        future_y = ball_y + vel_y * time_to_intersect

        # Account for bounces
        height = state.shape[1]
        while future_y < 0 or future_y >= height:
            if future_y < 0:
                future_y = -future_y
            if future_y >= height:
                future_y = 2 * (height - 1) - future_y

        return future_y

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    plt.style.use("dark_background")

    env = PongEnv()
    state = env.reset()
    right_agent = PongAgent("right")

    # Set up the figure and animation
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xticks([])
    ax.set_yticks([])
    img = ax.imshow(env.render(), cmap="gray", interpolation="nearest")

    def update(frame):
        right_action = right_agent.choose_action(env)
        left_action = np.random.choice([-1.0, 0.0, 1.0])

        state, reward, done = env.step(left_action, right_action)
        if done:
            env.reset()

        img.set_array(env.render())

        return [img]

    # Create animation (interval=50 means 20 FPS)
    anim = FuncAnimation(fig, update, frames=None, interval=50, blit=True)
    plt.show()


class PongDataset(Dataset):
    frames_per_sample: int = 4

    def __init__(self, num_samples=250, num_frames=500, frames_per_sample=4):
        self.data = self._generate_pong_dataset(num_samples, num_frames)
        self.frames_per_sample = frames_per_sample

    def _generate_pong_dataset(self, num_samples, num_frames):
        dataset = []
        env = PongEnv()
        right_agent = PongAgent("right")

        for _ in range(num_samples):
            state = env.reset()

            for _ in range(num_frames):
                right_action = right_agent.choose_action(env)
                left_action = np.random.choice([-1, 0, 1])

                next_state, reward, done = env.step(left_action, right_action)
                # Append the image of the current state and the actions
                dataset.append(
                    (
                        state.clone(),  # Image of the current state
                        torch.tensor(left_action, dtype=torch.int64),  # Left action
                        torch.tensor(right_action, dtype=torch.int64),
                    )  # Right action
                )
                state = next_state

                if done:
                    state = env.reset()

        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        samples = self.data[idx : idx + self.frames_per_sample]

        states = [sample[0] for sample in samples]
        left_actions = [sample[1] for sample in samples]
        right_actions = [sample[2] for sample in samples]

        stacked_states = torch.stack(states)

        return stacked_states, torch.stack(left_actions), torch.stack(right_actions)


dataset_path = "pong_dataset.pkl"

# Check if the dataset already exists
if os.path.exists(dataset_path):
    # Load the dataset from the file
    with open(dataset_path, "rb") as f:
        pong_dataset = pickle.load(f)
    print(f"Loaded existing dataset with {len(pong_dataset)} samples.")
else:
    # Generate the dataset and save it to a file
    pong_dataset = PongDataset()
    with open(dataset_path, "wb") as f:
        pickle.dump(pong_dataset, f)
    print(f"Generated and saved dataset with {len(pong_dataset)} samples.")
