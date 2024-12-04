import torch
from torch import nn


class FFN(nn.Module):
    def __init__(self, latent_dim=4, action_dim=1, hidden_dim=128):
        super(FFN, self).__init__()
        self.input_dim = latent_dim * 2 + action_dim  # input_mu, input_logvar, action
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Define the feedforward network layers
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc3_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3_mu.weight)
        nn.init.zeros_(self.fc3_mu.bias)
        nn.init.xavier_uniform_(self.fc3_logvar.weight)
        nn.init.zeros_(self.fc3_logvar.bias)

    def forward(self, input_mu, input_logvar, action):
        # Concatenate inputs
        x = torch.cat([input_mu, input_logvar, action], dim=-1)

        # Pass through the network
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Predict next latent dimensions
        pred_mu = self.fc3_mu(x)
        pred_logvar = self.fc3_logvar(x)

        return pred_mu, pred_logvar
