import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog


class CustomCNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Embedding layer (Assume the color encoding has 5 possible values)
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=16)  # 5 colors, embedding dim=16

        # Adjusted Convolutional layers for 10x10 input
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1)  # Output: 8x8x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)  # Output: 6x6x64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1)  # Output: 5x5x128

        # Calculate the flattened size after convolutional layers
        self._conv_out_size = self._get_conv_output_size(obs_space.shape)

        # Fully connected layers
        self.fc1 = nn.Linear(self._conv_out_size, 256)
        self.fc2 = nn.Linear(256, 256)

        # Output layer
        self.output_layer = nn.Linear(256, num_outputs)

    def _get_conv_output_size(self, shape):
        o = torch.zeros(1, *shape)
        o = self.embedding(o.long()).permute(0, 3, 1, 2)  # Simulate embedding output
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]
        x = self.embedding(x.long()).permute(0, 3, 1, 2)  # (batch_size, channels, height, width)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.output_layer(x)
        return output, state

