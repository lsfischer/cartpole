import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DQNet(nn.Module):
    def __init__(self):
        super(DQNet, self).__init__()

        # 4 being the observation space shape
        self.input_layer = nn.Linear(4, 32)
        self.hidden_layer = nn.Linear(32, 32)

        # 2 being the number of possible actions
        self.output_layer = nn.Linear(32, 2)

    def forward(self, inputs):
        x = self.input_layer(inputs)
        x = F.elu(x)

        x = self.hidden_layer(x)
        x = F.elu(x)

        return self.output_layer(x)


model = DQNet()


def epsilon_greedy_policy(state, epsilon=0):
    """
    Implements epsilon greedy exploration policy

    This exploration policy will choose a random action with probability epsilon, ensuring it continues to explore
    the entire environment, or it will choose a greedy action (the one the maximizes the Q values) with probability
    1 - epsilon. It's common to start with a very high epsilon (e.g. 1.0) and gradually reduce it (e.g. to 0.05)

    Params
    ------
        state: The current state of the environment to be fed to the policy network
        epsilon: The probability of choosing a random action over a greedy action

    Returns
    -------
        The action to be taken (0/1 or left/right in case of cartpole)
    """
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        q_values = model(torch.unsqueeze(torch.from_numpy(state), dim=0))

        # Choose the action the maximizes the predicted Q values for the current state
        return torch.argmax(q_values[0]).item()
