import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from util_functions import play_one_step, sample_experiences


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


env = gym.make("CartPole-v1")
model = DQNet()
batch_size = 32
discount_factor = 0.95
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = F.mse_loss


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


def training_step(batch_size):
    """
    Performs a training step of the Deep Q network

    It starts by sampling a batch of previous experiences from the replay buffer and computing the target
    Q values as the immediate rewards plus the max expected discounted future rewards (assuming the agent always
    plays optimally)

    Params
    ------
        batch_size: The number of experiences to draw from the replay buffer
    """

    # Zero out the gradients from a previous epoch
    optimizer.zero_grad()

    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences

    # Turn off gradient computation when predicting the expected future Q values
    with torch.no_grad():
        next_q_values = model(torch.from_numpy(next_states))

    max_q_values = torch.max(next_q_values, axis=1)

    # immediate rewards plus max discounted future expected Q values
    target_q_values = (
        rewards + (1 - dones) * discount_factor * max_q_values.values.numpy()
    )

    # Mask the Q values from actions not chosen from the agent
    mask = F.one_hot(torch.from_numpy(actions), 2)

    # Predict the next Q values
    all_q_values = model(torch.from_numpy(states))

    # Compute the loss between our predicted next Q values and the target Q values
    # maybe put keepdims=True here, but then target_q_values should be of different shape?
    q_values = torch.sum(all_q_values * mask, axis=1)
    loss = loss_fn(torch.from_numpy(target_q_values).float(), q_values)

    # backprop & gradient descent
    loss.backward()
    optimizer.step()


for episode in range(600):
    obs = env.reset()
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(
            env, obs, epsilon_greedy_policy, epsilon
        )
        if done:
            break
    if episode > 50:
        training_step(batch_size)
