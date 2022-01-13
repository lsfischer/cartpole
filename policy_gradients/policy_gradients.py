import gym
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from gym import wrappers
from gym.wrappers.time_limit import TimeLimit

from util_functions import play_multiple_episodes, discount_and_normalize_rewards


class NeuralPolicy(nn.Module):
    """
    Simple neural network policy chooser
    """

    def __init__(self):
        super(NeuralPolicy, self).__init__()

        # 4 being the observation space (cart position, velocity, pole angle and angular velocity)
        self.input_layer = nn.Linear(4, 5)
        self.output_layer = nn.Linear(5, 1)

    def forward(self, inputs):
        x = F.elu(self.input_layer(inputs))
        return torch.sigmoid(self.output_layer(x))


# Setup training constants
n_epochs = 150
episodes_per_epoch = 10
max_steps = 200
discount_factor = 0.95
env = gym.make("CartPole-v1")

model = NeuralPolicy()
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = F.binary_cross_entropy


def play_one_step(
    env: TimeLimit, obs: np.ndarray, model: NeuralPolicy, loss_fn: Callable
):
    """
    Performs one action in the game computing the gradients that would make the chosen action the most likely.

    This essentially implements the first step in the REINFORCE algorithm
    > Let the neural network policy play the game several times, at each step compute the gradients that would make
    the chose action even more likely, but don't apply those gradients just yet

    These gradients are returned to be stored so that we can tweak them according to whether or not the performed action
    gave a positive reward

    Params
    ------
        env: The game environment to be interacted with
        obs: The current state of the environment (which is fed to the neural network policy)
        model: The neural network policy
        loss_fn: The loss function used to train

    Returns
    -------
        obs: The new state of the environment after the action has been taken
        reward: The reward we got for the chose action (in cartpole the reward is always 1 while the pole is upright)
        done: Whether the game has ended
        grads: The gradients that would have made the chose action even more likely
    """
    # Get the probability of going left by feeding the network policy with the current state of the environment
    left_proba = model(torch.unsqueeze(torch.from_numpy(obs), dim=0))
    action = torch.rand((1, 1)) > left_proba

    # The target probability of going left. This is 1. is the action is 0 (going left) or 0. if the action was
    # 1 (going right)
    y_target = torch.Tensor([1.0]) - action.float()

    # Backprop the loss function
    loss = loss_fn(left_proba, y_target)
    loss.backward()

    # get gradients of loss function with respect to model's trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    grads = [torch.clone(t.grad) for t in trainable_params]

    # zero out the gradients to not pollute next computations
    model.zero_grad()

    obs, reward, done, info = env.step(int(action[0, 0].item()))

    return obs, reward, done, grads


# Training the neural policy
def train(
    env: TimeLimit,
    n_epochs: int,
    episodes_per_epoch: int,
    max_steps: int,
    model: NeuralPolicy,
    optimizer: torch.optim.Adam,
    loss_fn: Callable,
    discount_factor: float,
):
    """
    Trains the neural network by first accumulating a lot of experience from random episodes and then applying the
    gradients discounted by whether a specific action was a positive action to take

    Params
    ------
        env: The game environment to be interacted with
        n_epochs: Number of epochs to train the network
        episodes_per_epoch: How many replays of the game to do per epoch
        max_steps: The maximum number of steps in the game (200 in the case of cartpole)
        model: The model to be trained
        optimizer: the optimizer used
        loss_fn: The loss function used to computed gradients
        discount_factor: The discount factor to apply to each step
            essentially, how much do rewards in the future matter to the current step. Example with discount_factor=0.95
            rewards 13 steps into the future count half as much as intermediate rewards (0.95**13 ~= 0.5)
    """

    for epoch in range(n_epochs):

        # Zero out the gradients from a previous epoch
        optimizer.zero_grad()

        rewards, gradients = play_multiple_episodes(
            env, episodes_per_epoch, max_steps, model, play_one_step, loss_fn
        )
        discounted_rewards = discount_and_normalize_rewards(rewards, discount_factor)

        # For each trainable parameter, average out the gradients corresponding to that parameter
        for var_index in range(len(trainable_params)):
            mean_grads = torch.mean(
                torch.stack(
                    [
                        step_reward * gradients[episode_idx][step][var_index]
                        for episode_idx, episode_rewards in enumerate(
                            discounted_rewards
                        )
                        for step, step_reward in enumerate(episode_rewards)
                    ]
                ),
                axis=0,
            )

            # Update the gradients of the current trainable parameter to be the average of gradients multiplied by the
            # discounted rewards.
            trainable_params[var_index].grad = mean_grads

        # Do gradient descent and apply the previously computed gradients
        optimizer.step()

        # report how long on average the network is able to keep the pole upright
        print(
            f"epoch {epoch},  mean rewards per episode {np.mean([sum(episode_reward) for episode_reward in rewards])}"
        )


train(
    env,
    n_epochs,
    episodes_per_epoch,
    max_steps,
    model,
    optimizer,
    loss_fn,
    discount_factor,
)

# Running the trained neural policy
env = wrappers.Monitor(env, "./outputs/policy_gradients", force=True)
obs = env.reset()

# 200 steps is the maximum for this task, after that we've won
for step in range(200):
    model.eval()

    left_proba = model(torch.unsqueeze(torch.from_numpy(obs), dim=0))
    action = torch.rand((1, 1)) > left_proba
    obs, reward, done, info = env.step(int(action[0, 0].item()))

    if done:
        print(f"Episode finished after {step + 1} timesteps")
        break

env.close()
