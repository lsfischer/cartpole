import gym
import torch
import numpy as np
import torch.nn.functional as F
from gym import wrappers

from neural_policy import NeuralPolicy
from util_functions import play_multiple_episodes, discount_and_normalize_rewards

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

# Training the neural policy

for epoch in range(n_epochs):

    # Zero out the gradients from a previous epoch
    optimizer.zero_grad()

    rewards, gradients = play_multiple_episodes(
        env, episodes_per_epoch, max_steps, model, loss_fn
    )
    discounted_rewards = discount_and_normalize_rewards(rewards, discount_factor)

    # For each trainable parameter, average out the gradients corresponding to that parameter
    for var_index in range(len(trainable_params)):
        mean_grads = torch.mean(
            torch.stack(
                [
                    step_reward * gradients[episode_idx][step][var_index]
                    for episode_idx, episode_rewards in enumerate(discounted_rewards)
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


# Running the trained neural policy
env = wrappers.Monitor(env, "../../outputs/policy_gradients", force=True)
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
