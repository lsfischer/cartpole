import gym
import torch
import numpy as np
import torch.nn.functional as F

from neural_policy import NeuralPolicy


def play_one_step(env, obs, model, loss_fn):
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

    # The target probability of going left. This is 1. is the action is 0 (going left) or 0. if the action was going
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


def play_multiple_episodes(env, episodes, max_steps, model, loss_fn):
    """
    Plays the game multiple times resetting the environment everytime the game finishes.
    This serves to gather experience to train the neural network policy (through the form of storing its gradients)

    Params
    ------
        env: The game environment to be interacted with
        episodes: Number of times to repeat the game
        max_steps: Maximum number of steps to take in the game (for cartpole this is 200)
        model: The neural network policy
        loss_fn: The loss function used to train

    Returns
    -------
        rewards: A List[List] containing the rewards obtained for each action in each repetition of the game
        gradients: A List[List] containing the gradients obtained for each action in each repetition of the game
    """

    rewards, gradients = [], []

    # Repeat a complete game
    for episode in range(episodes):
        episode_rewards, episode_gradients = [], []

        # Reset the environment in the beginning of each episode
        obs = env.reset()

        # Perform a single action
        for step in range(max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            episode_rewards.append(reward)
            episode_gradients.append(grads)

            if done:
                break

        rewards.append(episode_rewards)
        gradients.append(episode_gradients)
    return rewards, gradients


def discount_rewards(episode_rewards, discount_factor):
    """
    Performs reward discount by traversing through the rewards backwards multiplying each step by discount_factor

    Example:
    >>> discount_rewards([10, 0, -50], discount_factor=0.8)
    array([-22, -40, -50])

    explenation:
        index 0 : 10 + 0.8 * 0 + 0.8**2 * -50 = -22
        index 1 : 0 + 0.8 * -50 = -40
        index 2 : -50 = -50

    Params
    ------
        episode_rewards: The list of rewards obtained for this episode
        discount_factor: The discount factor to apply to each step
            essentially, how much do rewards in the future matter to the current step. Example with discount_factor=0.95
            rewards 13 steps into the future count half as much as intermediate rewards (0.95**13 ~= 0.5)

    Returns
    -------
        discounted_rewards: A numpy array of the same shape as episode_rewards containing the discounted rewards for
        each step in the episode
    """
    discounted_rewards = np.array(episode_rewards)

    # iterate backwards through the rewards discounting it at each step
    for step in range(len(episode_rewards) - 2, -1, -1):
        discounted_rewards[step] += discounted_rewards[step + 1] * discount_factor
    return discounted_rewards


def discount_and_normalize_rewards(rewards, discount_factor):
    """
    Computes the discounted rewards for all episodes played and normalizes them

    Params
    ------
        rewards: The list of rewards obtained for each episode played
        discount_factor: the discount factor to be used

    Returns
    -------
        The normalized discounted rewards for each episode
    """
    # Get the discounted rewards for each episode played
    discounted_rewards = [
        discount_rewards(reward, discount_factor) for reward in rewards
    ]
    flat_rewards = np.concatenate(discounted_rewards)

    # Return the reward for every episode normalized based on all the rewards seen for every episode
    return [
        (reward - flat_rewards.mean()) / flat_rewards.std()
        for reward in discounted_rewards
    ]


n_epochs = 150
episodes_per_epoch = 10
max_steps = 200
discount_factor = 0.95
model = NeuralPolicy()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = F.binary_cross_entropy

env = gym.make("CartPole-v1")

trainable_params = [p for p in model.parameters() if p.requires_grad]

for epoch in range(n_epochs):
    optimizer.zero_grad()

    rewards, gradients = play_multiple_episodes(
        env, episodes_per_epoch, max_steps, model, loss_fn
    )
    discounted_rewards = discount_and_normalize_rewards(rewards, discount_factor)

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
        # discounted rewards
        trainable_params[var_index].grad = mean_grads

    # Do gradient descent and apply the gradients
    optimizer.step()

    # report how long on average the network is able to keep the pole upright
    print(
        f"epoch {epoch},  mean rewards per episode {np.mean([sum(episode_reward) for episode_reward in rewards])}"
    )
