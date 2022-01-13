import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Callable
from gym.wrappers.time_limit import TimeLimit

from util_functions import play_multiple_episodes, discount_and_normalize_rewards

# Create the model using Keras
model = keras.models.Sequential(
    [
        keras.layers.Dense(5, activation="elu", input_shape=[4]),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)

n_epochs = 150
episodes_per_epoch = 10
max_steps = 200
discount_factor = 0.95
optimizer = keras.optimizers.Adam(lr=0.01)
loss_fn = keras.losses.BinaryCrossentropy()

env = gym.make("CartPole-v1")


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

    # Define the context in which gradient computations should be captured
    with tf.GradientTape() as tape:

        # Get the probability of going left by feeding the network policy with the current state of the environment
        left_proba = model(obs[np.newaxis])
        action = tf.random.uniform([1, 1]) > left_proba

        # The target probability of going left. This is 1. is the action is 0 (going left) or 0. if the action was
        # 1 (going right)
        y_target = tf.constant([[1.0]]) - tf.cast(action, tf.float32)

        # Compute the loss value
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))

    # Compute the gradients of loss function with respect to model's trainable params
    grads = tape.gradient(loss, model.trainable_variables)

    obs, reward, done, info = env.step(int(action[0, 0].numpy()))

    return obs, reward, done, grads


def train(
    env: TimeLimit,
    n_epochs: int,
    episodes_per_epoch: int,
    max_steps: int,
    model: keras.models.Model,
    optimizer: keras.optimizers.Adam,
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

        rewards, gradients = play_multiple_episodes(
            env, episodes_per_epoch, max_steps, model, play_one_step, loss_fn
        )
        discounted_rewards = discount_and_normalize_rewards(rewards, discount_factor)

        all_mean_gradients = []
        for var_index in range(len(model.trainable_variables)):
            mean_grads = tf.reduce_mean(
                [
                    step_reward * gradients[episode_idx][step][var_index]
                    for episode_idx, episode_rewards in enumerate(discounted_rewards)
                    for step, step_reward in enumerate(episode_rewards)
                ],
                axis=0,
            )
            all_mean_gradients.append(mean_grads)
        optimizer.apply_gradients(zip(all_mean_gradients, model.trainable_variables))

        # Update the gradients of the current trainable parameter to be the average of gradients multiplied by the
        # discounted rewards

        # Do gradient descent and apply the gradients

        # report how long on average the network is able to keep the pole upright
        print(
            f"mean rewards per episode {np.mean([sum(episode_reward) for episode_reward in rewards])}"
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

# Run the trained network
obs = env.reset()

# 200 steps is the maximum for this task, after that we've won
for step in range(200):

    left_proba = model(obs[np.newaxis])
    action = tf.random.uniform([1, 1]) > left_proba
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))

    if done:
        print(f"Episode finished after {step + 1} timesteps")
        break

env.close()
