import gym
import numpy as np
from policies.basic_policy import basic_policy

env = gym.make("CartPole-v1")

totals = []
for episode in range(500):
    episode_rewards = 0

    # restart the environment each episode
    obs = env.reset()

    # 200 steps is the maximum for this task, after that we've won
    for step in range(200):
        # Act accordingly to the basic policy
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
        totals.append(episode_rewards)
