import gym
import numpy as np
from collections import deque

replay_buffer = deque(maxlen=2000)


def sample_experiences(batch_size):
    sampled_indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[i] for i in sampled_indices]

    # Separate the experiences in the batch into the 5 different aspects they contain (state, action, etc.)
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)
    ]

    return states, actions, rewards, next_states, dones


def play_one_step(env, state, epsilon_greedy_policy_fn, epsilon):
    action = epsilon_greedy_policy_fn(state, epsilon)
    next_state, reward, done, info = env.step(action)

    # For each step we add to our experiences the state we were on, the action we chose the reward we got for it
    # the state in which we ended up in and whether the episode finished
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info
