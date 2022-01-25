import numpy as np
from collections import deque

# Cache in which to store previous runs, to be sampled during training
replay_buffer = deque(maxlen=2000)


def sample_experiences(batch_size):
    """
    Samples `batch_size` examples from the replay buffer

    Params
    ------
        batch_size: The number of previous examples to sample

    Returns
    -------
        A tuple of 5 positions:
            position 0: contains states of the experiences sampled
            position 1: contains actions of the experiences sampled
            position 2: contains rewards of the experiences sampled
            position 3: contains immediate next states of the experiences sampled
            position 4: contains the information indicating whether the episode ended for the experiences sampled
    """

    sampled_indices = np.random.randint(len(replay_buffer), size=batch_size)
    batch = [replay_buffer[i] for i in sampled_indices]

    # Separate the experiences in the batch into the 5 different aspects they contain (state, action, etc.)
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)
    ]

    return states, actions, rewards, next_states, dones


def play_one_step(env, state, epsilon_greedy_policy_fn, epsilon):
    """
    Plays one full step in the environment by calling the epsilon greedy policy function and
    acting accordingly

    It also then appends the state to the replay buffer to be sampled in the future

    Params
    ------
        env: The catpole gym environment
        state: The snapshot observation of the current state of the environment
        epsilon_greedy_policy_fn: The epsilon greedy policy function to be called to choose an action
        epsilon: The epsilon to use in epsilon greedy policy

    Returns
    -------
        The resulting information of playing an action in the environment. A tuple with 4 positions
            position 0: The resulting state obtained from playing the chosen action
            position 1: The reward obtained by playing the chosen action
            position 2: Information about whether the episode has ended
            position 3: Additional information
    """
    action = epsilon_greedy_policy_fn(state, epsilon)
    next_state, reward, done, info = env.step(action)

    # For each step we add to our experiences the state we were on, the action we chose the reward we got for it
    # the state in which we ended up in and whether the episode finished
    replay_buffer.append((state, action, reward, next_state, done))
    return next_state, reward, done, info
