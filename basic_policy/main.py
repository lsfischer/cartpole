import gym
from gym import wrappers


def basic_policy(obs):
    """
    Basic policy that move the agent in the same direction in which the pole is falling into

    Params
    ------
        obs Environment observations, index 2 indicates the angle of the pole
    Returns
    -------
        The action to take. 0-> Accelerate left; 1-> Accelerate right
    """
    return 0 if obs[2] < 0 else 1


env = gym.make("CartPole-v1")
env = wrappers.Monitor(env, "../../outputs/basic_policy", force=True)
obs = env.reset()

# 200 steps is the maximum for this task, after that we've won
for step in range(200):
    # Act accordingly to the basic policy
    action = basic_policy(obs)
    obs, reward, done, info = env.step(action)
    if done:
        print(f"Episode finished after {step + 1} timesteps")
        break

env.close()
