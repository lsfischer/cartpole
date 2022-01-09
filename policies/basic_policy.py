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
