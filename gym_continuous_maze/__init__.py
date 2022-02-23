import gym

gym.register(
    id="ContinuousMaze-v0",
    entry_point="gym_continuous_maze.gym_continuous_maze:ContinuousMaze",
    max_episode_steps=200,
)
