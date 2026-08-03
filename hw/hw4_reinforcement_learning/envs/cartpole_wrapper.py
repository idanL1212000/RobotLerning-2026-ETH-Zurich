import gymnasium as gym
import numpy as np


class CartPoleWrapper:
    """
    A lightweight wrapper around Gymnasium's CartPole environment.

    This wrapper simplifies the Gymnasium API for this homework:
    - reset() -> state
    - step(action) -> next_state, reward, done, info

    It also:
    - converts observations to np.float32
    - merges `terminated` and `truncated` into a single `done` flag
    - exposes `state_dim` and `action_dim`
    """

    def __init__(self, env_name="CartPole-v1", seed=0, render_mode=None):
        """
        Create the environment.

        Args:
            env_name (str): name of the Gymnasium environment
            seed (int): random seed used for reproducibility
            render_mode (str or None): rendering mode, e.g. None or "rgb_array"
        """
        self.env_name = env_name
        self.seed = seed
        self.render_mode = render_mode

        # Create the underlying Gymnasium environment.
        self.env = gym.make(env_name, render_mode=render_mode)

        # Save observation and action spaces for later use.
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Set random seeds for reproducibility.
        self.env.reset(seed=seed)
        self.action_space.seed(seed)

        # For CartPole:
        # - observation is a 4-dimensional vector
        # - action space has 2 discrete actions
        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.n

    def reset(self):
        """
        Reset the environment and return the initial state.

        Returns:
            np.ndarray: initial state, shape (state_dim,), dtype np.float32
        """
        state, _ = self.env.reset()
        return np.asarray(state, dtype=np.float32)

    def step(self, action):
        """
        Take one step in the environment.

        Args:
            action (int): discrete action chosen by the agent

        Returns:
            next_state (np.ndarray): next observation, shape (state_dim,)
            reward (float): reward received after taking the action
            done (bool): whether the episode has ended
            info (dict): extra diagnostic information from Gymnasium
        """
        next_state, reward, terminated, truncated, info = self.env.step(action)

        # Combine the two Gymnasium termination signals into one `done` flag.
        done = terminated or truncated

        return (
            np.asarray(next_state, dtype=np.float32),
            float(reward),
            done,
            info,
        )

    def sample_action(self):
        """
        Sample a random action from the action space.

        Returns:
            int: a random discrete action
        """
        return int(self.action_space.sample())

    def close(self):
        """
        Close the environment and release resources.
        """
        self.env.close()