from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from envs.so100_mdp_utils import (
    compute_reward,
    get_obs,
    process_action,
    reset_robot,
    reset_target_position,
)


class SO100RLEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, xml_path: Path, render_mode=None):
        super().__init__()

        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None

        # Timing
        self.sim_timestep = self.model.opt.timestep
        self.ctrl_decimation = 50
        self.ctrl_timestep = self.sim_timestep * self.ctrl_decimation
        self.max_episode_length_s = 3
        self.max_episode_length = int(self.max_episode_length_s / self.ctrl_timestep)
        self.current_step = 0

        # Default robot configuration
        self.default_qpos = np.array(
            [0.0, -1.57, 1.0, 1.0, 0.0, 0.02239],
            dtype=np.float64,
        )

        # Evaluation metric
        self.ee_tracking_error = 0.0

        # Spaces
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        # Convenience attributes
        self.state_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = reset_robot(self.default_qpos)
        mujoco.mj_forward(self.model, self.data)

        base_pos = self.data.body("Base").xpos.copy()
        self.data.mocap_pos[0] = reset_target_position(base_pos)

        self.current_step = 0
        self.ee_tracking_error = 0.0

        obs = self._get_obs()
        info = {}
        return obs, info

    def _process_action(self, action: np.ndarray) -> np.ndarray:
        return process_action(action, self.model.jnt_range)

    def _compute_reward(self) -> float:
        q_vel = self.data.qvel.flat[:].copy()
        return compute_reward(self.ee_tracking_error, q_vel)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.data.ctrl[:] = self._process_action(action)

        for _ in range(self.ctrl_decimation):
            mujoco.mj_step(self.model, self.data)

        ee_pos = self.data.site("ee_site").xpos
        target_pos = self.data.mocap_pos[0]
        self.ee_tracking_error = float(np.linalg.norm(ee_pos - target_pos))

        reward = self._compute_reward()

        terminated = False
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_length

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        info = {
            "ee_tracking_error": self.ee_tracking_error,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.flat[:].copy()
        ee_pos_w = self.data.site("ee_site").xpos.copy()
        ee_rot_w = self.data.site("ee_site").xmat.reshape(3, 3).copy()
        base_pos_w = self.data.body("Base").xpos.copy()
        base_rot_w = self.data.body("Base").xmat.reshape(3, 3).copy()
        target_pos_w = self.data.mocap_pos[0].copy()

        return get_obs(
            qpos=qpos,
            ee_pos_w=ee_pos_w,
            ee_rot_w=ee_rot_w,
            base_pos_w=base_pos_w,
            base_rot_w=base_rot_w,
            target_pos_w=target_pos_w,
        )

    def render(self):
        if self.render_mode != "human":
            return

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None