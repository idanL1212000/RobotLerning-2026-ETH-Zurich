"""Thin wrapper around the MuJoCo SO-100 simulation for evaluation.

Provides a simple API to:
  - reset to a keyframe
  - set actuator targets (joint angles)
  - step the simulation
  - query the current state (joint angles, EE pose, cube pose)
  - render camera images

Usage:
    env = SO100SimEnv(xml_path, control_hz=10.0)
    env.reset()
    obs = env.get_obs()           # dict with "joints", "ee", "cube"
    env.set_targets(joint_angles) # set position-actuator targets
    env.step()                    # step the simulation
    img = env.render("angle")     # render a camera view (BGR)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw")
CUBE_JOINT_NAME = "red_box_joint"
CUBE_DIM = 7  # cube free joint: pos(3) + quat_wxyz(4)
BIN_BODY_NAME = "bin"
BIN_CENTER_SITE_NAME = "bin_center"
CUBE_HALF_EXTENT_XY = 0.02
BIN_HALF_EXTENT_XY = 0.05

OBSTACLE_BODY_NAME = "obstacle"
UPPER_OBSTACLE_BODY_NAME = "upper_obstacle"

# ── Obstacle randomization defaults DO NOT MODIFY ──────────────────────────────────
DEFAULT_OBSTACLE_POS_STD = 0.01  # metres; center-zone Gaussian noise
DEFAULT_ADVERSARIAL_OBSTACLE_POS_STD = 0.005  # metres; side-zone Gaussian noise
DEFAULT_OBSTACLE_SHIFT_X = 0.08  # metres; offset for adversarial side zones
ADVERSARIAL_CENTER_PROB = 0.2  # probability of center zone in adversarial mode
# ── Default cube position std ────────────────────────────────────────
DEFAULT_CUBE_POS_STD = 0.006  # metres; 0 = no randomization


# ── Multi-cube constants ─────────────────────────────────────────────

CUBE_COLORS: tuple[str, ...] = ("red", "green", "blue")
CUBE_JOINT_NAMES: tuple[str, ...] = (
    "red_box_joint",
    "green_box_joint",
    "blue_box_joint",
)
NUM_CUBES = len(CUBE_COLORS)
CUBE_FREE_DIM = 7  # pos(3) + quat_wxyz(4) per cube
ALL_CUBES_DIM = NUM_CUBES * CUBE_FREE_DIM  # 21

# One-hot goal encoding dimension
GOAL_DIM = NUM_CUBES  # 3


def build_multicube_slot_templates(
    default_cube_qpos: np.ndarray, default_bin_pos: np.ndarray
) -> np.ndarray:
    """Build qpos templates for the 3 cube slots + 1 bin slot."""
    bin_slot_qpos = default_cube_qpos[0].copy()
    bin_slot_qpos[:2] = default_bin_pos[:2]
    return np.concatenate([default_cube_qpos, bin_slot_qpos[None, :]], axis=0)


def xy_boxes_overlap(
    center_a: np.ndarray, half_a: float, center_b: np.ndarray, half_b: float
) -> bool:
    dxy = np.abs(center_a - center_b)
    return bool((dxy[0] < (half_a + half_b)) and (dxy[1] < (half_a + half_b)))


def multicube_layout_has_overlap(cube_xy: np.ndarray, bin_xy: np.ndarray) -> bool:
    for i in range(NUM_CUBES):
        for j in range(i + 1, NUM_CUBES):
            if xy_boxes_overlap(
                cube_xy[i], CUBE_HALF_EXTENT_XY, cube_xy[j], CUBE_HALF_EXTENT_XY
            ):
                return True
        if xy_boxes_overlap(cube_xy[i], CUBE_HALF_EXTENT_XY, bin_xy, BIN_HALF_EXTENT_XY):
            return True
    return False


def sample_multicube_layout(
    rng: np.random.Generator,
    default_cube_qpos: np.ndarray,
    default_bin_pos: np.ndarray,
    cube_pos_std: float,
    shuffle_cubes: bool,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Sample a non-overlapping multicube+bin layout."""
    slot_xy = np.concatenate(
        [default_cube_qpos[:, :2], default_bin_pos[None, :2]],
        axis=0,
    )
    num_slots = NUM_CUBES + 1
    perm = rng.permutation(num_slots) if shuffle_cubes else np.arange(num_slots)
    cube_slot_ids = perm[:NUM_CUBES]
    bin_slot_id = int(perm[NUM_CUBES])

    cube_slot_xy = slot_xy[cube_slot_ids]
    bin_slot_xy = slot_xy[bin_slot_id]
    while True:
        if cube_pos_std > 0:
            cube_noise = rng.normal(0.0, cube_pos_std, size=(NUM_CUBES, 2))
            bin_noise = rng.normal(0.0, cube_pos_std, size=2)
        else:
            cube_noise = np.zeros((NUM_CUBES, 2), dtype=np.float64)
            bin_noise = np.zeros(2, dtype=np.float64)
        cube_xy = cube_slot_xy + cube_noise
        bin_xy = bin_slot_xy + bin_noise
        if not multicube_layout_has_overlap(cube_xy, bin_xy):
            return cube_slot_ids, bin_slot_id, cube_xy, bin_xy


@dataclass
class BaseSO100SimEnv:
    """Common MuJoCo SO-100 simulation plumbing shared by all scene variants."""

    xml_path: Path
    control_hz: float = 10.0
    render_w: int = 640
    render_h: int = 480
    keyframe: str = "student_start"
    use_mocap: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(self.seed)

        if not self.use_mocap:
            self._disable_mocap_weld()

        self.dt_ctrl = 1.0 / self.control_hz
        self.sim_dt = float(self.model.opt.timestep)
        self.substeps = max(1, int(round(self.dt_ctrl / self.sim_dt)))

        self.qpos_idx = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )

        self.act_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.act_ids == -1):
            missing = [n for n, i in zip(JOINT_NAMES, self.act_ids) if i == -1]
            raise ValueError(f"Missing actuators: {missing}")

        self._jaw_idx = JOINT_NAMES.index("Jaw")

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        if self.ee_site_id == -1:
            raise ValueError("Site 'ee_site' not found in model.")
        self.bin_center_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, BIN_CENTER_SITE_NAME
        )
        if self.bin_center_site_id == -1:
            raise ValueError(f"Site '{BIN_CENTER_SITE_NAME}' not found in model.")

        self.mocap_id = 0

        self._init_scene_specific()

        self.renderer = mujoco.Renderer(
            self.model, height=self.render_h, width=self.render_w
        )

        self.reset()

    def _init_scene_specific(self) -> None:
        raise NotImplementedError

    def _apply_scene_reset_randomization(self) -> None:
        raise NotImplementedError

    def _disable_mocap_weld(self) -> None:
        """Disable weld constraints so position actuators drive the arm directly."""
        for i in range(self.model.neq):
            if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                self.model.eq_active0[i] = 0

    def reset(self, keyframe: str | None = None) -> dict[str, np.ndarray]:
        """Reset simulation to keyframe and return the initial observation."""
        key_name = keyframe or self.keyframe
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id == -1:
            raise ValueError(f"Keyframe '{key_name}' not found in XML.")

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        self._apply_scene_reset_randomization()
        mujoco.mj_forward(self.model, self.data)

        q = self.get_joint_angles()
        self.set_targets(q)

        self.data.mocap_pos[self.mocap_id] = self.get_ee_pos()
        self.data.mocap_quat[self.mocap_id] = self.get_ee_quat()

        return self.get_obs()

    # ── state queries ─────────────────────────────────────────────────

    def get_joint_angles(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def get_ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_ee_quat(self) -> np.ndarray:
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site_id])
        return quat

    def get_ee_state(self) -> np.ndarray:
        return np.concatenate([self.get_ee_pos(), self.get_ee_quat()])

    def get_cube_state(self) -> np.ndarray:
        raise NotImplementedError

    def get_obstacle_pos(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)

    def get_gripper_angle(self) -> float:
        return float(self.data.qpos[self.qpos_idx[self._jaw_idx]])

    def get_goal_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.bin_center_site_id].copy()

    def get_obs(self) -> dict[str, np.ndarray]:
        return {
            "joints": self.get_joint_angles(),
            "ee": self.get_ee_state(),
            "ee_pos": self.get_ee_pos(),
            "gripper": np.array([self.get_gripper_angle()], dtype=np.float64),
            "cube": self.get_cube_state(),
            "obstacle": self.get_obstacle_pos(),
            "goal_pos": self.get_goal_pos(),
        }

    # ── control ───────────────────────────────────────────────────────

    def set_targets(self, joint_targets: np.ndarray) -> None:
        joint_targets = np.asarray(joint_targets, dtype=np.float64)
        for i, act_id in enumerate(self.act_ids):
            if i == self._jaw_idx:
                continue
            self.data.ctrl[act_id] = joint_targets[i]
        self._clip_ctrl()

    def set_gripper(self, angle: float) -> None:
        self.data.ctrl[self.act_ids[self._jaw_idx]] = angle
        self._clip_ctrl()

    def set_mocap_pos(self, pos: np.ndarray) -> None:
        self.data.mocap_pos[self.mocap_id] = np.asarray(pos, dtype=np.float64)

    def set_mocap_quat(self, quat_wxyz: np.ndarray) -> None:
        self.data.mocap_quat[self.mocap_id] = np.asarray(quat_wxyz, dtype=np.float64)

    def set_mocap_pose(self, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        self.set_mocap_pos(pos)
        self.set_mocap_quat(quat_wxyz)

    def _clip_ctrl(self) -> None:
        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(self.data.ctrl, lo, hi)

    # ── simulation step ───────────────────────────────────────────────

    def step(self) -> dict[str, np.ndarray]:
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        return self.get_obs()

    # ── rendering ─────────────────────────────────────────────────────

    def render(self, camera_name: str = "angle") -> np.ndarray:
        import cv2

        self.renderer.update_scene(self.data, camera=camera_name)
        rgb = self.renderer.render()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def render_rgb(self, camera_name: str = "angle") -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render().copy()


@dataclass
class SO100SimEnv(BaseSO100SimEnv):
    """Single-cube scene with optional obstacle randomization."""

    cube_pos_std: float = DEFAULT_CUBE_POS_STD
    obstacle_pos_std: float = DEFAULT_OBSTACLE_POS_STD
    adversarial_obstacle_pos_std: float = DEFAULT_ADVERSARIAL_OBSTACLE_POS_STD
    obstacle_mode: str = "train"
    obstacle_shift_x: float = DEFAULT_OBSTACLE_SHIFT_X

    def _init_scene_specific(self) -> None:
        cube_jnt_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, CUBE_JOINT_NAME
        )
        if cube_jnt_id == -1:
            raise ValueError(f"Joint '{CUBE_JOINT_NAME}' not found in model.")
        cube_qpos_start = self.model.jnt_qposadr[cube_jnt_id]
        self.cube_qpos_idx = np.arange(cube_qpos_start, cube_qpos_start + CUBE_DIM)

        self.obstacle_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, OBSTACLE_BODY_NAME
        )
        if self.obstacle_body_id != -1:
            self._obstacle_default_pos = self.model.body_pos[self.obstacle_body_id].copy()
        else:
            self._obstacle_default_pos = None

        self.upper_obstacle_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, UPPER_OBSTACLE_BODY_NAME
        )
        if self.upper_obstacle_body_id != -1:
            self._upper_obstacle_default_pos = self.model.body_pos[
                self.upper_obstacle_body_id
            ].copy()
        else:
            self._upper_obstacle_default_pos = None

    def _apply_scene_reset_randomization(self) -> None:
        if self.cube_pos_std > 0:
            dx = self.rng.normal(0.0, self.cube_pos_std)
            dy = self.rng.normal(0.0, self.cube_pos_std)
            self.data.qpos[self.cube_qpos_idx[0]] += dx
            self.data.qpos[self.cube_qpos_idx[1]] += dy

        if self.obstacle_body_id == -1:
            return

        self.model.body_pos[self.obstacle_body_id] = self._obstacle_default_pos.copy()
        if self.upper_obstacle_body_id != -1:
            self.model.body_pos[self.upper_obstacle_body_id] = (
                self._upper_obstacle_default_pos.copy()
            )

        if self.obstacle_mode == "adversarial":
            r = self.rng.random()
            if r < ADVERSARIAL_CENTER_PROB:
                zone_offset = 0.0
                zone_std = self.obstacle_pos_std
            elif r < ADVERSARIAL_CENTER_PROB + (1 - ADVERSARIAL_CENTER_PROB) / 2:
                zone_offset = self.obstacle_shift_x
                zone_std = self.adversarial_obstacle_pos_std
            else:
                zone_offset = -self.obstacle_shift_x
                zone_std = self.adversarial_obstacle_pos_std

            self.model.body_pos[self.obstacle_body_id][0] += zone_offset
            if self.upper_obstacle_body_id != -1:
                self.model.body_pos[self.upper_obstacle_body_id][0] += zone_offset

            if zone_std > 0:
                dx = self.rng.normal(0.0, zone_std)
                self.model.body_pos[self.obstacle_body_id][0] += dx
                if self.upper_obstacle_body_id != -1:
                    self.model.body_pos[self.upper_obstacle_body_id][0] += dx
        elif self.obstacle_pos_std > 0:
            dx = self.rng.normal(0.0, self.obstacle_pos_std)
            self.model.body_pos[self.obstacle_body_id][0] += dx
            if self.upper_obstacle_body_id != -1:
                self.model.body_pos[self.upper_obstacle_body_id][0] += dx

    def get_cube_state(self) -> np.ndarray:
        return self.data.qpos[self.cube_qpos_idx].copy()

    def get_obstacle_pos(self) -> np.ndarray:
        if self.obstacle_body_id != -1:
            return self.data.xpos[self.obstacle_body_id].copy()
        return np.zeros(3, dtype=np.float64)


@dataclass
class SO100MulticubeSimEnv(BaseSO100SimEnv):
    """SO-100 scene with 3 cubes and goal conditioning."""

    goal_cube: str = "red"
    cube_pos_std: float = DEFAULT_CUBE_POS_STD
    shuffle_cubes: bool = True

    def _init_scene_specific(self) -> None:
        if self.goal_cube not in CUBE_COLORS:
            raise ValueError(
                f"goal_cube must be one of {CUBE_COLORS}, got {self.goal_cube!r}"
            )

        self.cube_qpos_slices: list[np.ndarray] = []
        for jname in CUBE_JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid == -1:
                raise ValueError(f"Joint '{jname}' not found in model.")
            start = self.model.jnt_qposadr[jid]
            self.cube_qpos_slices.append(np.arange(start, start + CUBE_FREE_DIM))

        self.bin_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, BIN_BODY_NAME
        )
        if self.bin_body_id == -1:
            raise ValueError(f"Body '{BIN_BODY_NAME}' not found in model.")
        self._default_bin_pos = self.model.body_pos[self.bin_body_id].copy()

        self._default_cube_qpos: np.ndarray | None = None
        self._cube_slot_qpos_templates: np.ndarray | None = None
        self._goal_index = CUBE_COLORS.index(self.goal_cube)
        self._goal_onehot = np.zeros(GOAL_DIM, dtype=np.float64)
        self._goal_onehot[self._goal_index] = 1.0

    def set_goal(self, cube_color: str) -> None:
        if cube_color not in CUBE_COLORS:
            raise ValueError(
                f"cube_color must be one of {CUBE_COLORS}, got {cube_color!r}"
            )
        self.goal_cube = cube_color
        self._goal_index = CUBE_COLORS.index(cube_color)
        self._goal_onehot = np.zeros(GOAL_DIM, dtype=np.float64)
        self._goal_onehot[self._goal_index] = 1.0

    def get_goal_onehot(self) -> np.ndarray:
        return self._goal_onehot.copy()

    def _randomize_layout(self) -> None:
        if self._default_cube_qpos is None:
            self._default_cube_qpos = np.array(
                [self.data.qpos[sl].copy() for sl in self.cube_qpos_slices]
            )
        if self._cube_slot_qpos_templates is None:
            self._cube_slot_qpos_templates = build_multicube_slot_templates(
                self._default_cube_qpos, self._default_bin_pos
            )

        cube_slot_ids, _, cube_xy, bin_xy = sample_multicube_layout(
            self.rng,
            self._default_cube_qpos,
            self._default_bin_pos,
            self.cube_pos_std,
            self.shuffle_cubes,
        )

        for cube_i, slot_i in enumerate(cube_slot_ids):
            qpos = self._cube_slot_qpos_templates[slot_i].copy()
            qpos[0] = cube_xy[cube_i, 0]
            qpos[1] = cube_xy[cube_i, 1]
            self.data.qpos[self.cube_qpos_slices[cube_i]] = qpos

        bin_pos = self._default_bin_pos.copy()
        bin_pos[0] = bin_xy[0]
        bin_pos[1] = bin_xy[1]
        self.model.body_pos[self.bin_body_id] = bin_pos

    def _apply_scene_reset_randomization(self) -> None:
        self._randomize_layout()

    def get_all_cubes_state(self) -> np.ndarray:
        parts = [self.data.qpos[sl].copy() for sl in self.cube_qpos_slices]
        return np.concatenate(parts)

    def get_all_cubes_xyz(self) -> np.ndarray:
        parts = [self.data.qpos[sl[:3]].copy() for sl in self.cube_qpos_slices]
        return np.concatenate(parts)

    def get_target_cube_state(self) -> np.ndarray:
        return self.data.qpos[self.cube_qpos_slices[self._goal_index]].copy()

    def get_cube_state(self) -> np.ndarray:
        return self.get_target_cube_state()

    def get_obstacle_pos(self) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)

    def get_obs(self) -> dict[str, np.ndarray]:
        obs = super().get_obs()
        obs.update(
            {
                "cubes": self.get_all_cubes_state(),
                "cubes_xyz": self.get_all_cubes_xyz(),
                "goal": self.get_goal_onehot(),
            }
        )
        return obs
