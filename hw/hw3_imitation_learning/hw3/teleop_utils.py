"""Shared utilities for teleop recording and DAgger data collection.

Provides:
- ``ZarrEpisodeWriter`` — incremental zarr writer for state/action data.
- ``rotate_quaternion`` — quaternion rotation helper.
- ``load_keymap`` — load a ``keymap.json`` into a ``{raw_keycode: action_name}`` dict.
- ``handle_teleop_key`` — apply a single teleop movement action to the sim.
- ``compose_camera_views`` — arrange rendered camera images in a 2‑row layout.
- Common constants (``JOINT_NAMES``, ``CAMERA_NAMES``, etc.)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mujoco
import numpy as np
import pyquaternion as pyq
import zarr

# ── constants ─────────────────────────────────────────────────────────

JOINT_NAMES: tuple[str, ...] = (
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
    "Jaw",
)
CAMERA_NAMES: tuple[str, ...] = ("left_wrist", "angle", "top")

DEFAULT_KEYMAP_PATH: Path = Path(__file__).resolve().parent / "keymap.json"

CUBE_JOINT_NAME: str = "red_box_joint"
CUBE_DIM: int = 7  # free joint: pos(3) + quat_wxyz(4)
OBSTACLE_DIM: int = 3  # obstacle xyz position


# ── quaternion rotation ───────────────────────────────────────────────


def rotate_quaternion(
    quat_wxyz: np.ndarray, axis_xyz, angle_deg: float
) -> np.ndarray:
    """Rotate *quat_wxyz* around *axis_xyz* by *angle_deg* degrees."""
    angle_rad = np.deg2rad(angle_deg)
    axis_xyz = np.asarray(axis_xyz, dtype=np.float64)
    axis_xyz = axis_xyz / np.linalg.norm(axis_xyz)
    q = pyq.Quaternion(quat_wxyz) * pyq.Quaternion(axis=axis_xyz, angle=angle_rad)
    q = q.normalised
    return q.elements  # wxyz


# ── keymap loading ────────────────────────────────────────────────────


def load_keymap(km_path: Path | None = None) -> dict[int, str]:
    """Load a ``keymap.json`` file and return ``{raw_keycode: action_name}``."""
    km_path = km_path or DEFAULT_KEYMAP_PATH
    if not km_path.exists():
        raise FileNotFoundError(
            f"Key mapping file not found: {km_path}\n"
            "Please run  python scripts/configure_keys.py  first."
        )
    with open(km_path) as f:
        km_data = json.load(f)
    return {int(entry["raw"]): action_name for action_name, entry in km_data.items()}


# ── teleop key dispatch ──────────────────────────────────────────────


def handle_teleop_key(
    action_name: str,
    data: mujoco.MjData,
    model: mujoco.MjModel,
    mocap_id: int,
    jaw_act_idx: int,
) -> None:
    """Apply a single teleop movement action to the MuJoCo simulation.

    Parameters
    ----------
    action_name : str
        Action identifier (``"move_up"``, ``"rot_x_pos"``, ``"gripper_open"``, …).
    data : mujoco.MjData
        Active simulation data (modified in-place).
    model : mujoco.MjModel
        MuJoCo model (used for ctrl range clamping).
    mocap_id : int
        Index into ``data.mocap_pos`` / ``data.mocap_quat``.
    jaw_act_idx : int
        Actuator index for the jaw/gripper.
    """
    if action_name == "move_up":
        data.mocap_pos[mocap_id, 2] += 0.01
    elif action_name == "move_down":
        data.mocap_pos[mocap_id, 2] -= 0.01
    elif action_name == "move_left":
        data.mocap_pos[mocap_id, 0] -= 0.01
    elif action_name == "move_right":
        data.mocap_pos[mocap_id, 0] += 0.01
    elif action_name == "move_forward":
        data.mocap_pos[mocap_id, 1] += 0.01
    elif action_name == "move_backward":
        data.mocap_pos[mocap_id, 1] -= 0.01
    elif action_name == "rot_x_pos":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [1, 0, 0], 10
        )
    elif action_name == "rot_x_neg":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [1, 0, 0], -10
        )
    elif action_name == "rot_y_pos":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [0, 1, 0], 10
        )
    elif action_name == "rot_y_neg":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [0, 1, 0], -10
        )
    elif action_name == "rot_z_pos":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [0, 0, 1], 10
        )
    elif action_name == "rot_z_neg":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [0, 0, 1], -10
        )
    elif action_name == "gripper_open":
        data.ctrl[jaw_act_idx] += 0.10
        lo = model.actuator_ctrlrange[:, 0]
        hi = model.actuator_ctrlrange[:, 1]
        data.ctrl[:] = np.clip(data.ctrl, lo, hi)
    elif action_name == "gripper_close":
        data.ctrl[jaw_act_idx] -= 0.10
        lo = model.actuator_ctrlrange[:, 0]
        hi = model.actuator_ctrlrange[:, 1]
        data.ctrl[:] = np.clip(data.ctrl, lo, hi)


# ── camera view composition ──────────────────────────────────────────


def compose_camera_views(
    images: dict[str, np.ndarray],
    camera_names: tuple[str, ...] = CAMERA_NAMES,
) -> np.ndarray:
    """Arrange rendered camera images in a 2‑row layout.

    Top row = first two cameras side-by-side, bottom = third camera (padded).
    Each image is labelled with the camera name.

    Parameters
    ----------
    images : dict[str, np.ndarray]
        Mapping from camera name → BGR uint8 image.
    camera_names : tuple[str, ...]
        Order of cameras (first two go on top, rest on bottom row).
    """
    views = []
    for cam in camera_names:
        img = images[cam].copy()
        cv2.putText(
            img, cam, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        views.append(img)

    top_row = np.concatenate(views[:2], axis=1)
    bottom = views[2]
    pad_w = top_row.shape[1] - bottom.shape[1]
    if pad_w > 0:
        padding = np.zeros((bottom.shape[0], pad_w, 3), dtype=bottom.dtype)
        bottom_row = np.concatenate([bottom, padding], axis=1)
    else:
        bottom_row = bottom
    return np.concatenate([top_row, bottom_row], axis=0)


# ── ZarrEpisodeWriter ────────────────────────────────────────────────


@dataclass
class ZarrEpisodeWriter:
    """Incremental zarr writer for teleop / DAgger state/action data.

    Buffers incoming timesteps and flushes to disk every *flush_every* steps.
    Supports ``end_episode()`` to finalise an episode and ``discard_episode()``
    to roll back all data since the last completed episode.
    """

    path: Path
    joint_dim: int = 6
    ee_dim: int = 7
    cube_dim: int = 7
    gripper_dim: int = 1
    obstacle_dim: int = 3
    flush_every: int = 12

    # ── internal state (populated by __post_init__) ───────────────────
    root: zarr.Group = field(init=False, repr=False)
    state_joints_arr: zarr.Array = field(init=False, repr=False)
    state_ee_arr: zarr.Array = field(init=False, repr=False)
    state_cube_arr: zarr.Array | None = field(init=False, repr=False, default=None)
    state_gripper_arr: zarr.Array = field(init=False, repr=False)
    action_gripper_arr: zarr.Array = field(init=False, repr=False)
    state_obstacle_arr: zarr.Array = field(init=False, repr=False)
    ep_ends_arr: zarr.Array = field(init=False, repr=False)

    _state_joints_buf: list[np.ndarray] = field(init=False, repr=False)
    _state_ee_buf: list[np.ndarray] = field(init=False, repr=False)
    _state_cube_buf: list[np.ndarray] = field(init=False, repr=False)
    _state_gripper_buf: list[np.ndarray] = field(init=False, repr=False)
    _action_gripper_buf: list[np.ndarray] = field(init=False, repr=False)
    _state_obstacle_buf: list[np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.root = zarr.open_group(str(self.path), mode="w", zarr_format=3)

        compressor = zarr.codecs.Blosc(cname="zstd", clevel=3, shuffle=2)
        compressors = (compressor,)

        data = self.root.require_group("data")
        meta = self.root.require_group("meta")

        self.state_joints_arr = data.require_array(
            "state_joints",
            shape=(0, self.joint_dim),
            chunks=(min(self.flush_every, 4096), self.joint_dim),
            dtype="f4",
            compressors=compressors,
        )
        self.state_ee_arr = data.require_array(
            "state_ee",
            shape=(0, self.ee_dim),
            chunks=(min(self.flush_every, 4096), self.ee_dim),
            dtype="f4",
            compressors=compressors,
        )
        if self.cube_dim > 0:
            self.state_cube_arr = data.require_array(
                "state_cube",
                shape=(0, self.cube_dim),
                chunks=(min(self.flush_every, 4096), self.cube_dim),
                dtype="f4",
                compressors=compressors,
            )
        self.state_gripper_arr = data.require_array(
            "state_gripper",
            shape=(0, self.gripper_dim),
            chunks=(min(self.flush_every, 4096), self.gripper_dim),
            dtype="f4",
            compressors=compressors,
        )
        self.action_gripper_arr = data.require_array(
            "action_gripper",
            shape=(0, self.gripper_dim),
            chunks=(min(self.flush_every, 4096), self.gripper_dim),
            dtype="f4",
            compressors=compressors,
        )
        self.state_obstacle_arr = data.require_array(
            "state_obstacle",
            shape=(0, self.obstacle_dim),
            chunks=(min(self.flush_every, 4096), self.obstacle_dim),
            dtype="f4",
            compressors=compressors,
        )
        self.ep_ends_arr = meta.require_array(
            "episode_ends",
            shape=(0,),
            chunks=(1024,),
            dtype="i8",
            compressors=compressors,
        )

        self._state_joints_buf = []
        self._state_ee_buf = []
        self._state_cube_buf = []
        self._state_gripper_buf = []
        self._action_gripper_buf = []
        self._state_obstacle_buf = []

    # ── convenience attributes ────────────────────────────────────────

    def set_attrs(self, **attrs) -> None:
        """Store arbitrary metadata on the zarr root group."""
        for k, v in attrs.items():
            self.root.attrs[k] = v

    @property
    def num_steps_total(self) -> int:
        """Total timesteps written (including unflushed buffer)."""
        return int(self.state_joints_arr.shape[0]) + len(self._state_joints_buf)

    @property
    def num_episodes(self) -> int:
        return int(self.ep_ends_arr.shape[0])

    # ── append / flush / episode management ───────────────────────────

    def append(
        self,
        state_joints: np.ndarray,
        state_ee: np.ndarray,
        state_cube: np.ndarray,
        state_gripper: np.ndarray,
        action_gripper: np.ndarray,
        state_obstacle: np.ndarray,
    ) -> None:
        """Buffer one timestep of data.  Flushes automatically every *flush_every* steps."""
        self._state_joints_buf.append(state_joints.astype(np.float32, copy=False))
        self._state_ee_buf.append(state_ee.astype(np.float32, copy=False))
        if self.cube_dim > 0:
            self._state_cube_buf.append(state_cube.astype(np.float32, copy=False))
        self._state_gripper_buf.append(state_gripper.astype(np.float32, copy=False))
        self._action_gripper_buf.append(action_gripper.astype(np.float32, copy=False))
        self._state_obstacle_buf.append(state_obstacle.astype(np.float32, copy=False))

        if len(self._state_joints_buf) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write buffered data to disk."""
        if not self._state_joints_buf:
            return

        joints = np.stack(self._state_joints_buf, axis=0)
        ee = np.stack(self._state_ee_buf, axis=0)
        gripper = np.stack(self._state_gripper_buf, axis=0)
        action_grip = np.stack(self._action_gripper_buf, axis=0)
        obstacle = np.stack(self._state_obstacle_buf, axis=0)

        n0 = self.state_joints_arr.shape[0]
        n1 = n0 + joints.shape[0]

        self.state_joints_arr.resize((n1, self.joint_dim))
        self.state_ee_arr.resize((n1, self.ee_dim))
        if self.state_cube_arr is not None:
            self.state_cube_arr.resize((n1, self.cube_dim))
        self.state_gripper_arr.resize((n1, self.gripper_dim))
        self.action_gripper_arr.resize((n1, self.gripper_dim))
        self.state_obstacle_arr.resize((n1, self.obstacle_dim))
        self.state_joints_arr[n0:n1] = joints
        self.state_ee_arr[n0:n1] = ee
        if self.state_cube_arr is not None:
            cube = np.stack(self._state_cube_buf, axis=0)
            self.state_cube_arr[n0:n1] = cube
        self.state_gripper_arr[n0:n1] = gripper
        self.action_gripper_arr[n0:n1] = action_grip
        self.state_obstacle_arr[n0:n1] = obstacle

        self._state_joints_buf.clear()
        self._state_ee_buf.clear()
        self._state_cube_buf.clear()
        self._state_gripper_buf.clear()
        self._action_gripper_buf.clear()
        self._state_obstacle_buf.clear()

    def end_episode(self) -> None:
        """Flush and record an episode boundary."""
        self.flush()
        end_idx = int(self.state_joints_arr.shape[0])
        m0 = self.ep_ends_arr.shape[0]
        self.ep_ends_arr.resize((m0 + 1,))
        self.ep_ends_arr[m0] = end_idx

    def discard_episode(self) -> None:
        """Roll back all data recorded since the last ``end_episode()`` call."""
        self._state_joints_buf.clear()
        self._state_ee_buf.clear()
        self._state_cube_buf.clear()
        self._state_gripper_buf.clear()
        self._action_gripper_buf.clear()
        self._state_obstacle_buf.clear()

        if self.ep_ends_arr.shape[0] > 0:
            rollback_to = int(self.ep_ends_arr[-1])
        else:
            rollback_to = 0

        if self.state_joints_arr.shape[0] > rollback_to:
            self.state_joints_arr.resize((rollback_to, self.joint_dim))
            self.state_ee_arr.resize((rollback_to, self.ee_dim))
            if self.state_cube_arr is not None:
                self.state_cube_arr.resize((rollback_to, self.cube_dim))
            self.state_gripper_arr.resize((rollback_to, self.gripper_dim))
            self.action_gripper_arr.resize((rollback_to, self.gripper_dim))
            self.state_obstacle_arr.resize((rollback_to, self.obstacle_dim))
