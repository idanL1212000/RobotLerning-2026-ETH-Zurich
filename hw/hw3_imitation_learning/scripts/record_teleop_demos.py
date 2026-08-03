from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import mujoco
import numpy as np
from hw.hw3_imitation_learning.hw3.sim_env import (
    BIN_BODY_NAME,
    CUBE_COLORS,
    DEFAULT_CUBE_POS_STD,
    DEFAULT_OBSTACLE_POS_STD,
    GOAL_DIM,
    NUM_CUBES,
    OBSTACLE_BODY_NAME,
    build_multicube_slot_templates,
    sample_multicube_layout,
)
from hw.hw3_imitation_learning.hw3.teleop_utils import (
    CAMERA_NAMES,
    CUBE_DIM,
    CUBE_JOINT_NAME,
    JOINT_NAMES,
    OBSTACLE_DIM,
    ZarrEpisodeWriter,
    compose_camera_views,
    handle_teleop_key,
    load_keymap,
)
from hw.hw3_imitation_learning.so101_gym.constants import ASSETS_DIR

MOCAP_INDEX = 0
CUBE_JOINT_NAMES: tuple[str, ...] = (
    "red_box_joint",
    "green_box_joint",
    "blue_box_joint",
)
CUBE_FREE_DIM = 7
ALL_CUBES_DIM = NUM_CUBES * CUBE_FREE_DIM


class BaseCv2TeleopRecorder:
    def __init__(
        self,
        xml_path: Path,
        out_zarr: Path,
        control_hz: float,
        render_w: int,
        render_h: int,
        window_name: str,
        keymap_path: Path | None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        if self.model.nmocap != 1:
            raise ValueError(
                f"Expected exactly 1 mocap body, got nmocap={self.model.nmocap}."
            )

        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site"
        )
        if self.ee_site_id == -1:
            raise ValueError("Site 'ee_site' not found in model.")

        for cam in CAMERA_NAMES:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
            if cam_id == -1:
                raise ValueError(f"Camera '{cam}' not found in loaded XML.")

        self.qpos_idx = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                ]
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )

        self.act_id = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in JOINT_NAMES
        }
        if any(v == -1 for v in self.act_id.values()):
            missing = [k for k, v in self.act_id.items() if v == -1]
            raise ValueError(f"Missing actuators: {missing}")

        out_zarr.parent.mkdir(parents=True, exist_ok=True)
        self.writer = self._build_writer(xml_path, out_zarr, control_hz)

        self.renderer = mujoco.Renderer(self.model, height=render_h, width=render_w)
        self.window_name = window_name

        self.control_hz = float(control_hz)
        self.dt_ctrl = 1.0 / self.control_hz
        self.sim_dt = float(self.model.opt.timestep)
        self.substeps = max(1, int(round(self.dt_ctrl / self.sim_dt)))

        self.episodes_done = 0
        self.recording = False
        self.running = True

        self._key_to_action = load_keymap(keymap_path)
        print(f"Loaded key mapping from {keymap_path or 'default'}")

    def _build_writer(
        self,
        xml_path: Path,
        out_zarr: Path,
        control_hz: float,
    ) -> ZarrEpisodeWriter:
        raise NotImplementedError

    def _reset_episode(self) -> None:
        raise NotImplementedError

    def _handle_key(self, k_raw: int, k_ascii: int) -> None:
        raise NotImplementedError

    def _record_step(self) -> None:
        raise NotImplementedError

    def _overlay_status(self, img_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _reset_to_keyframe(self, key_name: str = "student_start") -> None:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id == -1:
            raise ValueError(f"Keyframe '{key_name}' not found in XML.")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    def _get_q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def _clip_ctrl(self) -> None:
        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(self.data.ctrl, lo, hi)

    def _init_pose_and_targets(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        self.data.mocap_pos[MOCAP_INDEX] = self.data.site_xpos[self.ee_site_id].copy()

        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site_id])
        self.data.mocap_quat[MOCAP_INDEX] = quat

        q = self._get_q()
        for i, name in enumerate(JOINT_NAMES):
            self.data.ctrl[self.act_id[name]] = q[i]
        self._clip_ctrl()

        mujoco.mj_forward(self.model, self.data)

    def _get_ee_state(self) -> np.ndarray:
        pos = self.data.site_xpos[self.ee_site_id].copy()
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site_id])
        return np.concatenate([pos, quat])

    def _render_bgr(self, camera_name: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_name)
        return cv2.cvtColor(self.renderer.render(), cv2.COLOR_RGB2BGR)

    def _compose_views(self) -> np.ndarray:
        images = {cam: self._render_bgr(cam) for cam in CAMERA_NAMES}
        return compose_camera_views(images, CAMERA_NAMES)

    def _label_for(self, action: str) -> str:
        for code, act in self._key_to_action.items():
            if act == action:
                if 32 <= (code & 0xFF) <= 126:
                    ch = chr(code & 0xFF)
                    return ch if ch.strip() else "SPACE"
                if code & 0xFF == 27:
                    return "ESC"
                if code & 0xFF in (13, 10):
                    return "ENTER"
                return f"key:{code}"
        return "?"

    def _finalize_on_exit(self) -> None:
        if self.recording:
            self.writer.end_episode()
            self.episodes_done += 1
            print(f"Episode {self.episodes_done} saved on exit.")
            self.recording = False

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        last = time.perf_counter()
        try:
            while self.running:
                k_raw = cv2.waitKeyEx(1)
                if k_raw != -1:
                    k_ascii = k_raw & 0xFF
                    self._handle_key(k_raw, k_ascii)

                now = time.perf_counter()
                dt = now - last
                if dt < self.dt_ctrl:
                    time.sleep(self.dt_ctrl - dt)
                last = time.perf_counter()

                if self.recording:
                    self._record_step()

                for _ in range(self.substeps):
                    mujoco.mj_step(self.model, self.data)

                img = self._overlay_status(self._compose_views())
                cv2.imshow(self.window_name, img)
        finally:
            self._finalize_on_exit()
            self.writer.flush()
            cv2.destroyAllWindows()
            print(f"Flushed buffers. {self.episodes_done} episode(s) saved. Done.")


class SO100Cv2TeleopRecorder(BaseCv2TeleopRecorder):
    def __init__(
        self,
        xml_path: Path,
        out_zarr: Path,
        control_hz: float = 50.0,
        render_w: int = 640,
        render_h: int = 480,
        window_name: str = "SO100 Teleop",
        keymap_path: Path | None = None,
        cube_pos_std: float = DEFAULT_CUBE_POS_STD,
        obstacle_pos_std: float = DEFAULT_OBSTACLE_POS_STD,
    ) -> None:
        self.cube_pos_std = cube_pos_std
        self.obstacle_pos_std = obstacle_pos_std
        super().__init__(
            xml_path=xml_path,
            out_zarr=out_zarr,
            control_hz=control_hz,
            render_w=render_w,
            render_h=render_h,
            window_name=window_name,
            keymap_path=keymap_path,
        )

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
        self._obstacle_default_pos = (
            self.model.body_pos[self.obstacle_body_id].copy()
            if self.obstacle_body_id != -1
            else None
        )

        self._reset_episode()

    def _build_writer(
        self,
        xml_path: Path,
        out_zarr: Path,
        control_hz: float,
    ) -> ZarrEpisodeWriter:
        writer = ZarrEpisodeWriter(
            out_zarr,
            joint_dim=len(JOINT_NAMES),
            ee_dim=7,
            cube_dim=CUBE_DIM,
            gripper_dim=1,
            obstacle_dim=OBSTACLE_DIM,
            flush_every=12,
        )
        writer.set_attrs(
            xml=str(xml_path),
            joint_names=list(JOINT_NAMES),
            state_joints_spec="qpos(joints)",
            state_ee_spec="ee_pos(3) + ee_quat_wxyz(4)",
            state_cube_spec="cube_pos(3) + cube_quat_wxyz(4)",
            state_gripper_spec="gripper_angle(1)",
            action_gripper_spec="gripper_ctrl(1)",
            control_hz=float(control_hz),
            cameras_display=list(CAMERA_NAMES),
        )
        return writer

    def _get_cube_state(self) -> np.ndarray:
        return self.data.qpos[self.cube_qpos_idx].copy()

    def _get_obstacle_pos(self) -> np.ndarray:
        if self.obstacle_body_id != -1:
            return self.data.xpos[self.obstacle_body_id].copy().astype(np.float32)
        return np.zeros(OBSTACLE_DIM, dtype=np.float32)

    def _reset_episode(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._reset_to_keyframe("student_start")

        if self.cube_pos_std > 0:
            self.data.qpos[self.cube_qpos_idx[0]] += np.random.normal(
                0.0, self.cube_pos_std
            )
            self.data.qpos[self.cube_qpos_idx[1]] += np.random.normal(
                0.0, self.cube_pos_std
            )

        if self.obstacle_body_id != -1:
            self.model.body_pos[self.obstacle_body_id] = (
                self._obstacle_default_pos.copy()
            )
            if self.obstacle_pos_std > 0:
                dx = np.random.normal(0.0, self.obstacle_pos_std)
                self.model.body_pos[self.obstacle_body_id][0] += dx

        mujoco.mj_forward(self.model, self.data)
        self._init_pose_and_targets()

    def _handle_key(self, k_raw: int, _k_ascii: int) -> None:
        action = self._key_to_action.get(k_raw)

        if action == "escape":
            if self.recording:
                self.writer.end_episode()
                self.episodes_done += 1
                print(f"Episode {self.episodes_done} saved on exit.")
                self.recording = False
            self.running = False
            return

        if action == "record":
            self.recording = not self.recording
            print("RECORDING ON" if self.recording else "RECORDING OFF")
            return

        if action == "end_episode":
            if self.recording:
                self.writer.end_episode()
                self.episodes_done += 1
                print(f"Episode {self.episodes_done} saved.")
                self.recording = False
            self._reset_episode()
            return

        if action == "reset":
            if self.recording:
                self.writer.discard_episode()
                self.recording = False
                print(
                    "Episode DISCARDED. Press your record key to start a new recording."
                )
            self._reset_episode()
            return

        if action is None:
            return

        handle_teleop_key(
            action, self.data, self.model, MOCAP_INDEX, self.act_id["Jaw"]
        )

    def _record_step(self) -> None:
        state_joints = self._get_q()
        state_ee = self._get_ee_state()
        state_cube = self._get_cube_state()
        state_obstacle = self._get_obstacle_pos()
        state_gripper = np.array(
            [state_joints[JOINT_NAMES.index("Jaw")]], dtype=np.float32
        )
        action_gripper = np.array(
            [self.data.ctrl[self.act_id["Jaw"]]], dtype=np.float32
        )
        self.writer.append(
            state_joints,
            state_ee,
            state_cube,
            state_gripper,
            action_gripper,
            state_obstacle,
        )

    def _overlay_status(self, img_bgr: np.ndarray) -> np.ndarray:
        img = img_bgr.copy()
        status = f"{'REC' if self.recording else 'IDLE'} | ep {self.episodes_done} | substeps {self.substeps}"
        cv2.putText(
            img,
            status,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        hint = (
            f"{self._label_for('record')} rec | "
            f"{self._label_for('end_episode')} end ep | "
            f"{self._label_for('reset')} reset | "
            f"{self._label_for('escape')} quit"
        )
        cv2.putText(
            img,
            hint,
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return img


class MulticubeZarrWriter(ZarrEpisodeWriter):
    def __post_init__(self) -> None:
        super().__post_init__()
        import zarr

        compressor = zarr.codecs.Blosc(cname="zstd", clevel=3, shuffle=2)
        compressors = (compressor,)
        data = self.root.require_group("data")
        self.state_goal_arr = data.require_array(
            "state_goal",
            shape=(0, GOAL_DIM),
            chunks=(min(self.flush_every, 4096), GOAL_DIM),
            dtype="f4",
            compressors=compressors,
        )
        self.goal_pos_arr = data.require_array(
            "goal_pos",
            shape=(0, 3),
            chunks=(min(self.flush_every, 4096), 3),
            dtype="f4",
            compressors=compressors,
        )
        self.pos_cube_red_arr = data.require_array(
            "pos_cube_red",
            shape=(0, 7),
            chunks=(min(self.flush_every, 4096), 7),
            dtype="f4",
            compressors=compressors,
        )
        self.pos_cube_green_arr = data.require_array(
            "pos_cube_green",
            shape=(0, 7),
            chunks=(min(self.flush_every, 4096), 7),
            dtype="f4",
            compressors=compressors,
        )
        self.pos_cube_blue_arr = data.require_array(
            "pos_cube_blue",
            shape=(0, 7),
            chunks=(min(self.flush_every, 4096), 7),
            dtype="f4",
            compressors=compressors,
        )
        self._state_goal_buf: list[np.ndarray] = []
        self._goal_pos_buf: list[np.ndarray] = []
        self._pos_cube_red_buf: list[np.ndarray] = []
        self._pos_cube_green_buf: list[np.ndarray] = []
        self._pos_cube_blue_buf: list[np.ndarray] = []

    def append_with_goal(
        self,
        state_joints: np.ndarray,
        state_ee: np.ndarray,
        state_cube: np.ndarray,
        state_gripper: np.ndarray,
        action_gripper: np.ndarray,
        state_obstacle: np.ndarray,
        state_goal: np.ndarray,
        goal_pos: np.ndarray,
    ) -> None:
        self._state_goal_buf.append(state_goal.astype(np.float32, copy=False))
        self._goal_pos_buf.append(goal_pos.astype(np.float32, copy=False))
        # state_cube layout: [red(7), green(7), blue(7)].
        self._pos_cube_red_buf.append(state_cube[:7].astype(np.float32, copy=False))
        self._pos_cube_green_buf.append(
            state_cube[7:14].astype(np.float32, copy=False)
        )
        self._pos_cube_blue_buf.append(
            state_cube[14:21].astype(np.float32, copy=False)
        )
        self.append(
            state_joints,
            state_ee,
            state_cube,
            state_gripper,
            action_gripper,
            state_obstacle,
        )

    def flush(self) -> None:
        if not self._state_joints_buf:
            return

        n0 = self.state_goal_arr.shape[0]
        n_new = len(self._state_goal_buf)
        if n_new > 0:
            goal_data = np.stack(self._state_goal_buf, axis=0)
            n1 = n0 + n_new
            self.state_goal_arr.resize((n1, GOAL_DIM))
            self.state_goal_arr[n0:n1] = goal_data
            self._state_goal_buf.clear()

        n0_goal_pos = self.goal_pos_arr.shape[0]
        n_new_goal_pos = len(self._goal_pos_buf)
        if n_new_goal_pos > 0:
            goal_pos_data = np.stack(self._goal_pos_buf, axis=0)
            n1_goal_pos = n0_goal_pos + n_new_goal_pos
            self.goal_pos_arr.resize((n1_goal_pos, 3))
            self.goal_pos_arr[n0_goal_pos:n1_goal_pos] = goal_pos_data
            self._goal_pos_buf.clear()

        n0_pos = self.pos_cube_red_arr.shape[0]
        n_new_pos = len(self._pos_cube_red_buf)
        if n_new_pos > 0:
            red = np.stack(self._pos_cube_red_buf, axis=0)
            green = np.stack(self._pos_cube_green_buf, axis=0)
            blue = np.stack(self._pos_cube_blue_buf, axis=0)
            n1_pos = n0_pos + n_new_pos
            self.pos_cube_red_arr.resize((n1_pos, 7))
            self.pos_cube_green_arr.resize((n1_pos, 7))
            self.pos_cube_blue_arr.resize((n1_pos, 7))
            self.pos_cube_red_arr[n0_pos:n1_pos] = red
            self.pos_cube_green_arr[n0_pos:n1_pos] = green
            self.pos_cube_blue_arr[n0_pos:n1_pos] = blue
            self._pos_cube_red_buf.clear()
            self._pos_cube_green_buf.clear()
            self._pos_cube_blue_buf.clear()

        super().flush()

    def discard_episode(self) -> None:
        self._state_goal_buf.clear()
        self._goal_pos_buf.clear()
        self._pos_cube_red_buf.clear()
        self._pos_cube_green_buf.clear()
        self._pos_cube_blue_buf.clear()
        rollback_to = int(self.ep_ends_arr[-1]) if self.ep_ends_arr.shape[0] > 0 else 0
        if self.state_goal_arr.shape[0] > rollback_to:
            self.state_goal_arr.resize((rollback_to, GOAL_DIM))
        if self.goal_pos_arr.shape[0] > rollback_to:
            self.goal_pos_arr.resize((rollback_to, 3))
        if self.pos_cube_red_arr.shape[0] > rollback_to:
            self.pos_cube_red_arr.resize((rollback_to, 7))
            self.pos_cube_green_arr.resize((rollback_to, 7))
            self.pos_cube_blue_arr.resize((rollback_to, 7))
        super().discard_episode()


class MulticubeTeleopRecorder(BaseCv2TeleopRecorder):
    def __init__(
        self,
        xml_path: Path,
        out_zarr: Path,
        control_hz: float = 10.0,
        render_w: int = 640,
        render_h: int = 480,
        window_name: str = "SO100 Multicube Teleop",
        keymap_path: Path | None = None,
        seed: int | None = None,
        cube_pos_std: float = DEFAULT_CUBE_POS_STD,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.cube_pos_std = cube_pos_std

        super().__init__(
            xml_path=xml_path,
            out_zarr=out_zarr,
            control_hz=control_hz,
            render_w=render_w,
            render_h=render_h,
            window_name=window_name,
            keymap_path=keymap_path,
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
        self.bin_center_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "bin_center"
        )
        if self.bin_center_site_id == -1:
            raise ValueError("Site 'bin_center' not found in model.")

        self._default_bin_pos = self.model.body_pos[self.bin_body_id].copy()
        self._default_cube_qpos: np.ndarray | None = None
        self._cube_slot_qpos_templates: np.ndarray | None = None
        self._goal_index = 0
        self._goal_onehot = np.zeros(GOAL_DIM, dtype=np.float32)
        self._goal_onehot[0] = 1.0

        print(
            f"  Current goal cube: {CUBE_COLORS[self._goal_index]} "
            "(change with goal_cube_* keys before recording)"
        )

        self._reset_episode()

    def _build_writer(
        self,
        xml_path: Path,
        out_zarr: Path,
        control_hz: float,
    ) -> ZarrEpisodeWriter:
        writer = MulticubeZarrWriter(
            out_zarr,
            joint_dim=len(JOINT_NAMES),
            ee_dim=7,
            cube_dim=0,
            gripper_dim=1,
            obstacle_dim=3,
            flush_every=12,
        )
        writer.set_attrs(
            xml=str(xml_path),
            joint_names=list(JOINT_NAMES),
            cube_colors=list(CUBE_COLORS),
            cube_joint_names=list(CUBE_JOINT_NAMES),
            state_joints_spec="qpos(joints)",
            state_ee_spec="ee_pos(3) + ee_quat_wxyz(4)",
            state_cube_spec="not_stored_in_multicube_raw",
            pos_cube_red_spec="red_cube_pos(3) + red_cube_quat_wxyz(4)",
            pos_cube_green_spec="green_cube_pos(3) + green_cube_quat_wxyz(4)",
            pos_cube_blue_spec="blue_cube_pos(3) + blue_cube_quat_wxyz(4)",
            state_goal_spec="one_hot(red, green, blue) = 3",
            goal_pos_spec="bin_center_world_xyz(3)",
            state_gripper_spec="gripper_angle(1)",
            action_gripper_spec="gripper_ctrl(1)",
            control_hz=float(control_hz),
            cameras_display=list(CAMERA_NAMES),
        )
        return writer

    @property
    def goal_writer(self) -> MulticubeZarrWriter:
        return self.writer  # type: ignore[return-value]

    def _set_goal(self, index: int) -> None:
        self._goal_index = index
        self._goal_onehot = np.zeros(GOAL_DIM, dtype=np.float32)
        self._goal_onehot[index] = 1.0
        print(f"  Goal cube set to: {CUBE_COLORS[index]}")

    def _get_all_cubes_state(self) -> np.ndarray:
        parts = [self.data.qpos[sl].copy() for sl in self.cube_qpos_slices]
        return np.concatenate(parts)

    def _get_goal_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.bin_center_site_id].copy().astype(np.float32)

    def _randomize_layout(self) -> None:
        if self._default_cube_qpos is None:
            self._default_cube_qpos = np.array(
                [self.data.qpos[sl].copy() for sl in self.cube_qpos_slices]
            )
        if self._cube_slot_qpos_templates is None:
            self._cube_slot_qpos_templates = build_multicube_slot_templates(
                self._default_cube_qpos, self._default_bin_pos
            )

        cube_slot_ids, bin_slot_id, cube_xy, bin_xy = sample_multicube_layout(
            self.rng,
            self._default_cube_qpos,
            self._default_bin_pos,
            self.cube_pos_std,
            True,
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

        layout_labels: list[str] = []
        for slot_i in range(NUM_CUBES + 1):
            if slot_i == bin_slot_id:
                occupant = "bin"
            else:
                cube_i = int(np.where(cube_slot_ids == slot_i)[0][0])
                occupant = CUBE_COLORS[cube_i]
            layout_labels.append(f"slot {slot_i}: {occupant}")
        print(f"  Layout: {' | '.join(layout_labels)}")

    def _reset_episode(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._reset_to_keyframe("student_start")
        self._randomize_layout()
        mujoco.mj_forward(self.model, self.data)
        self._init_pose_and_targets()

    def _handle_key(self, k_raw: int, _k_ascii: int) -> None:
        action = self._key_to_action.get(k_raw)
        if action is None:
            return

        if action in ("goal_cube_red", "goal_cube_green", "goal_cube_blue"):
            if self.recording:
                print("  Cannot change goal cube while recording!")
                return
            goal_map = {
                "goal_cube_red": 0,
                "goal_cube_green": 1,
                "goal_cube_blue": 2,
            }
            self._set_goal(goal_map[action])
            return

        if action == "escape":
            if self.recording:
                self.writer.end_episode()
                self.episodes_done += 1
                print(f"Episode {self.episodes_done} saved on exit.")
                self.recording = False
            self.running = False
            return

        if action == "record":
            self.recording = not self.recording
            if self.recording:
                print(f"RECORDING ON  (goal: {CUBE_COLORS[self._goal_index]})")
            else:
                print("RECORDING OFF")
            return

        if action == "end_episode":
            if self.recording:
                self.writer.end_episode()
                self.episodes_done += 1
                print(
                    f"Episode {self.episodes_done} saved "
                    f"(goal was: {CUBE_COLORS[self._goal_index]})."
                )
                self.recording = False
            self._reset_episode()
            return

        if action == "reset":
            if self.recording:
                self.writer.discard_episode()
                self.recording = False
                print("Episode DISCARDED.")
            self._reset_episode()
            return

        handle_teleop_key(
            action, self.data, self.model, MOCAP_INDEX, self.act_id["Jaw"]
        )

    def _record_step(self) -> None:
        state_joints = self._get_q()
        state_ee = self._get_ee_state()
        state_cubes = self._get_all_cubes_state()
        state_gripper = np.array(
            [state_joints[JOINT_NAMES.index("Jaw")]], dtype=np.float32
        )
        action_gripper = np.array(
            [self.data.ctrl[self.act_id["Jaw"]]], dtype=np.float32
        )
        dummy_obstacle = np.zeros(3, dtype=np.float32)
        self.goal_writer.append_with_goal(
            state_joints,
            state_ee,
            state_cubes,
            state_gripper,
            action_gripper,
            dummy_obstacle,
            self._goal_onehot,
            self._get_goal_pos(),
        )

    def _overlay_status(self, img_bgr: np.ndarray) -> np.ndarray:
        img = img_bgr.copy()
        goal_color = CUBE_COLORS[self._goal_index]
        status = f"{'REC' if self.recording else 'IDLE'} | ep {self.episodes_done} | goal: {goal_color}"
        cv2.putText(
            img,
            status,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        hint = (
            f"{self._label_for('record')} rec | "
            f"{self._label_for('end_episode')} end ep | "
            f"{self._label_for('reset')} reset | "
            f"{self._label_for('escape')} quit | "
            f"{self._label_for('goal_cube_red')}/"
            f"{self._label_for('goal_cube_green')}/"
            f"{self._label_for('goal_cube_blue')} goal"
        )
        cv2.putText(
            img,
            hint,
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Record teleop demonstrations.")
    parser.add_argument(
        "--multicube",
        action="store_true",
        help="Record multicube goal-conditioned demonstrations.",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="Path to the MuJoCo XML scene file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible multicube shuffling.",
    )
    args = parser.parse_args()

    ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")

    if args.multicube:
        xml_path = args.xml or (ASSETS_DIR / "so100_multicube_ee.xml")
        run_dir = Path("./datasets/raw/multi_cube/teleop") / ts
        out = run_dir / "so100_multicube_teleop.zarr"

        MulticubeTeleopRecorder(
            xml_path=xml_path,
            out_zarr=out,
            control_hz=10.0,
            seed=args.seed,
        ).run()
        return

    xml_path = args.xml or (ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml")
    run_dir = Path("./datasets/raw/single_cube/teleop") / ts
    out = run_dir / "so100_transfer_cube_teleop.zarr"

    SO100Cv2TeleopRecorder(
        xml_path=xml_path,
        out_zarr=out,
        control_hz=10.0,
        render_w=640,
        render_h=480,
    ).run()


if __name__ == "__main__":
    main()
