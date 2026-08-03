"""Read recorded teleop .zarr datasets and compute actions for imitation learning.

Actions are defined as the relative change (delta) between consecutive states:
    a_t = s_{t+1} - s_{t}
The last timestep of every episode is dropped (no future state available).

For the gripper, actions are the control commands recorded during teleop since we need to 
push the gripper even more close to apply a force to the cube which can only be recorded from the 
control input not the state.

Three action spaces are supported (chosen via --action-space):

  ee       - EE xyz position delta (3-dim)
  ee_full  - EE full pose delta: delta_pos(3) + delta_euler(3) (6-dim)
  joints   - Joint angle deltas excluding Jaw (5-dim)

Gripper actions are stored as a separate ``action_gripper`` array in all modes.

Usage examples:
    python scripts/compute_actions.py --action-space ee
    python scripts/compute_actions.py --action-space ee_full
    python scripts/compute_actions.py --action-space joints
    python scripts/compute_actions.py --action-space joints --datasets-dir ./datasets/raw/multi_cube
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr

# ── quaternion helpers (wxyz convention) ──────────────────────────────


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate of unit quaternion(s) in wxyz format (= inverse for unit quats)."""
    return np.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternion arrays in wxyz format."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion(s) to Euler angles (roll, pitch, yaw)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([roll, pitch, yaw], axis=-1)


def _ee_full_delta(s_curr: np.ndarray, s_next: np.ndarray) -> np.ndarray:
    """Compute ee_full action: delta_pos(3) + delta_euler(3).

    States are (L, 7): [x, y, z, qw, qx, qy, qz].
    Returns (L, 6): [dx, dy, dz, droll, dpitch, dyaw].
    """
    pos_delta = s_next[:, :3] - s_curr[:, :3]
    q_rel = quat_multiply(s_next[:, 3:], quat_conjugate(s_curr[:, 3:]))
    euler_delta = quat_to_euler(q_rel)
    return np.concatenate([pos_delta, euler_delta], axis=-1)


# ── action-space naming config (do NOT modify) ────────────────────────
# Maps CLI choice → (action_label, state_label, key suffix for output arrays)
_ACTION_SPACE_LABELS: dict[str, tuple[str, str, str]] = {
    "ee": ("ee_pos_xyz(3)", "ee_pos_xyz(3)", "ee_xyz"),
    "ee_full": ("delta_pos(3)+delta_euler(3)", "ee_pos(3)+quat_wxyz(4)", "ee_full"),
    "joints": ("joint_angles(5)", "joint_angles(5)", "joints"),
}


def select_action_space(
    action_space: str, merged: dict[str, np.ndarray]
) -> tuple[np.ndarray, str, str, str]:
    """Select and slice the state array for the chosen action space.

    Parameters
    ----------
    action_space : str
        One of ``"ee"``, ``"ee_full"``, ``"joints"``.
    merged : dict
        Merged data from :func:`load_and_merge_zarrs`.  Relevant keys are
        ``"state_ee"`` (N, 7) with columns [x, y, z, qw, qx, qy, qz] and
        ``"state_joints"`` (N, 6) with columns for each joint angle
        (the last column is the Jaw / gripper and should be excluded).

    Returns
    -------
    raw_states : np.ndarray
        (N, D) array of the selected state columns.
    action_label : str
        Human-readable action dimension description.
    state_label : str
        Human-readable state dimension description.
    sa_suffix : str
        Key suffix for output arrays (provided, do not change).
    """
    action_label, state_label, sa_suffix = _ACTION_SPACE_LABELS[action_space]

    if action_space == "ee":
        raw_states = merged["state_ee"][:, :3]
    elif action_space == "ee_full":
        raw_states = merged["state_ee"]
    elif action_space == "joints":
        raw_states = merged["state_joints"][:, :5]
    else:
        raise ValueError(f"Unknown action space: {action_space!r}")

    return raw_states, action_label, state_label, sa_suffix


def get_episode_ranges(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) index pairs for each episode."""
    starts = np.concatenate([[0], episode_ends[:-1]])
    return list(zip(starts.tolist(), episode_ends.tolist()))


def compute_actions_for_episodes(
    states: np.ndarray,
    episode_ranges: list[tuple[int, int]],
    action_fn=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute delta actions and return aligned states, actions, episode_ends, and keep indices.

    For each episode of length L we keep L-1 transitions:
        state[t] -> action = action_fn(state[t], state[t+1])   for t in [start, end-2]
    Default (action_fn=None) uses simple subtraction: state[t+1] - state[t].

    Returns:
        out_states:       (N', D) - states with the last step of each episode removed
        out_actions:      (N', D) - corresponding delta actions
        out_episode_ends: (num_episodes,) - cumulative end indices for the trimmed data
        keep_idx:         (N',) - original indices of kept timesteps (for aligning other arrays)
    """
    out_states_list: list[np.ndarray] = []
    out_actions_list: list[np.ndarray] = []
    out_episode_ends: list[int] = []
    keep_idx_parts: list[np.ndarray] = []
    running = 0

    for start, end in episode_ranges:
        ep_states = states[start:end]  # (L, D)
        if ep_states.shape[0] < 2:
            continue  # skip degenerate episodes
        out_states_list.append(ep_states[:-1])
        if action_fn is not None:
            out_actions_list.append(action_fn(ep_states[:-1], ep_states[1:]))
        else:
            out_actions_list.append(ep_states[1:] - ep_states[:-1])
        keep_idx_parts.append(np.arange(start, end - 1))
        running += ep_states.shape[0] - 1
        out_episode_ends.append(running)

    out_states = np.concatenate(out_states_list, axis=0)
    out_actions = np.concatenate(out_actions_list, axis=0)
    keep_idx = np.concatenate(keep_idx_parts)
    return out_states, out_actions, np.array(out_episode_ends, dtype=np.int64), keep_idx


def trim_to_transitions(
    merged: dict[str, np.ndarray],
    keep_idx: np.ndarray,
    *,
    skip_keys: set[str],
) -> dict[str, np.ndarray]:
    """Trim auxiliary arrays to the kept transition indices.

    Applies ``keep_idx`` to every array in *merged* (except those in
    *skip_keys* and ``"episode_ends"``), and renames ``state_ee`` to
    ``state_ee_full`` (to avoid collisions with the ee-xyz state key).

    Parameters
    ----------
    merged : dict
        Raw merged data from :func:`load_and_merge_zarrs`.
    keep_idx : (N',) array
        Indices of kept timesteps (returned by :func:`compute_actions_for_episodes`).
    skip_keys : set[str]
        Keys that have already been written and should be skipped.

    Returns
    -------
    dict[str, np.ndarray]
        Trimmed arrays with their final destination names.
    """
    trimmed: dict[str, np.ndarray] = {}

    for key, arr in merged.items():
        if key == "episode_ends":
            continue
        if key.startswith("_"):
            continue  # skip internal metadata (e.g. _num_dagger_episodes)

        dest_name = key
        if dest_name == "state_ee":
            dest_name = "state_ee_full"  # avoid collision with state_ee_xyz
        elif dest_name in ("pos_cube_red", "pos_cube_green", "pos_cube_blue"):
            dest_name = f"original_{dest_name}"

        if dest_name in skip_keys:
            continue

        sliced = arr[keep_idx]

        trimmed[dest_name] = sliced
    return trimmed


def load_and_merge_zarrs(zarr_paths: list[Path]) -> dict[str, np.ndarray]:
    """Load and concatenate data from multiple zarr stores.

    Returns a dict with keys:
        state_joints, state_ee, state_cube, episode_ends,
        and images_<cam> for each camera found.
    """
    all_data: dict[str, list[np.ndarray]] = {}
    cumulative_offset = 0

    for zpath in sorted(zarr_paths):
        root = zarr.open_group(str(zpath), mode="r")
        data_grp = root["data"]
        meta_grp = root["meta"]

        ep_ends = np.asarray(meta_grp["episode_ends"])
        if ep_ends.size == 0:
            print(f"  Skipping {zpath} (no episodes)")
            continue

        n_steps = int(ep_ends[-1])
        is_dagger = "dagger" in str(zpath).lower()
        tag = " [dagger]" if is_dagger else ""
        print(f"  {zpath.name}: {ep_ends.size} episode(s), {n_steps} steps{tag}")

        # Shift episode_ends by the running offset
        all_data.setdefault("episode_ends", []).append(ep_ends + cumulative_offset)
        all_data.setdefault("_dagger_ep_counts", []).append(
            ep_ends.size if is_dagger else 0
        )

        for key in data_grp:
            arr = np.asarray(data_grp[key][:n_steps])
            all_data.setdefault(key, []).append(arr)

        cumulative_offset += n_steps

    merged: dict[str, np.ndarray] = {}
    for key, arrays in all_data.items():
        if key == "_dagger_ep_counts":
            continue  # not an array, handled separately
        merged[key] = np.concatenate(arrays, axis=0)

    # Propagate dagger episode count as a plain int (not an ndarray)
    merged["_num_dagger_episodes"] = sum(all_data.get("_dagger_ep_counts", [0]))

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute actions from recorded teleop zarr datasets."
    )
    parser.add_argument(
        "--action-space",
        choices=list(_ACTION_SPACE_LABELS),
        required=True,
        help="Action space: ee (xyz, 3D), ee_full (pos+euler, 6D), joints (5D).",
    )
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=Path("./datasets/raw/single_cube"),
        help="Root directory to search for .zarr stores (default: ./datasets/raw/single_cube).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output zarr path. Default: datasets/processed/<task>/processed_<action-space>.zarr",
    )
    args = parser.parse_args()

    # ── discover zarr stores ──────────────────────────────────────────
    zarr_paths = sorted(args.datasets_dir.rglob("*.zarr"))
    if not zarr_paths:
        print(f"No .zarr stores found under {args.datasets_dir}")
        return
    print(f"Found {len(zarr_paths)} zarr store(s):")

    # ── load & merge ──────────────────────────────────────────────────
    merged = load_and_merge_zarrs(zarr_paths)
    episode_ends = merged["episode_ends"]
    episode_ranges = get_episode_ranges(episode_ends)
    n_episodes = len(episode_ranges)
    n_dagger_episodes = int(merged.get("_num_dagger_episodes", 0))
    n_total = int(episode_ends[-1])
    print(
        f"\nMerged: {n_episodes} episodes ({n_dagger_episodes} dagger), {n_total} total steps"
    )

    # ── select state array for the chosen action space ────────────────
    raw_states, action_label, state_label, sa_suffix = select_action_space(
        args.action_space, merged
    )
    print(
        f"Action space: {args.action_space}, state_dim={raw_states.shape[1]} ({state_label}), action=({action_label})"
    )

    # ── compute next-state actions ────────────────────────────────────
    action_fn = _ee_full_delta if args.action_space == "ee_full" else None
    states, actions, new_ep_ends, keep_idx = compute_actions_for_episodes(
        raw_states,
        episode_ranges,
        action_fn=action_fn,
    )
    print(
        f"After action computation: {states.shape[0]} transitions "
        f"across {new_ep_ends.size} episodes"
    )

    # ── align gripper actions (recorded ctrl, not computed) ───────────
    raw_action_gripper = merged.get("action_gripper")
    if raw_action_gripper is not None:
        action_gripper_trimmed = raw_action_gripper[keep_idx]
    # ── write output zarr ─────────────────────────────────────────────
    if args.output is not None:
        out_path = args.output
    else:
        # Default: datasets/processed/<task>/processed_<action-space>.zarr
        base_dir = Path("./datasets/processed")
        if "multi_cube" in str(args.datasets_dir):
            base_dir = base_dir / "multi_cube"
        else:
            base_dir = base_dir / "single_cube"
        out_path = base_dir / f"processed_{sa_suffix}.zarr"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting to {out_path} ...")

    out_root = zarr.open_group(str(out_path), mode="w", zarr_format=3)
    compressor = zarr.codecs.Blosc(cname="zstd", clevel=3, shuffle=2)
    compressors = (compressor,)

    out_data = out_root.require_group("data")
    out_meta = out_root.require_group("meta")

    state_key = f"state_{sa_suffix}"
    action_key = f"action_{sa_suffix}"

    # State & action for ee/joints
    out_data.create_array(
        state_key, data=states.astype(np.float32), compressors=compressors
    )
    out_data.create_array(
        action_key, data=actions.astype(np.float32), compressors=compressors
    )

    # Gripper action (recorded control command, aligned to the same timesteps)
    out_data.create_array(
        "action_gripper",
        data=action_gripper_trimmed.astype(np.float32),
        compressors=compressors,
    )

    # Episode ends
    out_meta.create_array(
        "episode_ends", data=new_ep_ends.astype(np.int64), compressors=compressors
    )

    # Trim and copy auxiliary arrays (images, cube state, original states, gripper state)
    already_written = {state_key, action_key, "action_gripper"}
    aux_arrays = trim_to_transitions(merged, keep_idx, skip_keys=already_written)
    for dest_name, data in aux_arrays.items():
        out_data.create_array(dest_name, data=data, compressors=compressors)

    # Metadata
    out_root.attrs["action_space"] = args.action_space
    out_root.attrs["action_dim"] = int(actions.shape[1])
    out_root.attrs["action_spec"] = action_label
    out_root.attrs["state_spec"] = state_label
    out_root.attrs["state_key"] = state_key
    out_root.attrs["action_key"] = action_key
    out_root.attrs["num_episodes"] = int(new_ep_ends.size)
    out_root.attrs["num_dagger_episodes"] = n_dagger_episodes
    out_root.attrs["num_transitions"] = int(states.shape[0])
    out_root.attrs["source_zarrs"] = [str(p) for p in zarr_paths]

    print(f"Done. {states.shape[0]} transitions written.")
    print(f"  data/{state_key}:  {states.shape}")
    print(f"  data/{action_key}: {actions.shape}")
    print(f"  data/action_gripper: {action_gripper_trimmed.shape}")
    print(f"  meta/episode_ends: {new_ep_ends.shape}")


if __name__ == "__main__":
    main()
