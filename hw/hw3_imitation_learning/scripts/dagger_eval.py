"""DAgger interactive evaluation: run a trained policy with human takeover.

Runs policy inference in the obstacle scene.  At any time you (the expert) can press the takeover key to assume manual control.  While in
takeover mode every timestep is recorded into a zarr store (same format as
record_teleop_demos.py).  Pressing the takeover key again or ending the
episode hands control back to the policy.

Collected data is saved under datasets/raw/single_cube/dagger/ and can
later be merged with the original demonstrations for retraining via
compute_actions.py which will produce a merged .zarr dataset to train on.

Usage:
    python scripts/dagger_eval.py \\
        --checkpoint checkpoints/single_cube/best_model_ee_xyz_obstacle.pt \\
        --num-episodes 10
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from hw3.dataset import Normalizer
from hw3.eval_utils import (
    apply_action,
    check_cube_out_of_bounds,
    check_success,
    infer_action_chunk,
    load_checkpoint,
)
from hw3.sim_env import (
    SO100SimEnv,
)
from hw3.teleop_utils import (
    CAMERA_NAMES,
    DEFAULT_KEYMAP_PATH,
    ZarrEpisodeWriter,
    compose_camera_views,
    handle_teleop_key,
    load_keymap,
)
from so101_gym.constants import ASSETS_DIR

XML_PATH = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"


# ── main DAgger loop ─────────────────────────────────────────────────


def run_dagger_episode(
    env: SO100SimEnv,
    model: torch.nn.Module,
    normalizer: Normalizer,
    state_keys: list[str],
    action_keys: list[str],
    device: torch.device,
    writer: ZarrEpisodeWriter,
    key_to_action: dict[int, str],
    *,
    max_steps: int = 800,
    successes: int = 0,
    total: int = 0,
    headless: bool = False,
) -> tuple[bool, int, bool, bool]:
    """Run one DAgger episode: policy runs, human can take over at any time.

    Returns (success, n_takeover_steps, aborted, replay).
    """
    # Save RNG state so that the caller can restore it for a replay
    rng_state_before_reset = env.rng.bit_generator.state
    obs = env.reset()

    action_queue: list[np.ndarray] = []
    step = 0
    success = False
    human_control = False
    n_takeover_steps = 0
    recording_this_episode = False  # track if we recorded anything
    # Grace period: after success is detected during human control, keep
    # running for GRACE_SECS more so at least one full chunk is recorded.
    GRACE_SECS = 1.7
    grace_steps_remaining: int | None = None  # None = not in grace period

    while step < max_steps or human_control:
        # ── keyboard handling (skipped in headless mode) ─────────────
        if not headless:
            k_raw = cv2.waitKeyEx(1)
        else:
            k_raw = -1
        if k_raw != -1:
            action_name = key_to_action.get(k_raw)

            if action_name == "escape":
                # Discard current episode data and abort
                if recording_this_episode:
                    writer.discard_episode()
                    print("  Episode discarded on escape.")
                return success, n_takeover_steps, True, False  # aborted

            if action_name == "record":
                # Toggle human takeover mode
                human_control = not human_control
                if human_control:
                    print("  >>> HUMAN TAKEOVER — you are now controlling the arm")
                    print("      Press your 'record' key again to hand back to policy")
                    action_queue.clear()  # drop any queued policy actions
                    recording_this_episode = True
                else:
                    print("  <<< POLICY RESUMED")

            if action_name == "reset":
                # Replay: discard data and repeat with identical randomization
                if recording_this_episode:
                    writer.discard_episode()
                    print("  Episode discarded — replaying same scenario.")
                # Restore RNG so next reset() reproduces the same episode
                env.rng.bit_generator.state = rng_state_before_reset
                return False, 0, False, True  # replay=True

            # Enter key (13 / 0x0D) = skip to next episode
            if k_raw == 13 or k_raw == 0x0D:
                if recording_this_episode:
                    writer.discard_episode()
                    print("  Episode discarded — skipping to next.")
                return False, 0, False, False  # replay=False

            # If in human control, apply movement keys
            if human_control and action_name is not None:
                handle_teleop_key(
                    action_name,
                    env.data,
                    env.model,
                    env.mocap_id,
                    env.act_ids[env._jaw_idx],
                )

        # ── record state BEFORE step (if human is controlling) ────────
        if human_control:
            # Record current state for DAgger
            joints = env.get_joint_angles()
            ee_state = env.get_ee_state()
            cube_state = env.get_cube_state()
            gripper_state = np.array([env.get_gripper_angle()], dtype=np.float32)
            action_gripper = np.array(
                [env.data.ctrl[env.act_ids[env._jaw_idx]]], dtype=np.float32
            )
            obstacle_state = env.get_obstacle_pos()
            writer.append(
                joints,
                ee_state,
                cube_state,
                gripper_state,
                action_gripper,
                obstacle_state,
            )
            n_takeover_steps += 1

        # ── policy inference (if not in human control) ────────────────
        if not human_control:
            if not action_queue:
                chunk = infer_action_chunk(
                    model=model,
                    normalizer=normalizer,
                    obs=obs,
                    state_keys=state_keys,
                    device=device,
                )
                action_queue.extend(chunk)

            action = action_queue.pop(0)
            apply_action(env, action, action_keys)

        # ── step simulation ───────────────────────────────────────────
        obs = env.step()
        step += 1

        # ── check termination ─────────────────────────────────────────
        success = check_success(env)
        if success:
            if human_control and grace_steps_remaining is None:
                # Start grace period so we keep recording
                grace_steps_remaining = int(GRACE_SECS / env.dt_ctrl)
                print(f"  Cube in bin! Recording {grace_steps_remaining} more "
                      f"steps ({GRACE_SECS}s grace period)...")
            elif not human_control:
                # Policy mode — terminate immediately
                if recording_this_episode:
                    writer.end_episode()
                    print(f"  DAgger episode saved ({n_takeover_steps} takeover steps)")
                return success, n_takeover_steps, False, False

        # Tick down grace period
        if grace_steps_remaining is not None:
            grace_steps_remaining -= 1
            if grace_steps_remaining <= 0:
                if recording_this_episode:
                    writer.end_episode()
                    print(f"  DAgger episode saved ({n_takeover_steps} takeover steps)")
                return True, n_takeover_steps, False, False

        if check_cube_out_of_bounds(env):
            print("  Cube out of bounds — early termination.")
            if recording_this_episode:
                writer.end_episode()
                print(f"  DAgger episode saved ({n_takeover_steps} takeover steps)")
            return False, n_takeover_steps, False, False

        # ── render (skip in headless mode) ────────────────────────
        if headless:
            continue

        img = compose_camera_views({cam: env.render(cam) for cam in CAMERA_NAMES})
        status = f"Step {step}/{max_steps}"
        if human_control:
            status += " | HUMAN CONTROL"
        else:
            status += f" | POLICY (queue {len(action_queue)})"
        cv2.putText(
            img, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )

        # Success rate
        if total > 0:
            rate = successes / total * 100
            sr_text = f"Success: {successes}/{total} ({rate:.0f}%)"
        else:
            sr_text = "Success: -/-"
        color = (0, 255, 0) if success else (0, 0, 255)
        cv2.putText(img, sr_text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # DAgger info
        dagger_text = (
            f"DAgger steps: {n_takeover_steps} | Episodes saved: {writer.num_episodes}"
        )
        cv2.putText(
            img, dagger_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2
        )

        # Mode indicator
        if human_control:
            cv2.putText(
                img, "HUMAN", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
            )
        else:
            cv2.putText(
                img, "POLICY", (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
            )

        # Hint
        def _label_for(act):
            for code, a in key_to_action.items():
                if a == act:
                    if 32 <= (code & 0xFF) <= 126:
                        ch = chr(code & 0xFF)
                        return ch if ch.strip() else "SPACE"
                    if code & 0xFF == 27:
                        return "ESC"
                    return f"key:{code}"
            return "?"

        hint = (
            f"{_label_for('record')} takeover | "
            f"{_label_for('reset')} replay | "
            f"ENTER skip | "
            f"{_label_for('escape')} quit"
        )
        cv2.putText(
            img,
            hint,
            (10, img.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )

        cv2.imshow("DAgger Eval", img)
        time.sleep(env.dt_ctrl)

    # Episode ended by reaching max_steps
    if recording_this_episode:
        writer.end_episode()
        print(f"  DAgger episode saved ({n_takeover_steps} takeover steps)")
    return success, n_takeover_steps, False, False


def main():
    parser = argparse.ArgumentParser(
        description="DAgger interactive evaluation with human takeover."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the model checkpoint (.pt).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum steps per episode (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible cube spawns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for DAgger zarr data (default: datasets/raw/single_cube/dagger/<timestamp>).",
    )
    parser.add_argument(
        "--keymap",
        type=Path,
        default=None,
        help="Path to keymap.json (default: hw3/keymap.json).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without rendering or real-time pacing (faster batch eval). "
        "No human takeover is possible in this mode.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model, normalizer, chunk_size, state_keys, action_keys = load_checkpoint(
        args.checkpoint, device
    )

    # Decide on use_mocap
    use_mocap = not any("action_joints" in k for k in action_keys)

    # Scene
    print(f"Scene: {XML_PATH.name}")

    env = SO100SimEnv(
        xml_path=XML_PATH,
        render_w=640,
        render_h=480,
        use_mocap=use_mocap,
        obstacle_mode="adversarial",
        seed=args.seed,
    )

    # Keymap
    km_path = args.keymap or DEFAULT_KEYMAP_PATH
    key_to_action = load_keymap(km_path)
    print(f"Loaded keymap from {km_path}")

    # DAgger output zarr
    if args.output_dir:
        out_dir = args.output_dir
    else:
        ts = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = Path("./datasets/raw/single_cube/dagger") / ts
    out_zarr = out_dir / "so100_transfer_cube_teleop.zarr"
    print(f"DAgger data will be saved to: {out_zarr}")

    writer = ZarrEpisodeWriter(
        path=out_zarr,
    )

    if not args.headless:
        cv2.namedWindow("DAgger Eval", cv2.WINDOW_AUTOSIZE)

    successes = 0
    total_takeover_steps = 0
    try:
        ep = 0
        while ep < args.num_episodes:
            ep += 1
            print(f"\n═══ DAgger Episode {ep}/{args.num_episodes} ═══")
            print("  Policy is running. Press your 'record' key to take over control.")

            success, n_takeover, aborted, replay = run_dagger_episode(
                env,
                model,
                normalizer,
                state_keys,
                action_keys,
                device,
                writer,
                key_to_action,
                max_steps=args.max_steps,
                successes=successes,
                total=ep - 1,
                headless=args.headless,
            )

            if aborted:
                print("Aborted by user.")
                break

            if replay:
                # RNG already restored inside run_dagger_episode
                print("  Replaying same episode...")
                ep -= 1  # don't count this attempt
                continue

            total_takeover_steps += n_takeover
            if success:
                successes += 1
            rate = successes / ep * 100
            result = "SUCCESS" if success else "FAIL"
            print(f"Episode {ep}: {result} | takeover steps this ep: {n_takeover}")
            print(f"  Success rate: {successes}/{ep} ({rate:.0f}%)")

    finally:
        writer.flush()
        cv2.destroyAllWindows()

    n_eps = writer.num_episodes
    n_steps = writer.num_steps_total
    rate = successes / max(1, args.num_episodes) * 100
    print(f"\n{'=' * 50}")
    print("DAgger session complete.")
    print(f"  Episodes evaluated: {args.num_episodes}")
    print(f"  Success rate: {successes}/{args.num_episodes} ({rate:.0f}%)")
    print(f"  Total takeover steps: {total_takeover_steps}")
    print(f"  DAgger episodes saved: {n_eps} ({n_steps} total steps)")
    print(f"  Data saved to: {out_zarr}")
    print("\n If you collected data, you can now retrain your model with the additional episodes.")


if __name__ == "__main__":
    main()
