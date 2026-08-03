"""Evaluate a trained policy in the MuJoCo SO-100 simulation.

Supports both single-cube and multicube scenes. Use --multicube to run the
multicube goal-conditioned setup.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from hw.hw3_imitation_learning.hw3.eval_utils import (
    apply_action,
    check_cube_out_of_bounds,
    check_success,
    check_wrong_cube_in_bin,
    infer_action_chunk,
    load_checkpoint,
)
from hw.hw3_imitation_learning.hw3.sim_env import (
    CUBE_COLORS,
    SO100MulticubeSimEnv,
    SO100SimEnv,
)
from hw.hw3_imitation_learning.hw3.teleop_utils import CAMERA_NAMES, compose_camera_views
from hw.hw3_imitation_learning.so101_gym.constants import ASSETS_DIR

XML_PATH = ASSETS_DIR / "so100_transfer_cube_obstacle_ee.xml"
XML_PATH_MULTICUBE = ASSETS_DIR / "so100_multicube_ee.xml"


def compose_views(env) -> np.ndarray:
    images = {cam: env.render(cam) for cam in CAMERA_NAMES}
    return compose_camera_views(images, CAMERA_NAMES)


def run_episode(
    env,
    model: torch.nn.Module,
    normalizer,
    state_keys: list[str],
    action_keys: list[str],
    device: torch.device,
    max_steps: int,
    successes: int,
    total: int,
    headless: bool,
    multicube: bool,
) -> tuple[bool, bool, str | None]:
    """Run one evaluation episode.

    Returns (success, aborted, wrong_cube_color).
    wrong_cube_color is only set for multicube episodes.
    """
    obs = env.reset()
    action_queue: list[np.ndarray] = []
    step = 0

    while step < max_steps:
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
        obs = env.step()
        step += 1

        success = check_success(env)
        wrong_in_bin = check_wrong_cube_in_bin(env) if multicube else None

        if success:
            return True, False, wrong_in_bin

        if check_cube_out_of_bounds(env):
            if multicube:
                print(f"  [{env.goal_cube}] Cube out of bounds - early termination.")
            else:
                print("  Cube out of bounds - early termination (failure).")
            return False, False, wrong_in_bin

        if headless:
            continue

        img = compose_views(env)
        status = f"Step {step}/{max_steps} | queue {len(action_queue)}"
        if multicube:
            status = f"Goal: {env.goal_cube} | {status}"
        cv2.putText(
            img, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2
        )

        if total > 0:
            rate = successes / total * 100
            sr_text = f"Success: {successes}/{total} ({rate:.0f}%)"
        else:
            sr_text = "Success: -/-"
        cv2.putText(
            img,
            sr_text,
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if success else (0, 0, 255),
            2,
        )

        if multicube and wrong_in_bin:
            cv2.putText(
                img,
                f"WRONG CUBE IN BIN: {wrong_in_bin}!",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
        if not multicube and success:
            cv2.putText(
                img,
                "CUBE IN BIN",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

        cv2.imshow("SO100 Multicube Eval" if multicube else "SO100 Eval", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            return False, True, None
        if key == ord("r"):
            return False, False, wrong_in_bin
        if key == ord(" "):
            while True:
                k2 = cv2.waitKey(50) & 0xFF
                if k2 == ord(" ") or k2 == 27:
                    break

        time.sleep(env.dt_ctrl)

    return False, False, check_wrong_cube_in_bin(env) if multicube else None


def build_goal_schedule(goal_cube: str, num_episodes: int) -> list[str]:
    if goal_cube == "all":
        return [CUBE_COLORS[i % len(CUBE_COLORS)] for i in range(num_episodes)]
    return [goal_cube] * num_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy in simulation.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the model checkpoint (.pt).")
    parser.add_argument("--multicube", action="store_true", help="Evaluate in multicube scene.")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes (default: 10).")
    parser.add_argument("--max-steps", type=int, default=800, help="Maximum steps per episode (default: 800).")
    parser.add_argument("--headless", action="store_true", help="Run without rendering.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible cube spawns.")

    # single-cube args
    parser.add_argument("--adversarial-obstacle", action="store_true", help="Use adversarial three-zone obstacle placement.")

    # multicube args
    parser.add_argument("--goal-cube", type=str, default="all", choices=["red", "green", "blue", "all"], help="Goal colour for multicube ('all' cycles evenly).")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable multicube slot shuffling.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, normalizer, _chunk_size, state_keys, action_keys = load_checkpoint(
        args.checkpoint,
        device,
    )

    use_mocap = not any("action_joints" in k for k in action_keys)

    if args.multicube:
        goal_schedule = build_goal_schedule(args.goal_cube, args.num_episodes)
        env = SO100MulticubeSimEnv(
            xml_path=XML_PATH_MULTICUBE,
            render_w=640,
            render_h=480,
            use_mocap=use_mocap,
            goal_cube=goal_schedule[0],
            shuffle_cubes=not args.no_shuffle,
            seed=args.seed,
        )
    else:
        env = SO100SimEnv(
            xml_path=XML_PATH,
            render_w=640,
            render_h=480,
            use_mocap=use_mocap,
            obstacle_mode="adversarial" if args.adversarial_obstacle else "train",
            seed=args.seed,
        )

    if not args.headless:
        cv2.namedWindow(
            "SO100 Multicube Eval" if args.multicube else "SO100 Eval",
            cv2.WINDOW_AUTOSIZE,
        )

    successes = 0
    per_color: dict[str, dict[str, int]] | None = None
    if args.multicube:
        per_color = {c: {"success": 0, "total": 0} for c in CUBE_COLORS}

    episodes_run = 0
    try:
        for ep in range(1, args.num_episodes + 1):
            if args.multicube:
                goal = goal_schedule[ep - 1]
                env.set_goal(goal)
                print(f"\n═══ Episode {ep}/{args.num_episodes}  (goal: {goal}) ═══")
            else:
                print(f"\n═══ Episode {ep}/{args.num_episodes} ═══")

            success, aborted, wrong_cube_color = run_episode(
                env=env,
                model=model,
                normalizer=normalizer,
                state_keys=state_keys,
                action_keys=action_keys,
                device=device,
                max_steps=args.max_steps,
                successes=successes,
                total=ep - 1,
                headless=args.headless,
                multicube=args.multicube,
            )
            if aborted:
                print("Aborted by user.")
                break

            episodes_run = ep
            if success:
                successes += 1

            if args.multicube:
                assert per_color is not None
                goal = goal_schedule[ep - 1]
                per_color[goal]["total"] += 1
                if success:
                    per_color[goal]["success"] += 1

            rate = successes / ep * 100
            result = "SUCCESS" if success else "FAIL"
            print(f"Episode {ep} finished: {result}")
            print(f"  Success rate: {successes}/{ep} ({rate:.0f}%)")
            if args.multicube and wrong_cube_color:
                print(f"  WARNING: wrong cube in bin: {wrong_cube_color}")
    finally:
        cv2.destroyAllWindows()

    denom = max(episodes_run, 1)
    print(f"\nEvaluation complete. Success rate: {successes}/{denom} ({successes / denom * 100:.0f}%)")

    if args.multicube and per_color is not None:
        print(f"{'═' * 50}")
        for c in CUBE_COLORS:
            s = per_color[c]["success"]
            t = per_color[c]["total"]
            r = s / t * 100 if t > 0 else 0
            print(f"  {c:6s}: {s}/{t} ({r:.0f}%)")


if __name__ == "__main__":
    main()
