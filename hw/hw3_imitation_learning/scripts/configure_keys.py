"""Interactive key configuration for the SO-100 teleop recorder.

Opens a small OpenCV window and walks you through each action one-by-one.
Press the key you want to assign to each action (look for our recommended keys).  The result is saved as
a JSON file that is loaded automatically throughout the homework.

Usage:
    python scripts/configure_keys.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Default output location (next to this script)
DEFAULT_KEYMAP_PATH = Path(__file__).resolve().parent.parent / "hw3" / "keymap.json"

# Every configurable action with a human-readable description and its default
# (raw_code, ascii_code) — used both as documentation and as fallback.
ACTIONS: list[tuple[str, str]] = [
    ("move_up", "Move EE up (+Z) (recommended: UP ARROW)"),
    ("move_down", "Move EE down (-Z) (recommended: DOWN ARROW)"),
    ("move_left", "Move EE left (-X) (recommended: LEFT ARROW)"),
    ("move_right", "Move EE right (+X) (recommended: RIGHT ARROW)"),
    ("move_forward", "Move EE forward (+Y) (recommended: W)"),
    ("move_backward", "Move EE backward (-Y) (recommended: S)"),
    ("rot_x_pos", "Rotate EE +X axis"),
    ("rot_x_neg", "Rotate EE -X axis"),
    ("rot_y_pos", "Rotate EE +Y axis"),
    ("rot_y_neg", "Rotate EE -Y axis"),
    ("rot_z_pos", "Rotate EE +Z axis"),
    ("rot_z_neg", "Rotate EE -Z axis"),
    ("gripper_open", "Open gripper (Jaw +)"),
    ("gripper_close", "Close gripper (Jaw -)"),
    ("reset", "Reset environment (recommended: R)"),
    ("record", "Toggle recording on/off (recommended: Space)"),
    ("end_episode", "End recorded episode (recommended: Enter)"),
    ("escape", "Quit session (recommended: ESC)"),
    ("goal_cube_red", "Select RED cube as goal (multicube only)"),
    ("goal_cube_green", "Select GREEN cube as goal (multicube only)"),
    ("goal_cube_blue", "Select BLUE cube as goal (multicube only)"),
]

WINDOW_W = 640
WINDOW_H = 200


def draw_prompt(
    action_name: str, description: str, index: int, total: int
) -> np.ndarray:
    """Create a prompt image asking the user to press a key."""
    img = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    cv2.putText(
        img,
        f"Key config  ({index + 1}/{total})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (200, 200, 200),
        2,
    )
    cv2.putText(
        img,
        f"Action: {description}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        f"({action_name})",
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
    )
    cv2.putText(
        img,
        ">>> Press the key you want to use <<<",
        (10, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    return img


def draw_assigned(action_name: str, raw: int, ascii_code: int) -> np.ndarray:
    """Show confirmation after a key was captured."""
    img = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    label = chr(ascii_code) if 32 <= ascii_code <= 126 else "<special>"
    cv2.putText(
        img,
        f"Assigned: {action_name}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        f"Key: '{label}'  (raw={raw}, ascii={ascii_code})",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 200, 255),
        2,
    )
    cv2.putText(
        img,
        "Moving to next action...",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (150, 150, 150),
        1,
    )
    return img


def run_configuration(output_path: Path) -> None:
    """Walk through all actions and capture keys interactively."""
    cv2.namedWindow("Key Configuration", cv2.WINDOW_AUTOSIZE)

    keymap: dict[str, dict] = {}
    total = len(ACTIONS)

    for i, (action_name, description) in enumerate(ACTIONS):
        prompt = draw_prompt(action_name, description, i, total)
        cv2.imshow("Key Configuration", prompt)

        # Wait for a key press (block until a key is pressed)
        while True:
            k_raw = cv2.waitKeyEx(0)  # 0 = wait forever
            if k_raw != -1:
                break

        k_ascii = k_raw & 0xFF
        label = chr(k_ascii) if 32 <= k_ascii <= 126 else f"raw:{k_raw}"

        keymap[action_name] = {
            "raw": k_raw,
            "ascii": k_ascii,
            "label": label,
            "description": description,
        }

        print(f"  [{i + 1}/{total}] {action_name:20s} -> '{label}' (raw={k_raw})")

        # Brief confirmation
        confirm = draw_assigned(action_name, k_raw, k_ascii)
        cv2.imshow("Key Configuration", confirm)
        cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(keymap, f, indent=2)
    print(f"\nKey mapping saved to {output_path}")
    print(
        "You can now run record_teleop_demos.py — it will load this mapping automatically."
    )


def load_keymap(path: Path | None = None) -> dict[str, int]:
    """Load a keymap JSON and return {action_name: raw_keycode}.

    If no file exists, returns an empty dict (caller should use defaults).
    """
    if path is None:
        path = DEFAULT_KEYMAP_PATH
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {action: entry["raw"] for action, entry in data.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configure keyboard mapping for SO-100 teleop."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_KEYMAP_PATH,
        help=f"Output JSON file (default: {DEFAULT_KEYMAP_PATH}).",
    )
    args = parser.parse_args()
    run_configuration(args.output)
