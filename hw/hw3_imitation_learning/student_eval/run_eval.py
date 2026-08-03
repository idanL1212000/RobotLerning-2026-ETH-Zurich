#!/usr/bin/env python3
"""Student-facing evaluation script for HW3 Exercises 1–3.

Usage
-----
    python run_eval.py --exercise 1 --checkpoint ex1.pt
    python run_eval.py --exercise 2 --checkpoint ex2.pt
    python run_eval.py --exercise 3 --checkpoint ex3.pt

The script expects your ``model.py`` at ``hw3/model.py`` relative to the
project root (i.e. the parent directory of ``student_eval/``).

This script imports the **compiled** ``eval_harness`` module (.so / .pyd)
which lives in the same directory.  Do NOT modify or replace it.

The script will:
  1. Load your model definition from ``./model.py``
  2. Load the trained weights from the checkpoint
  3. Run 100 headless simulation episodes (seed=42)
  4. Print your success rate and score
  5. Write a signed ``ex{N}_result.hwresult`` file

Upload the ``.hwresult`` file(s) to Gradescope.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EX_INFO = {
    1: {"name": "Single-Cube Obstacle (train)", "default_ckpt": "ex1.pt"},
    2: {"name": "Single-Cube Obstacle (adversarial)", "default_ckpt": "ex2.pt"},
    3: {"name": "Multicube Goal-Conditioned", "default_ckpt": "ex3.pt"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HW3 – Local Evaluation (Exercises 1–3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exercise",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Exercise number to evaluate (1, 2, or 3).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to your checkpoint (default: ./ex{N}.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the signed result file "
        "(default: ./ex{N}_result.hwresult)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42). Do NOT change for official submission.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode progress output.",
    )
    args = parser.parse_args()

    ex = args.exercise
    info = _EX_INFO[ex]

    # Defaults that depend on exercise number
    ckpt = args.checkpoint or info["default_ckpt"]
    output = args.output or f"ex{ex}_result.hwresult"

    # Resolve paths – model.py is always at <project_root>/hw3/model.py
    # project_root = parent of student_eval/ (where this script lives)
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "hw3" / "model.py"
    ckpt_path = Path(ckpt).resolve()
    output_path = Path(output).resolve()

    if not model_path.exists():
        print(
            f"ERROR: model.py not found at {model_path}\n"
            "       Expected hw3/model.py relative to the project root.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    # Import the compiled harness (the .so / .pyd in this directory)
    harness_dir = Path(__file__).resolve().parent
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))

    try:
        import eval_harness  # noqa: E402  — compiled .so
    except ImportError as e:
        print(
            "ERROR: Could not import eval_harness.\n"
            "Make sure the compiled eval_harness*.so (or .pyd on Windows)\n"
            "is in the same directory as this script.\n"
            f"\nDetails: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    print()
    print("=" * 55)
    print(f"  HW3 Exercise {ex} – {info['name']}")
    print("=" * 55)
    print(f"  Model      : {model_path}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Output     : {output_path}")
    print(f"  Episodes   : {args.num_episodes}")
    print(f"  Seed       : {args.seed}")
    print("=" * 55)

    eval_harness.run_eval(
        exercise=ex,
        model_py=str(model_path),
        checkpoint=str(ckpt_path),
        output_path=str(output_path),
        num_episodes=args.num_episodes,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
