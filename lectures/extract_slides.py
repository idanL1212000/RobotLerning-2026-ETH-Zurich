#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pillow>=10.0",
#     "imageio-ffmpeg>=0.5",
# ]
# ///
"""Turn a lecture screen-recording into a slide deck PDF.

Drop the recording anywhere in `lectures/`, then give this script a week
number. It samples the video, discards the thousands of near-identical
frames, and writes the distinct slides to `lectures/weekNN/`.

    uv run lectures/extract_slides.py 3
    uv run lectures/extract_slides.py 3 --video lectures/robot_learning_w3.mp4
    uv run lectures/extract_slides.py 3 --threshold 1.5 --keep-frames

The work happens in two passes. The first samples small, cheap frames to
find *when* the slide changes; the second re-extracts only those moments at
full resolution, so the text in the PDF stays crisp.

On test recordings the gap between signal and noise was wide: a mouse
cursor and heavy compression moved 0.17% of the slide, while the faintest
real change - one bullet appearing mid-crossfade - moved 0.67%, and an
ordinary slide change moved 2-4%. Hence the 0.4% default.

That gap is what the defaults rely on, so the one thing that will break it
is a large moving overlay - a webcam inset of the lecturer, say. If a
recording has one, raise --threshold until the noise stops registering.

Slides with a video embedded in them need the second pass, --dedup, because
a playing video keeps crossing the change threshold and gets picked as a new
slide over and over. On lecture 1 that turned one slide into thirteen. The
duplicates sat under 1% apart while genuinely different slides were 3% or
more apart, with nothing in between, so 2% splits them cleanly.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

from PIL import Image, ImageChops, ImageStat

LECTURES_DIR = Path(__file__).resolve().parent
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".mpg", ".mpeg"}

# Frames are shrunk to this before being compared: big enough that one new
# bullet point registers, small enough that thousands of comparisons are free.
# Going finer than this measurably changed nothing on the test recordings.
SIGNATURE_SIZE = (256, 144)

# How far one pixel must move on a 0-255 grey scale to count as changed at
# all. Set well above the shimmer a lossy encoder leaves on a static slide.
PIXEL_DELTA = 32

# Width used for the throwaway detection pass. Detail doesn't matter here.
SCOUT_WIDTH = 480

# Slides are 16:9, so mapping each page to ~13.3in wide gives a PDF whose
# pages are the same shape and size as a widescreen PowerPoint slide.
PAGE_WIDTH_INCHES = 13.333


def find_ffmpeg() -> str:
    """Prefer a system ffmpeg, fall back to the one imageio-ffmpeg ships."""
    system = shutil.which("ffmpeg")
    if system:
        return system

    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def find_video(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            sys.exit(f"error: no such video: {explicit}")
        return explicit

    candidates = sorted(
        path
        for path in LECTURES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    if not candidates:
        sys.exit(
            f"error: no video in {LECTURES_DIR}\n"
            f"       drop the recording in there, or point at it with --video"
        )
    if len(candidates) > 1:
        listing = "\n".join(f"         {path.name}" for path in candidates)
        sys.exit(
            f"error: found several videos in {LECTURES_DIR}:\n{listing}\n"
            f"       choose one with --video"
        )
    return candidates[0]


def run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"error: ffmpeg failed\n{result.stderr.strip()}")


def scout_frames(ffmpeg: str, video: Path, out_dir: Path, fps: float) -> list[Path]:
    """Pass 1 - sample the whole video at low resolution."""
    run_ffmpeg(
        [
            ffmpeg,
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            f"fps={fps},scale={SCOUT_WIDTH}:-2:flags=bilinear",
            "-q:v",
            "5",
            str(out_dir / "scout_%06d.jpg"),
        ]
    )
    frames = sorted(out_dir.glob("scout_*.jpg"))
    if not frames:
        sys.exit(f"error: no frames came out of {video.name} - is it a valid video?")
    return frames


def signature(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L").resize(SIGNATURE_SIZE, Image.LANCZOS)


def difference(a: Image.Image, b: Image.Image) -> float:
    """How much of the slide changed, as a percentage of its area.

    Averaging the raw brightness difference instead would let a whole new
    slide of thin dark text register about as faintly as encoder noise. What
    separates the two cleanly is *area*: count the pixels that really moved
    and ignore how far they moved.
    """
    moved = ImageChops.difference(a, b).point(
        lambda value: 255 if value > PIXEL_DELTA else 0
    )
    return ImageStat.Stat(moved).mean[0] / 255 * 100


def group_into_runs(frames: list[Path], threshold: float) -> list[list[int]]:
    """Split the frames into runs that each show the same slide.

    Every frame is compared against the frame that *started* its run, not
    against its immediate neighbour. Slow crossfades drift a little per
    frame, which a neighbour-to-neighbour check would never notice.
    """
    runs: list[list[int]] = [[0]]
    anchor = signature(frames[0])

    for index in range(1, len(frames)):
        current = signature(frames[index])
        if difference(anchor, current) > threshold:
            runs.append([index])
            anchor = current
        else:
            runs[-1].append(index)

    return runs


def pick_slides(runs: list[list[int]], min_run: int) -> list[int]:
    """Keep one frame per run that held still long enough to be a real slide.

    Short runs are the in-between states of a fade or a build animation, and
    the middle of a long run is the frame least likely to be caught mid-wipe.
    """
    return [run[len(run) // 2] for run in runs if len(run) >= min_run]


def drop_repeats(
    frames: list[Path], chosen: list[int], threshold: float
) -> list[int]:
    """Collapse stretches where the same slide was picked more than once.

    A slide with a video embedded in it keeps crossing the change threshold
    as the video plays, so one slide can be picked a dozen times over. The
    later frame of any such pair replaces the earlier one, which also means a
    slide built up one bullet at a time collapses to its finished state
    rather than its emptiest.
    """
    survivors: list[int] = []
    previous: Image.Image | None = None

    for index in chosen:
        current = signature(frames[index])
        if previous is not None and difference(previous, current) < threshold:
            survivors[-1] = index
        else:
            survivors.append(index)
        previous = current

    return survivors


def extract_full_frames(
    ffmpeg: str,
    video: Path,
    timestamps: list[float],
    out_dir: Path,
    max_width: int,
) -> list[Path]:
    """Pass 2 - pull just the chosen moments, at full quality."""
    paths: list[Path] = []
    for number, timestamp in enumerate(timestamps, start=1):
        destination = out_dir / f"slide_{number:03d}.png"
        run_ffmpeg(
            [
                ffmpeg,
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                f"scale='min({max_width},iw)':-2:flags=lanczos",
                str(destination),
            ]
        )
        if destination.is_file():
            paths.append(destination)
        print(f"\r  extracted {number}/{len(timestamps)}", end="", flush=True)

    print()
    return paths


def as_pdf_pages(paths: list[Path]) -> Iterator[Image.Image]:
    """Open pages one at a time - a long lecture will not fit in memory."""
    for path in paths:
        with Image.open(path) as image:
            yield image.convert("RGB")


def write_pdf(paths: list[Path], destination: Path) -> None:
    pages = as_pdf_pages(paths)
    first = next(pages)
    resolution = max(1.0, first.width / PAGE_WIDTH_INCHES)
    first.save(
        destination,
        format="PDF",
        save_all=True,
        append_images=pages,
        resolution=resolution,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the slides from a lecture recording into a PDF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="example:  uv run lectures/extract_slides.py 3",
    )
    parser.add_argument("week", type=int, help="week number, e.g. 3 -> lectures/week03/")
    parser.add_argument(
        "--video",
        type=Path,
        help="path to the recording (default: the only video in lectures/)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="frames sampled per second while looking for slide changes (default: 1)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="percent of the slide that must change to count as a new slide "
        "(default: 0.4; lower catches subtler builds, higher ignores more)",
    )
    parser.add_argument(
        "--dedup",
        type=float,
        default=2.0,
        help="two kept slides less than this percent apart are treated as one "
        "(default: 2.0; 0 disables it)",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=2.0,
        help="how long a slide must stay on screen to be kept (default: 2)",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1920,
        help="cap on slide width in pixels in the PDF (default: 1920)",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="also leave the individual slide PNGs next to the PDF",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite the week folder if it already holds a PDF",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.week < 1:
        sys.exit("error: week number must be 1 or greater")

    week = f"week{args.week:02d}"
    week_dir = LECTURES_DIR / week
    pdf_path = week_dir / f"{week}_slides.pdf"

    if pdf_path.exists() and not args.force:
        sys.exit(f"error: {pdf_path} already exists - pass --force to replace it")

    ffmpeg = find_ffmpeg()
    video = find_video(args.video)
    min_run = max(1, round(args.min_seconds * args.fps))

    print(f"video    {video.name}")
    print(f"output   {week_dir}")

    week_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = week_dir / "slides"

    with tempfile.TemporaryDirectory(prefix="extract_slides_") as scratch:
        scratch_dir = Path(scratch)

        print(f"\nsampling at {args.fps} fps ...")
        scouts = scout_frames(ffmpeg, video, scratch_dir, args.fps)
        print(f"  {len(scouts)} frames to compare")

        print("\nlooking for slide changes ...")
        runs = group_into_runs(scouts, args.threshold)
        stable = pick_slides(runs, min_run)
        print(
            f"  {len(runs)} distinct screens, "
            f"{len(runs) - len(stable)} too brief to be slides"
        )

        kept = drop_repeats(scouts, stable, args.dedup)
        if len(kept) < len(stable):
            print(f"  {len(stable) - len(kept)} were the same slide seen again")

        if not kept:
            sys.exit(
                "error: no slide stayed on screen long enough\n"
                "       try a lower --min-seconds or a lower --threshold"
            )

        # Frame n of the sampled pass sits at (n - 1) / fps seconds in.
        timestamps = [index / args.fps for index in kept]

        print(f"\nre-extracting {len(kept)} slides at full quality ...")
        target_dir = frames_dir if args.keep_frames else scratch_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        # A re-run that finds fewer slides would otherwise leave the tail end
        # of the previous run's PNGs sitting there.
        for stale in target_dir.glob("slide_*.png"):
            stale.unlink()
        slides = extract_full_frames(
            ffmpeg, video, timestamps, target_dir, args.max_width
        )

        if not slides:
            sys.exit("error: could not re-extract any slide - try a different --fps")

        print("\nwriting PDF ...")
        write_pdf(slides, pdf_path)

    size_mb = pdf_path.stat().st_size / (1024 * 1024)
    print(f"\n{len(slides)} slides -> {pdf_path} ({size_mb:.1f} MB)")
    if args.keep_frames:
        print(f"{len(slides)} PNGs   -> {frames_dir}")


if __name__ == "__main__":
    main()
