"""Batch-run PaddleOCR video deidentification across input media files.

This script finds all .mp4/.mov files in the input directory and runs
remove_text_from_video_paddleocr.py on each one, writing outputs to the
output directory with a "_deid" suffix.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run remove_text_from_video_paddleocr.py on every .mp4/.mov in a directory"
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Directory containing source videos (default: input)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for deidentified videos (default: output)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Stop after this many videos are processed (skipped files do not count).",
    )
    return parser.parse_args()


def iter_videos(input_dir: Path) -> list[Path]:
    videos: list[Path] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".mp4", ".mov"}:
            videos.append(path)
    return videos


def main() -> int:
    args = parse_args()

    if args.max_videos is not None and args.max_videos <= 0:
        print("--max-videos must be greater than 0 when provided.", file=sys.stderr)
        return 2

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory does not exist or is not a directory: {input_dir}", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    video_paths = iter_videos(input_dir)

    if not video_paths:
        print(f"No .mp4 or .mov files found in {input_dir}")
        return 0

    script_path = Path(__file__).resolve().parent / "remove_text_from_video_paddleocr.py"

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for input_video in video_paths:
        if args.max_videos is not None and processed_count >= args.max_videos:
            print(f"Reached --max-videos limit ({args.max_videos}). Stopping.")
            break

        output_video = output_dir / f"{input_video.stem}_deid{input_video.suffix.lower()}"
        if output_video.exists():
            print(f"Skipping {input_video.name}: target already exists ({output_video.name})")
            skipped_count += 1
            continue

        cmd = [
            "uv",
            "run",
            str(script_path),
            "-i",
            str(input_video),
            "-o",
            str(output_video),
            "--interval",
            "4",
            "--extra-keyframes",
            "2",
            "--target-bitrate",
            "1000k",
        ]
        print(f"Processing {input_video.name} -> {output_video.name}")
        result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent)
        if result.returncode != 0:
            failed_count += 1
            print(f"Failed ({result.returncode}): {input_video.name}")
            continue

        processed_count += 1

    print(
        "Done. "
        f"Processed: {processed_count}, Skipped: {skipped_count}, Failed: {failed_count}"
    )
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
